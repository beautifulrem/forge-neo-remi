"""FLUX.3 Image Klein (Remi): detection, the DiT (references, flow-map target,
step caches, SLG hook, Spectrum seam), Diffusers folders and conversion, the
Qwen3-VL text window, on-the-fly quantization, LoRA key maps, the engine's
reference routing, per-block compile, workflow advice and the layout protocol.

Numerical parity with the official Diffusers ``Flux3Transformer2DModel`` was
checked separately (max |diff| <= 2e-6 in fp32 over t2i / references /
flow-map / fused-qkv); these tests pin Remi's behaviour on tiny models."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from backend import operations
from backend.args import dynamic_args

HIDDEN, HEADS, CTX = 256, 2, 64


def _tiny(*, flow_map=False, fused=False, depth=1, single=2, seed=0):
    from backend.nn.flux3 import Flux3

    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        dit = Flux3(hidden_size=HIDDEN, num_heads=HEADS, depth=depth, depth_single_blocks=single, fused_qkv=fused, flow_map=flow_map, context_in_dim=CTX).eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in dit.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    return dit


@pytest.fixture(autouse=True)
def _clean_dynamic_args(monkeypatch):
    monkeypatch.setattr(dynamic_args, "ref_latents", [])
    monkeypatch.setattr(dynamic_args, "flux3", False)
    yield


def _inputs(B=2, H=3, W=4, L=7, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, 128, H, W, generator=gen), torch.randn(B, L, CTX, generator=gen), torch.full((B,), 0.6)


# region detection and loading


def test_detection_model_list_and_engine():
    import huggingface_guess
    from huggingface_guess import model_list

    from backend.diffusion_engine.flux3 import Flux3 as Flux3Engine
    from backend.loader import possible_models

    for fused, flow_map in ((False, False), (True, True)):
        dit = _tiny(fused=fused, flow_map=flow_map)
        sd = {f"model.diffusion_model.{k}": v for k, v in dit.state_dict().items()}
        if fused:  # the BFL layout names the QK-norm weights "scale"
            sd = {(k[: -len("weight")] + "scale" if k.endswith(("query_norm.weight", "key_norm.weight")) else k): v for k, v in sd.items()}
        guess = huggingface_guess.guess(sd)
        assert isinstance(guess, model_list.Flux3Klein) and guess.huggingface_repo == "black-forest-labs/FLUX.3-Image-Klein"
        cfg = guess.unet_config
        assert (cfg["hidden_size"], cfg["num_heads"], cfg["depth"], cfg["depth_single_blocks"], cfg["context_in_dim"]) == (HIDDEN, HEADS, 1, 2, CTX)
        assert cfg["fused_qkv"] is fused and cfg["flow_map"] is flow_map
        assert guess.clip_target(sd) == {"qwen3vl_4b.transformer": "text_encoder"}
        # loads back, BFL "scale" keys included
        from backend.nn.flux3 import Flux3

        with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
            again = Flux3(**cfg)
        missing, unexpected = again.load_state_dict({k[len("model.diffusion_model.") :]: v for k, v in sd.items()}, strict=False)
        assert not missing and not unexpected
    assert Flux3Engine in possible_models and model_list.Flux3Klein in Flux3Engine.matched_guesses
    # mu = ln(6.93): the reference constant shift
    from backend.modules.k_prediction import PredictionFlux

    predictor = PredictionFlux(model_list.Flux3Klein({"image_model": "flux3"}))
    sigma = predictor.sigma(torch.tensor(0.5))
    assert abs(float(sigma) - 6.93 * 0.5 / (1 + 5.93 * 0.5)) < 1e-4


def test_hf_config_folder_ships_every_component():
    from diffusers import DiffusionPipeline

    from backend.loader import HF

    config = DiffusionPipeline.load_config(f"{HF}/black-forest-labs/FLUX.3-Image-Klein")
    assert config["transformer"] == ["diffusers", "Flux3Transformer2DModel"] and config["vae"] == ["diffusers", "AutoencoderKLFlux2"]
    assert config["text_encoder"] == ["transformers", "Qwen3VLModel"]
    with open(f"{HF}/black-forest-labs/FLUX.3-Image-Klein/scheduler/scheduler_config.json", encoding="utf-8") as file:
        scheduler = json.load(file)
    assert scheduler["shift"] == 6.93 and scheduler["use_dynamic_shifting"] is False


# endregion

# region the DiT


def test_forward_references_and_cond_cache():
    from backend.sampling.sampling_function import sampling_cleanup

    dit = _tiny()
    x, ctx, t = _inputs()
    with torch.inference_mode():
        plain = dit(x, t, ctx)
        assert plain.shape == x.shape and len(dit._cond_cache) == 1
        assert torch.equal(dit(x, t, ctx), plain) and len(dit._cond_cache) == 1  # cache hit
        dynamic_args.ref_latents = [torch.randn(1, 128, 2, 3), torch.randn(1, 128, 3, 2)]
        edited = dit(x, t, ctx)
        assert edited.shape == x.shape and not torch.allclose(edited, plain)
        assert len(dit._cond_cache) == 2  # references are part of the key
        dynamic_args.ref_latents = []
        # a weight change (a LoRA merged in place) invalidates the text cache
    with torch.no_grad():
        dit.txt_in.weight.add_(0.1)
    with torch.inference_mode():
        changed = dit(x, t, ctx)
        assert not torch.allclose(changed, plain)
        # online LoRA patches weights at call time: never cached
        dit.txt_in.weight_function = [lambda w: w]
        dit.clear_caches()
        dit(x, t, ctx)
        assert len(dit._cond_cache) == 0
        dit.txt_in.weight_function = []
        dit(x, t, ctx)
        assert len(dit._cond_cache) == 1
        sampling_cleanup(SimpleNamespace(list_controlnets=list, model=SimpleNamespace(diffusion_model=dit)))
        assert len(dit._cond_cache) == 0


def test_cached_conds_are_not_mutated_by_the_joint_blocks():
    dit = _tiny()
    x, ctx, t = _inputs()
    with torch.inference_mode():
        dit(x, t, ctx)
        (entry,) = dit._cond_cache.values()
        txt_before = entry[2].clone()
        dit(x, torch.full((2,), 0.2), ctx)
        assert torch.equal(entry[2], txt_before)


def test_flow_map_target_and_distilled_forward():
    from backend.nn.flux3 import flow_map_target

    sigmas = torch.tensor([1.0, 0.75, 0.4, 0.1, 0.0])
    assert float(flow_map_target(sigmas, torch.tensor([0.75]))) == pytest.approx(0.4)
    assert float(flow_map_target(sigmas, torch.tensor([0.1]))) == 0.0
    assert float(flow_map_target(sigmas, torch.tensor([0.0]))) == 0.0
    assert float(flow_map_target(None, torch.tensor([0.5]))) == 0.0

    dit = _tiny(flow_map=True)
    x, ctx, t = _inputs()
    t = torch.full((2,), 0.75)
    with torch.inference_mode():
        a = dit(x, t, ctx, transformer_options={"sampling_sigmas": sigmas})
        b = dit(x, t, ctx, transformer_options={"sampling_sigmas": torch.tensor([1.0, 0.75, 0.0])})
    assert not torch.allclose(a, b)  # the target time is embedded
    assert len(dit._cond_cache) == 0  # a flow map modulates the text blocks with each target


def test_slg_replace_hook_and_spectrum_seam():
    from backend.nn.dit_registry import get_dit_capability, match_spectrum_capability, resolve_spectrum_capture_module

    dit = _tiny(single=3)
    x, ctx, t = _inputs()
    calls = []

    def skip(args, extra):
        calls.append(sorted(args))
        return {"img": args["img"]}

    with torch.inference_mode():
        full = dit(x, t, ctx)
        skipped = dit(x, t, ctx, transformer_options={"patches_replace": {"dit": {("single_block", 1): skip}}})
    assert calls and "spans" in calls[0] and not torch.allclose(full, skipped)

    assert match_spectrum_capability(dit) is get_dit_capability("flux3")
    assert resolve_spectrum_capture_module(dit) is dit.final_input
    captured = {}
    handle = dit.final_input.register_forward_pre_hook(lambda module, args: captured.setdefault("feature", args[0].clone()))
    with torch.inference_mode():
        out = dit(x, t, ctx)
        again = dit.compute_final(captured["feature"], t, tuple(x.shape))
    handle.remove()
    torch.testing.assert_close(again, out, rtol=0, atol=1e-6)


def test_fused_qkv_matches_split_projections():
    dit = _tiny()
    fused = _tiny(fused=True)
    sd = dit.state_dict()
    for k in [k for k in sd if k.endswith("q_proj.weight")]:
        stem = k[: -len("q_proj.weight")]
        sd[stem + "qkv.weight"] = torch.cat([sd.pop(stem + "q_proj.weight"), sd.pop(stem + "k_proj.weight"), sd.pop(stem + "v_proj.weight")])
    fused.load_state_dict(sd)
    x, ctx, t = _inputs()
    with torch.inference_mode():
        torch.testing.assert_close(fused(x, t, ctx), dit(x, t, ctx), rtol=1e-5, atol=1e-5)


# endregion

# region Diffusers


def test_gguf_q8_checkpoint_runs_close_to_dense(monkeypatch):
    """A city96-style GGUF (Q8_0 block Linears, other tensors dense) loads into
    GGUF-native layers and runs close to the dense model."""

    import numpy as np
    from gguf import GGMLQuantizationType, quants

    from backend import memory_management, state_dict
    from backend.nn.flux3 import Flux3
    from backend.operations_gguf import ParameterGGUF

    monkeypatch.setattr(memory_management, "supports_nvfp4_compute", lambda device=None: False)
    dense = _tiny()
    sd = {}
    for k, v in dense.state_dict().items():
        if v.ndim == 2 and v.shape[1] % 32 == 0 and k.startswith(("single_blocks.", "txt_mode_blocks.", "content_mode_blocks.")):
            raw = torch.from_numpy(quants.quantize(v.float().numpy().astype(np.float32), GGMLQuantizationType.Q8_0))
            sd[k] = ParameterGGUF(raw, tensor_type=GGMLQuantizationType.Q8_0, tensor_shape=v.shape)
        else:
            sd[k] = v.clone()
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True, extra_dtype="gguf"):
        gguf = Flux3(hidden_size=HIDDEN, num_heads=HEADS, depth=1, depth_single_blocks=2, context_in_dim=CTX).eval()
    state_dict.load_state_dict(gguf, sd)
    memory_management.bake_gguf_model(gguf)
    assert type(gguf.single_blocks[0].q_proj.weight).__name__ == "ParameterGGUF"
    x, ctx, t = _inputs()
    with torch.inference_mode():
        expected, actual = dense(x, t, ctx), gguf(x, t, ctx).float()
    assert torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0) > 0.999


def _diffusers_name_map(native_sd):
    from backend.nn.flux3_diffusers import flux3_lora_key_map

    key_map = flux3_lora_key_map({f"diffusion_model.{k}": v for k, v in native_sd.items()})
    names = {}
    for lora_key, target in key_map.items():
        if isinstance(target, str) and lora_key.startswith("transformer."):
            native = target[len("diffusion_model.") : -len(".weight")]
            bare = lora_key[len("transformer.") :]
            if bare != native:
                names[native] = bare
    return names


def _to_diffusers(native_sd):
    names = _diffusers_name_map(native_sd)
    out = {}
    for k, v in native_sd.items():
        module, param = k.rsplit(".", 1)
        if module == "final_layer.image.adaLN_modulation.1":
            shift, scale = v.chunk(2)
            v = torch.cat([scale, shift])
        out[f"{names.get(module, module)}.{param}"] = v
    return out


def test_diffusers_conversion_round_trip_and_sidecars():
    from backend.nn.flux3_diffusers import convert_diffusers_flux3_transformer, is_diffusers_flux3_transformer, is_native_flux3_transformer

    native = _tiny(flow_map=True).state_dict()
    diff = _to_diffusers(native)
    assert "context_embedder.weight" in diff and "norm_out.linear.weight" in diff and "target_timestep_embedder.linear_1.weight" in diff
    assert "single_transformer_blocks.0.attn.to_out.0.weight" in diff and "early_stream_modulations.txt.linear.weight" in diff
    assert is_diffusers_flux3_transformer(diff) and not is_native_flux3_transformer(diff) and is_native_flux3_transformer(native)
    back = convert_diffusers_flux3_transformer(diff)
    assert set(back) == set(native)
    for k in native:
        assert torch.equal(back[k], native[k]), k
    # quantized sidecars follow their weight
    sidecars = convert_diffusers_flux3_transformer({"transformer.single_transformer_blocks.0.attn.to_q.weight_scale": torch.ones(1), "single_transformer_blocks.0.ff.linear_in.comfy_quant": torch.zeros(3, dtype=torch.uint8)}, output_prefix="p.")
    assert set(sidecars) == {"p.single_blocks.0.q_proj.weight_scale", "p.single_blocks.0.mlp_in.comfy_quant"}
    # the generic Diffusers single-file hook
    from modules_forge.packages.comfy.utils import convert_diffusers_mmdit

    converted = convert_diffusers_mmdit(dict(diff), "")
    assert converted is not None and set(converted) == set(native)


def test_diffusers_pipeline_folder_loads_through_split_state_dict(tmp_path):
    from backend.loader import split_state_dict
    from backend.nn.dit_registry import diffusers_checkpoint_info, is_diffusers_checkpoint_dir

    native = _tiny().state_dict()
    diff = _to_diffusers(native)
    root = tmp_path / "flux-3-image-klein"
    for sub in ("transformer", "text_encoder", "vae"):
        (root / sub).mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "Flux3ModularPipeline", "transformer": ["diffusers", "Flux3Transformer2DModel"]}))
    items = sorted(diff.items())
    half = len(items) // 2
    for i, part in enumerate((dict(items[:half]), dict(items[half:])), 1):
        safetensors.torch.save_file({k: v.contiguous() for k, v in part.items()}, str(root / "transformer" / f"diffusion_pytorch_model-0000{i}-of-00002.safetensors"))
    weight_map = {k: f"diffusion_pytorch_model-0000{1 if i < half else 2}-of-00002.safetensors" for i, (k, _) in enumerate(items)}
    (root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    safetensors.torch.save_file({"model.language_model.embed_tokens.weight": torch.zeros(4, 8), "lm_head.weight": torch.zeros(4, 8)}, str(root / "text_encoder" / "model.safetensors"))
    safetensors.torch.save_file({"decoder.conv_in.weight": torch.zeros(4, 32, 3, 3), "bn.running_mean": torch.zeros(128)}, str(root / "vae" / "diffusion_pytorch_model.safetensors"))

    assert is_diffusers_checkpoint_dir(root) and is_diffusers_checkpoint_dir(root / "transformer")
    arch, primary = diffusers_checkpoint_info(root)
    assert arch == "flux-3/diffusers" and primary.endswith("00001-of-00002.safetensors")

    state_dicts, guess = split_state_dict(str(root))
    assert "FLUX.3" in guess.huggingface_repo and guess.unet_config["depth_single_blocks"] == 2
    assert set(state_dicts["transformer"]) == set(native)
    assert "model.language_model.embed_tokens.weight" in state_dicts["text_encoder"]
    assert "bn.running_mean" in state_dicts["vae"]


# endregion

# region text window


class _StubQwen3VL(torch.nn.Module):
    """Stands in for Qwen3-VL: records the call and returns per-layer states."""

    num_layers = 36

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(151936, 4)
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, embeds=None, intermediate_output=None, position_ids=None, **kwargs):
        self.calls.append({"mask": attention_mask, "position_ids": position_ids, "layers": intermediate_output})
        B, L, _ = embeds.shape
        layers = torch.arange(len(intermediate_output), dtype=torch.float32).view(1, -1, 1, 1).expand(B, -1, L, 4)
        return embeds, layers + embeds.unsqueeze(1)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import Qwen2Tokenizer

    from backend.loader import HF

    return Qwen2Tokenizer.from_pretrained(f"{HF}/black-forest-labs/FLUX.3-Image-Klein/tokenizer")


def test_text_window_padding_positions_and_layers(qwen_tokenizer):
    from backend.text_processing.flux3_engine import HIDDEN_LAYERS, PAD_TOKEN, Flux3TextEngine, padded_length, padding_position_ids

    assert [padded_length(n) for n in (1, 80, 81, 8191, 9000)] == [80, 80, 160, 8192, 8192]
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    assert padding_position_ids(mask).tolist() == [[0, 1, 2, 1, 1]]

    stub = _StubQwen3VL()
    engine = Flux3TextEngine(stub, qwen_tokenizer)
    (window,) = engine.tokenize_padded("a red fox")
    ids = [t for t, _ in window]
    real = engine.tokenize("a red fox")
    assert ids[: len(real)] == real and set(ids[len(real) :]) == {PAD_TOKEN} and len(ids) % 80 == 0
    assert qwen_tokenizer.decode(real).startswith("<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n")

    (cond,) = engine(["a red fox"])
    assert cond.shape == (len(ids), len(HIDDEN_LAYERS) * 4)
    call = stub.calls[-1]
    assert call["layers"] == list(HIDDEN_LAYERS)
    assert call["mask"][0, : len(real)].all() and not call["mask"][0, len(real) :].any()
    assert call["position_ids"][0, len(real) :].eq(1).all() and call["position_ids"][0, : len(real)].tolist() == list(range(len(real)))
    # layer-major per token: channel block i holds layer i's state
    torch.testing.assert_close(cond[0, 4:8] - cond[0, :4], torch.ones(4))


# endregion

# region quantization, LoRA, compile, advice


def test_on_the_fly_quantization_follows_the_registry():
    from backend.nn.anima_quant import keep_sensitive_weights_precise, quantize_int8_convrot_state_dict, scale_fp8_state_dict, should_keep_sensitive_weights_precise, should_quantize_on_load

    assert should_quantize_on_load("scaled_fp8", "Flux3Transformer2DModel", None, False, torch.bfloat16)
    assert should_keep_sensitive_weights_precise("Flux3Transformer2DModel", torch.float8_e4m3fn, torch.bfloat16)

    sd = {k: v.to(torch.bfloat16) for k, v in _tiny().state_dict().items()}
    fp8 = dict(sd)
    n = scale_fp8_state_dict(fp8, fp8_matmul=False)
    block_linears = [k for k in sd if k.startswith(("content_mode_blocks.", "txt_mode_blocks.", "single_blocks.")) and sd[k].ndim == 2]
    assert n == len(block_linears) > 0
    assert fp8["single_blocks.0.q_proj.weight"].dtype == torch.float8_e4m3fn and fp8["single_blocks.0.q_proj.weight_scale"].shape == (HIDDEN, 1)
    assert json.loads(bytes(fp8["single_blocks.0.q_proj.comfy_quant"].tolist()))["format"] == "float8_e4m3fn_rowwise"
    for sensitive in ("txt_in.weight", "emb_in.image.weight", "early_stream_modulations.txt.lin.weight", "final_layer.image.linear.weight", "time_in.in_layer.weight"):
        assert fp8[sensitive].dtype == torch.bfloat16, sensitive

    int8 = dict(sd)
    assert quantize_int8_convrot_state_dict(int8) == len([k for k in block_linears if k.endswith(("_proj.weight", "attn_out.weight", "mlp_in.weight", "mlp_out.weight"))])
    assert int8["single_blocks.1.mlp_out.weight"].dtype == torch.int8

    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float8_e4m3fn, manual_cast_enabled=True):
        from backend.nn.flux3 import Flux3

        fp8_dit = Flux3(hidden_size=HIDDEN, num_heads=HEADS, depth=1, depth_single_blocks=1, context_in_dim=CTX)
    assert keep_sensitive_weights_precise(fp8_dit, torch.bfloat16) > 0
    assert fp8_dit.txt_in.weight.dtype == torch.bfloat16 and fp8_dit.final_layer["image"].linear.weight.dtype == torch.bfloat16
    assert fp8_dit.single_blocks[0].norm.query_norm.weight.dtype == torch.bfloat16
    assert fp8_dit.single_blocks[0].q_proj.weight.dtype == torch.float8_e4m3fn


def test_lora_keys_from_every_trainer_and_fused_qkv_slices():
    from modules_forge.packages.comfy.lora import load_lora, model_lora_keys_unet

    def fake(dit):
        return SimpleNamespace(state_dict=lambda: {f"diffusion_model.{k}": v for k, v in dit.state_dict().items()}, config=SimpleNamespace(huggingface_repo="black-forest-labs/FLUX.3-Image-Klein"), diffusion_model=SimpleNamespace(config={"image_model": "flux3"}))

    def pair(prefix, out_d, in_d):
        return {f"{prefix}.lora_A.weight": torch.randn(4, in_d), f"{prefix}.lora_B.weight": torch.randn(out_d, 4)}

    key_map = model_lora_keys_unet(fake(_tiny()), {})
    lora = {
        **pair("transformer.single_transformer_blocks.0.attn.to_q", HIDDEN, HIDDEN),  # Diffusers / PEFT
        **pair("transformer.mode_blocks.image.0.ff.linear_in", 6 * HIDDEN, HIDDEN),
        **pair("diffusion_model.single_blocks.1.mlp_out", HIDDEN, 3 * HIDDEN),  # ComfyUI / BFL native
        **pair("transformer.context_embedder", HIDDEN, CTX),
    }
    patches, rest = load_lora(lora, key_map)
    assert rest == {} and {"diffusion_model.single_blocks.0.q_proj.weight", "diffusion_model.content_mode_blocks.image.0.mlp_in.weight", "diffusion_model.single_blocks.1.mlp_out.weight", "diffusion_model.txt_in.weight"} <= set(patches)

    fused_map = model_lora_keys_unet(fake(_tiny(fused=True)), {})
    patches, rest = load_lora({**pair("transformer.single_transformer_blocks.0.attn.to_k", HIDDEN, HIDDEN)}, fused_map)
    assert rest == {} and next(iter(patches)) == ("diffusion_model.single_blocks.0.qkv.weight", (0, HIDDEN, HIDDEN))


def test_per_block_compile_installs_only_on_flux3():
    import importlib.util
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1] / "extensions-builtin" / "sd_forge_compile"
    sys.path.insert(0, str(root))
    from anima_block_compile import AnimaBlockCompileManager, _block_list

    spec = importlib.util.spec_from_file_location("compile_script_flux3_test", root / "scripts" / "compile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "FLUX.3 per-block" in module.PRESETS and "FLUX.3 per-block" in module._PER_BLOCK_PRESETS

    dit = _tiny()
    assert _block_list(dit) is dit.single_blocks
    kmodel = SimpleNamespace(diffusion_model=dit, apply_model=lambda value: value)
    script = module.TorchCompileForForge.__new__(module.TorchCompileForForge)
    p = SimpleNamespace(sd_model=SimpleNamespace(is_flux3_engine=False, forge_objects=SimpleNamespace(unet=SimpleNamespace(model=kmodel))), extra_generation_params={})
    script.process_batch(p, "FLUX.3 per-block")
    assert p.extra_generation_params["Torch compile"] == "FLUX.3 per-block ignored: non-FLUX.3 model"
    p.sd_model.is_flux3_engine = True
    script.process_batch(p, "FLUX.3 per-block")
    assert AnimaBlockCompileManager.is_installed(kmodel)
    AnimaBlockCompileManager.remove(kmodel)


def test_workflow_advice(monkeypatch):
    from backend.nn.dit_registry import collect_workflow_advice

    monkeypatch.setattr(dynamic_args, "flux3", True)
    monkeypatch.setattr(dynamic_args, "moss", False)
    monkeypatch.setattr(dynamic_args, "krea2", False)
    opts = SimpleNamespace(flux3_do_reference=True)
    long_prompt = "Full-color outdoor photograph of a red fox standing in fresh snow at the edge of a pine forest at dawn"

    distilled = SimpleNamespace(prompt=long_prompt, steps=16, cfg_scale=3.5, sd_model=SimpleNamespace(is_flux3_distilled=True, ref_latents=[], ini_latent=None))
    assert any("step-distilled" in note for note in collect_workflow_advice(distilled, opts=opts))
    base = SimpleNamespace(prompt=long_prompt, steps=16, cfg_scale=1.0, sd_model=SimpleNamespace(is_flux3_distilled=False, ref_latents=[], ini_latent=None))
    assert any("Base" in note for note in collect_workflow_advice(base, opts=opts))
    good = SimpleNamespace(prompt=long_prompt, steps=50, cfg_scale=3.5, sd_model=SimpleNamespace(is_flux3_distilled=False, ref_latents=[], ini_latent=None))
    assert collect_workflow_advice(good, opts=opts) == []
    short = SimpleNamespace(prompt="a fox", steps=16, cfg_scale=1.0, sd_model=SimpleNamespace(is_flux3_distilled=True, ref_latents=[object()] * 11, ini_latent=None))
    notes = collect_workflow_advice(short, opts=SimpleNamespace(flux3_do_reference=False))
    assert any("long, detailed captions" in n for n in notes) and any("at most 10" in n for n in notes) and any("Enable Reference is off" in n for n in notes)


def test_engine_routes_references_like_an_edit_model(monkeypatch):
    from backend import memory_management
    from backend.diffusion_engine import flux3 as engine_module
    from modules.prompt_parser import SdConditioning

    monkeypatch.setattr(memory_management, "load_model_gpu", lambda *a, **k: None)
    engine = engine_module.Flux3.__new__(engine_module.Flux3)
    engine.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=object()))
    seen = []
    engine.text_processing_engine_qwen = lambda prompt: seen.append(list(prompt)) or ["cond"]
    engine.ini_latent, engine.ref_latents = "init", ["stitched"]

    monkeypatch.setattr(engine_module, "opts", SimpleNamespace(flux3_do_reference=True))
    engine.get_learned_conditioning(SdConditioning(["make it snowy"]))
    assert dynamic_args.ref_latents == ["init", "stitched"] and engine.ini_latent is None
    engine.get_learned_conditioning(SdConditioning([""], is_negative_prompt=True))
    assert dynamic_args.ref_latents == ["init", "stitched"]  # the negative prompt keeps them

    monkeypatch.setattr(engine_module, "opts", SimpleNamespace(flux3_do_reference=False))
    engine.get_learned_conditioning(SdConditioning(["a fox"]))
    assert dynamic_args.ref_latents == []
    assert dynamic_args.can_reference() is False
    monkeypatch.setattr(dynamic_args, "flux3", True)
    assert dynamic_args.can_reference() is True


# endregion

# region layout protocol


def test_layout_protocol_split_reconcile_and_problems():
    from backend.nn.flux3_layout import boxes_to_canvas, layout_problems, reconcile_boxes, split_caption

    rows = [{"id": "fox_1", "bbox": [500, 100, 900, 600], "desc": "A red fox."}]
    prompt = f"Full-color photograph of a red fox <fox_1> in snow. {json.dumps(rows)}"
    caption, parsed = split_caption(prompt)
    assert caption.endswith("in snow.") and parsed == rows and layout_problems(prompt) == []
    assert layout_problems("plain caption without a table") == []
    bad = 'A fox. [{"id": "fox_1", "bbox": [0.1, 0.2, 0.5, 0.6], "desc": "x"}, {"id": "fox_1", "bbox": [10, 10, 20, 20], "desc": "y"}]'
    problems = layout_problems(bad)
    assert any("0-1000" in p for p in problems) and any("duplicate" in p for p in problems) and any("not cited" in p for p in problems)

    locked = boxes_to_canvas([{"id": "cat_1", "desc": "a cat", "bbox": [0.5, 0.1, 0.9, 0.6]}])
    assert locked == [{"id": "cat_1", "bbox": [500, 100, 900, 600], "desc": "a cat"}]
    answer = 'A tabby cat <kitty_1> on a rug <rug_1>. [{"id": "kitty_1", "bbox": [505, 100, 900, 600], "desc": "A tabby cat."}, {"id": "rug_1", "bbox": [800, 0, 1000, 1000], "desc": "A wool rug."}]'
    out_caption, out_rows = split_caption(reconcile_boxes(answer, locked))
    assert "<cat_1>" in out_caption and "<kitty_1>" not in out_caption
    assert out_rows[0] == {"id": "cat_1", "bbox": [500, 100, 900, 600], "desc": "A tabby cat."} and out_rows[1]["id"] == "rug_1"


def test_generate_uses_the_flux3_upsampler_prompt_when_flux3_is_loaded(monkeypatch):
    import remi.prompt_tools as pt
    from backend.nn.flux3_layout import SYSTEM_PROMPTS

    captured = {}
    monkeypatch.setattr(pt, "generate_prompt", lambda config, description, **kwargs: captured.update(kwargs) or "caption")
    tools = pt.PromptTools.__new__(pt.PromptTools)
    monkeypatch.setattr(pt, "registry", SimpleNamespace(tools_enabled=True))
    monkeypatch.setattr(pt.opt, "option", lambda name: "")
    monkeypatch.setattr(type(tools), "translation_config", lambda self: None)
    monkeypatch.setattr(dynamic_args, "flux3", True)
    tools.generate("a fox")
    assert captured["instructions"] == SYSTEM_PROMPTS["t2i"]
    tools.generate("blurry", negative=True)
    assert captured["instructions"] == ""
    monkeypatch.setattr(dynamic_args, "flux3", False)
    tools.generate("a fox")
    assert captured["instructions"] == ""


# endregion
