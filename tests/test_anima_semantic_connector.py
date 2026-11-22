"""Anima-3.8B: the Semantic Connector v2 and its Qwen3.5 4B text encoder, without checkpoints.

The manifests in ``tests/fixtures`` are the tensor names and shapes of the release files
(``Anima-3.8B-v1.1.safetensors``, ``qwen35_4b.safetensors``): the modules must load them with
nothing missing or left over, since the loader fills missing weights with zeros and a zeroed
connector would silently do nothing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from backend.loader import process_anima
from backend.nn.anima import Anima, LLMAdapter
from backend.nn.anima_connector import SemanticConnector, connector_config
from backend.nn.anima_qwen35 import AnimaQwen35
from backend.operations import using_forge_operations
from backend.sampling.condition import compile_conditions
from test_anima_text_engine_compat import load_engine

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SMALL = dict(source_dim=96, num_queries=8, resampler_blocks=2, resampler_dim=256, resampler_heads=4, mlp_hidden_dim=512)


def manifest(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def meta_shapes(tensors: dict) -> dict:
    return {key: torch.empty(shape, device="meta") for key, shape in tensors.items()}


def test_connector_config_is_read_from_the_release_weights():
    release = manifest("anima_3_8b_connector_manifest.json")
    config = connector_config(meta_shapes(release["tensors"]), metadata=release["metadata"])

    assert config == {
        "adapter_blocks": 6,
        "model_dim": 1024,
        "adapter_heads": 16,
        "source_dim": 2560,
        "layers": (7, 15, 23, 31),
        "num_queries": 64,
        "resampler_blocks": 6,
        "resampler_dim": 2048,
        "resampler_heads": 16,
        "mlp_hidden_dim": 5632,
    }
    assert connector_config({"blocks.0.mlp.layer1.weight": torch.empty(1)}) is None


def test_the_3_8b_dit_has_exactly_the_release_adapter_and_connector_tensors():
    release = manifest("anima_3_8b_connector_manifest.json")
    config = connector_config(meta_shapes(release["tensors"]), metadata=release["metadata"])
    with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
        model = Anima(16, 16, 2, 1, model_channels=2048, crossattn_emb_channels=1024, adaln_lora_dim=256, num_blocks=2, num_heads=16, semantic_connector=config)

    ours = {key: list(value.shape) for key, value in model.state_dict().items() if key.startswith(("anima_v2_connector.", "llm_adapter."))}
    assert ours == release["tensors"]


def test_the_qwen35_encoder_has_exactly_the_release_tensors():
    release = manifest("anima_qwen35_4b_manifest.json")
    with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
        model = AnimaQwen35()

    assert {key: list(value.shape) for key, value in model.state_dict().items()} == release["tensors"]
    assert model.layers[31].mlp is None and model.layers[30].mlp is not None


@pytest.fixture
def connector():
    torch.manual_seed(0)
    adapter = LLMAdapter().eval()
    model = SemanticConnector(**SMALL).eval()
    for module in (adapter, model):
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
    return adapter, model


def inputs():
    generator = torch.Generator().manual_seed(1)
    source = torch.randn(1, 13, 1024, generator=generator)
    ids = torch.randint(0, 32128, (1, 9), generator=generator)
    semantic = torch.randn(1, 4, 11, 96, generator=generator)
    return source, ids, semantic


@torch.inference_mode()
def test_the_connector_depends_on_the_timestep_and_reduces_to_the_adapter(connector):
    adapter, model = connector
    source, ids, semantic = inputs()

    early = model(adapter, source, ids, semantic, torch.tensor([0.9]))
    late = model(adapter, source, ids, semantic, torch.tensor([0.1]))
    native = model(adapter, source, ids, semantic, torch.tensor([0.5]), include_inserted_blocks=False)

    assert early.shape == (1, 9, 1024)
    assert (early - late).abs().max() > 1e-3
    assert torch.allclose(native, adapter(source, ids), atol=1e-6)
    assert (early - native).abs().max() > 1e-3


@torch.inference_mode()
def test_semantic_context_replaces_the_encoded_context_with_weights_padding_and_negpip(connector):
    adapter, model = connector
    dit = SimpleNamespace(llm_adapter=adapter, anima_v2_connector=model)
    source, ids, semantic = inputs()
    context = torch.zeros(1, 512, 1024)
    weights = torch.linspace(0.5, 1.5, 9)[None]
    negpip = torch.ones(1, 512, 1)
    negpip[0, 3] = -1
    kwargs = {"anima_source": source, "anima_ids": ids, "anima_weights": weights, "anima_semantic": semantic, "anima_semantic_mask": torch.ones(1, 11, dtype=torch.long)}

    got = Anima.semantic_context(dit, context, torch.tensor([0.4]), {**kwargs, "c_negpip_mask": negpip})
    want = model(adapter, source, ids, semantic, torch.tensor([0.4])) * weights[..., None]

    assert got.shape == context.shape
    assert torch.allclose(got[:, :9] * negpip[:, :9], want, atol=1e-6)
    assert not got[:, 9:].any()
    assert Anima.semantic_context(dit, context, torch.tensor([0.4]), {}) is context
    with pytest.raises(RuntimeError, match="semantic connector"):
        Anima.semantic_context(SimpleNamespace(anima_v2_connector=None), context, torch.tensor([0.4]), kwargs)


@torch.inference_mode()
def test_semantic_context_computes_in_the_dit_dtype_with_a_wider_embedding_table(connector):
    """On CUDA the weights are not cast to the input: a float32 T5 embedding (as stored next to
    bfloat16 weights) must not leave the connector's Linears a float32 input."""
    adapter, model = connector
    adapter, model = adapter.to(torch.bfloat16), model.to(torch.bfloat16)
    adapter.embed.float()
    dit = SimpleNamespace(llm_adapter=adapter, anima_v2_connector=model)
    source, ids, semantic = inputs()
    kwargs = {"anima_source": source.bfloat16(), "anima_ids": ids, "anima_weights": torch.ones(1, 9, dtype=torch.bfloat16), "anima_semantic": semantic.bfloat16()}

    got = Anima.semantic_context(dit, torch.zeros(1, 512, 1024, dtype=torch.bfloat16), torch.tensor([0.4]), kwargs)

    assert got.dtype == torch.bfloat16 and got[:, :9].abs().sum() > 0


def test_conditions_without_a_pooled_vector_pass_their_extra_tensors_to_the_model():
    cond = {"crossattn": torch.zeros(2, 512, 4), "anima_ids": torch.zeros(2, 9, dtype=torch.long), "anima_semantic": torch.zeros(2, 4, 11, 8)}

    (compiled,) = compile_conditions(cond)

    assert set(compiled["model_conds"]) == {"c_crossattn", "anima_ids", "anima_semantic"}
    assert compiled["model_conds"]["anima_ids"].cond is cond["anima_ids"]


def test_the_3_8b_adapter_is_copied_not_moved_to_the_text_encoder():
    dit = {"llm_adapter.embed.weight": torch.ones(2), "blocks.0.x": torch.ones(1)}
    enc = {}

    process_anima(dit, enc, keep_in_dit=True)

    assert set(dit) == {"llm_adapter.embed.weight", "blocks.0.x"} and set(enc) == {"llm_adapter.embed.weight"}


def test_qwen35_reads_the_prompt_without_emphasis_syntax_and_negatives_only_when_enabled():
    namespace, opts = load_engine()
    engine = namespace["Qwen06Engine"](object(), None, None, semantic_encoder=object(), semantic_layers=(7, 15, 23, 31))

    assert engine.semantic_text("a (red hair:1.2), [blue] BREAK eyes") == "a red hair, blue eyes"
    assert engine.uses_semantic() and not engine.uses_semantic(negative=True)
    opts.anima_semantic_negative = True
    assert engine.uses_semantic(negative=True)
    assert not namespace["Qwen06Engine"](object(), None, None).uses_semantic()


@torch.inference_mode()
def test_connector_static_cache_matches_uncached_and_reuses_kv_projections(connector):
    adapter, model = connector
    source, ids, semantic = inputs()
    mask = torch.ones(1, 11, dtype=torch.long)

    # Baseline with cache disabled via an online weight_function hook
    model.semantic_resampler.blocks[0].cross_attention.k_proj.weight_function = [lambda w: w]
    try:
        ref_t1 = model(adapter, source, ids, semantic, torch.tensor([0.9]), mask)
        ref_t2 = model(adapter, source, ids, semantic, torch.tensor([0.3]), mask)
    finally:
        del model.semantic_resampler.blocks[0].cross_attention.k_proj.weight_function

    resampler_norm_calls = 0
    anchor_norm_calls = 0
    orig_resampler_norm = model.semantic_resampler.blocks[0].source_norm.forward
    orig_anchor_norm = model.quality_anchor.source_norms[0].forward

    def count_resampler_norm(x):
        nonlocal resampler_norm_calls
        resampler_norm_calls += 1
        return orig_resampler_norm(x)

    def count_anchor_norm(x):
        nonlocal anchor_norm_calls
        anchor_norm_calls += 1
        return orig_anchor_norm(x)

    model.semantic_resampler.blocks[0].source_norm.forward = count_resampler_norm
    model.quality_anchor.source_norms[0].forward = count_anchor_norm
    try:
        out_t1 = model(adapter, source, ids, semantic, torch.tensor([0.9]), mask)
        out_t2 = model(adapter, source, ids, semantic, torch.tensor([0.3]), mask)
        out_t3 = model(adapter, source, ids, semantic, torch.tensor([0.1]), mask)
        assert resampler_norm_calls == 1 and anchor_norm_calls == 1
        assert torch.allclose(out_t1, ref_t1, atol=1e-6)
        assert torch.allclose(out_t2, ref_t2, atol=1e-6)
        assert (out_t1 - out_t3).abs().max() > 1e-3

        # Mutating a cached weight invalidates the cache immediately
        with torch.inference_mode(False), torch.no_grad():
            model.quality_anchor.layer_mix_logits[:, 0].add_(2.0)
        out_mutated = model(adapter, source, ids, semantic, torch.tensor([0.9]), mask)
        assert resampler_norm_calls == 2 and anchor_norm_calls == 2
        assert (out_mutated - out_t1).abs().max() > 1e-5

        # Moving the module (_apply) clears the cache
        model.cpu()
        assert len(model._static_cache) == 0
    finally:
        model.semantic_resampler.blocks[0].source_norm.forward = orig_resampler_norm
        model.quality_anchor.source_norms[0].forward = orig_anchor_norm


@torch.inference_mode()
def test_semantic_cond_and_uncond_batch_together_and_match_split_execution(connector, monkeypatch):
    from backend.sampling import sampling_function as sf

    adapter, model = connector
    dit = SimpleNamespace(llm_adapter=adapter, anima_v2_connector=model)
    source_p, ids_p, semantic_p = inputs()
    weights_p = torch.linspace(0.6, 1.4, 9)[None]
    ctx_p = torch.randn(1, 512, 1024)
    ctx_n = torch.randn(1, 512, 1024)
    negpip_p = torch.ones(1, 512, 1)
    negpip_p[0, 2] = -1
    negpip_n = torch.ones(1, 512, 1)

    # Case 1: anima_semantic_negative=False (cond has connector inputs, uncond does not) + NegPiP
    cond_p = compile_conditions(
        {
            "crossattn": ctx_p,
            "c_negpip_mask": negpip_p,
            "anima_source": source_p,
            "anima_ids": ids_p,
            "anima_weights": weights_p,
            "anima_semantic": semantic_p,
            "anima_semantic_mask": torch.ones(1, 11, dtype=torch.long),
        }
    )
    cond_n = compile_conditions({"crossattn": ctx_n, "c_negpip_mask": negpip_n})

    monkeypatch.setattr(sf, "shared", SimpleNamespace(batch_cond_uncond=True))
    monkeypatch.setattr(sf, "merges_by_free_memory", lambda _device: True)
    seen_calls = []

    class FakeModel:
        def memory_required(self, _shape):
            return 0.0

        def apply_model(self, x, t, **c):
            seen_calls.append(x.shape[0])
            ctx = Anima.semantic_context(dit, c["c_crossattn"], t, c)
            return ctx[:, :4, :4].reshape(x.shape[0], 1, 4, 4)

    out_cond, out_uncond = sf.calc_cond_uncond_batch(FakeModel(), cond_p, cond_n, torch.zeros(1, 1, 4, 4), torch.tensor([0.5]), {})
    assert seen_calls == [2]  # merged into a single DiT forward!

    want_p = Anima.semantic_context(
        dit,
        ctx_p,
        torch.tensor([0.5]),
        {
            "anima_source": source_p,
            "anima_ids": ids_p,
            "anima_weights": weights_p,
            "anima_semantic": semantic_p,
            "anima_semantic_mask": torch.ones(1, 11, dtype=torch.long),
            "c_negpip_mask": negpip_p,
        },
    )
    assert torch.allclose(out_cond, want_p[:, :4, :4].reshape(1, 1, 4, 4), atol=1e-6)
    assert torch.allclose(out_uncond, ctx_n[:, :4, :4].reshape(1, 1, 4, 4), atol=1e-6)

    # Case 2: anima_semantic_negative=True with different Qwen3 / T5 / Qwen3.5 sequence lengths
    gen = torch.Generator().manual_seed(7)
    source_n = torch.randn(1, 7, 1024, generator=gen)
    ids_n = torch.randint(0, 32128, (1, 5), generator=gen)
    weights_n = torch.ones(1, 5)
    semantic_n = torch.randn(1, 4, 6, 96, generator=gen)
    cond_n_sem = compile_conditions(
        {
            "crossattn": ctx_n,
            "c_negpip_mask": negpip_n,
            "anima_source": source_n,
            "anima_ids": ids_n,
            "anima_weights": weights_n,
            "anima_semantic": semantic_n,
            "anima_semantic_mask": torch.ones(1, 6, dtype=torch.long),
        }
    )
    seen_calls.clear()
    out_cond2, out_uncond2 = sf.calc_cond_uncond_batch(FakeModel(), cond_p, cond_n_sem, torch.zeros(1, 1, 4, 4), torch.tensor([0.5]), {})
    assert seen_calls == [2]
    want_n = Anima.semantic_context(
        dit,
        ctx_n,
        torch.tensor([0.5]),
        {
            "anima_source": source_n,
            "anima_ids": ids_n,
            "anima_weights": weights_n,
            "anima_semantic": semantic_n,
            "anima_semantic_mask": torch.ones(1, 6, dtype=torch.long),
            "c_negpip_mask": negpip_n,
        },
    )
    assert torch.allclose(out_cond2, want_p[:, :4, :4].reshape(1, 1, 4, 4), atol=1e-6)
    assert torch.allclose(out_uncond2, want_n[:, :4, :4].reshape(1, 1, 4, 4), atol=1e-6)


def test_qwen35_4b_normalization_detection_and_routing():
    from backend.loader import replace_state_dict
    from backend.nn.anima_qwen35 import is_qwen35_4b_state_dict, normalize_qwen35_4b_state_dict, qwen35_4b_needed_key

    release = manifest("anima_qwen35_4b_manifest.json")["tensors"]
    hf_sd = {f"model.language_model.{k}": torch.empty(shape, device="meta") for k, shape in release.items() if not k.startswith("norm.")}
    # Add extra HF Qwen/Qwen3.5-4B tensors that Anima-3.8B ignores
    hf_sd["model.language_model.layers.31.mlp.gate_proj.weight"] = torch.empty((9216, 2560), device="meta")
    hf_sd["model.language_model.layers.31.post_attention_layernorm.weight"] = torch.empty((2560,), device="meta")
    hf_sd["model.language_model.norm.weight"] = torch.empty((2560,), device="meta")
    hf_sd["model.visual.blocks.0.attn.qkv.weight"] = torch.empty((3072, 1152), device="meta")
    hf_sd["mtp.layers.0.mlp.gate_proj.weight"] = torch.empty((9216, 2560), device="meta")
    hf_sd["lm_head.weight"] = torch.empty((248320, 2560), device="meta")

    assert is_qwen35_4b_state_dict(hf_sd)
    normalized, has_release_head = normalize_qwen35_4b_state_dict(hf_sd)
    assert has_release_head is False

    with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
        no_head = AnimaQwen35(include_head=False)
    assert no_head.norm is None
    assert {k: list(v.shape) for k, v in normalized.items()} == {k: list(v.shape) for k, v in no_head.state_dict().items()}

    # Release head is preserved when present
    release_sd = meta_shapes(release)
    assert is_qwen35_4b_state_dict(release_sd)
    norm_rel, has_rel = normalize_qwen35_4b_state_dict(release_sd)
    assert has_rel is True and set(norm_rel) == set(release)

    # MOSS-Image's 9B (hidden_size=4096) must NOT be misidentified as Qwen3.5-4B
    moss_9b = {
        "model.language_model.layers.0.linear_attn.A_log": torch.empty(32, device="meta"),
        "model.language_model.layers.0.input_layernorm.weight": torch.empty(4096, device="meta"),
    }
    assert not is_qwen35_4b_state_dict(moss_9b)

    guess = SimpleNamespace(vae_key_prefix=["vae."], text_encoder_key_prefix=["text_encoders."], unet_key_prefix=["model.diffusion_model."])
    sd_4b: dict[str, torch.Tensor] = {}
    replace_state_dict(sd_4b, hf_sd, {}, guess, "Qwen3.5-4B")
    assert all(k.startswith("text_encoders.qwen35_4b.transformer.") for k in sd_4b)
    assert not any("visual" in k or "mtp." in k or "layers.31.mlp" in k for k in sd_4b)
    assert not qwen35_4b_needed_key("model.language_model.norm.weight")

    sd_9b: dict[str, torch.Tensor] = {}
    replace_state_dict(sd_9b, moss_9b, {}, guess, "qwen35_9b.safetensors")
    assert all(k.startswith("text_encoders.qwen35_9b.transformer.") for k in sd_9b)


@torch.inference_mode()
def test_anima_qwen35_loads_and_runs_from_hf_and_gguf_weights(monkeypatch):
    from dataclasses import dataclass, field

    import gguf

    from backend import loader
    from backend.loader_gguf import gguf_remapping
    from backend.nn import anima_qwen35
    from backend.operations_gguf import ParameterGGUF

    @dataclass
    class TinyQwen35Config(anima_qwen35.Qwen35_4BConfig):
        vocab_size: int = 64
        hidden_size: int = 64
        intermediate_size: int = 128
        num_hidden_layers: int = 4
        num_attention_heads: int = 4
        num_key_value_heads: int = 2
        head_dim: int = 32
        partial_rotary_factor: float = 0.25
        linear_num_key_heads: int = 2
        linear_num_value_heads: int = 4
        linear_key_head_dim: int = 16
        linear_value_head_dim: int = 16
        mrope_section: list = field(default_factory=lambda: [2, 1, 1])
        layer_types: list = field(default_factory=lambda: ["linear_attention", "linear_attention", "linear_attention", "full_attention"])
        no_mlp_layers: tuple = (3,)

    monkeypatch.setattr(anima_qwen35, "Qwen35_4BConfig", TinyQwen35Config)
    monkeypatch.setattr(loader.memory_management, "text_encoder_dtype", lambda: torch.float32)

    # 1. Build a reference HF Qwen3.5-4B state dict (with extra unused HF keys)
    torch.manual_seed(11)
    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        ref = anima_qwen35.AnimaQwen35(include_head=False).eval()
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.02)
    ref_sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    hf_sd = {f"model.language_model.{k}": v.clone() for k, v in ref_sd.items()}
    hf_sd["model.language_model.norm.weight"] = torch.zeros(64)
    hf_sd["model.language_model.layers.31.mlp.gate_proj.weight"] = torch.zeros(128, 64)
    hf_sd["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(16, 16)
    hf_sd["mtp.layers.0.mlp.gate_proj.weight"] = torch.zeros(16, 16)
    hf_sd["lm_head.weight"] = torch.zeros(64, 64)

    hf_model = loader.load_huggingface_component(None, "text_encoder_2", "transformers", "AnimaQwen35", "", hf_sd)
    assert hf_model.norm is None and hf_model.compute_dtype == torch.float32
    token_ids = torch.tensor([[1, 5, 12, 3]], dtype=torch.long)
    hf_states = hf_model(token_ids, layers=(0, 1, 2, 3), dtype=hf_model.compute_dtype)
    ref_states = ref(token_ids, layers=(0, 1, 2, 3), dtype=torch.float32)
    assert len(hf_states) == 4 and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(hf_states, ref_states))

    # 2. Build a synthetic Q8_0 / F32 GGUF state dict matching llama.cpp's qwen35 layout,
    #    remap it via gguf_remapping, load via AnimaQwen35, and run a real forward pass.
    def to_gguf_q8(tensor: torch.Tensor) -> ParameterGGUF:
        arr = tensor.detach().cpu().float().numpy()
        q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
        return ParameterGGUF(torch.from_numpy(q), tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=tensor.shape)

    def to_gguf_f32(tensor: torch.Tensor) -> ParameterGGUF:
        t = tensor.detach().cpu().float().contiguous()
        return ParameterGGUF(t, tensor_type=gguf.GGMLQuantizationType.F32, tensor_shape=t.shape)

    gguf_sd: dict[str, torch.Tensor] = {
        "token_embd.weight": to_gguf_q8(ref_sd["embed_tokens.weight"]),
        "output_norm.weight": to_gguf_f32(torch.ones(64)),
        "output.weight": to_gguf_q8(torch.zeros(64, 64)),
    }
    for idx in range(4):
        p = f"layers.{idx}."
        gguf_sd[f"blk.{idx}.attn_norm.weight"] = to_gguf_f32(ref_sd[f"{p}input_layernorm.weight"] + 1.0)
        if idx < 3:
            # llama.cpp stores GatedDeltaNet V-heads grouped as (v/k, k) instead of (k, v/k);
            # _reverse_qwen35 un-permutes them back to HF/Anima layout.
            h_perm = torch.arange(4).reshape(2, 2).transpose(0, 1).flatten()
            v_perm = torch.arange(64).reshape(2, 2, 16).transpose(0, 1).flatten()
            qkv_perm = torch.cat([torch.arange(64), 64 + v_perm])

            gguf_sd[f"blk.{idx}.post_attention_norm.weight"] = to_gguf_f32(ref_sd[f"{p}post_attention_layernorm.weight"] + 1.0)
            gguf_sd[f"blk.{idx}.ffn_up.weight"] = to_gguf_q8(ref_sd[f"{p}mlp.up_proj.weight"])
            gguf_sd[f"blk.{idx}.ffn_down.weight"] = to_gguf_q8(ref_sd[f"{p}mlp.down_proj.weight"])
            gguf_sd[f"blk.{idx}.ffn_gate.weight"] = to_gguf_q8(ref_sd[f"{p}mlp.gate_proj.weight"])

            gguf_sd[f"blk.{idx}.attn_qkv.weight"] = to_gguf_q8(ref_sd[f"{p}linear_attn.in_proj_qkv.weight"][qkv_perm])
            gguf_sd[f"blk.{idx}.attn_gate.weight"] = to_gguf_q8(ref_sd[f"{p}linear_attn.in_proj_z.weight"][v_perm])
            gguf_sd[f"blk.{idx}.ssm_alpha.weight"] = to_gguf_q8(ref_sd[f"{p}linear_attn.in_proj_a.weight"][h_perm])
            gguf_sd[f"blk.{idx}.ssm_beta.weight"] = to_gguf_q8(ref_sd[f"{p}linear_attn.in_proj_b.weight"][h_perm])
            gguf_sd[f"blk.{idx}.ssm_out.weight"] = to_gguf_q8(ref_sd[f"{p}linear_attn.out_proj.weight"][:, v_perm])
            gguf_sd[f"blk.{idx}.ssm_norm.weight"] = to_gguf_f32(ref_sd[f"{p}linear_attn.norm.weight"])
            gguf_sd[f"blk.{idx}.ssm_conv1d.weight"] = to_gguf_f32(ref_sd[f"{p}linear_attn.conv1d.weight"].squeeze(1)[qkv_perm])
            gguf_sd[f"blk.{idx}.ssm_dt.bias"] = to_gguf_f32(ref_sd[f"{p}linear_attn.dt_bias"][h_perm])
            gguf_sd[f"blk.{idx}.ssm_a"] = to_gguf_f32(-torch.exp(ref_sd[f"{p}linear_attn.A_log"][h_perm]))
        else:
            gguf_sd[f"blk.{idx}.attn_q.weight"] = to_gguf_q8(ref_sd[f"{p}self_attn.q_proj.weight"])
            gguf_sd[f"blk.{idx}.attn_k.weight"] = to_gguf_q8(ref_sd[f"{p}self_attn.k_proj.weight"])
            gguf_sd[f"blk.{idx}.attn_v.weight"] = to_gguf_q8(ref_sd[f"{p}self_attn.v_proj.weight"])
            gguf_sd[f"blk.{idx}.attn_output.weight"] = to_gguf_q8(ref_sd[f"{p}self_attn.o_proj.weight"])
            gguf_sd[f"blk.{idx}.attn_q_norm.weight"] = to_gguf_f32(ref_sd[f"{p}self_attn.q_norm.weight"] + 1.0)
            gguf_sd[f"blk.{idx}.attn_k_norm.weight"] = to_gguf_f32(ref_sd[f"{p}self_attn.k_norm.weight"] + 1.0)

    remapped = gguf_remapping(gguf_sd, {"general.architecture": "qwen35", "qwen35.ssm.group_count": 2})
    gguf_model = loader.load_huggingface_component(None, "text_encoder_2", "transformers", "AnimaQwen35", "", remapped)
    assert gguf_model.norm is None and gguf_model.compute_dtype == torch.float32
    gguf_states = gguf_model(token_ids, layers=(0, 1, 2, 3), dtype=gguf_model.compute_dtype)
    assert len(gguf_states) == 4
    assert all(state.shape == (1, 4, 64) and torch.allclose(state, ref_state, atol=2e-2) for state, ref_state in zip(gguf_states, ref_states))

    # Incomplete GGUF raises clearly instead of silently zero-filling weights
    incomplete = {k: v for k, v in remapped.items() if not k.endswith("layers.3.self_attn.q_proj.weight")}
    with pytest.raises(ValueError, match="Incomplete Qwen3.5 4B GGUF"):
        loader.load_huggingface_component(None, "text_encoder_2", "transformers", "AnimaQwen35", "", incomplete)
