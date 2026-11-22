"""Tests for the declarative DiT Capability Registry (``backend/nn/dit_registry.py``).

Verifies both built-in capabilities (Anima, MOSS-Image, Krea 2) and end-to-end
extensibility: registering a synthetic 4th DiT family automatically enables
on-the-fly quantization, Diffusers directory & single-file conversion, per-block
``torch.compile``, Spectrum pre-head forecasting, and workflow advice without
editing any consumer module.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import torch
from torch import nn

from backend.nn.anima_quant import (
    keep_sensitive_weights_precise,
    quantize_int8_convrot_state_dict,
    scale_fp8_state_dict,
    should_keep_sensitive_weights_precise,
    should_quantize_on_load,
)
from backend.nn.dit_registry import (
    DiTCapability,
    all_dit_capabilities,
    capability_for_compile_preset,
    capability_for_dit_cls,
    collect_workflow_advice,
    diffusers_checkpoint_info,
    get_dit_capability,
    is_diffusers_checkpoint_dir,
    load_diffusers_checkpoint_dir,
    match_spectrum_capability,
    per_block_compile_presets,
    register_dit_capability,
    registered_block_attrs,
    unregister_dit_capability,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_builtin_capabilities_cover_anima_moss_and_krea2():
    names = [cap.name for cap in all_dit_capabilities()]
    assert names[:4] == ["anima", "moss", "krea2", "flux3"]

    assert capability_for_dit_cls("CosmosTransformer3DModel") is get_dit_capability("anima")
    assert capability_for_dit_cls("MOSSImageTransformer") is get_dit_capability("moss")
    assert capability_for_dit_cls("Krea2Transformer2DModel") is get_dit_capability("krea2")
    assert capability_for_dit_cls("Flux3Transformer2DModel") is get_dit_capability("flux3")
    assert capability_for_dit_cls("Flux") is None

    assert per_block_compile_presets() == ("Anima per-block", "MOSS-Image per-block", "Krea-2 per-block", "FLUX.3 per-block")
    assert capability_for_compile_preset("Anima per-block").name == "anima"
    assert capability_for_compile_preset("MOSS-Image per-block").name == "moss"
    assert capability_for_compile_preset("Krea-2 per-block").name == "krea2"
    assert capability_for_compile_preset("FLUX.3 per-block").name == "flux3"
    assert capability_for_compile_preset("guard_filter_fn") is None
    assert "blocks" in registered_block_attrs() and "layers" in registered_block_attrs()


def test_registering_a_fourth_dit_flows_through_every_consumer(tmp_path):
    """Registering a synthetic 4th DiT (``toy_dit``) works across quantization,
    Diffusers loading, per-block compile, Spectrum pre-head caching, and
    workflow advice with zero edits to any consumer."""

    class _ToyHeadMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(4))

    class ToyDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.toy_in = nn.Linear(256, 256)
            self.toy_blocks = nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 256)])
            self.toy_head_in = nn.Identity()
            self.toy_head = nn.Linear(256, 256)
            self.toy_mod = _ToyHeadMod()
            self.fast_forward_calls = 0

        def forward(self, x):
            for blk in self.toy_blocks:
                x = blk(x)
            return self.toy_head(self.toy_head_in(x))

        def compute_toy_head(self, feat, t):
            self.fast_forward_calls += 1
            return self.toy_head(feat) + t.reshape(-1, 1) * 0.0

    toy_dir = tmp_path / "ToyDiT-Diffusers"
    toy_dir.mkdir()
    toy_marker = toy_dir / "toy_model.safetensors"
    toy_marker.write_bytes(b"placeholder")

    def _toy_fast_forward(dit, predictor, sigma, input_x, feature):
        return dit.compute_toy_head(feature, predictor.timestep(sigma))

    cap = DiTCapability(
        name="toy_dit",
        dit_cls_name="ToyDiTTransformer",
        model_type_name="ToyDiT",
        engine_label="ToyDiT",
        sensitive_prefixes=("toy_in.", "toy_head."),
        sensitive_module_types=("_ToyHeadMod",),
        matches_state_dict=lambda sd: "toy_in.weight" in sd,
        block_linear_filter=lambda k: k.startswith("toy_blocks.") and k.endswith(".weight"),
        fp8_rowwise=True,
        int8_convrot_filter=lambda k: k.startswith("toy_blocks.") and k.endswith(".weight"),
        diffusers_arch="toy-dit/diffusers",
        is_diffusers_dir=lambda p: pathlib.Path(p) == toy_dir,
        diffusers_primary_file=lambda p: str(toy_marker),
        load_diffusers_dir=lambda p: ({"model.diffusion_model.toy_in.weight": torch.zeros(4, 4)}, {"toy": "1"}),
        is_diffusers_single_file=lambda sd: "diffusers_toy_in.weight" in sd,
        convert_diffusers_single_file=lambda sd, prefix="": {f"{prefix}toy_in.weight": sd["diffusers_toy_in.weight"]},
        block_attr="toy_blocks",
        compile_preset="ToyDiT per-block",
        matches_engine=lambda model: bool(getattr(model, "is_toy_engine", False)),
        non_matching_engine_reason="ToyDiT per-block ignored: non-ToyDiT model",
        spectrum_required_attrs=("toy_head_in", "compute_toy_head"),
        spectrum_capture_attr="toy_head_in",
        spectrum_fast_forward=_toy_fast_forward,
        workflow_advice=lambda p: ["ToyDiT custom hint"] if getattr(p, "trigger_toy_hint", False) else [],
    )

    register_dit_capability(cap)
    try:
        # 1. On-the-fly FP8 sensitive weights + scaled FP8 (rowwise) + INT8-ConvRot
        assert should_keep_sensitive_weights_precise("ToyDiTTransformer", torch.float8_e4m3fn, torch.bfloat16) is True
        assert should_quantize_on_load("scaled_fp8", "ToyDiTTransformer", None, False, torch.bfloat16) is True

        fp8_toy = ToyDiT().to(torch.float8_e4m3fn)
        kept = keep_sensitive_weights_precise(fp8_toy, torch.bfloat16)
        assert kept > 0
        assert fp8_toy.toy_in.weight.dtype == torch.bfloat16
        assert fp8_toy.toy_head.weight.dtype == torch.bfloat16
        assert fp8_toy.toy_mod.scale.dtype == torch.bfloat16
        assert fp8_toy.toy_blocks[0].weight.dtype == torch.float8_e4m3fn

        bf16_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in ToyDiT().state_dict().items()}
        fp8_sd = dict(bf16_sd)
        assert scale_fp8_state_dict(fp8_sd, fp8_matmul=False) == 2
        assert fp8_sd["toy_blocks.0.weight"].dtype == torch.float8_e4m3fn
        assert fp8_sd["toy_blocks.0.weight_scale"].shape == (256, 1)
        assert "toy_in.comfy_quant" not in fp8_sd

        int8_sd = dict(bf16_sd)
        assert quantize_int8_convrot_state_dict(int8_sd) == 2
        assert int8_sd["toy_blocks.1.weight"].dtype == torch.int8

        # 2. Diffusers dir & single-file converter via comfy.utils.convert_diffusers_mmdit
        assert is_diffusers_checkpoint_dir(toy_dir) is True
        assert diffusers_checkpoint_info(toy_dir) == ("toy-dit/diffusers", str(toy_marker))
        loaded_sd, loaded_meta = load_diffusers_checkpoint_dir(toy_dir)
        assert "model.diffusion_model.toy_in.weight" in loaded_sd and loaded_meta == {"toy": "1"}

        from modules_forge.packages.comfy.utils import convert_diffusers_mmdit

        converted = convert_diffusers_mmdit({"diffusers_toy_in.weight": torch.ones(2, 2)}, output_prefix="p.")
        assert converted is not None and "p.toy_in.weight" in converted

        # 3. Per-block torch.compile (_block_list + TorchCompileForForge.process_batch)
        compile_root = REPO / "extensions-builtin" / "sd_forge_compile"
        if str(compile_root) not in sys.path:
            sys.path.insert(0, str(compile_root))
        from anima_block_compile import AnimaBlockCompileManager, _block_list

        toy = ToyDiT().eval()
        assert _block_list(toy) is toy.toy_blocks

        spec = importlib.util.spec_from_file_location("compile_script_registry_test", compile_root / "scripts" / "compile.py")
        compile_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compile_mod)
        compile_script = compile_mod.TorchCompileForForge.__new__(compile_mod.TorchCompileForForge)
        kmodel = SimpleNamespace(diffusion_model=toy, apply_model=lambda x: x)
        p_wrong = SimpleNamespace(
            sd_model=SimpleNamespace(is_toy_engine=False, forge_objects=SimpleNamespace(unet=SimpleNamespace(model=kmodel))),
            extra_generation_params={},
        )
        compile_script.process_batch(p_wrong, "ToyDiT per-block")
        assert not AnimaBlockCompileManager.is_installed(kmodel)
        assert p_wrong.extra_generation_params["Torch compile"] == "ToyDiT per-block ignored: non-ToyDiT model"

        p_right = SimpleNamespace(
            sd_model=SimpleNamespace(is_toy_engine=True, forge_objects=SimpleNamespace(unet=SimpleNamespace(model=kmodel))),
            extra_generation_params={},
        )
        compile_script.process_batch(p_right, "ToyDiT per-block")
        assert AnimaBlockCompileManager.is_installed(kmodel)
        AnimaBlockCompileManager.remove(kmodel)

        # 4. Spectrum pre-head feature capture & fast-forward
        spectrum_root = REPO / "extensions-builtin" / "sd_forge_spectrum"
        if str(spectrum_root) not in sys.path:
            sys.path.insert(0, str(spectrum_root))
        from lib_spectrum.forecaster import SpectrumNode

        assert match_spectrum_capability(toy) is cap

        class _FakePatcher:
            def __init__(self, model):
                self.model = model
                self.model_options = {}

            def clone(self):
                cloned = _FakePatcher(self.model)
                cloned.model_options = dict(self.model_options)
                return cloned

            def set_model_unet_function_wrapper(self, fn):
                self.model_options["model_function_wrapper"] = fn

        predictor = SimpleNamespace(timestep=lambda s: s)
        patcher = _FakePatcher(SimpleNamespace(diffusion_model=toy, predictor=predictor))
        patched = SpectrumNode.patch(
            patcher,
            steps=8,
            weight=0.5,
            degree=1,
            lam=1.0,
            window_size=2,
            flex_window=0.0,
            warmup_steps=3,
            stop_caching_step=1.0,
            tail_actual_steps=1,
            history_size=4,
            schedule="Window",
            refresh_ratio=0.5,
            sea_beta=0.9,
            compat_policy="Conservative",
            verbose=False,
            sea_cache_dir=str(tmp_path),
            sea_cache_context={},
        )
        wrapper = patched.model_options["model_function_wrapper"]
        x = torch.randn(1, 256)
        ctx = torch.randn(1, 4, 8)
        for step in range(5):
            sigma = torch.tensor([1.0 - step * 0.1])
            out = wrapper(
                lambda inp, _t, **_c: toy(inp),
                {"input": x, "timestep": sigma, "c": {"c_crossattn": ctx, "transformer_options": {}}, "cond_or_uncond": [0]},
            )
            assert out.shape == x.shape
        assert toy.fast_forward_calls >= 1

        # 5. Workflow advice collection
        p_hint = SimpleNamespace(trigger_toy_hint=True)
        assert "ToyDiT custom hint" in collect_workflow_advice(p_hint)
    finally:
        unregister_dit_capability("toy_dit")
        assert get_dit_capability("toy_dit") is None
