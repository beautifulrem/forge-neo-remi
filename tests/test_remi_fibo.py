"""Unit tests for Bria Fibo 2.0 (8.1B Dev & 4-step Edit Turbo):
VGL 2.0 JSON structured prompt protocol, CC BY-NC 4.0 license metadata,
DiT forward pass + multi-reference + cross-step caches, quantization,
Spectrum pre-head seam, Diffusers loading, and engine routing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from backend import memory_management, operations
from backend.args import dynamic_args
from backend.diffusion_engine.fibo import Fibo2
from backend.nn import dit_capabilities_ext
from backend.nn.anima_quant import (
    keep_sensitive_weights_precise,
    quantize_int8_convrot_state_dict,
    scale_fp8_state_dict,
)
from backend.nn.dit_registry import (
    capability_for_compile_preset,
    capability_for_dit_cls,
    capability_for_state_dict,
    get_dit_capability,
    match_spectrum_capability,
    resolve_spectrum_capture_module,
    unregister_dit_capability,
)
from backend.nn.fibo import (
    FIBO_COMMERCIAL_USE,
    FIBO_LICENSE,
    FIBO_LICENSE_NOTICE,
    VGL_REQUIRED_FIELDS,
    FiboTransformer2DModel,
    build_vgl_prompt,
    fast_forward_fibo,
    fibo_lora_key_map,
    fibo_workflow_advice,
    format_vgl_for_encoder,
    is_fibo_diffusers_dir,
    is_fibo_state_dict,
    load_fibo_diffusers_dir,
    parse_vgl_prompt,
    validate_vgl_prompt,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _register_ext_caps():
    preexisting = {name for name in dit_capabilities_ext.EXTENDED_DIT_NAMES if get_dit_capability(name) is not None}
    dit_capabilities_ext.register_extended_dit_capabilities()
    try:
        yield
    finally:
        for name in dit_capabilities_ext.EXTENDED_DIT_NAMES:
            if name not in preexisting:
                unregister_dit_capability(name)


def _tiny_fibo(*, hidden_size: int = 32, num_heads: int = 2, num_layers: int = 2, is_edit_turbo: bool = False) -> FiboTransformer2DModel:
    torch.manual_seed(0)
    head_dim = hidden_size // num_heads
    a0 = head_dim // 4
    a1 = (head_dim - a0) // 2
    a2 = head_dim - a0 - a1
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        m = FiboTransformer2DModel(
            in_channels=8,
            out_channels=8,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            text_in_dim=16,
            mlp_ratio=2.0,
            axes_dims_rope=(a0, a1, a2),
            is_edit_turbo=is_edit_turbo,
        ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.03)
    return m


def test_fibo_vgl_json_protocol_and_license_metadata():
    assert FIBO_LICENSE == "CC-BY-NC-4.0"
    assert FIBO_COMMERCIAL_USE is False
    assert "CC BY-NC 4.0" in FIBO_LICENSE_NOTICE
    assert FiboTransformer2DModel.license == "CC-BY-NC-4.0"
    assert Fibo2.license == "CC-BY-NC-4.0"
    assert Fibo2.commercial_use is False

    vgl = build_vgl_prompt(
        "A red ceramic teapot on a walnut table",
        objects=[{"description": "glossy red ceramic teapot", "location": "center", "relationship": "resting on table", "relative_size": "medium", "shape_and_color": "round red"}],
        style_medium="studio photograph",
    )
    assert set(VGL_REQUIRED_FIELDS) <= set(vgl.keys())
    assert validate_vgl_prompt(vgl) == []

    raw_json = json.dumps(vgl, indent=2)
    parsed = parse_vgl_prompt(f"```json\n{raw_json}\n```")
    assert parsed == vgl
    canonical = format_vgl_for_encoder(raw_json)
    assert json.loads(canonical) == vgl
    assert "\n" not in canonical
    assert format_vgl_for_encoder("plain text prompt") == "plain text prompt"

    # Invalid VGL detects missing fields & bad object items
    bad_issues = validate_vgl_prompt({"short_description": "", "objects": [{"location": "left"}]})
    assert len(bad_issues) >= 3


def test_fibo_dit_forward_caches_multiref_and_spectrum():
    dit_capabilities_ext.register_extended_dit_capabilities()
    dev_model = _tiny_fibo(hidden_size=32, num_heads=2, num_layers=2, is_edit_turbo=False)
    turbo_model = _tiny_fibo(hidden_size=32, num_heads=2, num_layers=2, is_edit_turbo=True)
    assert dev_model.guidance_in is not None
    assert turbo_model.guidance_in is None

    x = torch.randn(1, 8, 4, 4)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 5, 16)
    refs = [torch.randn(1, 8, 2, 2), torch.randn(1, 8, 3, 2)]

    with torch.inference_mode():
        y_dev = dev_model(x, t, ctx, guidance=torch.tensor([3.5]))
        y_turbo = turbo_model(x, t, ctx, ref_latents=refs)
        y_turbo_noref = turbo_model(x, t, ctx, ref_latents=[])
    assert y_dev.shape == x.shape
    assert y_turbo.shape == x.shape
    assert (y_turbo - y_turbo_noref).abs().max().item() > 1e-5
    assert len(turbo_model._text_cache) == 1 and len(turbo_model._rope_cache) == 2

    # Spectrum pre-head seam matches full forward bit-for-bit
    assert match_spectrum_capability(turbo_model) is get_dit_capability("fibo")
    assert resolve_spectrum_capture_module(turbo_model) is turbo_model.fibo_final_input
    captured = {}

    def hook(_mod, args):
        captured["feat"] = args[0].detach().clone()

    h = turbo_model.fibo_final_input.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            full = turbo_model(x, t, ctx, ref_latents=[])
            predictor = SimpleNamespace(timestep=lambda s: s, calculate_denoised=lambda s, pred, inp: pred)
            fast = fast_forward_fibo(turbo_model, predictor, t, x, captured["feat"])
        torch.testing.assert_close(fast, full, atol=1e-5, rtol=1e-5)
    finally:
        h.remove()


def test_fibo_quantization_diffusers_dir_lora_compile_and_engine(tmp_path, monkeypatch):
    dit_capabilities_ext.register_extended_dit_capabilities()
    cap = get_dit_capability("fibo")
    assert cap is not None
    assert capability_for_dit_cls("FiboTransformer2DModel") is cap
    assert capability_for_compile_preset("Fibo-2.0 per-block") is cap

    wide = _tiny_fibo(hidden_size=256, num_heads=4, num_layers=1, is_edit_turbo=True)
    fp8_model = _tiny_fibo(hidden_size=256, num_heads=4, num_layers=1, is_edit_turbo=True).to(torch.float8_e4m3fn)
    kept = keep_sensitive_weights_precise(fp8_model, torch.bfloat16)
    assert kept > 0
    assert fp8_model.img_in.weight.dtype == torch.bfloat16
    assert fp8_model.final_layer.linear.weight.dtype == torch.bfloat16
    assert fp8_model.transformer_blocks[0].attn.to_q.weight.dtype == torch.float8_e4m3fn

    bf16_sd = {k: v.detach().clone().to(torch.bfloat16) for k, v in wide.state_dict().items()}
    assert is_fibo_state_dict(bf16_sd) is True
    assert capability_for_state_dict(bf16_sd) is cap

    fp8_sd = dict(bf16_sd)
    assert scale_fp8_state_dict(fp8_sd, fp8_matmul=False) >= 7
    int8_sd = dict(bf16_sd)
    assert quantize_int8_convrot_state_dict(int8_sd) == 7

    # Diffusers dir load includes Qwen3-VL-4B, FLUX.2 VAE, and CC BY-NC 4.0 license metadata
    pipe_dir = tmp_path / "Bria-Fibo-2.0-Edit-Turbo"
    (pipe_dir / "transformer").mkdir(parents=True)
    (pipe_dir / "text_encoder").mkdir(parents=True)
    (pipe_dir / "vae").mkdir(parents=True)
    (pipe_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "FiboEditPipeline", "is_edit_turbo": True, "transformer": ["diffusers", "FiboTransformer2DModel"]}),
        encoding="utf-8",
    )
    save_file(bf16_sd, str(pipe_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    save_file({"language_model.embed_tokens.weight": torch.zeros(8, 2560)}, str(pipe_dir / "text_encoder" / "model.safetensors"))
    save_file({"decoder.post_quant_conv.weight": torch.zeros(32, 32, 1, 1)}, str(pipe_dir / "vae" / "diffusion_pytorch_model.safetensors"))

    assert is_fibo_diffusers_dir(pipe_dir) is True
    loaded_sd, meta = load_fibo_diffusers_dir(pipe_dir)
    assert meta["license"] == "CC-BY-NC-4.0" and meta["commercial_use"] == "false" and meta["fibo_variant"] == "edit_turbo"
    assert "text_encoders.qwen3vl_4b.transformer.model.language_model.embed_tokens.weight" in loaded_sd
    assert "vae.post_quant_conv.weight" in loaded_sd

    # LoRA key map & per-block compile
    km = fibo_lora_key_map({f"diffusion_model.{k}": v for k, v in wide.state_dict().items()})
    assert km["transformer.transformer_blocks.0.attn.to_q"] == "diffusion_model.transformer_blocks.0.attn.to_q.weight"
    compile_root = str(REPO / "extensions-builtin" / "sd_forge_compile")
    if compile_root not in sys.path:
        sys.path.insert(0, compile_root)
    from anima_block_compile import _block_list

    assert _block_list(wide) is wide.transformer_blocks

    # Engine conditioning canonicalizes VGL JSON and routes multi-references
    monkeypatch.setattr(memory_management, "load_model_gpu", lambda *_a, **_kw: None)
    monkeypatch.setattr(dynamic_args, "ref_latents", [])
    eng = Fibo2.__new__(Fibo2)
    eng.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=object()))
    seen = []
    eng.text_processing_engine_qwen = lambda prompts: seen.append(list(prompts)) or ["ok"]
    eng.ref_latents = [torch.zeros(1, 8, 2, 2)]
    eng.ini_latent = torch.ones(1, 8, 2, 2)
    eng.fibo_do_reference = True

    vgl_str = json.dumps(build_vgl_prompt("a lantern"), indent=2)
    eng.get_learned_conditioning([vgl_str])
    assert len(seen) == 1 and "\n" not in seen[0][0]
    assert len(dynamic_args.ref_latents) == 2 and eng.ini_latent is None

    # Advice for Edit Turbo + malformed JSON prompt
    p_bad = SimpleNamespace(
        sd_model=SimpleNamespace(is_fibo_engine=True, is_fibo_turbo=True, ref_latents=[], ini_latent=None, sd_checkpoint_info=SimpleNamespace(name="Fibo-2.0-Edit-Turbo")),
        prompt="{not valid json",
        steps=20,
        cfg_scale=3.5,
    )
    notes = fibo_workflow_advice(p_bad)
    assert any("4 steps" in n for n in notes)
    assert any("not valid JSON" in n for n in notes)
