"""Unit and pinning tests for ``sd_forge_ipadapter``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

EXT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = EXT_DIR / "scripts" / "forge_ipadapter.py"


@pytest.fixture(scope="module")
def ipadapter_mod():
    from remi.testing import webui_state

    ext_str = str(EXT_DIR)
    added = False
    if ext_str not in sys.path:
        sys.path.insert(0, ext_str)
        added = True
    try:
        with webui_state():
            spec = importlib.util.spec_from_file_location("forge_ipadapter_under_test", SCRIPT_PATH)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            yield mod
    finally:
        if added and ext_str in sys.path:
            sys.path.remove(ext_str)


def test_try_build_from_state_dict_supports_safetensors_sft_flat_and_nested(ipadapter_mod):
    patcher_cls = ipadapter_mod.IPAdapterPatcher
    flat_sd = {
        "image_proj.proj.weight": torch.ones(4, 4),
        "image_proj.image_proj.bias": torch.zeros(4),
        "ip_adapter.0.to_k_ip.weight": torch.ones(4, 4),
    }

    for fname in ("ip-adapter_sd15.safetensors", "ip-adapter-faceid-plusv2.sft", "ip-adapter_flat.bin"):
        built = patcher_cls.try_build_from_state_dict(flat_sd, fname)
        assert built is not None
        assert "proj.weight" in built.ip_adapter["image_proj"]
        # Exact removeprefix preserves nested 'image_proj.bias' key without stripping interior substring
        assert "image_proj.bias" in built.ip_adapter["image_proj"]
        assert "0.to_k_ip.weight" in built.ip_adapter["ip_adapter"]
        is_v2 = "v2" in fname
        assert built.faceid_v2 is is_v2
        assert built.weight_v2 is is_v2

    nested_sd = {
        "image_proj": {"proj.weight": torch.ones(2, 2)},
        "ip_adapter": {"0.to_k_ip.weight": torch.ones(2, 2)},
    }
    built_nested = patcher_cls.try_build_from_state_dict(nested_sd, "ip-adapter.bin")
    assert built_nested is not None

    assert patcher_cls.try_build_from_state_dict(None, "ip.safetensors") is None
    assert patcher_cls.try_build_from_state_dict({}, "ip.safetensors") is None
    assert patcher_cls.try_build_from_state_dict({"ip_adapter": torch.ones(1), "image_proj": {}}, "ip.bin") is None
    assert patcher_cls.try_build_from_state_dict({"ip_adapter": {"k": torch.ones(1)}, "image_proj": "bad"}, "ip.bin") is None


def test_process_before_every_sampling_validates_cond_and_mask_dims(ipadapter_mod, monkeypatch):
    patcher = ipadapter_mod.IPAdapterPatcher({"image_proj": {}, "ip_adapter": {"k": torch.ones(1)}})
    patcher.strength = 0.8
    patcher.start_percent = 0.0
    patcher.end_percent = 1.0

    sentinel_unet = object()
    process = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=sentinel_unet)))

    # Selecting IP-Adapter model with Preprocessor='None' passes a Tensor cond -> clear ValueError
    with pytest.raises(ValueError, match="IP-Adapter preprocessor"):
        patcher.process_before_every_sampling(process, torch.zeros(1, 3, 64, 64), None)

    recorded: list[dict[str, object]] = []

    def fake_apply_ipadapter(**kwargs):
        recorded.append(kwargs)
        return "patched_unet"

    monkeypatch.setattr(ipadapter_mod.IPAdapterApply, "apply_ipadapter", fake_apply_ipadapter)

    # 4D (B, 1, H, W) mask squeezes channel dim -> (B, H, W)
    mask_4d = torch.ones(2, 1, 8, 12)
    patcher.process_before_every_sampling(process, {"clip_vision": "cv", "image": torch.zeros(1, 8, 8, 3)}, mask_4d)
    assert process.sd_model.forge_objects.unet == "patched_unet"
    attn_4d = recorded[-1]["attn_mask"]
    assert isinstance(attn_4d, torch.Tensor) and attn_4d.shape == (2, 8, 12)

    # 3D (B, 1, W) mask is already 3D and must NOT have dim 1 squeezed away
    mask_3d = torch.ones(2, 1, 12)
    patcher.process_before_every_sampling(process, {"clip_vision": "cv", "image": torch.zeros(1, 8, 8, 3)}, mask_3d)
    attn_3d = recorded[-1]["attn_mask"]
    assert isinstance(attn_3d, torch.Tensor) and attn_3d.shape == (2, 1, 12)


def test_preprocessors_metadata_and_call_shapes(ipadapter_mod, monkeypatch):
    p_clip = ipadapter_mod.PreprocessorClipVisionForIPAdapter("test-clip", "https://example.invalid/m.safetensors", "m.safetensors")
    monkeypatch.setattr(p_clip, "load_clipvision", lambda: "mock_clip")
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    cond = p_clip(img, 256)
    assert cond["clip_vision"] == "mock_clip"
    assert cond["image"].shape == (1, 16, 16, 3)

    p_instant = ipadapter_mod.PreprocessorInsightFaceForInstantID("test-instantid")
    assert p_instant.sorting_priority == 10
    monkeypatch.setattr(ipadapter_mod.InsightFaceLoader, "load_insight_face", lambda name="buffalo_l": f"if_{name}")
    cond_id = p_instant(img, 256)
    assert cond_id["insightface"] == "if_antelopev2"
    assert cond_id["instant_id"] is True
