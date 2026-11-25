"""Unit tests for Apple Silicon MPS memory management & mixed-precision hot-path optimizations."""

from __future__ import annotations

import gc
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from backend import memory_management, operations  # noqa: F401 - breaks operations <-> operations_mixed_precision import cycle
from backend import operations_mixed_precision as omp
from backend.quant_ops import RowWiseFP8E4M3Layout, TensorCoreFP8E4M3Layout, TensorWiseINT8Layout


def _fake_mps_module(*, recommended: int, current: int, driver: int, empty_calls: list[str] | None = None):
    calls = empty_calls if empty_calls is not None else []

    def empty_cache():
        calls.append("empty_cache")

    return SimpleNamespace(
        recommended_max_memory=lambda: recommended,
        current_allocated_memory=lambda: current,
        driver_allocated_memory=lambda: driver,
        empty_cache=empty_cache,
    )


def test_same_device_treats_mps_and_mps0_as_equal():
    assert memory_management._same_device(torch.device("mps"), torch.device("mps:0"))
    assert memory_management._same_device(torch.device("mps:0"), torch.device("mps"))
    assert memory_management._same_device(torch.device("mps"), torch.device("mps"))
    assert not memory_management._same_device(torch.device("mps"), torch.device("cpu"))
    assert not memory_management._same_device(torch.device("cuda:0"), torch.device("cuda:1"))
    assert memory_management._canonical_device(torch.device("mps:0")) == torch.device("mps")
    assert memory_management._canonical_device(torch.device("cpu")) == torch.device("cpu")


def test_cast_to_avoids_copy_when_mps_and_mps0_match(monkeypatch):
    tensor = torch.randn(4, 4, dtype=torch.float32)

    class _FakeMPSParam:
        def __init__(self, inner: torch.Tensor):
            self.device = torch.device("mps:0")
            self.dtype = inner.dtype
            self.inner = inner
            self.to_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls += 1
            return self.inner.to(*args, **kwargs)

    fake = _FakeMPSParam(tensor)
    out = memory_management.cast_to(fake, dtype=torch.float32, device=torch.device("mps"), copy=False)
    assert out is fake
    assert fake.to_calls == 0


def test_mps_total_and_free_memory_respect_recommended_working_set(monkeypatch):
    gb = 1024**3
    fake_mps = _fake_mps_module(recommended=16 * gb, current=6 * gb, driver=8 * gb)
    monkeypatch.setattr(torch, "mps", fake_mps, raising=False)
    monkeypatch.setattr(memory_management.psutil, "virtual_memory", lambda: SimpleNamespace(total=24 * gb, available=12 * gb))

    total, driver = memory_management.get_total_memory(torch.device("mps"), torch_total_too=True)
    assert total == 16 * gb
    assert driver == 8 * gb

    # Headroom = 16 - 6 = 10 GiB; OS avail + reclaimable = 12 + 2 = 14 GiB => min = 10 GiB
    free_total, free_torch = memory_management.get_free_memory(torch.device("mps"), torch_free_too=True)
    assert free_total == 10 * gb
    assert free_torch == 2 * gb

    # Tight OS memory case: OS avail = 3 GiB + 2 GiB reclaimable = 5 GiB < 10 GiB headroom => min = 5 GiB
    monkeypatch.setattr(memory_management.psutil, "virtual_memory", lambda: SimpleNamespace(total=24 * gb, available=3 * gb))
    free_total_tight, free_torch_tight = memory_management.get_free_memory(torch.device("mps"), torch_free_too=True)
    assert free_total_tight == 5 * gb
    assert free_torch_tight == 2 * gb


def test_mps_free_memory_does_not_spam_empty_cache_when_reclaimable_is_zero(monkeypatch):
    gb = 1024**3
    calls: list[str] = []
    fake_mps = _fake_mps_module(recommended=16 * gb, current=4 * gb, driver=4 * gb, empty_calls=calls)
    monkeypatch.setattr(torch, "mps", fake_mps, raising=False)
    monkeypatch.setattr(memory_management, "cpu_state", memory_management.CPUState.MPS)
    monkeypatch.setattr(memory_management, "vram_state", memory_management.VRAMState.SHARED)
    monkeypatch.setattr(memory_management, "current_loaded_models", [])
    monkeypatch.setattr(memory_management, "signal_empty_cache", False)
    monkeypatch.setattr(memory_management.psutil, "virtual_memory", lambda: SimpleNamespace(total=24 * gb, available=14 * gb))
    monkeypatch.setattr(memory_management, "_CACHE_CLEAR_HOOKS", [])

    memory_management.free_memory(2 * gb, torch.device("mps"))
    assert calls == []

    # Direct soft_empty_cache(force=False) also skips Metal flush when driver == current
    memory_management.soft_empty_cache(force=False)
    assert calls == []

    # force=True always flushes
    memory_management.soft_empty_cache(force=True)
    assert calls == ["empty_cache"]


def test_mps_free_memory_clears_regenerable_caches_before_evicting_models(monkeypatch):
    gb = 1024**3
    calls: list[str] = []
    state = {"current": 14 * gb, "driver": 15 * gb, "cached_bytes": 2 * gb}

    def clear_hook():
        freed = state["cached_bytes"]
        state["cached_bytes"] = 0
        state["current"] -= freed
        return freed

    fake_mps = SimpleNamespace(
        recommended_max_memory=lambda: 16 * gb,
        current_allocated_memory=lambda: state["current"],
        driver_allocated_memory=lambda: state["driver"],
        empty_cache=lambda: (calls.append("empty_cache"), state.__setitem__("driver", state["current"])),
    )
    monkeypatch.setattr(torch, "mps", fake_mps, raising=False)
    monkeypatch.setattr(memory_management, "cpu_state", memory_management.CPUState.MPS)
    monkeypatch.setattr(memory_management, "vram_state", memory_management.VRAMState.SHARED)
    monkeypatch.setattr(memory_management, "DISABLE_SMART_MEMORY", False)
    monkeypatch.setattr(memory_management, "_CACHE_CLEAR_HOOKS", [clear_hook])
    monkeypatch.setattr(memory_management.psutil, "virtual_memory", lambda: SimpleNamespace(total=24 * gb, available=10 * gb))

    unloaded = []

    class _FakeLoaded:
        device = torch.device("mps:0")
        currently_used = True
        model = SimpleNamespace(model=SimpleNamespace())

        def is_dead(self):
            return False

        def model_offloaded_memory(self):
            return 0

        def model_memory(self):
            return 4 * gb

        def model_unload(self, memory_to_free=None):
            unloaded.append(memory_to_free)
            return True

    loaded = [_FakeLoaded()]
    monkeypatch.setattr(memory_management, "current_loaded_models", loaded)

    # Needs 3 GiB free; initially only 2 GiB headroom -> clears 2 GiB regenerable cache -> 4 GiB headroom -> model stays loaded!
    memory_management.free_memory(3 * gb, torch.device("mps"))
    assert unloaded == []
    assert len(loaded) == 1
    assert "empty_cache" in calls


def test_should_use_bf16_handles_none_mac_version(monkeypatch):
    monkeypatch.setattr(memory_management, "FORCE_FP32", False)
    monkeypatch.setattr(memory_management, "mac_version", lambda: None)
    assert memory_management.should_use_bf16(torch.device("mps")) is True

    monkeypatch.setattr(memory_management, "mac_version", lambda: (13, 6))
    assert memory_management.should_use_bf16(torch.device("mps")) is False

    monkeypatch.setattr(memory_management, "mac_version", lambda: (15, 0))
    assert memory_management.should_use_bf16(torch.device("mps")) is True


def _make_quant_linear(fmt: str, *, extra_conf: dict | None = None, out_f: int = 128, in_f: int = 256):
    gen = torch.Generator().manual_seed(42)
    w = (torch.randn(out_f, in_f, generator=gen) * 0.05).to(torch.float32)
    b = (torch.randn(out_f, generator=gen) * 0.01).to(torch.float32)

    ops = omp.mixed_precision_ops(compute_dtype=torch.float32, full_precision_mm=True, disabled=[])
    layer = ops.Linear(in_f, out_f, bias=True, device=torch.device("cpu"), dtype=torch.float32)

    conf = {"format": fmt, "full_precision_matrix_mult": True, **(extra_conf or {})}
    sd = {"bias": b, "comfy_quant": torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)}
    if fmt == "float8_e4m3fn":
        qdata, params = TensorCoreFP8E4M3Layout.quantize(w, scale="recalculate")
        sd["weight"], sd["weight_scale"] = qdata, params.scale
    elif fmt == "float8_e4m3fn_rowwise":
        qdata, params = RowWiseFP8E4M3Layout.quantize(w, scale="recalculate")
        sd["weight"], sd["weight_scale"] = qdata, params.scale
    elif fmt == "int8_tensorwise":
        qdata, params = TensorWiseINT8Layout.quantize(w, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
        sd["weight"], sd["weight_scale"] = qdata, params.scale.float()
    else:
        raise ValueError(fmt)
    layer.load_state_dict(sd)
    return layer


@pytest.mark.parametrize(
    ("fmt", "extra"),
    [
        ("float8_e4m3fn", {}),
        ("float8_e4m3fn_rowwise", {}),
        ("int8_tensorwise", {"convrot": True, "convrot_groupsize": 256}),
    ],
)
def test_mps_dequant_cache_hits_and_matches_uncached_output(monkeypatch, fmt, extra):
    omp.clear_mps_dequant_cache()
    omp._mps_dequant_cache.reset_stats()

    layer = _make_quant_linear(fmt, extra_conf=extra)
    x = torch.randn(1, 32, 256, dtype=torch.float32)

    # Uncached reference (normal CPU path)
    with torch.inference_mode():
        ref = layer(x)

    # Simulate resident MPS path with generous cache budget
    dequant_calls = []
    orig_dequant = omp.QuantizedTensor.dequantize

    def counted_dequant(self):
        dequant_calls.append(1)
        return orig_dequant(self)

    monkeypatch.setattr(omp, "is_device_mps", lambda dev: True)
    monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", "64")
    monkeypatch.setattr(omp.QuantizedTensor, "dequantize", counted_dequant)

    with torch.inference_mode():
        y1 = layer(x)
        y2 = layer(x)
        y3 = layer(x)

    assert torch.allclose(y1, ref, atol=1e-5, rtol=1e-4)
    assert torch.equal(y1, y2)
    assert torch.equal(y2, y3)
    # Dequantized once on miss, then served from cache on steps 2 & 3
    assert len(dequant_calls) == 1
    stats = omp.mps_dequant_cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 2
    assert stats["entries"] == 1

    # Clearing cache drops all entries and returns freed bytes
    freed = omp.clear_mps_dequant_cache()
    assert freed > 0
    assert omp.mps_dequant_cache_stats()["entries"] == 0


def test_an_invalid_mps_cache_override_is_warned_about_once(monkeypatch):
    """REMI_MPS_DEQUANT_CACHE_MB that is not a whole number fell back to the automatic budget,
    and a negative one turned caching off, both silently: one warning names the variable and
    the value. The budgets themselves are unchanged."""

    warnings = []
    monkeypatch.setattr(omp.logger, "warning", lambda message, *args: warnings.append(message % args))
    monkeypatch.setattr(omp, "_mps_cache_override_warned", False)
    gib = 1024**3
    monkeypatch.setattr(omp, "get_free_memory", lambda device: 8 * gib)
    monkeypatch.setattr(omp, "minimum_inference_memory", lambda: 1 * gib)
    automatic = int((8 * gib - 1 * gib - omp._MPS_DEQUANT_CACHE_SAFETY_MARGIN) * 0.25)

    monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", "lots")
    assert omp._mps_dequant_cache_budget_bytes() == omp._mps_dequant_cache_budget_bytes() == automatic
    assert len(warnings) == 1 and "REMI_MPS_DEQUANT_CACHE_MB" in warnings[0] and "'lots'" in warnings[0]
    monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", "-5")
    assert omp._mps_dequant_cache_budget_bytes() == 0 and len(warnings) == 1  # once per process
    monkeypatch.setattr(omp, "_mps_cache_override_warned", False)
    assert omp._mps_dequant_cache_budget_bytes() == 0
    assert len(warnings) == 2 and "REMI_MPS_DEQUANT_CACHE_MB" in warnings[1] and "'-5'" in warnings[1]
    monkeypatch.setattr(omp, "_mps_cache_override_warned", False)
    for valid, budget in (("64", 64 * 1024 * 1024), ("0", 0)):
        monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", valid)
        assert omp._mps_dequant_cache_budget_bytes() == budget
    assert len(warnings) == 2


def test_mps_dequant_cache_respects_zero_budget_and_lru_eviction(monkeypatch):
    omp.clear_mps_dequant_cache()
    omp._mps_dequant_cache.reset_stats()
    monkeypatch.setattr(omp, "is_device_mps", lambda dev: True)

    layer_a = _make_quant_linear("float8_e4m3fn", out_f=256, in_f=256)
    layer_b = _make_quant_linear("float8_e4m3fn", out_f=256, in_f=256)
    x = torch.randn(1, 8, 256, dtype=torch.float32)

    # Budget = 0 disables caching completely
    monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", "0")
    with torch.inference_mode():
        _ = layer_a(x)
    assert omp.mps_dequant_cache_stats()["entries"] == 0

    # Budget = 300 KiB fits one 256x256 fp32 weight (256 KiB), not two -> evicts layer_a when layer_b runs
    monkeypatch.delenv("REMI_MPS_DEQUANT_CACHE_MB", raising=False)
    monkeypatch.setattr(omp, "_mps_dequant_cache_budget_bytes", lambda: 300 * 1024)
    with torch.inference_mode():
        _ = layer_a(x)
        assert omp.mps_dequant_cache_stats()["entries"] == 1
        _ = layer_b(x)
    stats = omp.mps_dequant_cache_stats()
    assert stats["entries"] == 1
    assert stats["evictions"] >= 1
    omp.clear_mps_dequant_cache()


def test_mps_dequant_cache_invalidation_on_apply_set_weight_lora_and_gc(monkeypatch):
    omp.clear_mps_dequant_cache()
    monkeypatch.setattr(omp, "is_device_mps", lambda dev: True)
    monkeypatch.setenv("REMI_MPS_DEQUANT_CACHE_MB", "64")

    layer = _make_quant_linear("float8_e4m3fn", out_f=64, in_f=256)
    x = torch.randn(1, 4, 256, dtype=torch.float32)

    with torch.inference_mode():
        base_out = layer(x)
    assert omp.mps_dequant_cache_stats()["entries"] == 1

    # Moving / applying invalidates the layer's cache entries
    layer._apply(lambda t: t)
    assert omp.mps_dequant_cache_stats()["entries"] == 0

    # Re-populate, then attach a LoRA weight_function -> must bypass cache and reflect patch
    with torch.inference_mode():
        _ = layer(x)
    assert omp.mps_dequant_cache_stats()["entries"] == 1
    layer.weight_function = [lambda w: w * 2.0]
    with torch.inference_mode():
        patched_out = layer(x)
    assert not torch.allclose(patched_out, base_out)
    layer.weight_function = []

    # set_weight invalidates cached weight
    layer.set_weight(torch.ones(64, 256, dtype=torch.float32) * 0.1, seed=0)
    assert omp.mps_dequant_cache_stats()["entries"] == 0

    # GC finalizer reclaims entries when module goes out of scope
    temp_layer = _make_quant_linear("float8_e4m3fn", out_f=64, in_f=256)
    with torch.inference_mode():
        _ = temp_layer(x)
    assert omp.mps_dequant_cache_stats()["entries"] == 1
    del temp_layer
    gc.collect()
    assert omp.mps_dequant_cache_stats()["entries"] == 0


def test_convrot_linear_fp16_inplace_scaling_matches_fp32():
    layer = _make_quant_linear("int8_tensorwise", extra_conf={"convrot": True, "convrot_groupsize": 256}, out_f=128, in_f=256)
    x_fp32 = torch.randn(1, 8, 256, dtype=torch.float32)
    with torch.inference_mode():
        ref = omp._convrot_linear(x_fp32, layer.weight, layer.bias)
        out_fp16 = omp._convrot_linear(x_fp32.half(), layer.weight, layer.bias.half())
    assert out_fp16.dtype == torch.float16
    assert torch.allclose(out_fp16.float(), ref, atol=2e-2, rtol=2e-2)


def test_quantized_embedding_avoids_device_item_sync():
    ops = omp.mixed_precision_ops(compute_dtype=torch.float32, full_precision_mm=True, disabled=[])
    emb = ops.Embedding(16, 8)
    w = torch.randn(16, 8, dtype=torch.float32)
    qdata, _params = TensorCoreFP8E4M3Layout.quantize(w, scale=1.0)
    conf = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")), dtype=torch.uint8)
    emb.load_state_dict({"weight": qdata, "weight_scale": torch.ones((), dtype=torch.float32), "comfy_quant": conf})
    assert getattr(emb, "_scale_is_unit", False) is True

    # Ensure forward succeeds and produces finite values without calling .item() on a unit scale
    idx = torch.tensor([[0, 3, 7]], dtype=torch.long)
    with torch.inference_mode():
        out = emb(idx)
    assert out.shape == (1, 3, 8)
    assert torch.isfinite(out).all()


def test_no_gradio_imported_by_memory_or_mixed_precision():
    for mod in ("gradio", "gradio_client", "gradio_rangeslider"):
        assert mod not in sys.modules
