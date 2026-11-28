"""``is_oom`` recognises an out-of-memory error so callers can fall back (tiled VAE, split
attention, the queue's unload-and-report). An MPS out-of-memory error is a plain RuntimeError, and
the check must not raise on a device without CUDA."""

import torch

from backend import memory_management


def test_an_out_of_memory_error_is_recognised_without_cuda(monkeypatch):
    monkeypatch.setattr(memory_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: (_ for _ in ()).throw(AssertionError("Torch not compiled with CUDA enabled")))

    assert memory_management.is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 9.00 GiB)"))
    assert not memory_management.is_oom(RuntimeError("shape mismatch"))


def test_is_device_type_accepts_string_device_names():
    assert memory_management.is_device_mps("mps") is True
    assert memory_management.is_device_mps("mps:0") is True
    assert memory_management.is_device_cpu("cpu") is True
    assert memory_management.is_device_cuda("cuda:0") is True
    assert memory_management.supports_cast("mps", torch.float8_e4m3fn) is False
    assert memory_management.supports_cast("mps:0", torch.float8_e5m2) is False
