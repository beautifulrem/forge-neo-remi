"""Unit tests for tools/perf/bench_inference.py (cross-platform inference/memory/soak harness)."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("remi_perf_bench_inference", ROOT / "tools" / "perf" / "bench_inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_device_and_statistics_helpers(bench):
    assert bench.resolve_device("cpu") == torch.device("cpu")
    auto_dev = bench.resolve_device("auto")
    assert auto_dev.type in ("cuda", "mps", "cpu")
    with pytest.raises(ValueError, match="Unsupported device"):
        bench.resolve_device("tpu")

    assert bench.percentile([10.0, 20.0, 30.0, 40.0], 50.0) == pytest.approx(25.0)
    assert bench.percentile([10.0], 95.0) == 10.0
    assert bench.percentile([], 95.0) == 0.0
    assert bench.linear_slope([0.0, 1.0, 2.0, 3.0], [10.0, 12.0, 14.0, 16.0]) == pytest.approx(2.0)
    assert bench.linear_slope([1.0], [2.0]) == 0.0


def test_tensor_and_image_fidelity_metrics(bench):
    a = torch.linspace(-1.0, 1.0, 64).reshape(8, 8)
    same = bench.compute_tensor_fidelity(a, a)
    assert same["rel_l2_error"] == 0.0
    assert same["psnr_db"] == 99.99

    noisy = a + 0.01
    diff_t = bench.compute_tensor_fidelity(noisy, a)
    assert 0.0 < diff_t["rel_l2_error"] < 0.05
    assert diff_t["psnr_db"] > 35.0

    img1 = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8), mode="RGB")
    img2 = Image.fromarray(np.full((32, 32, 3), 124, dtype=np.uint8), mode="RGB")
    exact_img = bench.compute_image_fidelity(img1, img1)
    assert exact_img["psnr_db"] == 99.99
    assert exact_img["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert exact_img["mae"] == 0.0

    near_img = bench.compute_image_fidelity(img1, img2)
    assert 30.0 < near_img["psnr_db"] < 45.0
    assert near_img["mae"] == pytest.approx(4.0)
    assert near_img["max_abs_diff"] == pytest.approx(4.0)


def test_synthetic_inference_and_lifecycle_suites(bench):
    cpu = torch.device("cpu")
    inf = bench.run_synthetic_inference_suite(cpu, steps=4, warmup_steps=1, runs=2, tokens=64, dim=64, num_blocks=2, num_heads=4)
    assert inf["status"] == "ok"
    assert inf["it_per_s"] > 0
    assert inf["ms_per_step_p95"] >= inf["ms_per_step_median"] > 0
    assert inf["peak_rss_mb"] > 0

    life = bench.run_synthetic_lifecycle_suite(cpu, dim=64, num_blocks=2, num_heads=4)
    assert life["status"] == "ok"
    assert life["cold_load_s"] >= 0
    assert life["warm_load_s"] >= 0
    assert life["switch_s"] >= 0
    assert life["checkpoint_size_mb"] > 0


def test_synthetic_quantization_suite_covers_all_formats(bench):
    cpu = torch.device("cpu")
    q = bench.run_synthetic_quantization_suite(
        cpu,
        formats=("bf16", "fp8_e4m3", "fp8_rowwise", "int8_convrot", "gguf_q8_0", "svdquant_w4a4", "nvfp4"),
        in_features=256,
        out_features=256,
        tokens=32,
        steps=4,
        warmup=1,
    )
    assert q["status"] == "ok"
    formats = q["formats"]
    assert set(formats) == {"bf16", "fp8_e4m3", "fp8_rowwise", "int8_convrot", "gguf_q8_0", "svdquant_w4a4", "nvfp4"}
    for fmt in ("bf16", "fp8_e4m3", "fp8_rowwise", "int8_convrot", "gguf_q8_0", "nvfp4"):
        row = formats[fmt]
        assert row["status"] == "ok", f"{fmt}: {row}"
        assert row["ms_per_step_median"] > 0
        if fmt != "bf16":
            assert row["compression_vs_bf16"] >= 1.5
            assert row["rel_l2_error"] < 0.25


def test_synthetic_optimizations_suite(bench):
    cpu = torch.device("cpu")
    opt = bench.run_synthetic_optimizations_suite(cpu, tokens=64, heads=4, head_dim=32, steps=8, warmup=1)
    assert opt["status"] == "ok"
    assert opt["attention"]["basic"]["status"] == "ok"
    assert opt["attention"]["sdpa"]["status"] == "ok"
    assert opt["attention"]["sdpa"]["max_abs_diff"] < 1e-4
    # CUDA-only kernels report unsupported cleanly on CPU
    for name in ("sage", "flash", "xformers"):
        assert opt["attention"][name]["status"] == "unsupported"
    assert opt["accelerators"]["eager_baseline"]["status"] == "ok"
    assert opt["accelerators"]["spectrum_forecast"]["status"] == "ok"
    assert opt["accelerators"]["spectrum_forecast"]["forecast_steps"] > 0


def test_synthetic_soak_suite_runs_cleanly_without_leaks(bench):
    cpu = torch.device("cpu")
    soak = bench.run_synthetic_soak_suite(cpu, num_jobs=40, seed=7)
    assert soak["num_jobs"] == 40
    sc = soak["status_counts"]
    assert sc["queued"] == 0 and sc["running"] == 0
    assert sum(sc.values()) == 40
    assert sc["completed"] > 0
    assert sc["failed"] >= 1
    assert sc["interrupted"] >= 1
    assert sc["cancelled"] >= 1
    assert soak["model_switches"] >= 1
    assert soak["leak_detected"] is False


def test_real_checkpoint_harness_supports_remi_neo_and_comfyui(bench, monkeypatch, tmp_path):
    buf = io.BytesIO()
    ref_img = Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8), mode="RGB")
    ref_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(png_bytes)
    comfy_wf = tmp_path / "wf.json"
    comfy_wf.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}), encoding="utf-8")

    # 1. Remi protocol
    def fake_remi(method, url, body=None, timeout=600.0):
        if url.endswith("/api/v2/options"):
            return {"ok": True}
        if url.endswith("/api/v2/jobs") and method == "POST":
            return {"job_id": "job-1"}
        if url.endswith("/api/v2/jobs/job-1"):
            return {"status": "completed", "outputs": [{"id": "file-1"}]}
        if url.endswith("/api/v2/files/file-1"):
            return png_bytes
        if url.endswith("/sdapi/v1/memory"):
            return {"ram": {"used": 512 * 1024 * 1024}, "cuda": {"allocated": {"peak": 1024 * 1024 * 1024}}}
        raise AssertionError(f"Unexpected Remi request: {method} {url}")

    monkeypatch.setattr(bench, "_http_json", fake_remi)
    remi_suites = bench.run_real_checkpoint_benchmark(
        bench.RealBenchConfig(
            target="remi",
            base_url="http://127.0.0.1:7860",
            checkpoint="a.safetensors",
            second_checkpoint="b.safetensors",
            warmup_runs=1,
            runs=2,
            reference_image=ref_path,
        ),
        torch.device("cpu"),
    )
    assert remi_suites["model_lifecycle"]["switch_s"] >= 0
    assert remi_suites["inference"]["fidelity"]["psnr_db"] == 99.99
    assert remi_suites["inference"]["peak_device_mb"] == 1024.0

    # 2. Forge Neo protocol
    def fake_neo(method, url, body=None, timeout=600.0):
        if url.endswith("/sdapi/v1/options"):
            return {}
        if url.endswith("/sdapi/v1/txt2img"):
            return {"images": [png_b64], "info": "{}"}
        if url.endswith("/sdapi/v1/memory"):
            return {"ram": {"used": 600 * 1024 * 1024}}
        raise AssertionError(f"Unexpected Neo request: {method} {url}")

    monkeypatch.setattr(bench, "_http_json", fake_neo)
    neo_suites = bench.run_real_checkpoint_benchmark(
        bench.RealBenchConfig(target="forge-neo", base_url="http://127.0.0.1:7860", checkpoint="a.safetensors", warmup_runs=0, runs=2),
        torch.device("cpu"),
    )
    assert neo_suites["inference"]["target"] == "forge-neo"
    assert neo_suites["inference"]["runs"] == 2

    # 3. ComfyUI protocol
    def fake_comfy(method, url, body=None, timeout=600.0):
        if url.endswith("/prompt"):
            return {"prompt_id": "p-1"}
        if url.endswith("/history/p-1"):
            return {"p-1": {"outputs": {"9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}}
        if "/view?" in url:
            return png_bytes
        if url.endswith("/system_stats"):
            return {"devices": [{"vram_total": 8 * 1024**3, "vram_free": 5 * 1024**3}]}
        raise AssertionError(f"Unexpected ComfyUI request: {method} {url}")

    monkeypatch.setattr(bench, "_http_json", fake_comfy)
    comfy_suites = bench.run_real_checkpoint_benchmark(
        bench.RealBenchConfig(target="comfyui", base_url="http://127.0.0.1:8188", checkpoint="a.safetensors", comfy_workflow=comfy_wf, warmup_runs=0, runs=1),
        torch.device("cpu"),
    )
    assert comfy_suites["inference"]["target"] == "comfyui"
    assert comfy_suites["inference"]["peak_device_mb"] == 3072.0


def test_comparison_markdown_and_cli_roundtrip(bench, tmp_path):
    remi_json = tmp_path / "remi.json"
    neo_json = tmp_path / "neo.json"
    comfy_json = tmp_path / "comfyui.json"
    md_out = tmp_path / "comparison.md"

    assert (
        bench.main(
            [
                "--synthetic",
                "--device",
                "cpu",
                "--system",
                "remi",
                "--suites",
                "inference,model_lifecycle,quantization,optimizations,soak",
                "--quant-formats",
                "bf16,fp8_e4m3,int8_convrot",
                "--steps",
                "4",
                "--warmup-steps",
                "1",
                "--runs",
                "1",
                "--tokens",
                "64",
                "--dim",
                "64",
                "--linear-dim",
                "256",
                "--soak-jobs",
                "12",
                "--out",
                str(remi_json),
            ]
        )
        == 0
    )
    remi_data = json.loads(remi_json.read_text(encoding="utf-8"))
    assert remi_data["schema_version"] == 1
    assert remi_data["system"]["system_label"] == "remi"

    # Synthesize comparison peers for Forge Neo and ComfyUI using the exact same JSON schema
    neo_data = json.loads(json.dumps(remi_data))
    neo_data["system"]["system_label"] = "forge-neo"
    neo_data["suites"]["inference"]["it_per_s"] = round(remi_data["suites"]["inference"]["it_per_s"] * 0.95, 3)
    neo_json.write_text(json.dumps(neo_data), encoding="utf-8")

    comfy_data = json.loads(json.dumps(remi_data))
    comfy_data["system"]["system_label"] = "comfyui"
    comfy_data["suites"]["inference"]["it_per_s"] = round(remi_data["suites"]["inference"]["it_per_s"] * 1.05, 3)
    comfy_json.write_text(json.dumps(comfy_data), encoding="utf-8")

    assert bench.main(["--compare", str(remi_json), str(neo_json), str(comfy_json), "--markdown-out", str(md_out)]) == 0
    md = md_out.read_text(encoding="utf-8")
    assert "| **remi** |" in md
    assert "| **forge-neo** |" in md
    assert "| **comfyui** |" in md
    assert "## 2. Inference, Memory & Model Lifecycle" in md
    assert "## 3. Quantization Matrix" in md
    assert "## 5. Queue Soak Stability Summary" in md


def test_parity_drift_static_diagnosis(bench, tmp_path):
    out = tmp_path / "parity.json"
    assert bench.main(["--diagnose-parity", "--out", str(out)]) == 0
    diag = json.loads(out.read_text(encoding="utf-8"))
    assert diag["status"] == "diagnosed"
    assert diag["static_checks"]["get_attn_precision_returns_upcast"] is True
    assert diag["static_checks"]["macos_14_5_fp16_upcast_enabled"] is True
    assert diag["static_checks"]["wan_vae_striped_tiling_present"] is True
    assert diag["observed_drift"]["mean_rgb_diff"] == pytest.approx(8.3677)
    assert diag["observed_drift"]["max_rgb_diff"] == 247
    assert any("b59500ed" in c["commit"] for c in diag["root_causes"])


def test_bench_inference_does_not_import_gradio(bench):
    for mod in ("gradio", "gradio_client", "gradio_rangeslider"):
        assert mod not in sys.modules
