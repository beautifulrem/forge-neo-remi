"""Unit and integration tests for WorkerProtocol, RuntimeBridge, SubprocessWorker
crash recovery, Multi-GPU WorkerPool scheduling, MLX sidecar adapter, and worker routes."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from remi.files import FileStore
from remi.server.testing import FakeState, server_app
from remi.testing import webui_state
from remi.worker import (
    PROTOCOL_VERSION,
    InProcessWorker,
    JobEnvelope,
    MlxEnvironment,
    MlxSidecarWorker,
    PreviewChunk,
    RuntimeBridge,
    SubprocessWorker,
    WorkerCapabilities,
    WorkerMessage,
    WorkerPool,
    WorkerProgress,
    WorkerProtocol,
    WorkerResult,
    decode_message,
    detect_mlx_environment,
    encode_message,
    parse_worker_gpus,
)
from remi.worker.mlx_sidecar import build_mflux_command


def test_worker_protocol_round_trip_and_preview_frame():
    msg = WorkerMessage("run_job", JobEnvelope(job_id="j1", seq=4, tab="txt2img", core={"prompt": "hi"}).to_dict())
    raw = encode_message(msg)
    decoded = decode_message(raw)
    assert decoded.protocol_version == PROTOCOL_VERSION
    assert decoded.type == "run_job"
    env = JobEnvelope.from_dict(decoded.payload)
    assert env.job_id == "j1" and env.core["prompt"] == "hi"

    chunk = PreviewChunk.from_bytes("j1", 4, 3, b"imgbytes", fmt="webp")
    frame = PreviewChunk.from_dict(chunk.to_dict()).to_preview_frame()
    assert frame.job_seq == 4 and frame.step == 3 and frame.data == b"imgbytes" and frame.format_name == "webp"

    with pytest.raises(ValueError):
        decode_message('{"protocol_version": 999, "type": "ping", "payload": {}}')


def test_runtime_bridge_local_and_remote_modes():
    fake = FakeState()
    fake.job_count = 2
    fake.job_no = 1
    fake.sampling_steps = 10
    fake.sampling_step = 5
    bridge = RuntimeBridge(state_getter=lambda: fake)
    snap = bridge.progress_snapshot()
    assert snap["step"] == 5 and snap["steps"] == 10 and bridge.is_busy() is True
    bridge.interrupt()
    assert fake.interrupted is True

    # Remote worker snapshot overrides local state without reading shared.state
    cancelled = []
    remote_bridge = RuntimeBridge(
        remote_progress=lambda: WorkerProgress("j9", 1, 7, 10, progress=0.7, eta=1.5, text="remote"),
        remote_capabilities=lambda: WorkerCapabilities(backend="forge-pytorch", device="cuda:1", loaded_model="flux3.safetensors"),
        remote_cancel=lambda: cancelled.append(True),
    )
    assert remote_bridge.progress_snapshot()["text"] == "remote"
    assert remote_bridge.loaded_model() == "flux3.safetensors"
    remote_bridge.interrupt()
    assert cancelled == [True]


def test_subprocess_worker_real_ipc_and_crash_recovery(tmp_path):
    handler_file = tmp_path / "fake_worker_handler.py"
    handler_file.write_text(
        "import os, time\n"
        "from remi.worker.protocol import PreviewChunk, WorkerProgress, WorkerResult\n\n"
        "def handle(envelope, emit_progress, emit_preview, cancel_event):\n"
        "    if envelope.core.get('test_action') == 'crash':\n"
        "        os._exit(137)\n"
        "    emit_progress(WorkerProgress(job_id=envelope.job_id, seq=envelope.seq, step=1, steps=2, progress=0.5, text='sampling'))\n"
        "    emit_preview(PreviewChunk.from_bytes(envelope.job_id, envelope.seq, 1, b'\\xff\\xd8\\xff\\xd9', fmt='jpeg'))\n"
        "    emit_progress(WorkerProgress(job_id=envelope.job_id, seq=envelope.seq, step=2, steps=2, progress=1.0, text='done'))\n"
        "    return WorkerResult(\n"
        "        job_id=envelope.job_id,\n"
        "        status='completed',\n"
        "        output_file_ids=[f'out-{envelope.job_id}'],\n"
        "        outputs=[{'id': f'out-{envelope.job_id}', 'kind': 'image'}],\n"
        "        infotext=f'pid={os.getpid()} prompt={envelope.core.get(\"prompt\", \"\")}',\n"
        "    )\n",
        encoding="utf-8",
    )
    worker = SubprocessWorker(
        worker_id="worker-sub-test",
        handler_import=f"{handler_file}:handle",
        auto_restart=True,
        max_restarts=3,
        restart_backoff_s=0.05,
        startup_timeout=10.0,
    )
    assert isinstance(worker, WorkerProtocol)
    worker.start()
    try:
        hb1 = worker.status()
        assert hb1.state == "idle" and hb1.pid is not None
        first_pid = hb1.pid

        progress_seen: list[WorkerProgress] = []
        previews_seen: list[PreviewChunk] = []
        res1 = worker.execute(
            JobEnvelope(job_id="ok-1", seq=1, tab="txt2img", core={"prompt": "aurora"}),
            on_progress=progress_seen.append,
            on_preview=previews_seen.append,
        )
        assert res1.status == "completed"
        assert res1.output_file_ids == ["out-ok-1"]
        assert f"pid={first_pid}" in (res1.infotext or "")
        assert len(progress_seen) == 2 and len(previews_seen) == 1

        # Trigger OOM/SIGKILL crash inside child worker -> supervisor MUST recover automatically!
        res_crash = worker.execute(JobEnvelope(job_id="crash-1", seq=2, tab="txt2img", core={"test_action": "crash"}))
        assert res_crash.status == "failed"
        assert "crashed" in (res_crash.error or "").lower()
        assert "restarted" in (res_crash.error or "").lower()

        hb2 = worker.status()
        assert hb2.state == "idle" and hb2.restarts == 1 and hb2.pid is not None and hb2.pid != first_pid

        # Subsequent job succeeds cleanly on the newly respawned worker process
        res2 = worker.execute(JobEnvelope(job_id="ok-2", seq=3, tab="txt2img", core={"prompt": "after recovery"}))
        assert res2.status == "completed"
        assert f"pid={hb2.pid}" in (res2.infotext or "")
        assert worker.status().jobs_completed == 2 and worker.status().jobs_failed == 1
    finally:
        worker.stop()
        assert worker.status().state == "stopped"


# A child speaking the protocol by hand: noisy on stderr first, then it answers the ping
CHATTY_CHILD = """
import json, sys
sys.stderr.buffer.write(b"x" * 262144)
sys.stderr.flush()
for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "ping":
        print(json.dumps({"protocol_version": 1, "type": "pong", "payload": {}}), flush=True)
    elif message["type"] == "shutdown":
        break
"""


def test_a_worker_that_writes_a_lot_to_stderr_still_starts():
    # stderr was a pipe nobody read before the process died: a child writing more than the
    # pipe holds blocked, never answered the ping, and start() waited forever
    worker = SubprocessWorker(worker_id="w-chatty", command=[sys.executable, "-c", CHATTY_CHILD], auto_restart=False, startup_timeout=10.0)
    starter = threading.Thread(target=worker.start, daemon=True)
    try:
        starter.start()
        starter.join(timeout=12)
        assert not starter.is_alive() and worker.status().state == "idle"
    finally:
        worker.stop()
        starter.join(timeout=10)


def test_a_worker_that_never_answers_times_out_on_time():
    # the deadline was only checked between lines: a silent child kept start() in readline
    worker = SubprocessWorker(worker_id="w-silent", command=[sys.executable, "-c", "import time; time.sleep(6)"], auto_restart=False, startup_timeout=1.0)
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            worker.start()
        assert time.monotonic() - began < worker.startup_timeout + 1
        assert worker.status().state == "error"
    finally:
        worker.stop()


def test_a_restart_during_a_job_ends_the_job_and_leaves_one_worker(tmp_path):
    handler_file = tmp_path / "slow_handler.py"
    handler_file.write_text(
        "import time\n"
        "from remi.worker.protocol import WorkerResult\n\n"
        "def handle(envelope, emit_progress, emit_preview, cancel_event):\n"
        "    if envelope.core.get('hang'):\n"
        "        time.sleep(60)\n"
        "    return WorkerResult(job_id=envelope.job_id, status='completed')\n",
        encoding="utf-8",
    )
    worker = SubprocessWorker(worker_id="w-restart", handler_import=f"{handler_file}:handle", auto_restart=False, restart_backoff_s=0.0, startup_timeout=10.0)
    worker.start()
    # the job's crash handling lands after the restart started the new process (it used to forget that process)
    started = threading.Event()
    start, crashed = worker.start, worker._handle_crash
    worker.start = lambda: (start(), started.set())[0]
    worker._handle_crash = lambda *args, **kwargs: (started.wait(timeout=2), crashed(*args, **kwargs))[1]
    results = []
    try:
        job = threading.Thread(target=lambda: results.append(worker.execute(JobEnvelope(job_id="hung", seq=1, tab="txt2img", core={"hang": True}))), daemon=True)
        job.start()
        deadline = time.monotonic() + 10
        while worker.status().current_job_id != "hung" and time.monotonic() < deadline:
            time.sleep(0.02)
        restarted = worker.restart()  # the user's way out of a hung job (POST /workers/{id}/restart)
        job.join(timeout=10)
        assert not job.is_alive() and results[0].status == "failed"
        assert restarted.state == "idle" and restarted.pid is not None and worker.status().pid == restarted.pid
        assert worker.execute(JobEnvelope(job_id="next", seq=2, tab="txt2img", core={})).status == "completed"
    finally:
        worker.stop()


FAKE_MFLUX = """#!{python}
import sys
sys.stderr.write("progress " * 32768)  # 256 KB: more than a pipe holds
sys.stderr.flush()
output = sys.argv[sys.argv.index("--output") + 1]
open(output, "wb").write(b"png")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the fake mflux-generate is a script with a shebang")
def test_the_mlx_sidecar_reads_mflux_output_while_it_runs(tmp_path, monkeypatch):
    # stdout and stderr were pipes read only after the process ended: mflux's progress output
    # filled the stderr pipe and the generation never finished
    import os

    from remi.worker import mlx_sidecar

    tool = tmp_path / "bin" / "mflux-generate"
    tool.parent.mkdir()
    tool.write_text(FAKE_MFLUX.format(python=sys.executable), encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tool.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    cancel = threading.Event()
    results = []
    envelope = JobEnvelope(job_id="m-loud", seq=1, tab="txt2img", core={"prompt": "sun"})
    runner = threading.Thread(target=lambda: results.append(mlx_sidecar._default_mflux_runner(envelope, tmp_path / "o.png", None, cancel)), daemon=True)
    try:
        runner.start()
        runner.join(timeout=10)
        assert not runner.is_alive() and "MLX Sidecar" in results[0]["infotext"]
        assert (tmp_path / "o.png").read_bytes() == b"png"
    finally:
        cancel.set()
        runner.join(timeout=10)


def test_a_prompt_starting_with_a_dash_stays_the_prompt():
    command = build_mflux_command(JobEnvelope(job_id="m-dash", seq=1, tab="txt2img", core={"prompt": "-x"}), Path("out.png"))
    assert "--prompt=-x" in command and "-x" not in command


def test_multi_gpu_worker_pool_affinity_and_concurrent_dispatch():
    assert parse_worker_gpus("0, 1, 0, 2") == ["0", "1", "2"]
    assert parse_worker_gpus(None) == []
    # ids only (CUDA_VISIBLE_DEVICES takes indices and GPU / MIG UUIDs): anything else is dropped with a warning
    assert parse_worker_gpus("0, x, 1") == ["0", "1"]
    uuid = "GPU-8a4e1b1f-0c3d-4f5e-9a7b-1c2d3e4f5a6b"
    assert parse_worker_gpus(f"{uuid},MIG-1c2d3e4f-5a6b-7c8d-9e0f-1a2b3c4d5e6f,0 1,--x,-1") == [uuid, "MIG-1c2d3e4f-5a6b-7c8d-9e0f-1a2b3c4d5e6f"]

    active_barrier = threading.Barrier(2, timeout=5.0)
    ran_on: dict[str, str] = {}

    def make_gpu_worker(wid: str, loaded_model: str, vram_free: int) -> InProcessWorker:
        caps = WorkerCapabilities(
            backend="forge-pytorch",
            device=f"cuda:{wid[-1]}",
            gpu_name="Test GPU",
            vram_total_mb=24576,
            vram_free_mb=vram_free,
            supported_families=("anima", "flux3"),
            loaded_model=loaded_model,
        )
        bridge = RuntimeBridge(remote_capabilities=lambda: caps)

        def _exec(env: JobEnvelope) -> WorkerResult:
            ran_on[env.job_id] = wid
            if env.core.get("parallel"):
                active_barrier.wait()
            return WorkerResult(job_id=env.job_id, status="completed", output_file_ids=[f"{wid}-{env.job_id}"])

        return InProcessWorker(worker_id=wid, bridge=bridge, executor=_exec)

    w0 = make_gpu_worker("worker-gpu-0", loaded_model="anima.safetensors", vram_free=12000)
    w1 = make_gpu_worker("worker-gpu-1", loaded_model="flux3.safetensors", vram_free=20000)
    pool = WorkerPool([w0, w1], mode="pool")
    try:
        # Affinity selects worker-gpu-0 for anima.safetensors and worker-gpu-1 for flux3.safetensors
        assert pool.select_worker(preferred_model="anima.safetensors").worker_id == "worker-gpu-0"
        assert pool.select_worker(preferred_model="flux3.safetensors").worker_id == "worker-gpu-1"

        # Concurrent dispatch proves BOTH GPU workers execute simultaneously (Barrier(2) would timeout if serial!)
        results = pool.dispatch_concurrent(
            [
                JobEnvelope(job_id="p1", seq=1, tab="txt2img", core={"parallel": True}, model_checkpoint="anima.safetensors"),
                JobEnvelope(job_id="p2", seq=2, tab="txt2img", core={"parallel": True}, model_checkpoint="flux3.safetensors"),
            ]
        )
        assert all(r.status == "completed" for r in results)
        assert ran_on == {"p1": "worker-gpu-0", "p2": "worker-gpu-1"}
    finally:
        pool.shutdown()


def test_mlx_sidecar_worker_adapter_and_unavailable_fallback(tmp_path):
    # Unavailable environment degrades cleanly
    unavail = MlxSidecarWorker(env_probe=lambda: MlxEnvironment(available=False, platform_supported=False, reason="not macOS arm64"))
    assert unavail.status().state == "unavailable"
    fail_res = unavail.execute(JobEnvelope(job_id="m0", seq=1, tab="txt2img", core={"prompt": "test"}))
    assert fail_res.status == "failed" and "not macOS arm64" in (fail_res.error or "")

    # Command builder check
    cmd = build_mflux_command(JobEnvelope(job_id="m1", seq=1, tab="txt2img", core={"prompt": "cat", "width": 512, "height": 512}, scripts={"sampler": {"steps": 4}, "seed": {"seed": 42}}), Path("/tmp/out.png"))
    assert "--prompt=cat" in cmd and "--seed" in cmd and "42" in cmd

    # Available environment with mock runner writes output into FileStore cleanly
    files = FileStore(tmp_path / "files")
    try:
        def fake_mlx_runner(envelope, output_path, on_progress, cancel_event):
            Image.new("RGB", (16, 16), (0, 200, 120)).save(output_path, format="PNG")
            return {"infotext": "cat\nBackend: MLX", "model": "flux-schnell-4bit"}

        mlx = MlxSidecarWorker(
            env_probe=lambda: MlxEnvironment(available=True, platform_supported=True, mlx_version="0.22", mflux_version="0.6", unified_memory_mb=65536),
            runner=fake_mlx_runner,
        )
        assert isinstance(mlx, WorkerProtocol)
        assert mlx.status().state == "idle"
        res = mlx.execute(JobEnvelope(job_id="m2", seq=2, tab="txt2img", core={"prompt": "cat"}, file_store_root=str(files.root)))
        assert res.status == "completed" and len(res.output_file_ids) == 1
        assert files.get(res.output_file_ids[0]) is not None
        assert mlx.status().jobs_completed == 1
        assert detect_mlx_environment().to_dict()["supported_families"]
    finally:
        files.close()


def test_workers_http_routes(tmp_path):
    with webui_state(), server_app(tmp_path) as server:
        resp = server.client.get("/api/v2/workers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "inprocess"
        assert len(data["workers"]) == 1
        wid = data["workers"][0]["worker_id"]
        assert "mlx" in data and "available" in data["mlx"]

        one = server.client.get(f"/api/v2/workers/{wid}")
        assert one.status_code == 200 and one.json()["worker_id"] == wid

        restarted = server.client.post(f"/api/v2/workers/{wid}/restart")
        assert restarted.status_code == 200 and restarted.json()["ok"] is True
        assert restarted.json()["worker"]["restarts"] == 1

        assert server.client.post("/api/v2/workers/does-not-exist/restart").status_code == 404


def test_cli_worker_flags_parsing_and_mode_selection(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from modules import cmd_args
    from remi.jobs import run_inline
    from remi.server import launch
    from remi.server import services as services_mod
    from remi.worker import pool as pool_mod

    # 1. Verify argparse defaults & flag parsing in modules/cmd_args.py
    defaults = cmd_args.parser.parse_args([])
    assert defaults.worker_process is False
    assert defaults.worker_gpus is None
    assert defaults.worker_mlx is False

    parsed_sub = cmd_args.parser.parse_args(["--worker-process"])
    assert parsed_sub.worker_process is True and parsed_sub.worker_gpus is None and parsed_sub.worker_mlx is False

    parsed_pool = cmd_args.parser.parse_args(["--worker-gpus", "0,1", "--worker-mlx"])
    assert parsed_pool.worker_gpus == "0,1" and parsed_pool.worker_mlx is True

    # Clear env vars so tests deterministically check CLI options
    for key in ("REMI_WORKER_PROCESS", "REMI_WORKER_GPUS", "REMI_WORKER_MLX"):
        monkeypatch.delenv(key, raising=False)

    # 2. worker_options_from_cmd_opts + launch.worker_mode_summary; the line says the flags are a
    # preview (it read as if jobs ran in the isolated workers, which the queue does not dispatch to)
    assert services_mod.worker_options_from_cmd_opts(defaults) == (False, None, False)
    assert launch.worker_mode_summary(defaults) is None
    assert services_mod.worker_options_from_cmd_opts(parsed_sub) == (True, None, False)
    assert launch.worker_mode_summary(parsed_sub) == "Worker mode: subprocess (isolated worker) (preview: queued jobs still run in this process)"
    assert services_mod.worker_options_from_cmd_opts(parsed_pool) == (False, "0,1", True)
    assert launch.worker_mode_summary(parsed_pool) == "Worker mode: pool (GPUs: 0, 1), MLX sidecar enabled (preview: queued jobs still run in this process)"
    assert launch.worker_mode_summary(SimpleNamespace(worker_process=False, worker_gpus=None, worker_mlx=True)) == "Worker mode: inprocess, MLX sidecar enabled (preview: queued jobs still run in this process)"

    # 3. Services.create mode selection: default in-process, subprocess, multi-GPU pool, and MLX sidecar
    svc_default = services_mod.Services.create(tmp_path / "inproc", dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        assert svc_default.workers.mode == "inprocess"
        assert [type(w).__name__ for w in svc_default.workers.workers()] == ["InProcessWorker"]
    finally:
        svc_default.close()

    svc_sub = services_mod.Services.create(tmp_path / "sub", dispatch=run_inline, realtime=False, housekeeping=False, worker_process=True)
    try:
        assert svc_sub.workers.mode == "subprocess"
        assert [w.worker_id for w in svc_sub.workers.workers()] == ["worker-0"]
        assert isinstance(svc_sub.workers.get_worker("worker-0"), SubprocessWorker)
    finally:
        svc_sub.close()

    svc_pool = services_mod.Services.create(tmp_path / "pool", dispatch=run_inline, realtime=False, housekeeping=False, worker_gpus="0,1")
    try:
        assert svc_pool.workers.mode == "pool"
        assert [w.worker_id for w in svc_pool.workers.workers()] == ["worker-gpu-0", "worker-gpu-1"]
        assert all(isinstance(w, SubprocessWorker) for w in svc_pool.workers.workers())
    finally:
        svc_pool.close()

    # Force platform_supported=True to verify --worker-mlx attaches MlxSidecarWorker cleanly
    monkeypatch.setattr(
        pool_mod,
        "detect_mlx_environment",
        lambda: MlxEnvironment(available=True, platform_supported=True, mlx_version="0.22", mflux_version="0.6", unified_memory_mb=32768),
    )
    svc_mlx = services_mod.Services.create(tmp_path / "mlx", dispatch=run_inline, realtime=False, housekeeping=False, worker_mlx=True)
    try:
        assert svc_mlx.workers.mode == "inprocess"
        ids = [w.worker_id for w in svc_mlx.workers.workers()]
        assert "worker-inprocess" in ids and "worker-mlx-0" in ids
        assert isinstance(svc_mlx.workers.get_worker("worker-mlx-0"), MlxSidecarWorker)
    finally:
        svc_mlx.close()

    # 4. default_services() reads modules.shared_cmd_options.cmd_opts
    import modules.paths_internal
    import modules.shared_cmd_options
    import remi.server.settings

    monkeypatch.setattr(services_mod, "_default", None)
    monkeypatch.setattr(modules.paths_internal, "data_path", str(tmp_path / "default"))
    monkeypatch.setattr(remi.server.settings, "option", lambda name, default: default)
    monkeypatch.setattr(modules.shared_cmd_options, "cmd_opts", SimpleNamespace(worker_process=True, worker_gpus=None, worker_mlx=False))
    svc_cli = services_mod.default_services()
    try:
        assert svc_cli.workers.mode == "subprocess"
        assert isinstance(svc_cli.workers.get_worker("worker-0"), SubprocessWorker)
    finally:
        svc_cli.close()
        monkeypatch.setattr(services_mod, "_default", None)


def test_subprocess_worker_stdio_loop_helpers_and_edge_cases(tmp_path, monkeypatch):
    import io
    import subprocess
    from types import SimpleNamespace

    from remi.jobs.runner import JobResult
    from remi.worker import subprocess_worker as sw

    # 1. _resolve_handler: invalid format, .py path, dotted import, non-callable
    with pytest.raises(ValueError):
        sw._resolve_handler("no_colon_here")

    mod_file = tmp_path / "custom_handler.py"
    mod_file.write_text("NOT_CALLABLE = 123\ndef fn(*a): return None\n", encoding="utf-8")
    assert callable(sw._resolve_handler(f"{mod_file}:fn"))
    with pytest.raises(TypeError):
        sw._resolve_handler(f"{mod_file}:NOT_CALLABLE")
    assert callable(sw._resolve_handler("remi.worker.subprocess_worker:default_forge_handler"))

    # 2. run_worker_stdio_loop in-process (ping, heartbeat, invalid JSON, run_job ok + error, cancel, shutdown, EOF)
    seen_cancel: list[bool] = []

    def sample_handler(env, emit_progress, emit_preview, cancel_ev):
        if env.core.get("fail"):
            raise RuntimeError("boom in handler")
        emit_progress(WorkerProgress(job_id=env.job_id, seq=env.seq, step=1, steps=1, progress=1.0))
        emit_preview(PreviewChunk.from_bytes(env.job_id, env.seq, 1, b"abc", fmt="png"))
        seen_cancel.append(cancel_ev.is_set())
        return WorkerResult(job_id=env.job_id, status="completed", output_file_ids=["f1"])

    in_lines = b"".join(
        [
            b"not-valid-json\n",
            encode_message(WorkerMessage("ping", {"worker_id": "w1"})),
            encode_message(WorkerMessage("heartbeat", {})),
            encode_message(WorkerMessage("cancel", {"job_id": "j-none"})),
            encode_message(WorkerMessage("run_job", JobEnvelope(job_id="j1", seq=1, tab="txt2img", core={"prompt": "ok"}).to_dict())),
            encode_message(WorkerMessage("run_job", JobEnvelope(job_id="j2", seq=2, tab="txt2img", core={"fail": True}).to_dict())),
            encode_message(WorkerMessage("shutdown", {})),
        ]
    )
    out_buf = io.BytesIO()
    assert sw.run_worker_stdio_loop(sample_handler, worker_id="w1", stdin=io.BytesIO(in_lines), stdout=out_buf) == 0
    msgs = [decode_message(line) for line in out_buf.getvalue().splitlines() if line]
    types_seen = [m.type for m in msgs]
    assert types_seen == ["error", "pong", "heartbeat", "progress", "preview", "result", "result"]
    assert msgs[-1].payload["status"] == "failed" and "boom in handler" in msgs[-1].payload["error"]

    # EOF exits cleanly with 0
    assert sw.run_worker_stdio_loop(sample_handler, stdin=io.BytesIO(b""), stdout=io.BytesIO()) == 0

    # 3. main() CLI parser delegates to run_worker_stdio_loop
    main_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(sw, "run_worker_stdio_loop", lambda handler, *, worker_id, capabilities: main_calls.append((worker_id, capabilities.device)) or 0)
    assert sw.main(["--worker-id", "w-gpu", "--gpu", "2", "--handler", f"{mod_file}:fn"]) == 0
    assert main_calls == [("w-gpu", "cuda:2")]

    # 4. default_forge_handler with stubbed run_job (both saved-file and unsaved-image branches)
    files_root = tmp_path / "forge_files"
    saved_png = tmp_path / "already.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(saved_png, format="PNG")
    img_saved = Image.new("RGB", (8, 8), (10, 20, 30))
    img_saved.already_saved_as = str(saved_png)
    img_mem = Image.new("RGB", (8, 8), (40, 50, 60))

    import remi.jobs.runner as runner_mod

    monkeypatch.setattr(runner_mod, "run_job", lambda job, ctx: JobResult(images=[img_saved, img_mem], infotexts=["a", "b"], info="a"))
    res = sw.default_forge_handler(
        JobEnvelope(job_id="j-forge", seq=1, tab="txt2img", core={"prompt": "hi"}, file_store_root=str(files_root)),
        lambda _p: None,
        lambda _c: None,
        threading.Event(),
    )
    assert res.status == "completed" and len(res.output_file_ids) == 2

    # 5. SubprocessWorker command/env builders, capabilities, cancel, error/timeout/crash paths
    w_custom = SubprocessWorker(worker_id="w-gpu", gpu_id="1", command=["echo", "hi"], auto_restart=False)
    assert w_custom._build_command() == ["echo", "hi"]
    assert w_custom._build_env()["CUDA_VISIBLE_DEVICES"] == "1"
    assert w_custom.capabilities().device == "cuda:1"
    assert w_custom.cancel("missing") is False

    w_gpu_default = SubprocessWorker(worker_id="w-gpu2", gpu_id="3")
    assert "--gpu" in w_gpu_default._build_command()

    # Fake process for _handshake (capabilities + pong, EOF crash, timeout), execute (capabilities/failed/error/OSError), cancel, _terminate_proc
    class FakePipe(io.BytesIO):
        def __init__(self, initial: bytes = b"", *, fail_write: bool = False):
            super().__init__(initial)
            self.fail_write = fail_write

        def write(self, data):
            if self.fail_write:
                raise OSError("broken pipe")
            return super().write(data)

    class FakeProc:
        def __init__(self, stdout_bytes: bytes, *, rc: int | None = None, stderr_bytes: bytes = b"err", fail_write: bool = False, hang_wait: int = 0):
            self.pid = 4242
            self.stdin = FakePipe(fail_write=fail_write)
            self.stdout = FakePipe(stdout_bytes)
            self.stderr = FakePipe(stderr_bytes)
            self._rc = rc
            self._hang_wait = hang_wait
            self.terminated = False
            self.killed = False

        def poll(self):
            return self._rc

        def wait(self, timeout=None):
            if self._hang_wait > 0:
                self._hang_wait -= 1
                raise subprocess.TimeoutExpired("fake", timeout or 1.0)
            if self._rc is None:
                self._rc = 0
            return self._rc

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self._rc = -9

    # _handshake: capabilities frame followed by pong
    w_fake = SubprocessWorker(worker_id="w-fake", auto_restart=False, startup_timeout=1.0)
    hs_out = encode_message(WorkerMessage("capabilities", WorkerCapabilities(device="cuda:0", loaded_model="m1").to_dict())) + encode_message(WorkerMessage("pong", {}))
    proc_ok = FakeProc(hs_out)
    w_fake._handshake(proc_ok)
    assert w_fake.capabilities().loaded_model == "m1"

    # _handshake: immediate EOF raises RuntimeError; start() marks state error
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakeProc(b"", rc=1, stderr_bytes=b"startup boom"))
    with pytest.raises(RuntimeError, match="exited during startup"):
        w_fake.start()
    assert w_fake.status().state == "error"

    # _handshake: timeout raises TimeoutError
    w_to = SubprocessWorker(worker_id="w-to", startup_timeout=0.0)
    with pytest.raises(TimeoutError):
        w_to._handshake(FakeProc(b""))

    # _terminate_proc: timeout -> terminate -> timeout -> kill
    hung = FakeProc(b"", rc=None, hang_wait=2)
    SubprocessWorker._terminate_proc(hung, timeout=0.01)
    assert hung.terminated and hung.killed

    # execute: capabilities + failed result, error message, OSError crash, cancel True / False
    w_exec = SubprocessWorker(worker_id="w-ex", auto_restart=False)
    out_seq = (
        encode_message(WorkerMessage("capabilities", WorkerCapabilities(device="cuda:0", loaded_model="m2").to_dict()))
        + encode_message(WorkerMessage("result", WorkerResult(job_id="j-fail", status="failed", error="bad").to_dict()))
    )
    proc_exec = FakeProc(out_seq)
    w_exec._proc = proc_exec
    w_exec._state = "idle"
    w_exec.start()  # idempotent when already alive
    res_fail = w_exec.execute(JobEnvelope(job_id="j-fail", seq=1, tab="txt2img", core={}))
    assert res_fail.status == "failed" and w_exec.capabilities().loaded_model == "m2"

    proc_err = FakeProc(encode_message(WorkerMessage("error", {"error": "protocol fault"})))
    w_exec._proc = proc_err
    res_err = w_exec.execute(JobEnvelope(job_id="j-err", seq=2, tab="txt2img", core={}))
    assert res_err.status == "failed" and "protocol fault" in (res_err.error or "")

    # cancel while busy (ok & OSError)
    w_exec._proc = FakeProc(b"")
    w_exec._current_job_id = "j-active"
    assert w_exec.cancel("j-active") is True
    w_exec._proc = FakeProc(b"", fail_write=True)
    assert w_exec.cancel("j-active") is False

    # OSError during execute triggers _handle_crash (auto_restart=True with restart failure logs cleanly)
    w_crash = SubprocessWorker(worker_id="w-cr", auto_restart=True, max_restarts=2, restart_backoff_s=0.0)
    w_crash._proc = FakeProc(b"", fail_write=True)
    w_crash._state = "idle"
    monkeypatch.setattr(w_crash, "restart", lambda: (_ for _ in ()).throw(RuntimeError("respawn failed")))
    res_os = w_crash.execute(JobEnvelope(job_id="j-os", seq=3, tab="txt2img", core={}))
    assert res_os.status == "failed" and "broken pipe" in (res_os.error or "")

    # status() detects dead proc
    dead_proc = FakeProc(b"", rc=1)
    w_exec._proc = dead_proc
    w_exec._state = "idle"
    assert w_exec.status().state == "error"
    _ = SimpleNamespace()


def test_mlx_sidecar_pool_and_bridge_edge_cases(tmp_path, monkeypatch):
    import importlib.util
    import platform
    import shutil
    import subprocess
    import sys
    import types
    from types import SimpleNamespace

    from remi.worker import mlx_sidecar as mlx_mod
    from remi.worker.protocol import JobEnvelope, WorkerCapabilities, WorkerProgress, WorkerResult

    real_pkg_version = mlx_mod._pkg_version

    # 1. MLX environment detection branches
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert mlx_mod.detect_mlx_environment().platform_supported is False

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mlx_mod, "_pkg_version", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    env_no_mlx = mlx_mod.detect_mlx_environment()
    assert env_no_mlx.platform_supported and not env_no_mlx.available and "mlx" in (env_no_mlx.reason or "")

    monkeypatch.setattr(mlx_mod, "_pkg_version", lambda name: "0.22" if name == "mlx" else None)
    env_no_mflux = mlx_mod.detect_mlx_environment()
    assert not env_no_mflux.available and "mflux" in (env_no_mflux.reason or "")

    monkeypatch.setattr(mlx_mod, "_pkg_version", lambda name: "0.22" if name == "mlx" else "0.6")
    env_ok = mlx_mod.detect_mlx_environment()
    assert env_ok.available and env_ok.mlx_version == "0.22" and env_ok.mflux_version == "0.6"

    # _pkg_version fallback when metadata.version fails
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(mlx_mod, "_pkg_version", real_pkg_version)
    assert mlx_mod._pkg_version("nonexistent-pkg-meta-xyz") == "installed"

    # _default_mflux_runner success, non-zero exit, and cancel
    class FakeMfluxProc:
        def __init__(self, rc: int = 0, polls: list[int | None] | None = None, stderr: str = ""):
            self._polls = list(polls or [0])
            self.returncode = rc
            self._stderr = stderr
            self.stdout = None
            self.stderr = None
            self.terminated = False

        def poll(self):
            return self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]

        def communicate(self, timeout=None):
            return ("", self._stderr)

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    progress_list: list[WorkerProgress] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakeMfluxProc(rc=0, polls=[0]))
    info = mlx_mod._default_mflux_runner(JobEnvelope(job_id="m-run", seq=1, tab="txt2img", core={"prompt": "sun"}), tmp_path / "o.png", progress_list.append, threading.Event())
    assert "MLX Sidecar" in info["infotext"] and len(progress_list) == 2

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakeMfluxProc(rc=2, polls=[2], stderr="bad arg"))
    with pytest.raises(RuntimeError, match="mflux-generate failed"):
        mlx_mod._default_mflux_runner(JobEnvelope(job_id="m-fail", seq=1, tab="txt2img", core={"prompt": "sun"}), tmp_path / "o.png", None, threading.Event())

    ev_cancel = threading.Event()
    ev_cancel.set()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakeMfluxProc(rc=0, polls=[None, 0]))
    with pytest.raises(InterruptedError):
        mlx_mod._default_mflux_runner(JobEnvelope(job_id="m-can", seq=1, tab="txt2img", core={"prompt": "sun"}), tmp_path / "o.png", None, ev_cancel)

    # MlxSidecarWorker start/stop/restart/is_alive/cancel/execute error branches
    avail_env = MlxEnvironment(available=True, platform_supported=True, unified_memory_mb=16384)
    w_mlx = MlxSidecarWorker(env_probe=lambda: avail_env, runner=lambda env, out, prog, cev: {})
    assert w_mlx.is_alive() is True
    assert w_mlx.restart().restarts == 1
    assert w_mlx.cancel("nope") is False
    # runner completes without writing file -> failed
    r_missing = w_mlx.execute(JobEnvelope(job_id="m-nofile", seq=1, tab="txt2img", core={}))
    assert r_missing.status == "failed" and "without writing" in (r_missing.error or "")

    # runner triggers cancel_event -> cancelled
    def cancel_runner(env, out, prog, cev):
        assert w_mlx.cancel(env.job_id) is True
        return {}

    w_mlx._runner = cancel_runner
    r_can = w_mlx.execute(JobEnvelope(job_id="m-c1", seq=2, tab="txt2img", core={}))
    assert r_can.status == "cancelled"

    # runner raises InterruptedError -> cancelled
    w_mlx._runner = lambda env, out, prog, cev: (_ for _ in ()).throw(InterruptedError("stop"))
    assert w_mlx.execute(JobEnvelope(job_id="m-c2", seq=3, tab="txt2img", core={})).status == "cancelled"

    w_mlx._cancel_event = threading.Event()
    w_mlx.stop()
    assert w_mlx._cancel_event.is_set() and not w_mlx.is_alive()
    w_mlx._env_probe = lambda: MlxEnvironment(available=False, platform_supported=False)
    w_mlx.start()
    assert w_mlx.status().state == "unavailable"

    # 2. InProcessWorker & WorkerPool edge cases
    assert parse_worker_gpus([0, 1, 0]) == ["0", "1"]
    inproc = InProcessWorker(worker_id="ip-1")
    inproc.start()
    assert inproc.cancel("none") is False
    assert inproc.execute(JobEnvelope(job_id="j-noexec", seq=1, tab="txt2img", core={})).status == "failed"

    inproc_err = InProcessWorker(worker_id="ip-2", executor=lambda env: (_ for _ in ()).throw(ValueError("bad")))
    assert inproc_err.execute(JobEnvelope(job_id="j-err", seq=2, tab="txt2img", core={})).status == "failed"
    inproc_err._current_job_id = "j-live"
    assert inproc_err.cancel("j-live") is True

    pool = WorkerPool(mode="pool")
    pool.add_worker(inproc)
    pool.start_all()
    assert pool.select_worker(preferred_backend="mlx-mflux") is None
    assert pool.select_worker(model_family="unknown-family-xyz") is None
    assert pool.acquire_worker(preferred_backend="mlx-mflux", timeout=0.01) is None
    assert pool.execute(JobEnvelope(job_id="j-none", seq=1, tab="txt2img", core={}), preferred_backend="mlx-mflux", timeout=0.01).status == "failed"
    assert pool.dispatch_concurrent([]) == []
    inproc.stop()
    assert pool.select_worker() is None
    assert pool.remove_worker("ip-1") is True
    assert pool.remove_worker("ip-1") is False

    # 3. RuntimeBridge full coverage (None state, loaded_model_hash, loaded_weights, skip, stop_generating, CUDA & MPS torch snapshots)
    empty_bridge = RuntimeBridge(state_getter=lambda: None)
    assert empty_bridge.progress_snapshot()["step"] == 0
    assert empty_bridge.is_busy() is False

    calls: list[str] = []
    fake_state = SimpleNamespace(skip=lambda: calls.append("skip"), stop_generating=lambda: calls.append("stop"))
    b_state = RuntimeBridge(state_getter=lambda: fake_state)
    b_state.skip()
    b_state.stop_generating()
    assert calls == ["skip", "stop"]

    fake_sd_models = types.ModuleType("modules.sd_models")
    fake_sd_models.model_data = SimpleNamespace(sd_model=SimpleNamespace(sd_checkpoint_info=SimpleNamespace(title="ckpt.safetensors", sha256="abc123")))
    monkeypatch.setitem(sys.modules, "modules.sd_models", fake_sd_models)
    assert b_state.loaded_model_hash() == "abc123"

    class GoodEntry:
        model = SimpleNamespace(model=SimpleNamespace())
        device = "cuda:0"

        def model_loaded_memory(self):
            return 512 * 1024 * 1024

    class BadEntry:
        @property
        def model(self):
            raise RuntimeError("corrupt")

    fake_mm = types.ModuleType("backend.memory_management")
    fake_mm.current_loaded_models = [GoodEntry(), BadEntry()]
    monkeypatch.setitem(sys.modules, "backend.memory_management", fake_mm)
    weights = b_state.loaded_weights()
    assert len(weights) == 2 and weights[0]["memory_mb"] == 512 and "RuntimeError" in weights[1]["name"]

    # Torch CUDA & MPS capability probes
    fake_torch_cuda = types.ModuleType("torch")
    fake_torch_cuda.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "RTX 4090", mem_get_info=lambda i: (16 * 1024**3, 24 * 1024**3))
    monkeypatch.setitem(sys.modules, "torch", fake_torch_cuda)
    caps_cuda = b_state.capabilities_snapshot()
    assert caps_cuda.device == "cuda:0" and caps_cuda.gpu_name == "RTX 4090" and caps_cuda.vram_total_mb == 24576

    fake_torch_mps = types.ModuleType("torch")
    fake_torch_mps.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch_mps.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch_mps)
    assert b_state.capabilities_snapshot().device == "mps"

    fake_torch_err = types.ModuleType("torch")
    fake_torch_err.cuda = SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("no gpu")))
    monkeypatch.setitem(sys.modules, "torch", fake_torch_err)
    assert b_state.capabilities_snapshot().device == "cpu"

    # Protocol helper edge checks
    from remi.worker.protocol import WorkerHeartbeat

    hb = WorkerHeartbeat(worker_id="w", pid=None, state="idle", capabilities=WorkerCapabilities())
    assert hb.to_status_dict(iso=lambda ts: "2026-01-01T00:00:00Z" if ts else None)["worker_id"] == "w"
    assert WorkerProgress.from_dict(WorkerProgress(job_id="j", seq=1, step=1, steps=2).to_dict()).step == 1
    assert WorkerResult.from_dict(WorkerResult(job_id="j", status="completed").to_dict()).job_id == "j"


def test_subprocess_worker_stop_during_active_job_does_not_respawn(tmp_path):
    """SubprocessWorker.stop() while a job runs must preserve state='stopped' and never trigger auto-restart."""

    handler_file = tmp_path / "sleep_handler.py"
    handler_file.write_text(
        "import time\n"
        "from remi.worker.protocol import WorkerResult\n\n"
        "def handle(envelope, emit_progress, emit_preview, cancel_event):\n"
        "    time.sleep(30)\n"
        "    return WorkerResult(job_id=envelope.job_id, status='completed')\n",
        encoding="utf-8",
    )
    worker = SubprocessWorker(
        worker_id="w-stop-no-respawn",
        handler_import=f"{handler_file}:handle",
        auto_restart=True,
        max_restarts=3,
        restart_backoff_s=0.0,
        startup_timeout=10.0,
    )
    worker.start()
    results: list[WorkerResult] = []
    try:
        t = threading.Thread(target=lambda: results.append(worker.execute(JobEnvelope(job_id="j-stop", seq=1, tab="txt2img", core={}))), daemon=True)
        t.start()
        deadline = time.monotonic() + 10.0
        while worker.status().current_job_id != "j-stop" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert worker.status().current_job_id == "j-stop"
        worker.stop(timeout=0.3)
        t.join(timeout=10.0)
        assert not t.is_alive()
        assert results and results[0].status == "failed"
        assert worker.is_alive() is False
        hb = worker.status()
        assert hb.state == "stopped"
        assert hb.restarts == 0
        assert hb.pid is None
    finally:
        worker.stop()


def test_stdio_loop_concurrent_cancel_and_default_forge_handler_forwarding(tmp_path, monkeypatch):
    """run_worker_stdio_loop dispatches cancel mid-job and default_forge_handler forwards interrupt/progress/preview."""

    import io
    from types import SimpleNamespace

    from remi.jobs.runner import JobResult
    from remi.worker import subprocess_worker as sw

    # 1. Pipe run_job + cancel + shutdown where handler blocks on cancel_event.wait()
    seen_cancel: list[bool] = []

    def blocking_handler(env, emit_progress, emit_preview, cancel_ev):
        got = cancel_ev.wait(2.0)
        seen_cancel.append(got)
        return WorkerResult(job_id=env.job_id, status="cancelled" if got else "completed")

    in_bytes = b"".join(
        [
            encode_message(WorkerMessage("run_job", JobEnvelope(job_id="j-can", seq=1, tab="txt2img", core={}).to_dict())),
            encode_message(WorkerMessage("cancel", {"job_id": "j-can"})),
            encode_message(WorkerMessage("shutdown", {})),
        ]
    )
    out_buf = io.BytesIO()
    assert sw.run_worker_stdio_loop(blocking_handler, worker_id="w-can", stdin=io.BytesIO(in_bytes), stdout=out_buf) == 0
    assert seen_cancel == [True]
    msgs = [decode_message(line) for line in out_buf.getvalue().splitlines() if line]
    assert [m.type for m in msgs] == ["result"]
    assert msgs[0].payload["status"] == "cancelled"

    # 2. default_forge_handler wires cancel_event -> shared.state.interrupt() and forwards progress/preview safely
    import modules.shared as shared
    import remi.jobs.runner as runner_mod

    fake_state = FakeState()
    fake_opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=30)
    monkeypatch.setattr(shared, "state", fake_state)
    monkeypatch.setattr(shared, "opts", fake_opts)

    cancel_ev = threading.Event()
    progress_seen: list[WorkerProgress] = []
    previews_seen: list[PreviewChunk] = []

    def fake_run_job(job, ctx):
        fake_state.time_start = time.time()
        fake_state.job_count = 1
        fake_state.sampling_steps = 2
        fake_state.sampling_step = 1
        if ctx.on_begin is not None:
            ctx.on_begin()
        fake_state.assign_current_image(Image.new("RGB", (16, 16), (10, 20, 30)))
        cancel_ev.set()
        deadline = time.monotonic() + 2.0
        while not fake_state.interrupted and time.monotonic() < deadline:
            time.sleep(0.01)
        ctx.interrupted = bool(fake_state.interrupted)
        return JobResult(images=[], infotexts=[], info=None)

    monkeypatch.setattr(runner_mod, "run_job", fake_run_job)
    res = sw.default_forge_handler(
        JobEnvelope(job_id="j-dfh", seq=7, tab="txt2img", core={}, file_store_root=str(tmp_path / "files")),
        lambda p: (progress_seen.append(p), (_ for _ in ()).throw(RuntimeError("ignore callback fail")))[0],
        previews_seen.append,
        cancel_ev,
    )
    assert fake_state.interrupted is True
    assert res.status == "cancelled"
    assert progress_seen and previews_seen


def test_subprocess_worker_timeout_non_json_stdout_finally_cleanup_and_single_restart(tmp_path, monkeypatch):
    """DEFAULT_JOB_TIMEOUT enforcement, non-JSON stdout resilience, finally cleanup, and no _restarts double-count."""

    # 1. Job timeout triggers structured failure and cleans up worker state
    hang_handler = tmp_path / "hang_handler.py"
    hang_handler.write_text(
        "import time\n"
        "from remi.worker.protocol import WorkerResult\n\n"
        "def handle(envelope, emit_progress, emit_preview, cancel_event):\n"
        "    time.sleep(30)\n"
        "    return WorkerResult(job_id=envelope.job_id, status='completed')\n",
        encoding="utf-8",
    )
    w_to = SubprocessWorker(
        worker_id="w-job-timeout",
        handler_import=f"{hang_handler}:handle",
        auto_restart=False,
        startup_timeout=10.0,
        job_timeout=0.2,
    )
    w_to.start()
    try:
        res_to = w_to.execute(JobEnvelope(job_id="j-to", seq=1, tab="txt2img", core={}))
        assert res_to.status == "failed"
        assert "timed out" in (res_to.error or "")
        assert w_to.status().state == "error"
        assert w_to.status().current_job_id is None
    finally:
        w_to.stop()

    # 2. Non-JSON stdout lines during startup & execution are ignored cleanly
    noisy_child = (
        "import json, sys\n"
        "print('PyTorch diagnostic warning', flush=True)\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if msg['type'] == 'ping':\n"
        "        print(json.dumps({'protocol_version': 1, 'type': 'pong', 'payload': {}}), flush=True)\n"
        "    elif msg['type'] == 'run_job':\n"
        "        print('CUDA kernel info line', flush=True)\n"
        "        jid = msg['payload']['job_id']\n"
        "        print(json.dumps({'protocol_version': 1, 'type': 'result', 'payload': {'job_id': jid, 'status': 'completed'}}), flush=True)\n"
        "    elif msg['type'] == 'shutdown':\n"
        "        break\n"
    )
    w_noisy = SubprocessWorker(worker_id="w-noisy", command=[sys.executable, "-c", noisy_child], auto_restart=False, startup_timeout=10.0)
    w_noisy.start()
    try:
        res_ok = w_noisy.execute(JobEnvelope(job_id="j-noisy", seq=1, tab="txt2img", core={}))
        assert res_ok.status == "completed"
        assert w_noisy.status().state == "idle"
        assert w_noisy.status().current_job_id is None
    finally:
        w_noisy.stop()

    # 3. Unexpected callback error still resets _state and _current_job_id in finally
    import io

    prog_child = (
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if msg['type'] == 'ping':\n"
        "        print(json.dumps({'protocol_version': 1, 'type': 'pong', 'payload': {}}), flush=True)\n"
        "    elif msg['type'] == 'run_job':\n"
        "        jid = msg['payload']['job_id']\n"
        "        print(json.dumps({'protocol_version': 1, 'type': 'progress', 'payload': {'job_id': jid, 'seq': 1, 'step': 1, 'steps': 2}}), flush=True)\n"
        "    elif msg['type'] == 'shutdown':\n"
        "        break\n"
    )
    w_cb = SubprocessWorker(worker_id="w-cb", command=[sys.executable, "-c", prog_child], auto_restart=False, startup_timeout=10.0)
    w_cb.start()
    try:
        with pytest.raises(RuntimeError, match="cb boom"):
            w_cb.execute(
                JobEnvelope(job_id="j-cb", seq=1, tab="txt2img", core={}),
                on_progress=lambda _p: (_ for _ in ()).throw(RuntimeError("cb boom")),
            )
        assert w_cb.status().state == "idle"
        assert w_cb.status().current_job_id is None
    finally:
        w_cb.stop()
        _ = io  # keep unused import away

    # 4. Concurrent restart() during execute() with auto_restart=True increments _restarts only once
    w_conc = SubprocessWorker(
        worker_id="w-conc-restart",
        handler_import=f"{hang_handler}:handle",
        auto_restart=True,
        max_restarts=5,
        restart_backoff_s=0.0,
        startup_timeout=10.0,
    )
    w_conc.start()
    results_conc: list[WorkerResult] = []
    try:
        t = threading.Thread(target=lambda: results_conc.append(w_conc.execute(JobEnvelope(job_id="j-hang", seq=1, tab="txt2img", core={}))), daemon=True)
        t.start()
        deadline = time.monotonic() + 10.0
        while w_conc.status().current_job_id != "j-hang" and time.monotonic() < deadline:
            time.sleep(0.02)
        hb_after = w_conc.restart()
        t.join(timeout=10.0)
        assert results_conc and results_conc[0].status == "failed"
        assert hb_after.state == "idle"
        assert w_conc.status().restarts == 1
    finally:
        w_conc.stop()


def test_mlx_stop_state_preserved_and_pool_releases_cv_before_status():
    """MlxSidecarWorker.execute() preserves 'stopped' and WorkerPool does not hold _cv during status()."""

    avail_env = MlxEnvironment(available=True, platform_supported=True, unified_memory_mb=16384)
    entered = threading.Event()
    release = threading.Event()

    def slow_runner(env, out, prog, cancel_ev):
        entered.set()
        release.wait(timeout=5.0)
        return {}

    w_mlx = MlxSidecarWorker(env_probe=lambda: avail_env, runner=slow_runner)
    res_box: list[WorkerResult] = []
    t = threading.Thread(target=lambda: res_box.append(w_mlx.execute(JobEnvelope(job_id="m-stop", seq=1, tab="txt2img", core={}))), daemon=True)
    t.start()
    assert entered.wait(timeout=5.0)
    w_mlx.stop()
    release.set()
    t.join(timeout=5.0)
    assert res_box and res_box[0].status == "cancelled"
    assert w_mlx.status().state == "stopped"
    assert w_mlx.is_alive() is False

    # WorkerPool.acquire_worker() must not hold _cv across worker.status()/is_alive()
    lock_held_during_status: list[bool] = []
    pool: WorkerPool

    class CheckingWorker(InProcessWorker):
        def status(self):
            lock_held_during_status.append(pool._cv._is_owned())
            return super().status()

    w_check = CheckingWorker(worker_id="w-lock-check")
    pool = WorkerPool([w_check], mode="pool")
    chosen = pool.acquire_worker(timeout=1.0)
    assert chosen is w_check
    assert lock_held_during_status == [False]
    pool.release_worker(w_check.worker_id)


def test_workers_route_reports_dispatch_active_and_refuses_preview_subprocess_restart(tmp_path):
    """/api/v2/workers exposes dispatch_active/preview and refuses idle subprocess spawn when inactive."""

    with webui_state(), server_app(tmp_path) as server:
        data = server.client.get("/api/v2/workers").json()
        assert data["dispatch_active"] is False
        assert data["preview"] is False
        assert data["note"]

        # Swap pool to preview subprocess mode (dispatch_active=False)
        sub = SubprocessWorker(worker_id="worker-preview-0", auto_restart=False)
        server.services.workers = WorkerPool([sub], mode="subprocess", dispatch_active=False)
        try:
            preview_data = server.client.get("/api/v2/workers").json()
            assert preview_data["mode"] == "subprocess"
            assert preview_data["dispatch_active"] is False
            assert preview_data["preview"] is True
            assert "preview" in (preview_data["note"] or "").lower()

            refused = server.client.post("/api/v2/workers/worker-preview-0/restart")
            assert refused.status_code == 409
            assert sub.is_alive() is False
            assert sub.status().restarts == 0
        finally:
            server.services.workers.shutdown()
