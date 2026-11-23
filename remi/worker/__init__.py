"""Process-isolated inference worker protocol, runtime bridge, supervised
subprocess worker, local multi-GPU worker pool, and optional Apple Silicon
MLX / ``mflux`` sidecar adapter."""

from remi.worker.bridge import RuntimeBridge
from remi.worker.mlx_sidecar import MlxEnvironment, MlxSidecarWorker, detect_mlx_environment
from remi.worker.pool import InProcessWorker, WorkerPool, create_default_pool, parse_worker_gpus
from remi.worker.protocol import (
    PROTOCOL_VERSION,
    JobEnvelope,
    PreviewChunk,
    WorkerCapabilities,
    WorkerHeartbeat,
    WorkerMessage,
    WorkerProgress,
    WorkerProtocol,
    WorkerResult,
    decode_message,
    encode_message,
)
from remi.worker.subprocess_worker import SubprocessWorker, run_worker_stdio_loop

__all__ = [
    "PROTOCOL_VERSION",
    "InProcessWorker",
    "JobEnvelope",
    "MlxEnvironment",
    "MlxSidecarWorker",
    "PreviewChunk",
    "RuntimeBridge",
    "SubprocessWorker",
    "WorkerCapabilities",
    "WorkerHeartbeat",
    "WorkerMessage",
    "WorkerPool",
    "WorkerProgress",
    "WorkerProtocol",
    "WorkerResult",
    "create_default_pool",
    "decode_message",
    "detect_mlx_environment",
    "encode_message",
    "parse_worker_gpus",
    "run_worker_stdio_loop",
]
