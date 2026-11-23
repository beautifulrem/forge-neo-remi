"""Non-node linear multi-stage pipelines (:class:`JobChain` & :class:`StageRun`)
with content-addressed stage caching and zero-node partial re-execution.

Supported ordered stage kinds:
- ``prompt_upsample`` — optional multimodal/local Prompt Upsampler (PUP)
- ``generate`` — base generation (``txt2img`` / ``img2img``)
- ``refine`` — refinement pass (``img2img`` / Inpaint / ADetailer)
- ``finish`` — finishing pass (Hires ``upscale_from`` or ``extras``)

Each stage's deterministic ``stage_hash`` hashes its canonical parameters,
its upstream stage's hash/output file IDs/prompt, and the active model
fingerprint. Re-running a chain after tweaking a downstream stage (e.g.
``refine`` or ``finish``) automatically reuses cached upstream stage output
``file_id``s without re-executing unchanged upstream stages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remi.files import FileStore, connect, decode_cursor, encode_cursor
from remi.jobs.queue import JobError, JobQueue, Submission
from remi.jobs.store import CANCELLED, COMPLETED, FAILED, INTERRUPTED, Job, JobStore
from remi.realtime import EventBus

log = logging.getLogger(__name__)

STAGE_KINDS = ("prompt_upsample", "generate", "refine", "finish")
DEFAULT_STAGE_TABS = {
    "prompt_upsample": "prompt_upsample",
    "generate": "txt2img",
    "refine": "img2img",
    "finish": "extras",
}
CHAIN_STATUSES = ("queued", "running", "paused", "completed", "failed", "cancelled")
STAGE_STATUSES = ("pending", "running", "cached", "completed", "failed", "skipped", "cancelled")
MAX_STAGES = 8


class ChainError(JobError):
    """A chain validation or state error with an HTTP status code."""


@dataclass
class StageSpec:
    kind: str
    name: str = ""
    enabled: bool = True
    tab: str = ""
    core: dict[str, Any] = field(default_factory=dict)
    scripts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    index: int = 0

    def __post_init__(self) -> None:
        if self.kind not in STAGE_KINDS:
            raise ChainError(f"stage kind must be one of {', '.join(STAGE_KINDS)}", status=422)
        if not self.tab:
            self.tab = DEFAULT_STAGE_TABS[self.kind]
        if not self.name:
            self.name = self.kind.replace("_", " ").title()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "name": self.name,
            "enabled": self.enabled,
            "tab": self.tab,
            "core": dict(self.core),
            "scripts": {k: dict(v) if isinstance(v, Mapping) else v for k, v in self.scripts.items()},
            "config": dict(self.config),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], index: int = 0) -> StageSpec:
        kind = str(data.get("kind") or "")
        raw_index = data.get("index")
        return cls(
            index=index if raw_index is None else int(raw_index),
            kind=kind,
            name=str(data.get("name") or ""),
            enabled=bool(data.get("enabled", True)),
            tab=str(data.get("tab") or ""),
            core=dict(data.get("core") or {}),
            scripts=dict(data.get("scripts") or {}),
            config=dict(data.get("config") or {}),
        )


@dataclass
class StageRun:
    chain_id: str
    index: int
    kind: str
    name: str
    enabled: bool
    tab: str
    core: dict[str, Any] = field(default_factory=dict)
    scripts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    stage_hash: str | None = None
    cached: bool = False
    job_id: str | None = None
    output_file_ids: list[str] = field(default_factory=list)
    output_prompt: str | None = None
    infotext: str | None = None
    error: str | None = None
    started: float | None = None
    finished: float | None = None

    def to_spec(self) -> StageSpec:
        return StageSpec(
            index=self.index,
            kind=self.kind,
            name=self.name,
            enabled=self.enabled,
            tab=self.tab,
            core=dict(self.core),
            scripts=dict(self.scripts),
            config=dict(self.config),
        )


@dataclass
class JobChain:
    id: str
    name: str
    status: str
    current_stage: int
    stages: list[StageRun] = field(default_factory=list)
    priority: int = 0
    user: str | None = None
    model_fingerprint: str = ""
    error: str | None = None
    created: float = 0.0
    updated: float = 0.0
    started: float | None = None
    finished: float | None = None


@dataclass
class ChainTemplate:
    id: str
    name: str
    description: str = ""
    model_family: str | None = None
    stages: list[StageSpec] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created: float = 0.0
    updated: float = 0.0


@dataclass(frozen=True)
class CachedStageOutput:
    stage_hash: str
    kind: str
    output_file_ids: tuple[str, ...]
    output_prompt: str | None
    infotext: str | None
    job_id: str | None
    created: float


def compute_stage_hash(
    spec: StageSpec,
    *,
    upstream_hash: str = "",
    upstream_outputs: Sequence[str] = (),
    upstream_prompt: str | None = None,
    model_fingerprint: str = "",
) -> str:
    """Deterministic content-addressed hash of a stage's effective inputs."""

    payload = {
        "kind": spec.kind,
        "enabled": bool(spec.enabled),
        "tab": spec.tab,
        "core": spec.core,
        "scripts": spec.scripts,
        "config": spec.config,
        "upstream_hash": upstream_hash,
        "upstream_outputs": list(upstream_outputs),
        "upstream_prompt": upstream_prompt,
        "model_fingerprint": model_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def stage_json(stage: StageRun, iso: Callable[[float | None], str | None]) -> dict[str, Any]:
    return {
        "index": stage.index,
        "kind": stage.kind,
        "name": stage.name,
        "enabled": stage.enabled,
        "tab": stage.tab,
        "core": stage.core,
        "scripts": stage.scripts,
        "config": stage.config,
        "status": stage.status,
        "stage_hash": stage.stage_hash,
        "cached": stage.cached,
        "job_id": stage.job_id,
        "output_file_ids": list(stage.output_file_ids),
        "output_prompt": stage.output_prompt,
        "infotext": stage.infotext,
        "error": stage.error,
        "started_at": iso(stage.started),
        "finished_at": iso(stage.finished),
    }


def chain_json(chain: JobChain, iso: Callable[[float | None], str | None]) -> dict[str, Any]:
    return {
        "id": chain.id,
        "name": chain.name,
        "status": chain.status,
        "current_stage": chain.current_stage,
        "total_stages": len(chain.stages),
        "stages": [stage_json(stage, iso) for stage in chain.stages],
        "created_at": iso(chain.created) or "",
        "updated_at": iso(chain.updated) or "",
        "started_at": iso(chain.started),
        "finished_at": iso(chain.finished),
        "error": chain.error,
    }


def template_json(template: ChainTemplate, iso: Callable[[float | None], str | None]) -> dict[str, Any]:
    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "model_family": template.model_family,
        "stages": [spec.to_dict() for spec in template.stages],
        "tags": list(template.tags),
        "created_at": iso(template.created) or "",
        "updated_at": iso(template.updated) or "",
    }


_STANDALONE_SCHEMA = """
CREATE TABLE IF NOT EXISTS chains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    current_stage INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 0,
    user TEXT,
    model_fingerprint TEXT NOT NULL DEFAULT '',
    error TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    started REAL,
    finished REAL
);
CREATE INDEX IF NOT EXISTS chains_created ON chains(created DESC);
CREATE TABLE IF NOT EXISTS stage_runs (
    chain_id TEXT NOT NULL,
    stage_index INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    tab TEXT NOT NULL,
    core TEXT NOT NULL DEFAULT '{}',
    scripts TEXT NOT NULL DEFAULT '{}',
    config TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    stage_hash TEXT,
    cached INTEGER NOT NULL DEFAULT 0,
    job_id TEXT,
    output_file_ids TEXT NOT NULL DEFAULT '[]',
    output_prompt TEXT,
    infotext TEXT,
    error TEXT,
    started REAL,
    finished REAL,
    PRIMARY KEY (chain_id, stage_index)
);
CREATE INDEX IF NOT EXISTS stage_runs_job ON stage_runs(job_id);
CREATE TABLE IF NOT EXISTS stage_cache (
    stage_hash TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    output_file_ids TEXT NOT NULL DEFAULT '[]',
    output_prompt TEXT,
    infotext TEXT,
    job_id TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chain_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    model_family TEXT,
    stages TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
"""


class ChainStore:
    """SQLite persistence for :class:`JobChain`, :class:`StageRun`, stage cache and templates.

    Accepts either an existing :class:`JobStore` (sharing its connection and lock)
    or a standalone database path (closed explicitly by :meth:`close`).
    """

    def __init__(self, target: JobStore | str | Path) -> None:
        if isinstance(target, JobStore):
            self._db = target._db
            self._lock = target._lock
            self._owns_db = False
        else:
            self._lock = threading.RLock()
            self._db = connect(Path(target))
            self._owns_db = True
        with self._lock:
            self._db.executescript(_STANDALONE_SCHEMA)

    def close(self) -> None:
        if self._owns_db:
            with self._lock:
                self._db.close()

    def __enter__(self) -> ChainStore:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _load_stages(self, chain_id: str) -> list[StageRun]:
        rows = self._db.execute(
            """
            SELECT chain_id, stage_index, kind, name, enabled, tab, core, scripts, config,
                   status, stage_hash, cached, job_id, output_file_ids, output_prompt,
                   infotext, error, started, finished
            FROM stage_runs WHERE chain_id = ? ORDER BY stage_index ASC
            """,
            (chain_id,),
        ).fetchall()
        stages: list[StageRun] = []
        for row in rows:
            stages.append(
                StageRun(
                    chain_id=row[0],
                    index=int(row[1]),
                    kind=row[2],
                    name=row[3],
                    enabled=bool(row[4]),
                    tab=row[5],
                    core=json.loads(row[6]),
                    scripts=json.loads(row[7]),
                    config=json.loads(row[8]),
                    status=row[9],
                    stage_hash=row[10],
                    cached=bool(row[11]),
                    job_id=row[12],
                    output_file_ids=json.loads(row[13]),
                    output_prompt=row[14],
                    infotext=row[15],
                    error=row[16],
                    started=row[17],
                    finished=row[18],
                )
            )
        return stages

    def _row_to_chain(self, row: tuple[Any, ...]) -> JobChain:
        chain_id = row[0]
        return JobChain(
            id=chain_id,
            name=row[1],
            status=row[2],
            current_stage=int(row[3]),
            priority=int(row[4]),
            user=row[5],
            model_fingerprint=row[6],
            error=row[7],
            created=float(row[8]),
            updated=float(row[9]),
            started=row[10],
            finished=row[11],
            stages=self._load_stages(chain_id),
        )

    def create_chain(
        self,
        stages: Sequence[StageSpec],
        *,
        name: str = "",
        priority: int = 0,
        user: str | None = None,
        model_fingerprint: str = "",
        status: str = "queued",
    ) -> JobChain:
        if not stages:
            raise ChainError("a chain requires at least one stage", status=422)
        if len(stages) > MAX_STAGES:
            raise ChainError(f"a chain may have at most {MAX_STAGES} stages", status=422)
        chain_id = secrets.token_hex(8)
        now = time.time()
        clean_name = (name or "").strip()[:200] or f"Pipeline {chain_id[:6]}"
        with self._lock:
            self._db.execute(
                """
                INSERT INTO chains (id, name, status, current_stage, priority, user, model_fingerprint, error, created, updated)
                VALUES (?, ?, ?, 0, ?, ?, ?, NULL, ?, ?)
                """,
                (chain_id, clean_name, status, int(priority), user, model_fingerprint, now, now),
            )
            for idx, spec in enumerate(stages):
                self._db.execute(
                    """
                    INSERT INTO stage_runs (chain_id, stage_index, kind, name, enabled, tab, core, scripts, config, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chain_id,
                        idx,
                        spec.kind,
                        spec.name,
                        int(spec.enabled),
                        spec.tab,
                        json.dumps(spec.core, ensure_ascii=False),
                        json.dumps(spec.scripts, ensure_ascii=False),
                        json.dumps(spec.config, ensure_ascii=False),
                        "pending" if spec.enabled else "skipped",
                    ),
                )
        created = self.get_chain(chain_id)
        assert created is not None
        return created

    def get_chain(self, chain_id: str) -> JobChain | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, name, status, current_stage, priority, user, model_fingerprint, error, created, updated, started, finished FROM chains WHERE id = ?",
                (chain_id,),
            ).fetchone()
            return self._row_to_chain(row) if row is not None else None

    def list_chains(self, *, status: str | None = None, cursor: str | None = None, limit: int = 50) -> tuple[list[JobChain], str | None, int]:
        where = ["1=1"]
        args: list[Any] = []
        if status:
            wanted = [item.strip() for item in status.split(",") if item.strip() in CHAIN_STATUSES]
            where.append(f"status IN ({','.join('?' * len(wanted))})" if wanted else "0=1")
            args.extend(wanted)
        filters = " AND ".join(where)
        page_where = filters
        page_args = list(args)
        pos = decode_cursor(cursor, 2)
        if pos is not None:
            try:
                created_val = float(pos[0])
                cid = pos[1]
            except ValueError:
                pass
            else:
                page_where += " AND (created < ? OR (created = ? AND id < ?))"
                page_args.extend([created_val, created_val, cid])
        limit = min(max(int(limit), 1), 200)
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM chains WHERE {filters}", args).fetchone()[0]
            rows = self._db.execute(
                f"SELECT id, name, status, current_stage, priority, user, model_fingerprint, error, created, updated, started, finished FROM chains WHERE {page_where} ORDER BY created DESC, id DESC LIMIT ?",
                [*page_args, limit + 1],
            ).fetchall()
            items = [self._row_to_chain(row) for row in rows[:limit]]
        next_cursor = encode_cursor(repr(items[-1].created), items[-1].id) if len(rows) > limit and items else None
        return items, next_cursor, int(total)

    def update_chain(
        self,
        chain_id: str,
        *,
        status: str | None = None,
        current_stage: int | None = None,
        model_fingerprint: str | None = None,
        error: Any = ...,
        started: Any = ...,
        finished: Any = ...,
    ) -> JobChain | None:
        sets = ["updated = ?"]
        args: list[Any] = [time.time()]
        if status is not None:
            sets.append("status = ?")
            args.append(status)
        if current_stage is not None:
            sets.append("current_stage = ?")
            args.append(int(current_stage))
        if model_fingerprint is not None:
            sets.append("model_fingerprint = ?")
            args.append(model_fingerprint)
        if error is not ...:
            sets.append("error = ?")
            args.append(error)
        if started is not ...:
            sets.append("started = ?")
            args.append(started)
        if finished is not ...:
            sets.append("finished = ?")
            args.append(finished)
        args.append(chain_id)
        with self._lock:
            self._db.execute(f"UPDATE chains SET {', '.join(sets)} WHERE id = ?", args)
        return self.get_chain(chain_id)

    def replace_stages(self, chain_id: str, stages: Sequence[StageSpec]) -> None:
        if not stages or len(stages) > MAX_STAGES:
            raise ChainError(f"stages count must be between 1 and {MAX_STAGES}", status=422)
        with self._lock:
            self._db.execute("DELETE FROM stage_runs WHERE chain_id = ?", (chain_id,))
            for idx, spec in enumerate(stages):
                self._db.execute(
                    """
                    INSERT INTO stage_runs (chain_id, stage_index, kind, name, enabled, tab, core, scripts, config, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chain_id,
                        idx,
                        spec.kind,
                        spec.name,
                        int(spec.enabled),
                        spec.tab,
                        json.dumps(spec.core, ensure_ascii=False),
                        json.dumps(spec.scripts, ensure_ascii=False),
                        json.dumps(spec.config, ensure_ascii=False),
                        "pending" if spec.enabled else "skipped",
                    ),
                )

    def update_stage(
        self,
        chain_id: str,
        stage_index: int,
        *,
        status: str | None = None,
        stage_hash: Any = ...,
        cached: bool | None = None,
        job_id: Any = ...,
        output_file_ids: Sequence[str] | None = None,
        output_prompt: Any = ...,
        infotext: Any = ...,
        error: Any = ...,
        started: Any = ...,
        finished: Any = ...,
    ) -> None:
        sets: list[str] = []
        args: list[Any] = []
        if status is not None:
            sets.append("status = ?")
            args.append(status)
        if stage_hash is not ...:
            sets.append("stage_hash = ?")
            args.append(stage_hash)
        if cached is not None:
            sets.append("cached = ?")
            args.append(int(cached))
        if job_id is not ...:
            sets.append("job_id = ?")
            args.append(job_id)
        if output_file_ids is not None:
            sets.append("output_file_ids = ?")
            args.append(json.dumps(list(output_file_ids), ensure_ascii=False))
        if output_prompt is not ...:
            sets.append("output_prompt = ?")
            args.append(output_prompt)
        if infotext is not ...:
            sets.append("infotext = ?")
            args.append(infotext)
        if error is not ...:
            sets.append("error = ?")
            args.append(error)
        if started is not ...:
            sets.append("started = ?")
            args.append(started)
        if finished is not ...:
            sets.append("finished = ?")
            args.append(finished)
        if not sets:
            return
        args.extend([chain_id, int(stage_index)])
        with self._lock:
            self._db.execute(f"UPDATE stage_runs SET {', '.join(sets)} WHERE chain_id = ? AND stage_index = ?", args)
            self._db.execute("UPDATE chains SET updated = ? WHERE id = ?", (time.time(), chain_id))

    def find_stage_by_job(self, job_id: str) -> tuple[str, int] | None:
        with self._lock:
            row = self._db.execute("SELECT chain_id, stage_index FROM stage_runs WHERE job_id = ?", (job_id,)).fetchone()
        return (row[0], int(row[1])) if row is not None else None

    def delete_chain(self, chain_id: str) -> bool:
        with self._lock:
            self._db.execute("DELETE FROM stage_runs WHERE chain_id = ?", (chain_id,))
            return self._db.execute("DELETE FROM chains WHERE id = ?", (chain_id,)).rowcount > 0

    # region Stage cache

    def get_cached(self, stage_hash: str) -> CachedStageOutput | None:
        with self._lock:
            row = self._db.execute(
                "SELECT stage_hash, kind, output_file_ids, output_prompt, infotext, job_id, created FROM stage_cache WHERE stage_hash = ?",
                (stage_hash,),
            ).fetchone()
        if row is None:
            return None
        return CachedStageOutput(
            stage_hash=row[0],
            kind=row[1],
            output_file_ids=tuple(json.loads(row[2])),
            output_prompt=row[3],
            infotext=row[4],
            job_id=row[5],
            created=float(row[6]),
        )

    def put_cached(
        self,
        stage_hash: str,
        kind: str,
        *,
        output_file_ids: Sequence[str] = (),
        output_prompt: str | None = None,
        infotext: str | None = None,
        job_id: str | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO stage_cache (stage_hash, kind, output_file_ids, output_prompt, infotext, job_id, created)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (stage_hash, kind, json.dumps(list(output_file_ids), ensure_ascii=False), output_prompt, infotext, job_id, time.time()),
            )

    # endregion

    # region Chain templates

    def _row_to_template(self, row: tuple[Any, ...]) -> ChainTemplate:
        raw_stages = json.loads(row[4])
        stages = [StageSpec.from_mapping(item, idx) for idx, item in enumerate(raw_stages)]
        return ChainTemplate(
            id=row[0],
            name=row[1],
            description=row[2],
            model_family=row[3],
            stages=stages,
            tags=json.loads(row[5]),
            created=float(row[6]),
            updated=float(row[7]),
        )

    def list_templates(self) -> list[ChainTemplate]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, description, model_family, stages, tags, created, updated FROM chain_templates ORDER BY updated DESC, name ASC"
            ).fetchall()
        return [self._row_to_template(row) for row in rows]

    def get_template(self, template_id: str) -> ChainTemplate | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, name, description, model_family, stages, tags, created, updated FROM chain_templates WHERE id = ?",
                (template_id,),
            ).fetchone()
        return self._row_to_template(row) if row is not None else None

    def create_template(
        self,
        name: str,
        stages: Sequence[StageSpec],
        *,
        description: str = "",
        model_family: str | None = None,
        tags: Sequence[str] = (),
    ) -> ChainTemplate:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ChainError("template name is required", status=422)
        if not stages or len(stages) > MAX_STAGES:
            raise ChainError(f"template must contain 1..{MAX_STAGES} stages", status=422)
        template_id = secrets.token_hex(8)
        now = time.time()
        clean_tags = list(dict.fromkeys(str(t).strip() for t in tags if str(t).strip()))
        with self._lock:
            try:
                self._db.execute(
                    """
                    INSERT INTO chain_templates (id, name, description, model_family, stages, tags, created, updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template_id,
                        clean_name[:200],
                        (description or "").strip()[:2000],
                        model_family.strip() if isinstance(model_family, str) and model_family.strip() else None,
                        json.dumps([spec.to_dict() for spec in stages], ensure_ascii=False),
                        json.dumps(clean_tags, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ChainError(f"a chain template named {clean_name!r} already exists", status=409) from error
        tmpl = self.get_template(template_id)
        assert tmpl is not None
        return tmpl

    def delete_template(self, template_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM chain_templates WHERE id = ?", (template_id,)).rowcount > 0

    # endregion


def _default_model_fingerprint() -> str:
    try:
        from modules import shared

        opts = shared.opts
        parts = [
            getattr(opts, "sd_model_checkpoint", "") or "",
            getattr(opts, "forge_preset", "") or "",
            getattr(opts, "forge_unet_storage_dtype", "") or "",
        ]
        return "|".join(str(p) for p in parts)
    except Exception:
        return ""


UpsamplerFunc = Callable[[StageRun, str | None, Sequence[str]], str]


def _default_upsampler(stage: StageRun, upstream_prompt: str | None, upstream_outputs: Sequence[str]) -> str:
    """Default ``prompt_upsample`` stage handler (uses ``remi.prompt_tools.pup`` if configured)."""

    base_prompt = str(stage.core.get("prompt") or upstream_prompt or "")
    if not base_prompt.strip():
        return base_prompt
    try:
        from remi.prompt_tools.pup import upsample_prompt

        cfg = stage.config or {}
        res = upsample_prompt(
            base_prompt,
            model_family=cfg.get("model_family"),
            aspect_ratio=cfg.get("aspect_ratio"),
            bbox_draft=cfg.get("bbox_draft"),
            mode=str(cfg.get("mode") or "auto"),
            local_model_path=cfg.get("local_model_path"),
        )
        return res.prompt
    except Exception:
        return base_prompt


class ChainRunner:
    """Orchestrates linear multi-stage pipelines over :class:`JobQueue` with stage caching."""

    def __init__(
        self,
        store: ChainStore,
        queue: JobQueue,
        files: FileStore,
        bus: EventBus,
        *,
        upsampler: UpsamplerFunc | None = None,
        model_fingerprint: Callable[[], str] = _default_model_fingerprint,
    ) -> None:
        self.store = store
        self.queue = queue
        self.files = files
        self.bus = bus
        self.upsampler = upsampler or _default_upsampler
        self.model_fingerprint = model_fingerprint
        self._lock = threading.RLock()
        self._force_stages: dict[str, set[int]] = {}
        self._unhook = queue.add_completion_hook(self.on_job_finished)

    def close(self) -> None:
        if self._unhook is not None:
            self._unhook()
            self._unhook = None
        self.store.close()

    def _cache_valid(self, entry: CachedStageOutput | None) -> bool:
        if entry is None:
            return False
        if entry.kind == "prompt_upsample":
            return entry.output_prompt is not None
        if not entry.output_file_ids:
            return False
        return all(self.files.get(fid) is not None for fid in entry.output_file_ids)

    def plan_chain(self, stages: Sequence[StageSpec], *, model_fingerprint: str | None = None) -> list[dict[str, Any]]:
        """Dry-run cache evaluation: reports per-stage ``stage_hash`` and ``cache_hit``."""

        fp = model_fingerprint if model_fingerprint is not None else self.model_fingerprint()
        plan: list[dict[str, Any]] = []
        upstream_hash = ""
        upstream_outputs: list[str] = []
        upstream_prompt: str | None = None
        chain_cache_intact = True

        for idx, spec in enumerate(stages):
            stage_hash = compute_stage_hash(
                spec,
                upstream_hash=upstream_hash,
                upstream_outputs=upstream_outputs,
                upstream_prompt=upstream_prompt,
                model_fingerprint=fp,
            )
            if not spec.enabled:
                plan.append(
                    {
                        "index": idx,
                        "kind": spec.kind,
                        "name": spec.name,
                        "enabled": False,
                        "stage_hash": stage_hash,
                        "cache_hit": False,
                        "cached_outputs": [],
                        "cached_prompt": None,
                    }
                )
                upstream_hash = stage_hash
                continue

            cached = self.store.get_cached(stage_hash) if chain_cache_intact else None
            hit = self._cache_valid(cached)
            if hit and cached is not None:
                upstream_hash = stage_hash
                if cached.output_file_ids:
                    upstream_outputs = list(cached.output_file_ids)
                if cached.output_prompt is not None:
                    upstream_prompt = cached.output_prompt
                plan.append(
                    {
                        "index": idx,
                        "kind": spec.kind,
                        "name": spec.name,
                        "enabled": True,
                        "stage_hash": stage_hash,
                        "cache_hit": True,
                        "cached_outputs": list(cached.output_file_ids),
                        "cached_prompt": cached.output_prompt,
                    }
                )
            else:
                chain_cache_intact = False
                upstream_hash = stage_hash
                upstream_outputs = []
                plan.append(
                    {
                        "index": idx,
                        "kind": spec.kind,
                        "name": spec.name,
                        "enabled": True,
                        "stage_hash": stage_hash,
                        "cache_hit": False,
                        "cached_outputs": [],
                        "cached_prompt": None,
                    }
                )
        return plan

    def create_chain(
        self,
        stages: Sequence[StageSpec],
        *,
        name: str = "",
        run_now: bool = True,
        priority: int = 0,
        user: str | None = None,
        model_fingerprint: str | None = None,
    ) -> JobChain:
        fp = model_fingerprint if model_fingerprint is not None else self.model_fingerprint()
        chain = self.store.create_chain(stages, name=name, priority=priority, user=user, model_fingerprint=fp, status="queued" if run_now else "paused")
        self._publish_chain(chain)
        if run_now:
            return self._advance(chain.id)
        return chain

    def rerun_chain(
        self,
        chain_id: str,
        *,
        stages: Sequence[StageSpec] | None = None,
        from_stage: int | None = None,
        force_stages: Sequence[int] = (),
        model_fingerprint: str | None = None,
    ) -> JobChain:
        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None:
                raise ChainError(f"no chain {chain_id!r}", status=404)
            if chain.status == "running":
                raise ChainError("cancel or pause the running chain before re-running", status=409)
            if stages is not None:
                self.store.replace_stages(chain_id, stages)
            else:
                # Reset existing stages to pending/skipped so cache check runs cleanly
                for st in chain.stages:
                    self.store.update_stage(
                        chain_id,
                        st.index,
                        status="pending" if st.enabled else "skipped",
                        cached=False,
                        job_id=None,
                        output_file_ids=[],
                        output_prompt=None,
                        infotext=None,
                        error=None,
                        started=None,
                        finished=None,
                    )
            forced = set(force_stages)
            if from_stage is not None:
                total = len(stages) if stages is not None else len(chain.stages)
                forced |= set(range(max(0, int(from_stage)), total))
            self._force_stages[chain_id] = forced
            fp = model_fingerprint if model_fingerprint is not None else chain.model_fingerprint or self.model_fingerprint()
            self.store.update_chain(chain_id, status="queued", current_stage=0, model_fingerprint=fp, error=None, started=None, finished=None)
        return self._advance(chain_id)

    def pause_chain(self, chain_id: str) -> JobChain:
        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None:
                raise ChainError(f"no chain {chain_id!r}", status=404)
            if chain.status in ("completed", "failed", "cancelled"):
                raise ChainError(f"cannot pause chain in status {chain.status!r}", status=409)
            updated = self.store.update_chain(chain_id, status="paused")
            assert updated is not None
            self._publish_chain(updated)
            return updated

    def resume_chain(self, chain_id: str) -> JobChain:
        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None:
                raise ChainError(f"no chain {chain_id!r}", status=404)
            if chain.status != "paused":
                raise ChainError(f"chain is {chain.status!r}, not paused", status=409)
            self.store.update_chain(chain_id, status="queued")
        return self._advance(chain_id)

    def cancel_chain(self, chain_id: str) -> JobChain:
        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None:
                raise ChainError(f"no chain {chain_id!r}", status=404)
            now = time.time()
            active_jobs = [st.job_id for st in chain.stages if st.status == "running" and st.job_id]
            for st in chain.stages:
                if st.status in ("pending", "running"):
                    self.store.update_stage(chain_id, st.index, status="cancelled", finished=now)
            updated = self.store.update_chain(chain_id, status="cancelled", finished=now)
            self._force_stages.pop(chain_id, None)
        for job_id in active_jobs:
            try:
                self.queue.interrupt(job_id)
            except Exception:
                pass
        assert updated is not None
        self._publish_chain(updated)
        return updated

    def delete_chain(self, chain_id: str) -> bool:
        chain = self.store.get_chain(chain_id)
        if chain is None:
            return False
        if chain.status == "running":
            raise ChainError("cannot delete a running chain; cancel it first", status=409)
        self._force_stages.pop(chain_id, None)
        return self.store.delete_chain(chain_id)

    def _upstream_state(self, chain: JobChain, before_index: int) -> tuple[str, list[str], str | None]:
        upstream_hash = ""
        upstream_outputs: list[str] = []
        upstream_prompt: str | None = None
        for st in chain.stages:
            if st.index >= before_index:
                break
            if st.stage_hash:
                upstream_hash = st.stage_hash
            if st.output_file_ids:
                upstream_outputs = list(st.output_file_ids)
            if st.output_prompt is not None:
                upstream_prompt = st.output_prompt
        return upstream_hash, upstream_outputs, upstream_prompt

    def _build_stage_submission(self, chain: JobChain, stage: StageRun, upstream_outputs: list[str], upstream_prompt: str | None) -> Submission:
        core = dict(stage.core)
        scripts = {k: dict(v) if isinstance(v, Mapping) else v for k, v in stage.scripts.items()}
        config = stage.config or {}
        tab = stage.tab if stage.tab in ("txt2img", "img2img", "extras") else DEFAULT_STAGE_TABS.get(stage.kind, "txt2img")
        upscale_from: str | None = None
        primary_img = upstream_outputs[0] if upstream_outputs else None

        if upstream_prompt and not core.get("prompt") and tab in ("txt2img", "img2img"):
            core["prompt"] = upstream_prompt

        if stage.kind == "refine" and primary_img:
            target_field = str(config.get("input_field") or "init_img")
            if not core.get(target_field):
                core[target_field] = primary_img
        elif stage.kind == "finish" and primary_img:
            if tab == "extras":
                if not core.get("image"):
                    core["image"] = primary_img
            elif tab == "txt2img":
                upscale_from = primary_img
            elif tab == "img2img" and not core.get("init_img"):
                core["init_img"] = primary_img

        meta: dict[str, Any] = {
            "chain_id": chain.id,
            "stage_index": stage.index,
            "stage_kind": stage.kind,
        }
        if upstream_outputs:
            meta["upstream_file_ids"] = list(upstream_outputs)
        return Submission(tab=tab, core=core, scripts=scripts, priority=chain.priority, upscale_from=upscale_from, meta=meta)

    def _advance(self, chain_id: str) -> JobChain:
        while True:
            with self._lock:
                chain = self.store.get_chain(chain_id)
                if chain is None:
                    raise ChainError(f"no chain {chain_id!r}", status=404)
                if chain.status in ("paused", "cancelled", "completed", "failed"):
                    return chain
                if any(st.status == "running" for st in chain.stages):
                    return chain

                next_stage = next((st for st in chain.stages if st.status == "pending"), None)
                if next_stage is None:
                    done = self.store.update_chain(chain_id, status="completed", finished=time.time())
                    self._force_stages.pop(chain_id, None)
                    assert done is not None
                    self._publish_chain(done)
                    return done

                now = time.time()
                if chain.started is None:
                    self.store.update_chain(chain_id, status="running", current_stage=next_stage.index, started=now)
                else:
                    self.store.update_chain(chain_id, status="running", current_stage=next_stage.index)

                upstream_hash, upstream_outputs, upstream_prompt = self._upstream_state(chain, next_stage.index)
                stage_hash = compute_stage_hash(
                    next_stage.to_spec(),
                    upstream_hash=upstream_hash,
                    upstream_outputs=upstream_outputs,
                    upstream_prompt=upstream_prompt,
                    model_fingerprint=chain.model_fingerprint,
                )

                forced = self._force_stages.get(chain_id, set())
                cached_entry = None if next_stage.index in forced else self.store.get_cached(stage_hash)
                if self._cache_valid(cached_entry) and cached_entry is not None:
                    self.store.update_stage(
                        chain_id,
                        next_stage.index,
                        status="cached",
                        stage_hash=stage_hash,
                        cached=True,
                        job_id=cached_entry.job_id,
                        output_file_ids=list(cached_entry.output_file_ids),
                        output_prompt=cached_entry.output_prompt,
                        infotext=cached_entry.infotext,
                        started=now,
                        finished=now,
                    )
                    updated_chain = self.store.get_chain(chain_id)
                    assert updated_chain is not None
                    self._publish_stage_completed(updated_chain.stages[next_stage.index])
                    self._publish_chain(updated_chain)
                    continue

                # Execute prompt_upsample inline
                if next_stage.kind == "prompt_upsample":
                    self.store.update_stage(chain_id, next_stage.index, status="running", stage_hash=stage_hash, started=now)
                    try:
                        upsampled = self.upsampler(next_stage, upstream_prompt, upstream_outputs)
                    except Exception as error:
                        msg = f"{type(error).__name__}: {error}"
                        self.store.update_stage(chain_id, next_stage.index, status="failed", error=msg, finished=time.time())
                        failed = self.store.update_chain(chain_id, status="failed", error=msg, finished=time.time())
                        self._force_stages.pop(chain_id, None)
                        assert failed is not None
                        self._publish_stage_completed(failed.stages[next_stage.index])
                        self._publish_chain(failed)
                        return failed
                    done_time = time.time()
                    self.store.update_stage(
                        chain_id,
                        next_stage.index,
                        status="completed",
                        stage_hash=stage_hash,
                        cached=False,
                        output_prompt=upsampled,
                        finished=done_time,
                    )
                    self.store.put_cached(stage_hash, next_stage.kind, output_prompt=upsampled)
                    updated_chain = self.store.get_chain(chain_id)
                    assert updated_chain is not None
                    self._publish_stage_completed(updated_chain.stages[next_stage.index])
                    self._publish_chain(updated_chain)
                    continue

                # Queue a generation stage job
                try:
                    submission = self._build_stage_submission(chain, next_stage, upstream_outputs, upstream_prompt)
                    jobs = self.queue.submit(submission, user=chain.user)
                except Exception as error:
                    msg = f"{type(error).__name__}: {error}"
                    self.store.update_stage(chain_id, next_stage.index, status="failed", stage_hash=stage_hash, error=msg, started=now, finished=time.time())
                    failed = self.store.update_chain(chain_id, status="failed", error=msg, finished=time.time())
                    self._force_stages.pop(chain_id, None)
                    assert failed is not None
                    self._publish_stage_completed(failed.stages[next_stage.index])
                    self._publish_chain(failed)
                    return failed

                job = jobs[0]
                self.store.update_stage(
                    chain_id,
                    next_stage.index,
                    status="running",
                    stage_hash=stage_hash,
                    cached=False,
                    job_id=job.id,
                    started=now,
                )
                updated_chain = self.store.get_chain(chain_id)
                assert updated_chain is not None
                self._publish_chain(updated_chain)
                return updated_chain

    def on_job_finished(self, job: Job) -> None:
        """Completion hook called by :class:`JobQueue` when a job reaches a final status."""

        found = self.store.find_stage_by_job(job.id)
        if found is None:
            return
        chain_id, stage_index = found
        should_advance = False
        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None or stage_index >= len(chain.stages):
                return
            stage = chain.stages[stage_index]
            if stage.status != "running":
                return
            now = time.time()
            output_ids = [out["id"] for out in (job.outputs or []) if isinstance(out, Mapping) and out.get("kind", "image") == "image" and isinstance(out.get("id"), str)]
            if job.status == COMPLETED:
                self.store.update_stage(
                    chain_id,
                    stage_index,
                    status="completed",
                    output_file_ids=output_ids,
                    infotext=job.infotext,
                    finished=now,
                )
                if stage.stage_hash and output_ids:
                    self.store.put_cached(
                        stage.stage_hash,
                        stage.kind,
                        output_file_ids=output_ids,
                        infotext=job.infotext,
                        job_id=job.id,
                    )
                updated = self.store.get_chain(chain_id)
                assert updated is not None
                self._publish_stage_completed(updated.stages[stage_index])
                self._publish_chain(updated)
                should_advance = updated.status == "running"
            elif job.status in (CANCELLED, INTERRUPTED):
                self.store.update_stage(chain_id, stage_index, status="cancelled", error=job.error or job.status, finished=now)
                updated = self.store.update_chain(chain_id, status="cancelled", error=job.error or job.status, finished=now)
                self._force_stages.pop(chain_id, None)
                assert updated is not None
                self._publish_stage_completed(updated.stages[stage_index])
                self._publish_chain(updated)
            elif job.status == FAILED:
                self.store.update_stage(chain_id, stage_index, status="failed", error=job.error or "stage job failed", finished=now)
                updated = self.store.update_chain(chain_id, status="failed", error=job.error or "stage job failed", finished=now)
                self._force_stages.pop(chain_id, None)
                assert updated is not None
                self._publish_stage_completed(updated.stages[stage_index])
                self._publish_chain(updated)
        if should_advance:
            self._advance(chain_id)

    def _publish_stage_completed(self, stage: StageRun) -> None:
        self.bus.publish(
            "chain.stage_completed",
            chain_id=stage.chain_id,
            stage_index=stage.index,
            kind=stage.kind,
            status=stage.status,
            cached=stage.cached,
            job_id=stage.job_id,
            output_file_ids=list(stage.output_file_ids),
            output_prompt=stage.output_prompt,
        )

    def _publish_chain(self, chain: JobChain) -> None:
        from datetime import UTC, datetime

        def _iso(ts: float | None) -> str | None:
            return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None

        self.bus.publish(
            "chain.updated",
            chain_id=chain.id,
            name=chain.name,
            status=chain.status,
            current_stage=chain.current_stage,
            total_stages=len(chain.stages),
            stages=[stage_json(st, _iso) for st in chain.stages],
        )


__all__ = [
    "CHAIN_STATUSES",
    "DEFAULT_STAGE_TABS",
    "STAGE_KINDS",
    "STAGE_STATUSES",
    "CachedStageOutput",
    "ChainError",
    "ChainRunner",
    "ChainStore",
    "ChainTemplate",
    "JobChain",
    "StageRun",
    "StageSpec",
    "chain_json",
    "compute_stage_hash",
    "stage_json",
    "template_json",
]
