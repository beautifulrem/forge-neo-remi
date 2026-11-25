"""Non-node linear multi-stage pipelines (:class:`JobChain` & :class:`StageRun`)
with content-addressed stage caching and zero-node partial re-execution.

Supported ordered stage kinds:
- ``prompt_upsample`` — optional multimodal/local Prompt Upsampler (PUP)
- ``generate`` — base generation (``txt2img`` / ``img2img``)
- ``refine`` — refinement pass (``img2img`` / Inpaint / ADetailer)
- ``finish`` — finishing pass (Hires ``upscale_from`` or ``extras``)

Each stage's deterministic ``stage_hash`` hashes its canonical parameters,
its upstream stage's hash/output file IDs/prompt, and the fingerprint of the
model loaded when the stage runs. Re-running a chain after tweaking a downstream
stage (e.g. ``refine`` or ``finish``) automatically reuses cached upstream stage
output ``file_id``s without re-executing unchanged upstream stages.

A stage whose seed is random (``-1`` or unset) makes a new image every time,
so its hash is salted with the chain id: re-running that chain reuses it, a
new chain does not.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remi.files import FileStore, connect, decode_cursor, encode_cursor, transaction
from remi.jobs.queue import JobError, JobQueue, Submission
from remi.jobs.store import CANCELLED, CHAIN_SCHEMA, COMPLETED, FAILED, INTERRUPTED, Job, JobStore
from remi.realtime import EventBus
from remi.ui.sentence import Sentence

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
TERMINAL_CHAIN_STATUSES = ("completed", "failed", "cancelled")
MAX_STAGES = 8
STAGE_HASH_VERSION = 2
"""Bumped when the hash inputs change, so entries keyed the old way never hit."""
SEEDED_TABS = ("txt2img", "img2img")
"""Tabs whose output depends on the seed (``extras`` and prompt upsampling do not)."""


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
            raise ChainError(Sentence("stage kind must be one of {{kinds}}", kinds=", ".join(STAGE_KINDS)), status=422)
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
    created: float  # stored, or last reused by a pipeline


def _check_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ChainError(Sentence("{{path}} must not contain NaN or Infinity", path=path), status=422)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")


def check_stages_finite(stages: Sequence[StageSpec]) -> None:
    """Refuse NaN and Infinity in what a chain or template stores (JSON has
    neither: the stored stage would no longer read back as it was sent)."""

    for index, spec in enumerate(stages):
        for part in ("core", "scripts", "config"):
            _check_finite(getattr(spec, part), f"stages[{index}].{part}")


def _random_seed(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        return int(float(value)) == -1
    except (TypeError, ValueError):
        return False


def uses_random_seed(spec: StageSpec) -> bool:
    """Whether the stage draws a new seed each time it runs (``-1`` or no seed
    given, which the form defaults to ``-1``; also a random variation seed
    with a non-zero strength)."""

    if spec.tab not in SEEDED_TABS:
        return False
    seed_values = spec.scripts.get("seed")
    seed_values = seed_values if isinstance(seed_values, Mapping) else {}
    if _random_seed(seed_values.get("seed", spec.core.get("seed"))):
        return True
    try:
        strength = float(seed_values.get("subseed_strength") or 0)
    except (TypeError, ValueError):
        strength = 0.0
    return strength > 0 and _random_seed(seed_values.get("subseed"))


def compute_stage_hash(
    spec: StageSpec,
    *,
    upstream_hash: str = "",
    upstream_outputs: Sequence[str] = (),
    upstream_prompt: str | None = None,
    model_fingerprint: str = "",
    salt: str = "",
) -> str:
    """Deterministic content-addressed hash of a stage's effective inputs
    (``salt``: what else the output depends on, e.g. the chain of a random seed)."""

    payload = {
        "v": STAGE_HASH_VERSION,
        "salt": salt,
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
            self._db.executescript(CHAIN_SCHEMA)

    def close(self) -> None:
        if self._owns_db:
            with self._lock:
                self._db.close()

    def __enter__(self) -> ChainStore:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """Several writes that land together or not at all (the connection
        autocommits otherwise)."""

        with self._lock, transaction(self._db, "BEGIN"):
            yield

    def _insert_stages(self, chain_id: str, stages: Sequence[StageSpec]) -> None:
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

    def _load_stages(self, chain_ids: Sequence[str]) -> dict[str, list[StageRun]]:
        """The stages of each chain (one query for a page of chains)."""

        stages: dict[str, list[StageRun]] = {chain_id: [] for chain_id in chain_ids}
        if not stages:
            return stages
        rows = self._db.execute(
            f"""
            SELECT chain_id, stage_index, kind, name, enabled, tab, core, scripts, config,
                   status, stage_hash, cached, job_id, output_file_ids, output_prompt,
                   infotext, error, started, finished
            FROM stage_runs WHERE chain_id IN ({",".join("?" * len(stages))}) ORDER BY chain_id, stage_index ASC
            """,
            list(stages),
        ).fetchall()
        for row in rows:
            stages[row[0]].append(
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

    def _row_to_chain(self, row: tuple[Any, ...], stages: list[StageRun]) -> JobChain:
        return JobChain(
            id=row[0],
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
            stages=stages,
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
            raise ChainError(Sentence("a chain requires at least one stage"), status=422)
        if len(stages) > MAX_STAGES:
            raise ChainError(Sentence("a chain may have at most {{maximum}} stages", maximum=MAX_STAGES), status=422)
        check_stages_finite(stages)
        chain_id = secrets.token_hex(8)
        now = time.time()
        clean_name = (name or "").strip()[:200] or f"Pipeline {chain_id[:6]}"
        with self._transaction():
            self._db.execute(
                """
                INSERT INTO chains (id, name, status, current_stage, priority, user, model_fingerprint, error, created, updated)
                VALUES (?, ?, ?, 0, ?, ?, ?, NULL, ?, ?)
                """,
                (chain_id, clean_name, status, int(priority), user, model_fingerprint, now, now),
            )
            self._insert_stages(chain_id, stages)
        created = self.get_chain(chain_id)
        assert created is not None
        return created

    def get_chain(self, chain_id: str) -> JobChain | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, name, status, current_stage, priority, user, model_fingerprint, error, created, updated, started, finished FROM chains WHERE id = ?",
                (chain_id,),
            ).fetchone()
            return self._row_to_chain(row, self._load_stages([chain_id])[chain_id]) if row is not None else None

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
            stages = self._load_stages([row[0] for row in rows[:limit]])
            items = [self._row_to_chain(row, stages[row[0]]) for row in rows[:limit]]
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
            raise ChainError(Sentence("stages count must be between 1 and {{maximum}}", maximum=MAX_STAGES), status=422)
        check_stages_finite(stages)
        with self._transaction():
            self._db.execute("DELETE FROM stage_runs WHERE chain_id = ?", (chain_id,))
            self._insert_stages(chain_id, stages)

    def reset_stages(self, chain_id: str) -> None:
        """Every stage back to pending (skipped when disabled), results cleared."""

        with self._transaction():
            self._db.execute(
                """
                UPDATE stage_runs SET status = CASE WHEN enabled THEN 'pending' ELSE 'skipped' END,
                    stage_hash = NULL, cached = 0, job_id = NULL, output_file_ids = '[]', output_prompt = NULL,
                    infotext = NULL, error = NULL, started = NULL, finished = NULL
                WHERE chain_id = ?
                """,
                (chain_id,),
            )
            self._db.execute("UPDATE chains SET updated = ? WHERE id = ?", (time.time(), chain_id))

    def get_forced(self, chain_id: str) -> set[int]:
        """Stages a re-run must execute even when their output is cached."""

        with self._lock:
            row = self._db.execute("SELECT stages FROM chain_force WHERE chain_id = ?", (chain_id,)).fetchone()
        return {int(i) for i in json.loads(row[0])} if row else set()

    def set_forced(self, chain_id: str, stages: set[int]) -> None:
        with self._lock:
            if stages:
                self._db.execute(
                    "INSERT INTO chain_force (chain_id, stages) VALUES (?, ?) ON CONFLICT(chain_id) DO UPDATE SET stages = excluded.stages",
                    (chain_id, json.dumps(sorted(stages))),
                )
            else:
                self._db.execute("DELETE FROM chain_force WHERE chain_id = ?", (chain_id,))

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
        with self._transaction():
            self._db.execute("DELETE FROM stage_runs WHERE chain_id = ?", (chain_id,))
            self._db.execute("DELETE FROM chain_force WHERE chain_id = ?", (chain_id,))
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

    def touch_cached(self, stage_hash: str, when: float) -> None:
        """A pipeline reused this entry at ``when``: its ``created`` is the
        last use from now on, so :meth:`prune_cache` keeps the entries in use."""

        with self._lock:
            self._db.execute("UPDATE stage_cache SET created = ? WHERE stage_hash = ?", (when, stage_hash))

    def prune_cache(self, older_than: float) -> int:
        """Forget the cached stage outputs stored or last reused before
        ``older_than`` (a time); how many went. A pipeline that needs one runs
        the stage again."""

        with self._lock:
            return self._db.execute("DELETE FROM stage_cache WHERE created < ?", (older_than,)).rowcount

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
            rows = self._db.execute("SELECT id, name, description, model_family, stages, tags, created, updated FROM chain_templates ORDER BY updated DESC, name ASC").fetchall()
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
            raise ChainError(Sentence("template name is required"), status=422)
        if not stages or len(stages) > MAX_STAGES:
            raise ChainError(Sentence("template must contain 1..{{maximum}} stages", maximum=MAX_STAGES), status=422)
        check_stages_finite(stages)
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
                raise ChainError(Sentence("a chain template named {{name}} already exists", name=repr(clean_name)), status=409) from error
        tmpl = self.get_template(template_id)
        assert tmpl is not None
        return tmpl

    def delete_template(self, template_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM chain_templates WHERE id = ?", (template_id,)).rowcount > 0

    # endregion


def _file_identity(path: Any) -> str:
    try:
        import os

        stat = os.stat(path)
    except (OSError, TypeError, ValueError):
        return ""
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _default_model_fingerprint() -> str:
    """What the loaded model and the settings that change every image amount to.

    The checkpoint goes in by name and file identity (size and mtime), so a
    file replaced under the same name does not reuse the old outputs."""

    try:
        from remi.setting_values import setting

        checkpoint = str(setting("sd_model_checkpoint", "") or "")
        identity = ""
        try:
            from modules import sd_models

            info = sd_models.get_closet_checkpoint_match(checkpoint) if checkpoint else None
            if info is not None:
                checkpoint = str(getattr(info, "name", None) or checkpoint)
                identity = _file_identity(getattr(info, "filename", None))
        except Exception:
            pass
        modules = setting("forge_additional_modules", [])
        parts = [
            checkpoint,
            identity,
            setting("forge_preset", "") or "",
            setting("forge_unet_storage_dtype", "") or "",
            setting("sd_vae", "") or "",
            ",".join(sorted(f"{m}@{_file_identity(m)}" for m in modules)),
            setting("CLIP_stop_at_last_layers") or "",
            setting("emphasis") or "",
            setting("eta_noise_seed_delta") or "",
            setting("randn_source") or "",
        ]
        return "|".join(str(p) for p in parts)
    except Exception:
        return ""


UpsamplerFunc = Callable[[StageRun, str | None, Sequence[str]], str]


def _default_upsampler(stage: StageRun, upstream_prompt: str | None, upstream_outputs: Sequence[str], *, config: Any = None) -> str:
    """Default ``prompt_upsample`` stage handler (``remi.prompt_tools.pup``);
    ``config``: the prompt tools' provider (endpoint, model, key), as
    ``POST /prompt-tools/upsample`` passes it.

    Errors propagate: the stage fails instead of passing the base prompt on
    as if it had been upsampled (and caching that)."""

    base_prompt = str(stage.core.get("prompt") or upstream_prompt or "")
    if not base_prompt.strip():
        return base_prompt
    cfg = stage.config or {}
    local_model_path = cfg.get("local_model_path")
    if local_model_path:
        from modules import shared

        # the same guard as the upsample route: a client-given path is a server folder
        if getattr(getattr(shared, "cmd_opts", None), "hide_ui_dir_config", False):
            raise PermissionError("local_model_path is disabled under --hide-ui-dir-config")
    else:
        from remi.prompt_tools import TranslationError, registry

        # the endpoint is the prompt tools' provider: off with them, as the upsample route is
        if not registry.tools_enabled:
            raise TranslationError(Sentence("the prompt tools are disabled"), 409)
    from remi.prompt_tools.pup import upsample_prompt

    res = upsample_prompt(
        base_prompt,
        model_family=cfg.get("model_family"),
        aspect_ratio=cfg.get("aspect_ratio"),
        bbox_draft=cfg.get("bbox_draft"),
        mode=str(cfg.get("mode") or "auto"),
        local_model_path=local_model_path,
        config=config,
    )
    return res.prompt


def _stage_images(job: Job) -> list[str]:
    """The images a stage hands on: its results, not the grid or extra images
    (ControlNet maps) that come with them."""

    return [out["id"] for out in (job.outputs or []) if isinstance(out, Mapping) and out.get("kind", "image") == "image" and not out.get("role") and isinstance(out.get("id"), str)]


class ChainRunner:
    """Orchestrates linear multi-stage pipelines over :class:`JobQueue` with stage caching.

    ``model_fingerprint`` describes the model loaded now; it is read when a
    stage is looked up and again when its job finishes (a model picked while
    it waited is the one it ran on). A chain's stored ``model_fingerprint`` is
    the caller's extra salt, never a substitute for it."""

    def __init__(
        self,
        store: ChainStore,
        queue: JobQueue,
        files: FileStore,
        bus: EventBus,
        *,
        upsampler: UpsamplerFunc | None = None,
        model_fingerprint: Callable[[], str] = _default_model_fingerprint,
        upsample_identity: Callable[[], str] = lambda: "",
    ) -> None:
        self.store = store
        self.queue = queue
        self.files = files
        self.bus = bus
        self.upsampler = upsampler or _default_upsampler
        self.model_fingerprint = model_fingerprint
        self.upsample_identity = upsample_identity
        """the endpoint and model ``upsampler`` asks (never its API key): part of
        a ``prompt_upsample`` stage's hash, as the model is of a generation's"""
        self._lock = threading.RLock()
        self._threads: set[threading.Thread] = set()
        self._run_fingerprints: dict[str, str] = {}
        self._local_upsample_guard = threading.Lock()
        self._local_upsample: str | None = None
        """the chain whose local model's upsample stage runs now, behind the generation lock (one at
        a time): set while the queue's state is that stage's to flip, so Cancel stops it as Interrupt does"""
        """job id → the model fingerprint when the stage job started"""
        self._unhook = queue.add_completion_hook(self.on_job_finished)
        self._unhook_start: Callable[[], None] | None = queue.add_start_hook(self._on_job_started)
        self._unhook_busy: Callable[[], None] | None = queue.add_busy_check(self.busy)

    def close(self) -> None:
        if self._unhook is not None:
            self._unhook()
            self._unhook = None
        if self._unhook_start is not None:
            self._unhook_start()
            self._unhook_start = None
        if self._unhook_busy is not None:
            self._unhook_busy()
            self._unhook_busy = None
        for thread in list(self._threads):
            thread.join(timeout=10)
        self.store.close()

    def busy(self) -> bool:
        """Whether a pipeline is still queued or running (its next stage may
        not be a queued job yet: prompt upsampling runs on its own thread)."""

        _chains, _cursor, total = self.store.list_chains(status="queued,running", limit=1)
        return total > 0

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Wait for inline stages (prompt upsampling) running in the background."""

        deadline = time.monotonic() + timeout
        for thread in list(self._threads):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    def _fingerprint(self, salt: str = "", *, model: str | None = None) -> str:
        fp = self.model_fingerprint() if model is None else model
        return f"{fp}#{salt}" if salt else fp

    def _on_job_started(self, job: Job) -> None:
        # the model is in place now (a pick held while the job waited was applied just before);
        # settings changed while it runs need not have reached it, so this is the fingerprint it ran with
        if job.meta.get("chain_id"):
            self._run_fingerprints[job.id] = self.model_fingerprint()

    def _stage_hash(
        self,
        spec: StageSpec,
        upstream: tuple[str, Sequence[str], str | None],
        *,
        fingerprint: str,
        chain_id: str | None,
    ) -> str:
        upstream_hash, upstream_outputs, upstream_prompt = upstream
        salt = ""
        if uses_random_seed(spec):
            # a new chain draws a new seed; without a chain (a preview) nothing can match
            salt = f"chain:{chain_id}" if chain_id else f"once:{secrets.token_hex(8)}"
        if spec.kind == "prompt_upsample":
            # another endpoint or model writes another prompt (a local model is in the config)
            provider = "local" if spec.config.get("local_model_path") else self.upsample_identity()
            fingerprint = f"{fingerprint}|upsample:{provider}"
        return compute_stage_hash(
            spec,
            upstream_hash=upstream_hash,
            upstream_outputs=upstream_outputs,
            upstream_prompt=upstream_prompt,
            model_fingerprint=fingerprint,
            salt=salt,
        )

    def _cache_valid(self, entry: CachedStageOutput | None) -> bool:
        if entry is None:
            return False
        if entry.kind == "prompt_upsample":
            return entry.output_prompt is not None
        if not entry.output_file_ids:
            return False
        return all(self.files.get(fid) is not None for fid in entry.output_file_ids)

    def plan_chain(
        self,
        stages: Sequence[StageSpec],
        *,
        model_fingerprint: str | None = None,
        chain_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dry-run cache evaluation: reports per-stage ``stage_hash`` and ``cache_hit``.

        ``chain_id``: plan a re-run of that chain (its random-seed stages can
        hit); without it a random-seed stage always runs. Disabled stages pass
        their upstream on untouched, as :meth:`_advance` does."""

        salt = model_fingerprint
        if salt is None and chain_id:
            existing = self.store.get_chain(chain_id)
            salt = existing.model_fingerprint if existing is not None else ""
        fp = self._fingerprint(salt or "")
        plan: list[dict[str, Any]] = []
        upstream_hash = ""
        upstream_outputs: list[str] = []
        upstream_prompt: str | None = None
        chain_cache_intact = True

        for idx, spec in enumerate(stages):
            stage_hash = self._stage_hash(spec, (upstream_hash, upstream_outputs, upstream_prompt), fingerprint=fp, chain_id=chain_id)
            entry = {"index": idx, "kind": spec.kind, "name": spec.name, "enabled": bool(spec.enabled), "stage_hash": stage_hash, "cache_hit": False, "cached_outputs": [], "cached_prompt": None}
            if not spec.enabled:
                plan.append(entry)
                continue
            cached = self.store.get_cached(stage_hash) if chain_cache_intact else None
            upstream_hash = stage_hash
            if self._cache_valid(cached) and cached is not None:
                if cached.output_file_ids:
                    upstream_outputs = list(cached.output_file_ids)
                if cached.output_prompt is not None:
                    upstream_prompt = cached.output_prompt
                entry.update(cache_hit=True, cached_outputs=list(cached.output_file_ids), cached_prompt=cached.output_prompt)
            else:
                chain_cache_intact = False
                upstream_outputs = []
            plan.append(entry)
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
        """``model_fingerprint``: extra salt for the stage hashes (what else the
        caller knows the output depends on), added to the loaded model's."""

        chain = self.store.create_chain(stages, name=name, priority=priority, user=user, model_fingerprint=model_fingerprint or "", status="queued" if run_now else "paused")
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
                raise ChainError(Sentence("no chain {{id}}", id=repr(chain_id)), status=404)
            if chain.status == "running" or any(st.status == "running" for st in chain.stages):
                raise ChainError(Sentence("cancel or pause the running chain before re-running"), status=409)
            if stages is not None:
                self.store.replace_stages(chain_id, stages)
            else:
                self.store.reset_stages(chain_id)
            forced = {int(i) for i in force_stages}
            if from_stage is not None:
                total = len(stages) if stages is not None else len(chain.stages)
                forced |= set(range(max(0, int(from_stage)), total))
            self.store.set_forced(chain_id, forced)
            salt = model_fingerprint if model_fingerprint is not None else chain.model_fingerprint
            self.store.update_chain(chain_id, status="queued", current_stage=0, model_fingerprint=salt, error=None, started=None, finished=None)
        return self._advance(chain_id)

    def pause_chain(self, chain_id: str) -> JobChain:
        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None:
                raise ChainError(Sentence("no chain {{id}}", id=repr(chain_id)), status=404)
            if chain.status in TERMINAL_CHAIN_STATUSES:
                raise ChainError(Sentence("cannot pause chain in status {{status}}", status=repr(chain.status)), status=409)
            updated = self.store.update_chain(chain_id, status="paused")
            assert updated is not None
            self._publish_chain(updated)
            return updated

    def resume_chain(self, chain_id: str) -> JobChain:
        """Continue a paused chain (a stage still running finishes first)."""

        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None:
                raise ChainError(Sentence("no chain {{id}}", id=repr(chain_id)), status=404)
            if chain.status != "paused":
                raise ChainError(Sentence("chain is {{status}}, not paused", status=repr(chain.status)), status=409)
            running = any(st.status == "running" for st in chain.stages)
            updated = self.store.update_chain(chain_id, status="running" if running else "queued")
            assert updated is not None
            if running:
                self._publish_chain(updated)
                return updated
        return self._advance(chain_id)

    def cancel_chain(self, chain_id: str) -> JobChain:
        with self._lock:
            chain = self.store.get_chain(chain_id)
            if chain is None:
                raise ChainError(Sentence("no chain {{id}}", id=repr(chain_id)), status=404)
            if chain.status in TERMINAL_CHAIN_STATUSES:
                raise ChainError(Sentence("chain is already {{status}}", status=chain.status), status=409)
            now = time.time()
            active_jobs = [st.job_id for st in chain.stages if st.status == "running" and st.job_id]
            for st in chain.stages:
                if st.status in ("pending", "running"):
                    self.store.update_stage(chain_id, st.index, status="cancelled", finished=now)
            updated = self.store.update_chain(chain_id, status="cancelled", finished=now)
            self.store.set_forced(chain_id, set())
        for job_id in active_jobs:
            try:
                self.queue.interrupt(job_id)
            except Exception:
                pass
        # a local model's upsample stage is no job: it stops on the queue's state, which is its own
        # while it runs (it holds the generation lock); a stage asking an endpoint runs beside jobs
        with self._local_upsample_guard:
            if self._local_upsample == chain_id:
                state = self.queue.state()
                if state is not None:
                    state.interrupt()
        assert updated is not None
        self._publish_chain(updated)
        return updated

    def delete_chain(self, chain_id: str) -> bool:
        with self._lock:  # not between a resume or re-run and the stage job it queues
            chain = self.store.get_chain(chain_id)
            if chain is None:
                return False
            if chain.status == "running" or any(st.status == "running" for st in chain.stages):
                raise ChainError(Sentence("cannot delete a running chain; cancel it first"), status=409)
            return self.store.delete_chain(chain_id)

    def reconcile(self) -> int:
        """Settle chains a previous process left mid-run (call once the queue
        has started, so jobs it left running are already marked interrupted).

        A stage whose job finished is recorded as if the job had just ended; a
        stage whose job is gone, or an inline stage nobody runs any more,
        cancels its chain (also a paused one); queued chains continue. Returns
        how many chains moved."""

        moved = 0
        cursor: str | None = None
        while True:
            chains, cursor, _total = self.store.list_chains(status="queued,running,paused", cursor=cursor, limit=200)
            for chain in chains:
                if self._reconcile_one(chain):
                    moved += 1
            if cursor is None:
                return moved

    def _reconcile_one(self, chain: JobChain) -> bool:
        from remi.jobs.store import FINISHED

        finished_jobs: list[Job] = []
        with self._lock:
            now = time.time()
            for stage in chain.stages:
                if stage.status != "running":
                    continue
                job = self.queue.store.get(stage.job_id) if stage.job_id and stage.kind != "prompt_upsample" else None
                if job is not None and job.status in FINISHED:
                    finished_jobs.append(job)
                elif job is None:
                    error = "the server stopped while the stage ran" if stage.kind == "prompt_upsample" or not stage.job_id else "the stage job was deleted"
                    self.store.update_stage(chain.id, stage.index, status="cancelled", error=error, finished=now)
                    updated = self.store.update_chain(chain.id, status="cancelled", error=error, finished=now)
                    self.store.set_forced(chain.id, set())
                    assert updated is not None
                    self._publish_chain(updated)
                    return True
                else:
                    return False  # still queued: its hook will move the chain on
        if not finished_jobs and chain.status == "paused":
            return False  # nothing was running: it waits for its resume
        for job in finished_jobs:
            self.on_job_finished(job)
        if not finished_jobs:
            self._advance(chain.id)
        return True

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

    def _fail_chain(self, chain_id: str, stage_index: int, message: str, *, status: str = "failed", **stage_fields: Any) -> JobChain:
        """Record a stage that ended the chain (``status``: failed or cancelled); the caller holds the lock."""

        now = time.time()
        self.store.update_stage(chain_id, stage_index, status=status, error=message, finished=now, **stage_fields)
        updated = self.store.update_chain(chain_id, status=status, error=message, finished=now)
        self.store.set_forced(chain_id, set())
        assert updated is not None
        self._publish_stage_completed(updated.stages[stage_index])
        self._publish_chain(updated)
        return updated

    def _advance(self, chain_id: str) -> JobChain:
        while True:
            with self._lock:
                chain = self.store.get_chain(chain_id)
                if chain is None:
                    raise ChainError(Sentence("no chain {{id}}", id=repr(chain_id)), status=404)
                if chain.status in ("paused", *TERMINAL_CHAIN_STATUSES):
                    return chain
                if any(st.status == "running" for st in chain.stages):
                    return chain

                next_stage = next((st for st in chain.stages if st.status == "pending"), None)
                if next_stage is None:
                    done = self.store.update_chain(chain_id, status="completed", finished=time.time())
                    self.store.set_forced(chain_id, set())
                    assert done is not None
                    self._publish_chain(done)
                    return done

                now = time.time()
                if chain.started is None:
                    self.store.update_chain(chain_id, status="running", current_stage=next_stage.index, started=now)
                else:
                    self.store.update_chain(chain_id, status="running", current_stage=next_stage.index)
                try:
                    started = self._start_stage(chain, next_stage, now)
                except Exception as error:
                    # a chain left running with no stage running would never move again
                    log.exception("starting stage %d of chain %s", next_stage.index, chain_id)
                    return self._fail_chain(chain_id, next_stage.index, f"{type(error).__name__}: {error}")
                if started is not None:
                    return started

    def _start_stage(self, chain: JobChain, next_stage: StageRun, now: float) -> JobChain | None:
        """Under the lock: answer the stage from the cache (None: go on to the
        next one), or start it (the chain as it is now)."""

        chain_id = chain.id
        upstream = self._upstream_state(chain, next_stage.index)
        _upstream_hash, upstream_outputs, upstream_prompt = upstream
        stage_hash = self._stage_hash(next_stage.to_spec(), upstream, fingerprint=self._fingerprint(chain.model_fingerprint), chain_id=chain.id)

        forced = self.store.get_forced(chain_id)
        cached_entry = None if next_stage.index in forced else self.store.get_cached(stage_hash)
        if self._cache_valid(cached_entry) and cached_entry is not None:
            self.store.touch_cached(stage_hash, now)
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
            return None

        if next_stage.kind == "prompt_upsample":
            # Runs in the background: a request or the queue worker must not wait for it.
            # A local model (VRAM) also runs behind the generation lock (_run_upsample).
            self.store.update_stage(chain_id, next_stage.index, status="running", stage_hash=stage_hash, cached=False, started=now)
            thread = threading.Thread(
                target=self._run_upsample,
                args=(chain_id, next_stage, stage_hash, now, upstream_prompt, list(upstream_outputs)),
                name=f"remi-chain-upsample-{chain_id[:6]}",
                daemon=True,
            )
            self._threads.add(thread)
            try:
                thread.start()
            except BaseException:
                self._threads.discard(thread)
                raise
            updated_chain = self.store.get_chain(chain_id)
            assert updated_chain is not None
            self._publish_chain(updated_chain)
            return updated_chain

        # Queue a generation stage job (the completion hook waits for this lock,
        # so it sees the job id even when the job ends at once)
        try:
            submission = self._build_stage_submission(chain, next_stage, upstream_outputs, upstream_prompt)
            jobs = self.queue.submit(submission, user=chain.user)
        except Exception as error:
            return self._fail_chain(chain_id, next_stage.index, f"{type(error).__name__}: {error}", stage_hash=stage_hash, started=now)

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

    def _run_upsample(self, chain_id: str, stage: StageRun, stage_hash: str, started: float, upstream_prompt: str | None, upstream_outputs: list[str]) -> None:
        from remi.prompt_tools.translate import GenerationInterrupted

        def run_local() -> str:
            # Interrupt flips the queue's state while this holds the lock: the local model stops on it
            with self.queue.interruptible():
                with self._local_upsample_guard:
                    self._local_upsample = chain_id
                try:
                    return self.upsampler(stage, upstream_prompt, upstream_outputs)
                finally:
                    # before interruptible() clears the flag: one Cancel set cannot reach what runs next
                    with self._local_upsample_guard:
                        self._local_upsample = None

        try:
            result: str | None = None
            error: str | None = None
            interrupted = False
            try:
                if stage.config.get("local_model_path"):
                    # a local model loads into VRAM: behind the generation lock, on the main thread
                    with self.queue.lock:
                        if self._stage_running(chain_id, stage.index, stage_hash, started):
                            result = self.queue.dispatch(run_local)
                elif self._stage_running(chain_id, stage.index, stage_hash, started):
                    # the prompt tools' endpoint is only a network request: jobs go on meanwhile
                    result = self.upsampler(stage, upstream_prompt, upstream_outputs)
            except GenerationInterrupted as exc:
                error, interrupted = str(exc), True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            should_advance = False
            with self._lock:
                if not self._stage_running(chain_id, stage.index, stage_hash, started):
                    return  # cancelled, deleted, or re-run (another thread runs it now)
                if interrupted:  # as an interrupted generation stage: the pipeline is cancelled, nothing cached
                    self._fail_chain(chain_id, stage.index, error or "", status="cancelled")
                    self.queue.recheck_done()
                    return
                if error is not None or result is None:
                    self._fail_chain(chain_id, stage.index, error or "prompt upsampling returned nothing")
                    self.queue.recheck_done()  # the pipeline ended here, where no job's end looks at the queue
                    return
                self.store.update_stage(chain_id, stage.index, status="completed", output_prompt=result, finished=time.time())
                self.store.put_cached(stage_hash, stage.kind, output_prompt=result)
                updated = self.store.get_chain(chain_id)
                assert updated is not None
                self._publish_stage_completed(updated.stages[stage.index])
                self._publish_chain(updated)
                should_advance = updated.status in ("running", "queued")
            if should_advance and self._advance(chain_id).status in ("completed", "failed"):
                self.queue.recheck_done()  # the last stage, or the next one refused: as above
        except Exception:
            log.exception("prompt upsampling stage of chain %s", chain_id)
        finally:
            self._threads.discard(threading.current_thread())

    def _stage_running(self, chain_id: str, stage_index: int, stage_hash: str, started: float) -> bool:
        """Whether this run of the stage is still the one wanted. A re-run of the
        chain starts the stage over, maybe with the same parameters: ``started``
        (when this run began) tells a cancelled run's late answer from the re-run's."""

        chain = self.store.get_chain(chain_id)
        if chain is None or chain.status in TERMINAL_CHAIN_STATUSES or stage_index >= len(chain.stages):
            return False
        stage = chain.stages[stage_index]
        return stage.status == "running" and stage.stage_hash == stage_hash and stage.started == started

    def on_job_finished(self, job: Job) -> None:
        """Completion hook called by :class:`JobQueue` when a job reaches a final status
        (also with a cancelled copy of a queued job that was deleted)."""

        should_advance = False
        run_fingerprint = self._run_fingerprints.pop(job.id, None)
        with self._lock:
            found = self.store.find_stage_by_job(job.id)
            if found is None:
                return
            chain_id, stage_index = found
            chain = self.store.get_chain(chain_id)
            if chain is None or stage_index >= len(chain.stages):
                return
            stage = chain.stages[stage_index]
            if stage.status != "running":
                return
            output_ids = _stage_images(job)
            if job.status == COMPLETED and not output_ids:
                self._fail_chain(chain_id, stage_index, "the stage made no image to pass on")
                return
            if job.status == COMPLETED:
                # keyed by the model it ran on, not the one loaded when it was queued; when that is
                # unknown (it ended before a restart), by what was expected at submission
                stage_hash = stage.stage_hash
                if run_fingerprint is not None or not stage_hash:
                    stage_hash = self._stage_hash(
                        stage.to_spec(),
                        self._upstream_state(chain, stage_index),
                        fingerprint=self._fingerprint(chain.model_fingerprint, model=run_fingerprint),
                        chain_id=chain.id,
                    )
                self.store.update_stage(
                    chain_id,
                    stage_index,
                    status="completed",
                    stage_hash=stage_hash,
                    output_file_ids=output_ids,
                    infotext=job.infotext,
                    finished=time.time(),
                )
                self.store.put_cached(stage_hash, stage.kind, output_file_ids=output_ids, infotext=job.infotext, job_id=job.id)
                updated = self.store.get_chain(chain_id)
                assert updated is not None
                self._publish_stage_completed(updated.stages[stage_index])
                self._publish_chain(updated)
                should_advance = updated.status in ("running", "queued")
            elif job.status in (CANCELLED, INTERRUPTED):
                self._fail_chain(chain_id, stage_index, job.error or job.status, status="cancelled")
            elif job.status == FAILED:
                self._fail_chain(chain_id, stage_index, job.error or "stage job failed")
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
        """``chain.updated`` carries the whole chain, as ``GET /chains/{id}`` returns it."""

        from datetime import UTC, datetime

        def _iso(ts: float | None) -> str | None:
            return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None

        self.bus.publish("chain.updated", chain_id=chain.id, **chain_json(chain, _iso))


__all__ = [
    "CHAIN_STATUSES",
    "DEFAULT_STAGE_TABS",
    "STAGE_KINDS",
    "STAGE_STATUSES",
    "TERMINAL_CHAIN_STATUSES",
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
    "uses_random_seed",
]
