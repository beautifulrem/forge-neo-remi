"""Non-node linear multi-stage pipelines with stage caching: ``/api/v2/chains``."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Annotated, Any, Literal
from weakref import WeakKeyDictionary

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import Field

from remi.jobs.chains import StageSpec, chain_json, check_stages_finite, template_json
from remi.server import models as m
from remi.server.deps import ServicesDep, UserDep, iso
from remi.ui.errors import Refused
from remi.ui.sentence import Sentence

router = APIRouter(prefix="/chains")

ERRORS = {404: {"model": m.Detail}, 409: {"model": m.Detail}, 422: {"model": m.Detail | m.ValidationErrorBody}}


class _RecentChains:
    """Recent keyed ``POST /chains`` submissions per runner instance."""

    def __init__(self, *, ttl: float = 600.0, limit: int = 1000, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl, self.limit, self.clock = ttl, limit, clock
        self.lock = threading.Lock()
        self._entries: OrderedDict[tuple[str | None, str], tuple[float, str, str, dict[str, Any]]] = OrderedDict()

    def _expire(self) -> None:
        now = self.clock()
        while self._entries:
            key, (recorded, _digest, _chain_id, _snapshot) = next(iter(self._entries.items()))
            if now - recorded < self.ttl:
                break
            del self._entries[key]

    def get(self, user: str | None, key: str) -> tuple[str, str, dict[str, Any]] | None:
        self._expire()
        entry = self._entries.get((user, key))
        if entry is None:
            return None
        _recorded, digest, chain_id, snapshot = entry
        return digest, chain_id, snapshot

    def put(self, user: str | None, key: str, digest: str, chain_id: str, snapshot: dict[str, Any]) -> None:
        self._expire()
        self._entries[(user, key)] = (self.clock(), digest, chain_id, snapshot)
        while len(self._entries) > self.limit:
            self._entries.popitem(last=False)


_RECENT_LOCK = threading.Lock()
_RECENT_BY_RUNNER: WeakKeyDictionary[Any, _RecentChains] = WeakKeyDictionary()


def _recent_for(runner: Any) -> _RecentChains:
    with _RECENT_LOCK:
        recent = _RECENT_BY_RUNNER.get(runner)
        if recent is None:
            recent = _RECENT_BY_RUNNER[runner] = _RecentChains()
        return recent


def _request_digest(body: ChainCreateRequest) -> str:
    payload = json.dumps(body.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

StageKind = Literal["prompt_upsample", "generate", "refine", "finish"]
ChainStatus = Literal["queued", "running", "paused", "completed", "failed", "cancelled"]
StageStatus = Literal["pending", "running", "cached", "completed", "failed", "skipped", "cancelled"]


class StageSpecModel(m.Model):
    index: int | None = None
    kind: StageKind
    name: str = ""
    enabled: bool = True
    tab: str = ""
    core: dict[str, Any] = {}
    scripts: dict[str, Any] = {}
    config: dict[str, Any] = {}


class StageRunModel(m.Model):
    index: int
    kind: StageKind
    name: str
    enabled: bool
    tab: str
    core: dict[str, Any] = {}
    scripts: dict[str, Any] = {}
    config: dict[str, Any] = {}
    status: StageStatus
    stage_hash: str | None = None
    cached: bool = False
    job_id: str | None = None
    output_file_ids: list[str] = []
    output_prompt: str | None = None
    infotext: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class JobChainModel(m.Model):
    id: str
    name: str
    status: ChainStatus
    current_stage: int
    total_stages: int
    stages: list[StageRunModel]
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class ChainCreateRequest(m.Model):
    name: str = Field("", max_length=200)
    stages: list[StageSpecModel] = Field(min_length=1, max_length=8)
    run_now: bool = True
    priority: int = Field(0, ge=-1000, le=1000)
    model_fingerprint: str | None = None


class ChainPreviewRequest(m.Model):
    stages: list[StageSpecModel] = Field(min_length=1, max_length=8)
    model_fingerprint: str | None = Field(None, description="Extra salt for the stage hashes, added to the loaded model's fingerprint")
    chain_id: str | None = Field(None, description="Plan a re-run of this chain: its random-seed stages can reuse their outputs")


class StageCachePlan(m.Model):
    index: int
    kind: StageKind
    name: str
    enabled: bool
    stage_hash: str
    cache_hit: bool
    cached_outputs: list[str] = []
    cached_prompt: str | None = None


class ChainPreviewResult(m.Model):
    stages: list[StageCachePlan]


class ChainRerunRequest(m.Model):
    stages: list[StageSpecModel] | None = Field(None, min_length=1, max_length=8)
    from_stage: int | None = Field(None, ge=0, le=8)
    force_stages: list[int] = []
    model_fingerprint: str | None = None


class ChainTemplateModel(m.Model):
    id: str
    name: str
    description: str = ""
    model_family: str | None = None
    stages: list[StageSpecModel]
    tags: list[str] = []
    created_at: str
    updated_at: str


class ChainTemplateList(m.Model):
    items: list[ChainTemplateModel]


class ChainTemplateCreate(m.Model):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=2000)
    model_family: str | None = Field(None, max_length=100)
    stages: list[StageSpecModel] = Field(min_length=1, max_length=8)
    tags: list[str] = []


def _to_specs(items: list[StageSpecModel]) -> list[StageSpec]:
    return [StageSpec.from_mapping(item.model_dump(), index=idx) for idx, item in enumerate(items)]


def _chain_response(data: dict[str, Any], *, status_code: int = 200) -> Any:
    from remi.setting_values import nested_deeper

    if nested_deeper(data, 200):
        from remi.server.app import TextSafeResponse

        return TextSafeResponse(content=data, status_code=status_code)
    return data


@router.get("", response_model=m.Page[JobChainModel], summary="List multi-stage pipeline chains")
def list_chains(
    services: ServicesDep,
    status: str | None = None,
    cursor: str | None = None,
    limit: int = Query(50, ge=1, le=200),
) -> Any:
    chains, next_cursor, total = services.chains.store.list_chains(status=status, cursor=cursor, limit=limit)
    return _chain_response({"items": [chain_json(ch, iso) for ch in chains], "next_cursor": next_cursor, "total": total})


@router.post("", response_model=JobChainModel, status_code=201, responses=ERRORS, summary="Create (and optionally run) a linear multi-stage chain")
def create_chain(
    body: ChainCreateRequest,
    services: ServicesDep,
    user: UserDep,
    idempotency_key: Annotated[
        str | None,
        Header(
            min_length=1,
            max_length=255,
            description="a fresh value per submission (a UUID): the same key sent again within 10 minutes creates nothing and answers the first chain",
        ),
    ] = None,
) -> Any:
    specs = _to_specs(body.stages)
    check_stages_finite(specs)
    if idempotency_key is None:
        chain = services.chains.create_chain(
            specs,
            name=body.name,
            run_now=body.run_now,
            priority=body.priority,
            user=user,
            model_fingerprint=body.model_fingerprint,
        )
        return _chain_response(chain_json(chain, iso), status_code=201)

    digest = _request_digest(body)
    recent = _recent_for(services.chains)
    with recent.lock:
        found = recent.get(user, idempotency_key)
        if found is not None:
            prev_digest, chain_id, snapshot = found
            if prev_digest != digest:
                raise Refused(Sentence("Idempotency-Key was already used with different parameters"), 409)
            current = services.chains.store.get_chain(chain_id)
            return _chain_response(chain_json(current, iso) if current is not None else snapshot, status_code=201)

        chain = services.chains.create_chain(
            specs,
            name=body.name,
            run_now=body.run_now,
            priority=body.priority,
            user=user,
            model_fingerprint=body.model_fingerprint,
        )
        snapshot = chain_json(chain, iso)
        recent.put(user, idempotency_key, digest, chain.id, snapshot)
        return _chain_response(snapshot, status_code=201)


@router.post("/preview", response_model=ChainPreviewResult, responses=ERRORS, summary="Dry-run cache plan showing which stages will reuse cached outputs")
def preview_chain(body: ChainPreviewRequest, services: ServicesDep) -> dict[str, Any]:
    plan = services.chains.plan_chain(_to_specs(body.stages), model_fingerprint=body.model_fingerprint, chain_id=body.chain_id)
    return {"stages": plan}


@router.get("/templates", response_model=ChainTemplateList, summary="List saved pipeline templates")
def list_templates(services: ServicesDep) -> Any:
    items = services.chains.store.list_templates()
    return _chain_response({"items": [template_json(tmpl, iso) for tmpl in items]})


@router.post("/templates", response_model=ChainTemplateModel, status_code=201, responses=ERRORS, summary="Save a reusable pipeline template")
def create_template(body: ChainTemplateCreate, services: ServicesDep) -> Any:
    tmpl = services.chains.store.create_template(
        body.name,
        _to_specs(body.stages),
        description=body.description,
        model_family=body.model_family,
        tags=body.tags,
    )
    return _chain_response(template_json(tmpl, iso), status_code=201)


@router.get("/templates/{template_id}", response_model=ChainTemplateModel, responses=ERRORS, summary="Get a pipeline template")
def get_template(template_id: str, services: ServicesDep) -> Any:
    tmpl = services.chains.store.get_template(template_id)
    if tmpl is None:
        raise HTTPException(status_code=404, detail=f"no chain template {template_id!r}")
    return _chain_response(template_json(tmpl, iso))


@router.delete("/templates/{template_id}", status_code=204, responses=ERRORS, summary="Delete a pipeline template")
def delete_template(template_id: str, services: ServicesDep) -> Response:
    if not services.chains.store.delete_template(template_id):
        raise HTTPException(status_code=404, detail=f"no chain template {template_id!r}")
    return Response(status_code=204)


@router.get("/{chain_id}", response_model=JobChainModel, responses=ERRORS, summary="Get a chain and all stage runs")
def get_chain(chain_id: str, services: ServicesDep) -> Any:
    chain = services.chains.store.get_chain(chain_id)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"no chain {chain_id!r}")
    return _chain_response(chain_json(chain, iso))


@router.post("/{chain_id}/rerun", response_model=JobChainModel, responses=ERRORS, summary="Re-run a chain, automatically reusing unchanged cached upstream stages")
def rerun_chain(chain_id: str, services: ServicesDep, body: ChainRerunRequest | None = None) -> Any:
    specs = _to_specs(body.stages) if (body is not None and body.stages is not None) else None
    chain = services.chains.rerun_chain(
        chain_id,
        stages=specs,
        from_stage=body.from_stage if body else None,
        force_stages=body.force_stages if body else (),
        model_fingerprint=body.model_fingerprint if body else None,
    )
    return _chain_response(chain_json(chain, iso))


@router.post("/{chain_id}/pause", response_model=JobChainModel, responses=ERRORS, summary="Pause a chain between stages")
def pause_chain(chain_id: str, services: ServicesDep) -> Any:
    return _chain_response(chain_json(services.chains.pause_chain(chain_id), iso))


@router.post("/{chain_id}/resume", response_model=JobChainModel, responses=ERRORS, summary="Resume a paused chain")
def resume_chain(chain_id: str, services: ServicesDep) -> Any:
    return _chain_response(chain_json(services.chains.resume_chain(chain_id), iso))


@router.post("/{chain_id}/cancel", response_model=JobChainModel, responses=ERRORS, summary="Cancel a chain and any active stage job")
def cancel_chain(chain_id: str, services: ServicesDep) -> Any:
    return _chain_response(chain_json(services.chains.cancel_chain(chain_id), iso))


@router.delete("/{chain_id}", status_code=204, responses=ERRORS, summary="Delete a finished or paused chain")
def delete_chain(chain_id: str, services: ServicesDep) -> Response:
    if not services.chains.delete_chain(chain_id):
        raise HTTPException(status_code=404, detail=f"no chain {chain_id!r}")
    return Response(status_code=204)
