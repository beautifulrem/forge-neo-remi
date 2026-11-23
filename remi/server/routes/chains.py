"""Non-node linear multi-stage pipelines with stage caching: ``/api/v2/chains``."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import Field

from remi.jobs.chains import StageSpec, chain_json, template_json
from remi.server import models as m
from remi.server.deps import ServicesDep, UserDep, iso

router = APIRouter(prefix="/chains")

ERRORS = {404: {"model": m.Detail}, 409: {"model": m.Detail}, 422: {"model": m.Detail}}

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


@router.get("", response_model=m.Page[JobChainModel], summary="List multi-stage pipeline chains")
def list_chains(
    services: ServicesDep,
    status: str | None = None,
    cursor: str | None = None,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    chains, next_cursor, total = services.chains.store.list_chains(status=status, cursor=cursor, limit=limit)
    return {"items": [chain_json(ch, iso) for ch in chains], "next_cursor": next_cursor, "total": total}


@router.post("", response_model=JobChainModel, status_code=201, responses=ERRORS, summary="Create (and optionally run) a linear multi-stage chain")
def create_chain(body: ChainCreateRequest, services: ServicesDep, user: UserDep) -> dict[str, Any]:
    chain = services.chains.create_chain(
        _to_specs(body.stages),
        name=body.name,
        run_now=body.run_now,
        priority=body.priority,
        user=user,
        model_fingerprint=body.model_fingerprint,
    )
    return chain_json(chain, iso)


@router.post("/preview", response_model=ChainPreviewResult, responses=ERRORS, summary="Dry-run cache plan showing which stages will reuse cached outputs")
def preview_chain(body: ChainPreviewRequest, services: ServicesDep) -> dict[str, Any]:
    plan = services.chains.plan_chain(_to_specs(body.stages), model_fingerprint=body.model_fingerprint, chain_id=body.chain_id)
    return {"stages": plan}


@router.get("/templates", response_model=ChainTemplateList, summary="List saved pipeline templates")
def list_templates(services: ServicesDep) -> dict[str, Any]:
    items = services.chains.store.list_templates()
    return {"items": [template_json(tmpl, iso) for tmpl in items]}


@router.post("/templates", response_model=ChainTemplateModel, status_code=201, responses=ERRORS, summary="Save a reusable pipeline template")
def create_template(body: ChainTemplateCreate, services: ServicesDep) -> dict[str, Any]:
    tmpl = services.chains.store.create_template(
        body.name,
        _to_specs(body.stages),
        description=body.description,
        model_family=body.model_family,
        tags=body.tags,
    )
    return template_json(tmpl, iso)


@router.get("/templates/{template_id}", response_model=ChainTemplateModel, responses=ERRORS, summary="Get a pipeline template")
def get_template(template_id: str, services: ServicesDep) -> dict[str, Any]:
    tmpl = services.chains.store.get_template(template_id)
    if tmpl is None:
        raise HTTPException(status_code=404, detail=f"no chain template {template_id!r}")
    return template_json(tmpl, iso)


@router.delete("/templates/{template_id}", status_code=204, responses=ERRORS, summary="Delete a pipeline template")
def delete_template(template_id: str, services: ServicesDep) -> Response:
    if not services.chains.store.delete_template(template_id):
        raise HTTPException(status_code=404, detail=f"no chain template {template_id!r}")
    return Response(status_code=204)


@router.get("/{chain_id}", response_model=JobChainModel, responses=ERRORS, summary="Get a chain and all stage runs")
def get_chain(chain_id: str, services: ServicesDep) -> dict[str, Any]:
    chain = services.chains.store.get_chain(chain_id)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"no chain {chain_id!r}")
    return chain_json(chain, iso)


@router.post("/{chain_id}/rerun", response_model=JobChainModel, responses=ERRORS, summary="Re-run a chain, automatically reusing unchanged cached upstream stages")
def rerun_chain(chain_id: str, services: ServicesDep, body: ChainRerunRequest | None = None) -> dict[str, Any]:
    specs = _to_specs(body.stages) if (body is not None and body.stages is not None) else None
    chain = services.chains.rerun_chain(
        chain_id,
        stages=specs,
        from_stage=body.from_stage if body else None,
        force_stages=body.force_stages if body else (),
        model_fingerprint=body.model_fingerprint if body else None,
    )
    return chain_json(chain, iso)


@router.post("/{chain_id}/pause", response_model=JobChainModel, responses=ERRORS, summary="Pause a chain between stages")
def pause_chain(chain_id: str, services: ServicesDep) -> dict[str, Any]:
    return chain_json(services.chains.pause_chain(chain_id), iso)


@router.post("/{chain_id}/resume", response_model=JobChainModel, responses=ERRORS, summary="Resume a paused chain")
def resume_chain(chain_id: str, services: ServicesDep) -> dict[str, Any]:
    return chain_json(services.chains.resume_chain(chain_id), iso)


@router.post("/{chain_id}/cancel", response_model=JobChainModel, responses=ERRORS, summary="Cancel a chain and any active stage job")
def cancel_chain(chain_id: str, services: ServicesDep) -> dict[str, Any]:
    return chain_json(services.chains.cancel_chain(chain_id), iso)


@router.delete("/{chain_id}", status_code=204, responses=ERRORS, summary="Delete a finished or paused chain")
def delete_chain(chain_id: str, services: ServicesDep) -> Response:
    if not services.chains.delete_chain(chain_id):
        raise HTTPException(status_code=404, detail=f"no chain {chain_id!r}")
    return Response(status_code=204)
