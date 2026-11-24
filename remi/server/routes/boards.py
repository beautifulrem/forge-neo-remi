"""Gallery Boards, Asset Lineage, and Structured Parameter Diff routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import Field

from remi.image_meta import MAX_INFOTEXT_CHARS
from remi.library import BoardError, board_json, compare_outputs, gallery_for
from remi.server import models as m
from remi.server.deps import ServicesDep, iso

router = APIRouter()

ERRORS = {404: {"model": m.Detail}, 409: {"model": m.Detail}, 422: {"model": m.Detail}}


# region Models


class BoardItem(m.Model):
    id: str
    name: str
    description: str = ""
    color: str | None = None
    item_count: int = 0
    cover_file_id: str | None = None
    created_at: str
    updated_at: str


class BoardList(m.Model):
    items: list[BoardItem]


class BoardCreate(m.Model):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    color: str | None = Field(None, max_length=32)


class BoardUpdate(m.Model):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=1000)
    color: str | None = Field(None, max_length=32)


class BoardItemsRequest(m.Model):
    file_ids: list[str] = Field(min_length=1, max_length=2000)


class BoardItemsResult(m.Model):
    board_id: str
    count: int


class LineageNode(m.Model):
    file_id: str
    relation: str
    depth: int = 0
    job_id: str | None = None
    chain_id: str | None = None
    stage_index: int | None = None
    name: str | None = None
    width: int | None = None
    height: int | None = None
    tab: str | None = None
    model: str | None = None
    created_at: str | None = None
    exists: bool = False


class AssetLineage(m.Model):
    file_id: str
    item: LineageNode | None = None
    ancestors: list[LineageNode] = []
    siblings: list[LineageNode] = []
    descendants: list[LineageNode] = []


class CompareRequest(m.Model):
    id_a: str | None = None
    id_b: str | None = None
    infotext_a: str | None = Field(None, max_length=MAX_INFOTEXT_CHARS)
    infotext_b: str | None = Field(None, max_length=MAX_INFOTEXT_CHARS)


class PromptTokenDiff(m.Model):
    a: str
    b: str
    added: list[str] = []
    removed: list[str] = []
    unchanged: list[str] = []


class CoreParamDiff(m.Model):
    key: str
    field: str | None = None
    a: Any = None
    b: Any = None
    kind: Literal["changed", "added", "removed"]


class LoraDiff(m.Model):
    name: str
    a: Any = None
    b: Any = None
    kind: Literal["changed", "added", "removed"]


class ExtensionParamDiff(m.Model):
    group: str
    key: str
    a: Any = None
    b: Any = None
    kind: Literal["changed", "added", "removed"]


class AdoptPatch(m.Model):
    core: dict[str, Any] = {}
    scripts: dict[str, dict[str, Any]] = {}
    raw_params: dict[str, Any] = {}


class CompareResult(m.Model):
    id_a: str | None = None
    id_b: str | None = None
    infotext_a: str = ""
    infotext_b: str = ""
    identical: bool
    prompt_diff: PromptTokenDiff
    negative_diff: PromptTokenDiff
    core_diffs: list[CoreParamDiff] = []
    lora_diffs: list[LoraDiff] = []
    extension_diffs: list[ExtensionParamDiff] = []
    adopt_b_patch: AdoptPatch
    adopt_a_patch: AdoptPatch


# endregion


def _call(fn: Any) -> Any:
    try:
        return fn()
    except BoardError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error


# region Boards routes


@router.get("/boards", response_model=BoardList, summary="List gallery project boards")
def list_boards(services: ServicesDep) -> dict[str, Any]:
    boards = gallery_for(services.files).list_boards()
    return {"items": [board_json(b, iso) for b in boards]}


@router.post("/boards", response_model=BoardItem, status_code=201, responses=ERRORS, summary="Create a gallery board")
def create_board(body: BoardCreate, services: ServicesDep) -> dict[str, Any]:
    index = gallery_for(services.files)
    board = _call(lambda: index.create_board(body.name, description=body.description, color=body.color))
    return board_json(board, iso)


@router.get("/boards/{board_id}", response_model=BoardItem, responses=ERRORS, summary="Get a board")
def get_board(board_id: str, services: ServicesDep) -> dict[str, Any]:
    board = gallery_for(services.files).get_board(board_id)
    if board is None:
        raise HTTPException(status_code=404, detail=f"no board {board_id!r}")
    return board_json(board, iso)


@router.patch("/boards/{board_id}", response_model=BoardItem, responses=ERRORS, summary="Rename or update a board")
def update_board(board_id: str, body: BoardUpdate, services: ServicesDep) -> dict[str, Any]:
    index = gallery_for(services.files)
    kwargs: dict[str, Any] = {}
    if "name" in body.model_fields_set:
        kwargs["name"] = body.name
    if "description" in body.model_fields_set:
        kwargs["description"] = body.description
    if "color" in body.model_fields_set:
        kwargs["color"] = body.color
    board = _call(lambda: index.update_board(board_id, **kwargs))
    return board_json(board, iso)


@router.delete("/boards/{board_id}", status_code=204, responses=ERRORS, summary="Delete a board (images stay in gallery)")
def delete_board(board_id: str, services: ServicesDep) -> Response:
    if not gallery_for(services.files).delete_board(board_id):
        raise HTTPException(status_code=404, detail=f"no board {board_id!r}")
    return Response(status_code=204)


@router.post("/boards/{board_id}/items", response_model=BoardItemsResult, responses=ERRORS, summary="Add gallery outputs to a board")
def add_board_items(board_id: str, body: BoardItemsRequest, services: ServicesDep) -> dict[str, Any]:
    index = gallery_for(services.files)
    count = _call(lambda: index.add_to_board(board_id, body.file_ids))
    return {"board_id": board_id, "count": count}


@router.post("/boards/{board_id}/items/remove", response_model=BoardItemsResult, responses=ERRORS, summary="Remove gallery outputs from a board")
def remove_board_items(board_id: str, body: BoardItemsRequest, services: ServicesDep) -> dict[str, Any]:
    index = gallery_for(services.files)
    count = _call(lambda: index.remove_from_board(board_id, body.file_ids))
    return {"board_id": board_id, "count": count}


@router.delete("/boards/{board_id}/items/{file_id}", status_code=204, responses=ERRORS, summary="Remove one output from a board")
def remove_single_board_item(board_id: str, file_id: str, services: ServicesDep) -> Response:
    index = gallery_for(services.files)
    _call(lambda: index.remove_from_board(board_id, [file_id]))
    return Response(status_code=204)


# endregion

# region Lineage & Compare routes


@router.get("/gallery/{file_id}/lineage", response_model=AssetLineage, responses=ERRORS, summary="Ancestors, siblings, and descendants of a gallery output")
def gallery_lineage(file_id: str, services: ServicesDep) -> dict[str, Any]:
    index = gallery_for(services.files)
    result = index.lineage_for(file_id, job_store=services.store, iso=iso)
    if result["item"] is None and not result["ancestors"] and not result["descendants"]:
        raise HTTPException(status_code=404, detail=f"no output {file_id!r}")
    return result


@router.post("/gallery/compare", response_model=CompareResult, responses=ERRORS, summary="Structured parameter diff between two gallery outputs or infotexts")
def gallery_compare(body: CompareRequest, services: ServicesDep) -> dict[str, Any]:
    if body.id_a is None and body.infotext_a is None and body.id_b is None and body.infotext_b is None:
        raise HTTPException(status_code=422, detail="provide id_a/id_b or infotext_a/infotext_b")
    return _call(
        lambda: compare_outputs(
            services.files,
            id_a=body.id_a,
            id_b=body.id_b,
            infotext_a=body.infotext_a,
            infotext_b=body.infotext_b,
        )
    )


# endregion
