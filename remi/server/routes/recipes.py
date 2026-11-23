"""Cross-model reusable parameter Recipes: ``/api/v2/recipes``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import Field

from remi.recipes import RECIPE_EXPORT_VERSION, RecipeError, apply_recipe, recipe_json, recipes_for
from remi.server import models as m
from remi.server.deps import ServicesDep, iso

router = APIRouter(prefix="/recipes")

ERRORS = {404: {"model": m.Detail}, 409: {"model": m.Detail}, 422: {"model": m.Detail}}


class RecipeItem(m.Model):
    id: str
    name: str
    description: str = ""
    model_family: str | None = None
    capabilities: list[str] = []
    tab: m.FormTab = "txt2img"
    core_patch: dict[str, Any] = {}
    scripts_patch: dict[str, dict[str, Any]] = {}
    tags: list[str] = []
    created_at: str
    updated_at: str


class RecipeList(m.Model):
    items: list[RecipeItem]
    total: int


class RecipeCreate(m.Model):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=2000)
    model_family: str | None = Field(None, max_length=100)
    capabilities: list[str] = []
    tab: m.FormTab = "txt2img"
    core_patch: dict[str, Any] = {}
    scripts_patch: dict[str, dict[str, Any]] = {}
    tags: list[str] = []


class RecipeUpdate(m.Model):
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    model_family: str | None = Field(None, max_length=100)
    capabilities: list[str] | None = None
    tab: m.FormTab | None = None
    core_patch: dict[str, Any] | None = None
    scripts_patch: dict[str, dict[str, Any]] | None = None
    tags: list[str] | None = None


class RecipeExport(m.Model):
    version: int = RECIPE_EXPORT_VERSION
    recipes: list[dict[str, Any]]


class RecipeImportRequest(m.Model):
    version: int = RECIPE_EXPORT_VERSION
    recipes: list[dict[str, Any]]
    overwrite: bool = False


class RecipeImportResult(m.Model):
    added: int
    updated: int
    skipped: int
    items: list[RecipeItem]


class RecipeApplyRequest(m.Model):
    tab: m.FormTab | None = None
    core: dict[str, Any] = {}
    scripts: dict[str, dict[str, Any]] = {}
    check_conflicts: bool = True


class RecipeApplyResult(m.Model):
    recipe: RecipeItem
    tab: m.FormTab
    patch: dict[str, Any]
    core: dict[str, Any]
    scripts: dict[str, dict[str, Any]]
    conflicts: dict[str, Any]
    warnings: list[str]


def _call(fn: Any) -> Any:
    try:
        return fn()
    except RecipeError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error


@router.get("", response_model=RecipeList, summary="List parameter recipes")
def list_recipes(
    services: ServicesDep,
    q: str | None = Query(None, max_length=200),
    model_family: str | None = Query(None, max_length=100),
    tab: m.FormTab | None = None,
    tag: str | None = Query(None, max_length=100),
    capability: str | None = Query(None, max_length=100),
) -> dict[str, Any]:
    store = recipes_for(services.files)
    items = store.list(query=q, model_family=model_family, tab=tab, tag=tag, capability=capability)
    return {"items": [recipe_json(r, iso) for r in items], "total": len(items)}


@router.post("", response_model=RecipeItem, status_code=201, responses=ERRORS, summary="Create a parameter recipe")
def create_recipe(body: RecipeCreate, services: ServicesDep) -> dict[str, Any]:
    store = recipes_for(services.files)
    recipe = _call(
        lambda: store.create(
            body.name,
            description=body.description,
            model_family=body.model_family,
            capabilities=body.capabilities,
            tab=body.tab,
            core_patch=body.core_patch,
            scripts_patch=body.scripts_patch,
            tags=body.tags,
        )
    )
    return recipe_json(recipe, iso)


@router.get("/export", response_model=RecipeExport, summary="Export all recipes as JSON")
def export_recipes(services: ServicesDep, response: Response) -> dict[str, Any]:
    response.headers["Content-Disposition"] = 'attachment; filename="remi-recipes.json"'
    return recipes_for(services.files).export_all()


@router.post("/import", response_model=RecipeImportResult, responses=ERRORS, summary="Import recipes from JSON")
def import_recipes(body: RecipeImportRequest, services: ServicesDep) -> dict[str, Any]:
    store = recipes_for(services.files)
    res = _call(lambda: store.import_recipes({"version": body.version, "recipes": body.recipes}, overwrite=body.overwrite))
    return {
        "added": res["added"],
        "updated": res["updated"],
        "skipped": res["skipped"],
        "items": [recipe_json(r, iso) for r in res["items"]],
    }


@router.get("/{recipe_id}", response_model=RecipeItem, responses=ERRORS, summary="Get one recipe")
def get_recipe(recipe_id: str, services: ServicesDep) -> dict[str, Any]:
    recipe = recipes_for(services.files).get(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"no recipe {recipe_id!r}")
    return recipe_json(recipe, iso)


@router.patch("/{recipe_id}", response_model=RecipeItem, responses=ERRORS, summary="Update a recipe")
def update_recipe(recipe_id: str, body: RecipeUpdate, services: ServicesDep) -> dict[str, Any]:
    store = recipes_for(services.files)
    changes = {field: getattr(body, field) for field in body.model_fields_set}
    recipe = _call(lambda: store.update(recipe_id, changes))
    return recipe_json(recipe, iso)


@router.delete("/{recipe_id}", status_code=204, responses=ERRORS, summary="Delete a recipe")
def delete_recipe(recipe_id: str, services: ServicesDep) -> Response:
    if not recipes_for(services.files).delete(recipe_id):
        raise HTTPException(status_code=404, detail=f"no recipe {recipe_id!r}")
    return Response(status_code=204)


@router.post("/{recipe_id}/apply", response_model=RecipeApplyResult, responses=ERRORS, summary="Apply a recipe onto form values with conflict & capability checks")
def apply_recipe_route(recipe_id: str, body: RecipeApplyRequest, services: ServicesDep) -> dict[str, Any]:
    recipe = recipes_for(services.files).get(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"no recipe {recipe_id!r}")
    applied = _call(
        lambda: apply_recipe(
            recipe,
            tab=body.tab,
            core=body.core,
            scripts=body.scripts,
            check_conflicts=body.check_conflicts,
        )
    )
    return {"recipe": recipe_json(recipe, iso), **applied}
