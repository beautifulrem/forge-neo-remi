"""Cross-model reusable parameter Recipes (SQLite-backed CRUD, JSON import/export, apply).

A Recipe captures a named partial patch of ``core`` and ``scripts`` parameters
with optional ``model_family``, ``capabilities`` and ``tags`` metadata so users
can save, share, and apply parameter combinations across models with automatic
conflict and capability checks.
"""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remi.files import FileStore, connect

RECIPE_EXPORT_VERSION = 1
VALID_TABS = ("txt2img", "img2img", "extras")
MAX_IMPORT_RECIPES = 500


class RecipeError(ValueError):
    """A recipe validation or state error with an HTTP status code."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Recipe:
    id: str
    name: str
    description: str = ""
    model_family: str | None = None
    capabilities: list[str] = field(default_factory=list)
    tab: str = "txt2img"
    core_patch: dict[str, Any] = field(default_factory=dict)
    scripts_patch: dict[str, dict[str, Any]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created: float = 0.0
    updated: float = 0.0

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model_family": self.model_family,
            "capabilities": list(self.capabilities),
            "tab": self.tab,
            "core_patch": dict(self.core_patch),
            "scripts_patch": {k: dict(v) for k, v in self.scripts_patch.items()},
            "tags": list(self.tags),
        }


def recipe_json(recipe: Recipe, iso: Callable[[float | None], str | None]) -> dict[str, Any]:
    return {
        "id": recipe.id,
        "name": recipe.name,
        "description": recipe.description,
        "model_family": recipe.model_family,
        "capabilities": list(recipe.capabilities),
        "tab": recipe.tab,
        "core_patch": dict(recipe.core_patch),
        "scripts_patch": {k: dict(v) for k, v in recipe.scripts_patch.items()},
        "tags": list(recipe.tags),
        "created_at": iso(recipe.created) or "",
        "updated_at": iso(recipe.updated) or "",
    }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    model_family TEXT,
    capabilities TEXT NOT NULL DEFAULT '[]',
    tab TEXT NOT NULL DEFAULT 'txt2img',
    core_patch TEXT NOT NULL DEFAULT '{}',
    scripts_patch TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS recipes_family ON recipes(model_family);
CREATE INDEX IF NOT EXISTS recipes_tab ON recipes(tab);
CREATE INDEX IF NOT EXISTS recipes_updated ON recipes(updated DESC);
"""

_COLUMNS = "id, name, description, model_family, capabilities, tab, core_patch, scripts_patch, tags, created, updated"


def _check_finite(value: Any, path: str) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise RecipeError(f"{path} must not contain NaN or Infinity")
    if isinstance(value, Mapping):
        return {str(k): _check_finite(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_check_finite(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return value


def _clean_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise RecipeError("recipe name is required")
    cleaned = name.strip()
    if len(cleaned) > 200:
        raise RecipeError("recipe name must be at most 200 characters")
    return cleaned


def _clean_tab(tab: Any) -> str:
    if not isinstance(tab, str) or tab not in VALID_TABS:
        raise RecipeError(f"tab must be one of {', '.join(VALID_TABS)}")
    return tab


def _clean_str_list(values: Any, field_name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise RecipeError(f"{field_name} must be a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise RecipeError(f"{field_name} entries must be strings")
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _clean_core_patch(patch: Any) -> dict[str, Any]:
    if patch is None:
        return {}
    if not isinstance(patch, Mapping):
        raise RecipeError("core_patch must be an object")
    return _check_finite(dict(patch), "core_patch")


def _clean_scripts_patch(patch: Any) -> dict[str, dict[str, Any]]:
    if patch is None:
        return {}
    if not isinstance(patch, Mapping):
        raise RecipeError("scripts_patch must be an object")
    cleaned: dict[str, dict[str, Any]] = {}
    for script_id, values in patch.items():
        if not isinstance(script_id, str) or not script_id.strip():
            raise RecipeError("scripts_patch keys must be non-empty script ids")
        if not isinstance(values, Mapping):
            raise RecipeError(f"scripts_patch[{script_id!r}] must be an object")
        cleaned[script_id.strip()] = _check_finite(dict(values), f"scripts_patch.{script_id}")
    return cleaned


def _row_to_recipe(row: tuple[Any, ...]) -> Recipe:
    return Recipe(
        id=row[0],
        name=row[1],
        description=row[2],
        model_family=row[3],
        capabilities=json.loads(row[4]),
        tab=row[5],
        core_patch=json.loads(row[6]),
        scripts_patch=json.loads(row[7]),
        tags=json.loads(row[8]),
        created=float(row[9]),
        updated=float(row[10]),
    )


class RecipeStore:
    """SQLite-backed CRUD store for cross-model parameter recipes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._db = connect(self.path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> RecipeStore:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def get(self, recipe_id: str) -> Recipe | None:
        with self._lock:
            row = self._db.execute(f"SELECT {_COLUMNS} FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        return _row_to_recipe(row) if row is not None else None

    def get_by_name(self, name: str) -> Recipe | None:
        with self._lock:
            row = self._db.execute(f"SELECT {_COLUMNS} FROM recipes WHERE name = ?", (name.strip(),)).fetchone()
        return _row_to_recipe(row) if row is not None else None

    def list(
        self,
        *,
        query: str | None = None,
        model_family: str | None = None,
        tab: str | None = None,
        tag: str | None = None,
        capability: str | None = None,
    ) -> list[Recipe]:
        where = ["1=1"]
        args: list[Any] = []
        if model_family:
            where.append("(model_family = ? OR model_family IS NULL)")
            args.append(model_family.strip())
        if tab:
            where.append("tab = ?")
            args.append(tab)
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM recipes WHERE {' AND '.join(where)} ORDER BY updated DESC, name ASC",
                args,
            ).fetchall()
        recipes = [_row_to_recipe(row) for row in rows]
        if tag:
            wanted_tag = tag.strip().lower()
            recipes = [r for r in recipes if any(t.lower() == wanted_tag for t in r.tags)]
        if capability:
            wanted_cap = capability.strip().lower()
            recipes = [r for r in recipes if any(c.lower() == wanted_cap for c in r.capabilities)]
        if query:
            terms = [term.lower() for term in query.split() if term.strip()]
            if terms:
                filtered: list[Recipe] = []
                for recipe in recipes:
                    haystack = " ".join([recipe.name, recipe.description, recipe.model_family or "", *recipe.tags, *recipe.capabilities]).lower()
                    if all(term in haystack for term in terms):
                        filtered.append(recipe)
                recipes = filtered
        return recipes

    def create(
        self,
        name: str,
        *,
        description: str = "",
        model_family: str | None = None,
        capabilities: Iterable[str] = (),
        tab: str = "txt2img",
        core_patch: Mapping[str, Any] | None = None,
        scripts_patch: Mapping[str, Mapping[str, Any]] | None = None,
        tags: Iterable[str] = (),
    ) -> Recipe:
        cleaned_name = _clean_name(name)
        cleaned_desc = str(description or "").strip()[:2000]
        cleaned_family = model_family.strip() if isinstance(model_family, str) and model_family.strip() else None
        cleaned_caps = _clean_str_list(list(capabilities), "capabilities")
        cleaned_tab = _clean_tab(tab)
        cleaned_core = _clean_core_patch(core_patch)
        cleaned_scripts = _clean_scripts_patch(scripts_patch)
        cleaned_tags = _clean_str_list(list(tags), "tags")
        recipe_id = secrets.token_hex(8)
        now = time.time()
        with self._lock:
            try:
                self._db.execute(
                    f"INSERT INTO recipes ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        recipe_id,
                        cleaned_name,
                        cleaned_desc,
                        cleaned_family,
                        json.dumps(cleaned_caps, ensure_ascii=False),
                        cleaned_tab,
                        json.dumps(cleaned_core, ensure_ascii=False),
                        json.dumps(cleaned_scripts, ensure_ascii=False),
                        json.dumps(cleaned_tags, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise RecipeError(f"a recipe named {cleaned_name!r} already exists", status=409) from error
        created = self.get(recipe_id)
        assert created is not None
        return created

    def update(self, recipe_id: str, changes: Mapping[str, Any]) -> Recipe:
        existing = self.get(recipe_id)
        if existing is None:
            raise RecipeError(f"no recipe {recipe_id!r}", status=404)
        name = _clean_name(changes["name"]) if "name" in changes and changes["name"] is not None else existing.name
        description = str(changes["description"]).strip()[:2000] if "description" in changes and changes["description"] is not None else existing.description
        if "model_family" in changes:
            raw_family = changes["model_family"]
            model_family = raw_family.strip() if isinstance(raw_family, str) and raw_family.strip() else None
        else:
            model_family = existing.model_family
        capabilities = _clean_str_list(changes["capabilities"], "capabilities") if "capabilities" in changes and changes["capabilities"] is not None else existing.capabilities
        tab = _clean_tab(changes["tab"]) if "tab" in changes and changes["tab"] is not None else existing.tab
        core_patch = _clean_core_patch(changes["core_patch"]) if "core_patch" in changes and changes["core_patch"] is not None else existing.core_patch
        scripts_patch = _clean_scripts_patch(changes["scripts_patch"]) if "scripts_patch" in changes and changes["scripts_patch"] is not None else existing.scripts_patch
        tags = _clean_str_list(changes["tags"], "tags") if "tags" in changes and changes["tags"] is not None else existing.tags
        now = time.time()
        with self._lock:
            try:
                self._db.execute(
                    "UPDATE recipes SET name = ?, description = ?, model_family = ?, capabilities = ?, tab = ?, core_patch = ?, scripts_patch = ?, tags = ?, updated = ? WHERE id = ?",
                    (
                        name,
                        description,
                        model_family,
                        json.dumps(capabilities, ensure_ascii=False),
                        tab,
                        json.dumps(core_patch, ensure_ascii=False),
                        json.dumps(scripts_patch, ensure_ascii=False),
                        json.dumps(tags, ensure_ascii=False),
                        now,
                        recipe_id,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise RecipeError(f"a recipe named {name!r} already exists", status=409) from error
        updated = self.get(recipe_id)
        assert updated is not None
        return updated

    def delete(self, recipe_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,)).rowcount > 0

    def export_all(self) -> dict[str, Any]:
        return {
            "version": RECIPE_EXPORT_VERSION,
            "recipes": [recipe.to_export_dict() for recipe in self.list()],
        }

    def import_recipes(self, payload: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise RecipeError("import payload must be a JSON object")
        items = payload.get("recipes")
        if not isinstance(items, list):
            raise RecipeError("import payload must include a 'recipes' list")
        if len(items) > MAX_IMPORT_RECIPES:
            raise RecipeError(f"cannot import more than {MAX_IMPORT_RECIPES} recipes at once")
        # one transaction: an entry that fails leaves none of the earlier ones behind
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                result = self._import_entries(items, overwrite=overwrite)
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")
        return result

    def _import_entries(self, items: list[Any], *, overwrite: bool) -> dict[str, Any]:
        added = updated = skipped = 0
        imported: list[Recipe] = []
        for entry in items:
            if not isinstance(entry, Mapping):
                raise RecipeError("each imported recipe must be an object")
            name = _clean_name(entry.get("name"))
            existing = self.get_by_name(name)
            fields = {
                "description": entry.get("description", ""),
                "model_family": entry.get("model_family"),
                "capabilities": entry.get("capabilities") or [],
                "tab": entry.get("tab") or "txt2img",
                "core_patch": entry.get("core_patch") or {},
                "scripts_patch": entry.get("scripts_patch") or {},
                "tags": entry.get("tags") or [],
            }
            if existing is not None:
                if overwrite:
                    recipe = self.update(existing.id, {"name": name, **fields})
                    updated += 1
                    imported.append(recipe)
                else:
                    skipped += 1
            else:
                recipe = self.create(name, **fields)
                added += 1
                imported.append(recipe)
        return {"added": added, "updated": updated, "skipped": skipped, "items": imported}


def _active_model_capabilities() -> set[str] | None:
    try:
        from remi.core.capabilities import active_capability_summary

        summary = active_capability_summary()
        if isinstance(summary, Mapping):
            caps = summary.get("capabilities")
            if isinstance(caps, (list, tuple, set, frozenset)):
                return {str(c) for c in caps}
    except Exception:
        return None
    return None


def apply_recipe(
    recipe: Recipe,
    *,
    tab: str | None = None,
    core: Mapping[str, Any] | None = None,
    scripts: Mapping[str, Mapping[str, Any]] | None = None,
    check_conflicts: bool = True,
    active_capabilities: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build an action patch and merged form values for applying ``recipe``."""

    effective_tab = _clean_tab(tab or recipe.tab)
    merged_core = dict(core or {})
    merged_core.update(recipe.core_patch)
    merged_scripts: dict[str, dict[str, Any]] = {sid: dict(vals) for sid, vals in (scripts or {}).items()}
    for sid, spatch in recipe.scripts_patch.items():
        merged_scripts.setdefault(sid, {}).update(spatch)

    patch: dict[str, Any] = {}
    for key, value in recipe.core_patch.items():
        patch[f"core.{key}"] = {"value": value}
    for sid, spatch in recipe.scripts_patch.items():
        for key, value in spatch.items():
            patch[f"{sid}.{key}"] = {"value": value}

    warnings: list[str] = []
    caps = set(active_capabilities) if active_capabilities is not None else _active_model_capabilities()
    if caps is not None and recipe.capabilities:
        missing = [cap for cap in recipe.capabilities if cap not in caps]
        if missing:
            warnings.append(f"Active model lacks required capabilities: {', '.join(missing)}")

    conflicts: dict[str, Any] = {}
    if check_conflicts and effective_tab in ("txt2img", "img2img"):
        try:
            from remi.core.conflicts import conflict_patch

            derived = conflict_patch(effective_tab, {**merged_scripts, "core": merged_core})
            if isinstance(derived, Mapping):
                conflicts = dict(derived)
                # The conflict patch is the rules' whole state (every group member, every role,
                # core params bare). The form writes each patch entry and counts it as applied,
                # so only a control whose value the rules actually change joins the recipe's
                # own values; the recipe's values stay as the recipe has them (the form resolves
                # its conflicts again once they are written).
                for target, entry in conflicts.items():
                    if target.startswith("__") or not isinstance(entry, Mapping) or "value" not in entry:
                        continue
                    key = target if "." in target else f"core.{target}"
                    if key in patch:
                        continue
                    owner, param = key.split(".", 1)
                    values = merged_core if owner == "core" else merged_scripts.get(owner)
                    if values is None or param not in values or values[param] == entry["value"]:
                        continue
                    patch[key] = {"value": entry["value"]}
                    values[param] = entry["value"]
        except Exception:
            pass

    return {
        "tab": effective_tab,
        "patch": patch,
        "core": merged_core,
        "scripts": merged_scripts,
        "conflicts": conflicts,
        "warnings": warnings,
    }


_stores: weakref.WeakKeyDictionary[FileStore, RecipeStore] = weakref.WeakKeyDictionary()
_stores_lock = threading.Lock()


def recipes_for(files: FileStore) -> RecipeStore:
    """One recipe store per file store (closed by :func:`close_recipes`)."""

    with _stores_lock:
        store = _stores.get(files)
        if store is None:
            store = _stores[files] = RecipeStore(files.root / "recipes.db")
        return store


def close_recipes(files: FileStore) -> None:
    """Close the recipe store associated with ``files``, if opened."""

    with _stores_lock:
        store = _stores.pop(files, None)
    if store is not None:
        store.close()


__all__ = [
    "RECIPE_EXPORT_VERSION",
    "VALID_TABS",
    "Recipe",
    "RecipeError",
    "RecipeStore",
    "apply_recipe",
    "close_recipes",
    "recipe_json",
    "recipes_for",
]
