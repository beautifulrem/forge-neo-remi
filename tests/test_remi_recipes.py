"""Unit and route tests for cross-model parameter Recipes (``remi/recipes.py``)."""

from __future__ import annotations

import pytest

from remi import recipes as recipes_module
from remi.recipes import RECIPE_EXPORT_VERSION, RecipeError, RecipeStore, apply_recipe
from remi.server.testing import server_app
from remi.testing import webui_state


def test_recipe_store_crud_and_filtering(tmp_path):
    with RecipeStore(tmp_path / "recipes.db") as store:
        r1 = store.create(
            "Anima Clean Lineart",
            description="Balanced 28-step Euler recipe for Anima",
            model_family="anima",
            capabilities=["anima_hooks", "spectrum"],
            tab="txt2img",
            core_patch={"cfg_scale": 4.5, "width": 832, "height": 1216},
            scripts_patch={"sampler": {"steps": 28, "sampler_name": "Euler"}},
            tags=["anime", "lineart"],
        )
        r2 = store.create(
            "Universal Hires Crisp",
            description="Works across all families",
            model_family=None,
            capabilities=[],
            tab="txt2img",
            core_patch={"enable_hr": True, "hr_scale": 1.5},
            tags=["hires"],
        )
        r3 = store.create(
            "Qwen Edit RGBA",
            model_family="qwen21",
            capabilities=["multi_reference", "rgba_output"],
            tab="img2img",
            core_patch={"denoising_strength": 0.65},
            tags=["edit", "rgba"],
        )

        # Duplicate name rejected with 409
        with pytest.raises(RecipeError) as dup_err:
            store.create("Anima Clean Lineart")
        assert dup_err.value.status == 409

        # Invalid tab / NaN rejected
        with pytest.raises(RecipeError):
            store.create("Bad Tab", tab="unknown")
        with pytest.raises(RecipeError):
            store.create("Bad Float", core_patch={"cfg_scale": float("nan")})

        # Filtering
        assert len(store.list()) == 3
        assert {r.id for r in store.list(model_family="anima")} == {r1.id, r2.id}
        assert [r.id for r in store.list(tab="img2img")] == [r3.id]
        assert [r.id for r in store.list(tag="lineart")] == [r1.id]
        assert [r.id for r in store.list(capability="rgba_output")] == [r3.id]
        assert [r.id for r in store.list(query="balanced euler")] == [r1.id]

        # Update
        updated = store.update(r1.id, {"description": "Updated desc", "tags": ["anime", "clean"]})
        assert updated.description == "Updated desc"
        assert updated.tags == ["anime", "clean"]
        assert store.get(r1.id).description == "Updated desc"

        # Delete
        assert store.delete(r2.id) is True
        assert store.delete(r2.id) is False
        assert store.get(r2.id) is None


def test_recipe_export_import_and_apply(tmp_path):
    with RecipeStore(tmp_path / "src.db") as src:
        src.create(
            "FLUX3 Klein Fast",
            model_family="flux3",
            capabilities=["multi_reference"],
            core_patch={"cfg_scale": 1.0, "width": 1024, "height": 1024},
            scripts_patch={"sampler": {"steps": 8}},
            tags=["turbo"],
        )
        exported = src.export_all()
        assert exported["version"] == RECIPE_EXPORT_VERSION
        assert len(exported["recipes"]) == 1

    with RecipeStore(tmp_path / "dst.db") as dst:
        first = dst.import_recipes(exported)
        assert first["added"] == 1 and first["updated"] == 0 and first["skipped"] == 0
        # Re-importing without overwrite skips
        second = dst.import_recipes(exported, overwrite=False)
        assert second["added"] == 0 and second["skipped"] == 1
        # Re-importing with overwrite updates
        exported["recipes"][0]["description"] = "Overwritten!"
        third = dst.import_recipes(exported, overwrite=True)
        assert third["updated"] == 1 and third["items"][0].description == "Overwritten!"

        recipe = third["items"][0]
        applied = apply_recipe(
            recipe,
            core={"prompt": "a skyline", "cfg_scale": 7.0},
            scripts={"sampler": {"steps": 30, "sampler_name": "Euler"}},
            check_conflicts=False,
            active_capabilities=["anima_hooks"],
        )
        assert applied["core"]["prompt"] == "a skyline"
        assert applied["core"]["cfg_scale"] == 1.0
        assert applied["scripts"]["sampler"]["steps"] == 8
        assert applied["scripts"]["sampler"]["sampler_name"] == "Euler"
        assert applied["patch"]["core.cfg_scale"] == {"value": 1.0}
        assert applied["patch"]["sampler.steps"] == {"value": 8}
        # the English text names the capability; the note keeps the template and the names for the UI to translate
        assert applied["warnings"] == ["Active model lacks required capabilities: Multi-reference conditioning"]
        assert applied["warning_notes"] == [{"message": "Active model lacks required capabilities: {{capabilities}}", "values": {"capabilities": ["Multi-reference conditioning"]}}]


def test_the_capability_warning_is_in_the_core_catalog():
    """The apply toast shows the warning: a sentence formatted on the server could never be translated."""

    import json
    import re
    from pathlib import Path

    catalog = json.loads((Path(__file__).resolve().parents[1] / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    template = "Active model lacks required capabilities: {{capabilities}}"
    assert re.findall(r"\{\{(\w+)\}\}", catalog.get(template, "")) == ["capabilities"]


def test_an_import_that_fails_part_way_adds_nothing(tmp_path):
    """A rejected entry fails the whole import: the client is told it failed,
    so none of the entries before it may stay behind."""

    with RecipeStore(tmp_path / "recipes.db") as store:
        store.create("kept", tab="txt2img", core_patch={"steps": 4})
        payload = {
            "recipes": [
                {"name": "a", "tab": "txt2img"},
                {"name": "kept", "tab": "img2img", "description": "changed"},
                {"name": "b", "tab": "nope"},
            ]
        }
        with pytest.raises(RecipeError):
            store.import_recipes(payload, overwrite=True)
        assert [recipe.name for recipe in store.list()] == ["kept"]
        assert store.get_by_name("kept").tab == "txt2img" and store.get_by_name("kept").description == ""
        # the store still works after the rollback
        assert store.import_recipes({"recipes": [{"name": "a", "tab": "txt2img"}]})["added"] == 1


def test_recipes_http_routes(tmp_path, monkeypatch):
    with webui_state(), server_app(tmp_path) as server:
        created = server.client.post(
            "/api/v2/recipes",
            json={
                "name": "Studio Portrait",
                "description": "Soft light portrait",
                "model_family": "krea2",
                "capabilities": ["spectrum"],
                "tab": "txt2img",
                "core_patch": {"cfg_scale": 3.5, "width": 896, "height": 1152},
                "scripts_patch": {"sampler": {"steps": 24}},
                "tags": ["portrait"],
            },
        )
        assert created.status_code == 201
        recipe = created.json()
        rid = recipe["id"]
        assert recipe["name"] == "Studio Portrait" and recipe["created_at"]

        listed = server.client.get("/api/v2/recipes", params={"tag": "portrait"}).json()
        assert listed["total"] == 1 and listed["items"][0]["id"] == rid

        fetched = server.client.get(f"/api/v2/recipes/{rid}")
        assert fetched.status_code == 200 and fetched.json()["model_family"] == "krea2"

        patched = server.client.patch(f"/api/v2/recipes/{rid}", json={"description": "Crisp studio"})
        assert patched.status_code == 200 and patched.json()["description"] == "Crisp studio"

        applied = server.client.post(
            f"/api/v2/recipes/{rid}/apply",
            json={"tab": "txt2img", "core": {"prompt": "portrait"}, "scripts": {}, "check_conflicts": True},
        )
        assert applied.status_code == 200
        body = applied.json()
        assert body["core"]["cfg_scale"] == 3.5
        assert body["scripts"]["sampler"]["steps"] == 24
        assert body["patch"]["core.cfg_scale"]["value"] == 3.5

        # a capability the active model lacks reaches the client as a note too
        monkeypatch.setattr(recipes_module, "_active_model_capabilities", lambda: set())
        warned = server.client.post(f"/api/v2/recipes/{rid}/apply", json={"tab": "txt2img"}).json()
        assert warned["warnings"] == ["Active model lacks required capabilities: Spectrum forecasting"]
        assert warned["warning_notes"] == [{"message": "Active model lacks required capabilities: {{capabilities}}", "values": {"capabilities": ["Spectrum forecasting"]}}]

        exported = server.client.get("/api/v2/recipes/export")
        assert exported.status_code == 200 and len(exported.json()["recipes"]) == 1

        deleted = server.client.delete(f"/api/v2/recipes/{rid}")
        assert deleted.status_code == 204
        assert server.client.get(f"/api/v2/recipes/{rid}").status_code == 404

        reimported = server.client.post("/api/v2/recipes/import", json=exported.json())
        assert reimported.status_code == 200 and reimported.json()["added"] == 1

        # Route 409/404 error paths
        assert server.client.post("/api/v2/recipes", json={"name": "Studio Portrait"}).status_code == 409
        assert server.client.delete("/api/v2/recipes/missing-id").status_code == 404
        assert server.client.post("/api/v2/recipes/missing-id/apply", json={}).status_code == 404

    # Store validation & update/import edge cases
    with RecipeStore(tmp_path / "edge.db") as store:
        with pytest.raises(RecipeError):
            store.create("   ")
        with pytest.raises(RecipeError):
            store.create("x" * 201)
        with pytest.raises(RecipeError):
            store.create("Bad Core", core_patch="not-a-dict")  # type: ignore[arg-type]
        with pytest.raises(RecipeError):
            store.create("Bad Scripts", scripts_patch={"s": "not-a-dict"})  # type: ignore[dict-item]
        with pytest.raises(RecipeError):
            store.create("Bad List Float", core_patch={"weights": [1.0, float("inf")]})

        a = store.create("Alpha", tags=["t1"])
        b = store.create("Beta", tags=["t2"])
        with pytest.raises(RecipeError) as e404:
            store.update("missing", {"description": "x"})
        assert e404.value.status == 404
        with pytest.raises(RecipeError) as e409:
            store.update(b.id, {"name": "Alpha"})
        assert e409.value.status == 409
        assert store.update(a.id, {"unknown_field": 1}).id == a.id

        with pytest.raises(RecipeError):
            store.import_recipes("bad")  # type: ignore[arg-type]
        with pytest.raises(RecipeError):
            store.import_recipes({"recipes": "bad"})
        with pytest.raises(RecipeError):
            store.import_recipes({"recipes": ["not-a-dict"]})
        imp = store.import_recipes({"recipes": [{"name": "Gamma", "core_patch": {"cfg_scale": 5}}]})
        assert imp["added"] == 1 and imp["skipped"] == 0


def test_concurrent_partial_updates_keep_each_others_fields(tmp_path, monkeypatch):
    """Two PATCHes of different fields: the later write must not put back the
    value the earlier one replaced."""

    import threading

    from remi import recipes as recipes_module

    with RecipeStore(tmp_path / "recipes.db") as store:
        recipe = store.create("r", core_patch={"steps": 4}, tags=["old"])
        inside, release = threading.Event(), threading.Event()
        clean = recipes_module._clean_core_patch

        def slow_clean(patch):
            inside.set()
            assert release.wait(5)
            return clean(patch)

        monkeypatch.setattr(recipes_module, "_clean_core_patch", slow_clean)
        first = threading.Thread(target=store.update, args=(recipe.id, {"core_patch": {"steps": 8}}))
        first.start()
        assert inside.wait(5)
        second = threading.Thread(target=store.update, args=(recipe.id, {"tags": ["new"]}))
        second.start()
        second.join(0.3)
        release.set()
        first.join(5)
        second.join(5)
        final = store.get(recipe.id)
        assert final.core_patch == {"steps": 8} and final.tags == ["new"]


@pytest.mark.parametrize(
    "entry",
    [{"capabilities": "abc"}, {"capabilities": 5}, {"tags": "abc"}, {"model_family": "f" * 101}, {"model_family": 5}],
)
def test_an_import_with_a_malformed_field_is_refused(tmp_path, entry):
    """The import takes raw JSON: a string is not a list of one-letter
    capabilities, a number is not a server error, and the family has the
    length limit the create route has."""

    with webui_state(), server_app(tmp_path) as server:
        answer = server.client.post("/api/v2/recipes/import", json={"recipes": [{"name": "ok"}, {"name": "bad", **entry}]})
        assert answer.status_code == 422
        assert server.client.get("/api/v2/recipes").json()["total"] == 0


def test_an_update_checks_the_family_too(tmp_path):
    with RecipeStore(tmp_path / "recipes.db") as store:
        recipe = store.create("r", model_family="anima")
        with pytest.raises(RecipeError):
            store.update(recipe.id, {"model_family": "f" * 101})
        assert store.update(recipe.id, {"model_family": "  "}).model_family is None


def test_a_patch_nested_deeper_than_the_limit_is_refused_not_a_server_error(tmp_path):
    """The NaN check walked a patch recursively: about a thousand levels (the JSON parser
    goes deeper) raised RecursionError, a 500."""

    deep: object = 1
    for _ in range(900):
        deep = {"a": deep}
    with webui_state(), server_app(tmp_path) as server:
        for body in ({"name": "Deep core", "core_patch": {"x": deep}}, {"name": "Deep scripts", "scripts_patch": {"s": {"x": deep}}}):
            answer = server.client.post("/api/v2/recipes", json=body)
            assert answer.status_code == 422, answer.text
        assert server.client.post("/api/v2/recipes", json={"name": "Shallow", "core_patch": {"x": [[{"a": 1}]]}}).status_code == 201
