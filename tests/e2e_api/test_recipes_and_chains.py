"""Opaque-box E2E tests for F5 (Recipes) and F6 (Linear Pipelines / Chains & Stage Cache).

Every test exercises only public HTTP endpoints via ``httpx.Client`` against
live isolated servers (``live_server``, ``gated_server``, ``failing_server``,
``make_live_server``) with zero internal ``remi.*``/``modules.*`` imports.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest


def _deep_nested_dict(depth: int) -> dict[str, Any]:
    node: dict[str, Any] = {"leaf": 1}
    for _ in range(depth):
        node = {"nested": node}
    return node


def _stage_generate(
    *,
    prompt: str = "1girl, test",
    seed: int = 1234,
    steps: int = 2,
    width: int = 64,
    height: int = 64,
    enabled: bool = True,
    name: str = "Generate",
) -> dict[str, Any]:
    return {
        "kind": "generate",
        "name": name,
        "enabled": enabled,
        "tab": "txt2img",
        "core": {"prompt": prompt, "width": width, "height": height},
        "scripts": {
            "sampler": {"steps": steps},
            "seed": {"seed": seed},
        },
    }


def _stage_refine(
    *,
    prompt: str = "1girl, refined",
    seed: int = 5678,
    steps: int = 2,
    denoising_strength: float = 0.45,
    width: int = 64,
    height: int = 64,
    enabled: bool = True,
    name: str = "Refine",
) -> dict[str, Any]:
    return {
        "kind": "refine",
        "name": name,
        "enabled": enabled,
        "tab": "img2img",
        "core": {
            "prompt": prompt,
            "denoising_strength": denoising_strength,
            "width": width,
            "height": height,
        },
        "scripts": {
            "sampler": {"steps": steps},
            "seed": {"seed": seed},
        },
    }


# =============================================================================
# Recipes CRUD, Export/Import & Apply (Happy Path)
# =============================================================================


def test_recipe_crud_lifecycle(live_server: Any) -> None:
    create_resp = live_server.http.post(
        "/api/v2/recipes",
        json={
            "name": "Clean Anima Portrait",
            "description": "Balanced portrait starter",
            "model_family": "anima",
            "capabilities": ["anima_hooks"],
            "tab": "txt2img",
            "core_patch": {"prompt": "1girl, portrait", "width": 768, "height": 1024},
            "scripts_patch": {"sampler": {"steps": 24}},
            "tags": ["portrait", "clean"],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    created = create_resp.json()
    recipe_id = created["id"]
    assert created["name"] == "Clean Anima Portrait"
    assert created["model_family"] == "anima"
    assert created["capabilities"] == ["anima_hooks"]
    assert created["core_patch"]["width"] == 768
    assert created["scripts_patch"] == {"sampler": {"steps": 24}}
    assert created["tags"] == ["portrait", "clean"]
    assert created["created_at"]
    assert created["updated_at"]

    list_resp = live_server.http.get("/api/v2/recipes")
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert listed["total"] == 1
    assert [item["id"] for item in listed["items"]] == [recipe_id]

    get_resp = live_server.http.get(f"/api/v2/recipes/{recipe_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == recipe_id

    patch_resp = live_server.http.patch(
        f"/api/v2/recipes/{recipe_id}",
        json={
            "name": "Clean Anima Portrait v2",
            "description": "Updated description",
            "tags": ["portrait", "v2"],
        },
    )
    assert patch_resp.status_code == 200, patch_resp.text
    patched = patch_resp.json()
    assert patched["name"] == "Clean Anima Portrait v2"
    assert patched["description"] == "Updated description"
    assert patched["tags"] == ["portrait", "v2"]
    assert patched["core_patch"] == {"prompt": "1girl, portrait", "width": 768, "height": 1024}

    del_resp = live_server.http.delete(f"/api/v2/recipes/{recipe_id}")
    assert del_resp.status_code == 204

    after_del = live_server.http.get(f"/api/v2/recipes/{recipe_id}")
    assert after_del.status_code == 404


def test_filter_recipes_includes_universal_family(live_server: Any) -> None:
    anima_match = live_server.http.post(
        "/api/v2/recipes",
        json={
            "name": "Anima Clean Portrait",
            "description": "Crisp studio lighting",
            "model_family": "anima",
            "capabilities": ["anima_hooks"],
            "tab": "txt2img",
            "tags": ["portrait", "studio"],
        },
    )
    universal_match = live_server.http.post(
        "/api/v2/recipes",
        json={
            "name": "Universal Clean Portrait",
            "description": "Cross-model clean look",
            "model_family": None,
            "capabilities": ["anima_hooks"],
            "tab": "txt2img",
            "tags": ["portrait"],
        },
    )
    flux_other = live_server.http.post(
        "/api/v2/recipes",
        json={
            "name": "Flux Clean Portrait",
            "description": "Flux specific",
            "model_family": "flux",
            "capabilities": ["anima_hooks"],
            "tab": "txt2img",
            "tags": ["portrait"],
        },
    )
    anima_other_tag = live_server.http.post(
        "/api/v2/recipes",
        json={
            "name": "Anima Clean Landscape",
            "model_family": "anima",
            "capabilities": ["anima_hooks"],
            "tab": "txt2img",
            "tags": ["landscape"],
        },
    )
    for resp in (anima_match, universal_match, flux_other, anima_other_tag):
        assert resp.status_code == 201, resp.text

    filtered = live_server.http.get(
        "/api/v2/recipes",
        params={
            "model_family": "anima",
            "tag": "portrait",
            "capability": "anima_hooks",
            "q": "clean",
        },
    )
    assert filtered.status_code == 200
    body = filtered.json()
    names = {item["name"] for item in body["items"]}
    assert body["total"] == 2
    assert names == {"Anima Clean Portrait", "Universal Clean Portrait"}


def test_export_delete_all_and_import_restores(live_server: Any) -> None:
    payloads = [
        {"name": "Recipe Alpha", "model_family": "anima", "core_patch": {"width": 512}, "tags": ["a"]},
        {"name": "Recipe Beta", "model_family": None, "tab": "img2img", "core_patch": {"denoising_strength": 0.5}},
        {"name": "Recipe Gamma", "scripts_patch": {"sampler": {"steps": 18}}, "capabilities": ["anima_hooks"]},
    ]
    created_ids: list[str] = []
    for item in payloads:
        resp = live_server.http.post("/api/v2/recipes", json=item)
        assert resp.status_code == 201, resp.text
        created_ids.append(resp.json()["id"])

    export_resp = live_server.http.get("/api/v2/recipes/export")
    assert export_resp.status_code == 200
    assert 'filename="remi-recipes.json"' in export_resp.headers.get("content-disposition", "")
    exported = export_resp.json()
    assert exported["version"] == 1
    assert len(exported["recipes"]) == 3

    for rid in created_ids:
        assert live_server.http.delete(f"/api/v2/recipes/{rid}").status_code == 204
    assert live_server.http.get("/api/v2/recipes").json()["total"] == 0

    import_resp = live_server.http.post(
        "/api/v2/recipes/import",
        json={"version": 1, "recipes": exported["recipes"], "overwrite": False},
    )
    assert import_resp.status_code == 200, import_resp.text
    imported = import_resp.json()
    assert imported["added"] == 3
    assert imported["updated"] == 0
    assert imported["skipped"] == 0
    assert len(imported["items"]) == 3

    listed = live_server.http.get("/api/v2/recipes").json()
    assert listed["total"] == 3
    by_name = {item["name"]: item for item in listed["items"]}
    assert by_name["Recipe Alpha"]["core_patch"] == {"width": 512}
    assert by_name["Recipe Beta"]["tab"] == "img2img"
    assert by_name["Recipe Gamma"]["scripts_patch"] == {"sampler": {"steps": 18}}


def test_import_overwrite_false_skips_and_true_updates(live_server: Any) -> None:
    create_resp = live_server.http.post(
        "/api/v2/recipes",
        json={"name": "Shared Style", "description": "Original", "core_patch": {"width": 512}},
    )
    assert create_resp.status_code == 201
    existing_id = create_resp.json()["id"]

    skip_resp = live_server.http.post(
        "/api/v2/recipes/import",
        json={
            "version": 1,
            "overwrite": False,
            "recipes": [
                {"name": "Shared Style", "description": "Modified", "core_patch": {"width": 1024}},
            ],
        },
    )
    assert skip_resp.status_code == 200, skip_resp.text
    skip_data = skip_resp.json()
    assert skip_data["added"] == 0
    assert skip_data["updated"] == 0
    assert skip_data["skipped"] == 1
    assert skip_data["items"] == []
    assert live_server.http.get(f"/api/v2/recipes/{existing_id}").json()["core_patch"] == {"width": 512}

    update_resp = live_server.http.post(
        "/api/v2/recipes/import",
        json={
            "version": 1,
            "overwrite": True,
            "recipes": [
                {"name": "Shared Style", "description": "Modified", "core_patch": {"width": 1024}},
            ],
        },
    )
    assert update_resp.status_code == 200, update_resp.text
    update_data = update_resp.json()
    assert update_data["added"] == 0
    assert update_data["updated"] == 1
    assert update_data["skipped"] == 0
    assert len(update_data["items"]) == 1
    assert update_data["items"][0]["id"] == existing_id
    assert update_data["items"][0]["core_patch"] == {"width": 1024}
    assert update_data["items"][0]["description"] == "Modified"


def test_apply_merges_core_and_scripts_over_form_values(live_server: Any) -> None:
    create_resp = live_server.http.post(
        "/api/v2/recipes",
        json={
            "name": "Apply Merge Recipe",
            "tab": "txt2img",
            "core_patch": {"prompt": "recipe prompt", "cfg_scale": 5.5},
            "scripts_patch": {"sampler": {"steps": 28}},
        },
    )
    assert create_resp.status_code == 201
    recipe_id = create_resp.json()["id"]

    apply_resp = live_server.http.post(
        f"/api/v2/recipes/{recipe_id}/apply",
        json={
            "tab": "txt2img",
            "core": {"prompt": "draft prompt", "width": 768, "height": 512},
            "scripts": {"sampler": {"sampler_name": "Euler", "steps": 12}, "seed": {"seed": 99}},
            "check_conflicts": False,
        },
    )
    assert apply_resp.status_code == 200, apply_resp.text
    applied = apply_resp.json()
    assert applied["tab"] == "txt2img"
    assert applied["recipe"]["id"] == recipe_id
    assert applied["patch"]["core.prompt"] == {"value": "recipe prompt"}
    assert applied["patch"]["core.cfg_scale"] == {"value": 5.5}
    assert applied["patch"]["sampler.steps"] == {"value": 28}
    assert applied["core"] == {
        "prompt": "recipe prompt",
        "width": 768,
        "height": 512,
        "cfg_scale": 5.5,
    }
    assert applied["scripts"]["sampler"] == {"sampler_name": "Euler", "steps": 28}
    assert applied["scripts"]["seed"] == {"seed": 99}


# =============================================================================
# Recipes CRUD, Export/Import & Apply (Boundary / Negative)
# =============================================================================


def test_create_recipe_rejects_blank_duplicate_nonfinite_and_deep_nesting(live_server: Any) -> None:
    blank_resp = live_server.http.post("/api/v2/recipes", json={"name": "   "})
    assert blank_resp.status_code == 422

    first_resp = live_server.http.post("/api/v2/recipes", json={"name": "Unique Recipe"})
    assert first_resp.status_code == 201
    dup_resp = live_server.http.post("/api/v2/recipes", json={"name": "  Unique Recipe  "})
    assert dup_resp.status_code == 409

    for bad_literal in ("NaN", "Infinity", "-Infinity", "1e999"):
        raw = f'{{"name": "NonFinite {bad_literal}", "core_patch": {{"cfg_scale": {bad_literal}}}}}'.encode()
        resp = live_server.http.post(
            "/api/v2/recipes",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, f"expected 422 for {bad_literal}, got {resp.status_code}: {resp.text}"

    deep_resp = live_server.http.post(
        "/api/v2/recipes",
        json={"name": "Deeply Nested", "core_patch": _deep_nested_dict(65)},
    )
    assert deep_resp.status_code == 422


@pytest.mark.parametrize(
    "scripts_patch",
    [
        {"": {"steps": 20}},
        {"   ": {"steps": 20}},
        {"sampler": "not-an-object"},
        {"sampler": ["not", "an", "object"]},
        {"sampler": 42},
    ],
)
def test_create_recipe_rejects_invalid_scripts_patch(live_server: Any, scripts_patch: dict[str, Any]) -> None:
    resp = live_server.http.post(
        "/api/v2/recipes",
        json={"name": "Bad Scripts Patch", "scripts_patch": scripts_patch},
    )
    assert resp.status_code == 422, resp.text


def test_import_rejects_bad_version_over_limit_and_rolls_back_atomically(live_server: Any) -> None:
    bad_version = live_server.http.post(
        "/api/v2/recipes/import",
        json={"version": 99, "recipes": [{"name": "Should Not Import"}]},
    )
    assert bad_version.status_code == 422

    too_many = live_server.http.post(
        "/api/v2/recipes/import",
        json={"version": 1, "recipes": [{"name": f"Bulk {i}"} for i in range(501)]},
    )
    assert too_many.status_code == 422

    partial_bad = live_server.http.post(
        "/api/v2/recipes/import",
        json={
            "version": 1,
            "recipes": [
                {"name": "Valid First Recipe", "core_patch": {"width": 640}},
                {"name": "   ", "core_patch": {"width": 512}},
            ],
        },
    )
    assert partial_bad.status_code == 422

    listed = live_server.http.get("/api/v2/recipes").json()
    assert listed["total"] == 0
    assert listed["items"] == []


def test_nonexistent_recipe_returns_404(live_server: Any) -> None:
    missing_id = "nonexistent-recipe-0000"
    assert live_server.http.get(f"/api/v2/recipes/{missing_id}").status_code == 404
    assert live_server.http.patch(f"/api/v2/recipes/{missing_id}", json={"name": "Nope"}).status_code == 404
    assert live_server.http.delete(f"/api/v2/recipes/{missing_id}").status_code == 404
    assert live_server.http.post(f"/api/v2/recipes/{missing_id}/apply", json={}).status_code == 404


def test_apply_warns_when_active_model_lacks_capabilities(make_live_server: Any) -> None:
    server = make_live_server(steps=2, step_time=0.01, options={"forge_preset": "sd"})
    create_resp = server.http.post(
        "/api/v2/recipes",
        json={
            "name": "Anima Only Recipe",
            "model_family": "anima",
            "capabilities": ["anima_hooks", "control_blocks"],
            "tab": "txt2img",
            "core_patch": {"prompt": "1girl"},
        },
    )
    assert create_resp.status_code == 201
    recipe_id = create_resp.json()["id"]

    apply_resp = server.http.post(f"/api/v2/recipes/{recipe_id}/apply", json={"tab": "txt2img"})
    assert apply_resp.status_code == 200, apply_resp.text
    data = apply_resp.json()
    assert len(data["warnings"]) >= 1
    assert any("Anima engine hooks" in msg and "DiT Control Blocks" in msg for msg in data["warnings"])
    assert len(data["warning_notes"]) >= 1
    note = data["warning_notes"][0]
    assert "capabilities" in note["values"]
    assert "Anima engine hooks" in note["values"]["capabilities"]
    assert "DiT Control Blocks" in note["values"]["capabilities"]


# =============================================================================
# Linear Pipelines / Chains & Stage Cache (Happy Path)
# =============================================================================


def test_two_stage_chain_flows_primary_output_and_skips_disabled(live_server: Any) -> None:
    create_resp = live_server.http.post(
        "/api/v2/chains",
        json={
            "name": "Generate Then Refine",
            "stages": [
                _stage_generate(prompt="1girl, stage0", seed=101),
                _stage_refine(prompt="1girl, stage1", seed=202),
            ],
            "run_now": True,
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    chain_id = create_resp.json()["id"]

    completed = live_server.wait_chain(chain_id)
    assert completed["status"] == "completed", completed
    assert completed["total_stages"] == 2
    stage0, stage1 = completed["stages"]
    assert stage0["status"] == "completed"
    assert stage1["status"] == "completed"
    assert len(stage0["output_file_ids"]) >= 1
    assert len(stage1["output_file_ids"]) >= 1
    assert stage1["job_id"] is not None

    stage1_job_resp = live_server.http.get(f"/api/v2/jobs/{stage1['job_id']}")
    assert stage1_job_resp.status_code == 200
    stage1_job = stage1_job_resp.json()
    assert stage1_job["core"]["init_img"] == stage0["output_file_ids"][0]

    disabled_resp = live_server.http.post(
        "/api/v2/chains",
        json={
            "name": "Chain With Disabled Stage",
            "stages": [
                _stage_generate(prompt="1girl, base", seed=303),
                _stage_refine(prompt="1girl, skipped", seed=404, enabled=False, name="Skipped Refine"),
                _stage_refine(prompt="1girl, final", seed=505, enabled=True, name="Active Refine"),
            ],
            "run_now": True,
        },
    )
    assert disabled_resp.status_code == 201, disabled_resp.text
    disabled_done = live_server.wait_chain(disabled_resp.json()["id"])
    assert disabled_done["status"] == "completed"
    st0, st1, st2 = disabled_done["stages"]
    assert st0["status"] == "completed"
    assert st1["status"] == "skipped"
    assert st1["job_id"] is None
    assert st2["status"] == "completed"
    st2_job = live_server.http.get(f"/api/v2/jobs/{st2['job_id']}").json()
    assert st2_job["core"]["init_img"] == st0["output_file_ids"][0]


def test_preview_cache_hits_and_downstream_invalidation(live_server: Any) -> None:
    stages = [
        _stage_generate(prompt="1girl, preview base", seed=777),
        _stage_refine(prompt="1girl, preview refine", seed=888),
    ]

    before_resp = live_server.http.post("/api/v2/chains/preview", json={"stages": stages})
    assert before_resp.status_code == 200, before_resp.text
    before_stages = before_resp.json()["stages"]
    assert len(before_stages) == 2
    assert before_stages[0]["cache_hit"] is False
    assert before_stages[1]["cache_hit"] is False

    create_resp = live_server.http.post(
        "/api/v2/chains",
        json={"name": "Preview Cache Chain", "stages": stages, "run_now": True},
    )
    assert create_resp.status_code == 201
    done = live_server.wait_chain(create_resp.json()["id"])
    assert done["status"] == "completed"

    after_resp = live_server.http.post(
        "/api/v2/chains/preview",
        json={"stages": stages, "chain_id": done["id"]},
    )
    assert after_resp.status_code == 200, after_resp.text
    after_stages = after_resp.json()["stages"]
    assert after_stages[0]["cache_hit"] is True
    assert after_stages[1]["cache_hit"] is True
    assert after_stages[0]["cached_outputs"] == done["stages"][0]["output_file_ids"]
    assert after_stages[1]["cached_outputs"] == done["stages"][1]["output_file_ids"]

    modified_stages = [
        stages[0],
        _stage_refine(prompt="1girl, preview refine modified", seed=888),
    ]
    partial_resp = live_server.http.post(
        "/api/v2/chains/preview",
        json={"stages": modified_stages, "chain_id": done["id"]},
    )
    assert partial_resp.status_code == 200, partial_resp.text
    partial_stages = partial_resp.json()["stages"]
    assert partial_stages[0]["cache_hit"] is True
    assert partial_stages[1]["cache_hit"] is False


def test_rerun_caches_unchanged_stage_zero_and_runs_stage_one(live_server: Any) -> None:
    stage0 = _stage_generate(prompt="1girl, rerun base", seed=4242)
    stage1_v1 = _stage_refine(prompt="1girl, rerun refine v1", seed=4343)

    create_resp = live_server.http.post(
        "/api/v2/chains",
        json={"name": "Rerun Cache Chain", "stages": [stage0, stage1_v1], "run_now": True},
    )
    assert create_resp.status_code == 201
    first = live_server.wait_chain(create_resp.json()["id"])
    assert first["status"] == "completed"
    first_st0_outputs = first["stages"][0]["output_file_ids"]
    first_st0_job = first["stages"][0]["job_id"]
    first_st1_job = first["stages"][1]["job_id"]

    stage1_v2 = _stage_refine(prompt="1girl, rerun refine v2", seed=4343)
    rerun_resp = live_server.http.post(
        f"/api/v2/chains/{first['id']}/rerun",
        json={"stages": [stage0, stage1_v2]},
    )
    assert rerun_resp.status_code == 200, rerun_resp.text
    second = live_server.wait_chain(first["id"])
    assert second["status"] == "completed"

    st0, st1 = second["stages"]
    assert st0["status"] == "cached"
    assert st0["cached"] is True
    assert st0["output_file_ids"] == first_st0_outputs
    assert st0["job_id"] == first_st0_job

    assert st1["status"] == "completed"
    assert st1["cached"] is False
    assert st1["job_id"] is not None
    assert st1["job_id"] != first_st1_job
    st1_job_detail = live_server.http.get(f"/api/v2/jobs/{st1['job_id']}").json()
    assert st1_job_detail["core"]["init_img"] == first_st0_outputs[0]


def test_create_paused_chain_and_resume_to_completion(live_server: Any) -> None:
    create_resp = live_server.http.post(
        "/api/v2/chains",
        json={
            "name": "Deferred Chain",
            "stages": [_stage_generate(prompt="1girl, paused", seed=909)],
            "run_now": False,
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    paused = create_resp.json()
    chain_id = paused["id"]
    assert paused["status"] == "paused"
    assert paused["stages"][0]["status"] == "pending"
    assert paused["stages"][0]["job_id"] is None

    resume_resp = live_server.http.post(f"/api/v2/chains/{chain_id}/resume")
    assert resume_resp.status_code == 200, resume_resp.text
    completed = live_server.wait_chain(chain_id)
    assert completed["status"] == "completed"
    assert completed["stages"][0]["status"] == "completed"
    assert len(completed["stages"][0]["output_file_ids"]) >= 1


def test_chain_templates_crud_lifecycle(live_server: Any) -> None:
    create_resp = live_server.http.post(
        "/api/v2/chains/templates",
        json={
            "name": "Portrait Two-Stage Template",
            "description": "Generate then refine portrait",
            "model_family": "anima",
            "stages": [_stage_generate(seed=111), _stage_refine(seed=222)],
            "tags": ["portrait", "two-stage"],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    tmpl = create_resp.json()
    tmpl_id = tmpl["id"]
    assert tmpl["name"] == "Portrait Two-Stage Template"
    assert tmpl["model_family"] == "anima"
    assert len(tmpl["stages"]) == 2
    assert tmpl["tags"] == ["portrait", "two-stage"]

    list_resp = live_server.http.get("/api/v2/chains/templates")
    assert list_resp.status_code == 200
    assert any(item["id"] == tmpl_id for item in list_resp.json()["items"])

    get_resp = live_server.http.get(f"/api/v2/chains/templates/{tmpl_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == tmpl_id

    del_resp = live_server.http.delete(f"/api/v2/chains/templates/{tmpl_id}")
    assert del_resp.status_code == 204

    assert live_server.http.get(f"/api/v2/chains/templates/{tmpl_id}").status_code == 404


# =============================================================================
# Linear Pipelines / Chains & Stage Cache (Boundary / Negative)
# =============================================================================


def test_create_chain_rejects_bad_stage_counts_kind_and_nan(live_server: Any) -> None:
    zero_stages = live_server.http.post("/api/v2/chains", json={"name": "Empty", "stages": []})
    assert zero_stages.status_code == 422

    nine_stages = live_server.http.post(
        "/api/v2/chains",
        json={"name": "Too Many", "stages": [_stage_generate(seed=i) for i in range(9)]},
    )
    assert nine_stages.status_code == 422

    bad_kind = live_server.http.post(
        "/api/v2/chains",
        json={"name": "Bad Kind", "stages": [{"kind": "upscale_magic", "core": {}}]},
    )
    assert bad_kind.status_code == 422

    assert math.isnan(float("nan"))
    nan_payload = json.dumps({"name": "NaN Core", "stages": [{"kind": "generate", "core": {"cfg_scale": 1.0}}]})
    nan_bytes = nan_payload.replace("1.0", "NaN").encode("utf-8")
    nan_resp = live_server.http.post(
        "/api/v2/chains",
        content=nan_bytes,
        headers={"Content-Type": "application/json"},
    )
    assert nan_resp.status_code == 422


def test_idempotency_key_reused_with_different_params_returns_409(live_server: Any) -> None:
    key = "chain-idem-key-001"
    payload_a = {
        "name": "Idem Chain",
        "stages": [_stage_generate(prompt="1girl, idem A", seed=100)],
        "run_now": False,
    }
    first = live_server.http.post("/api/v2/chains", json=payload_a, headers={"Idempotency-Key": key})
    assert first.status_code == 201, first.text
    chain_id = first.json()["id"]

    replay = live_server.http.post("/api/v2/chains", json=payload_a, headers={"Idempotency-Key": key})
    assert replay.status_code == 201
    assert replay.json()["id"] == chain_id

    payload_b = {
        "name": "Idem Chain",
        "stages": [_stage_generate(prompt="1girl, idem B", seed=100)],
        "run_now": False,
    }
    conflict = live_server.http.post("/api/v2/chains", json=payload_b, headers={"Idempotency-Key": key})
    assert conflict.status_code == 409


def test_invalid_state_transitions_return_409(live_server: Any, gated_server: Any) -> None:
    done_resp = live_server.http.post(
        "/api/v2/chains",
        json={"name": "Done Chain", "stages": [_stage_generate(seed=11)], "run_now": True},
    )
    assert done_resp.status_code == 201
    done_chain = live_server.wait_chain(done_resp.json()["id"])
    assert done_chain["status"] == "completed"

    assert live_server.http.post(f"/api/v2/chains/{done_chain['id']}/pause").status_code == 409
    assert live_server.http.post(f"/api/v2/chains/{done_chain['id']}/cancel").status_code == 409
    assert live_server.http.post(f"/api/v2/chains/{done_chain['id']}/resume").status_code == 409

    assert gated_server.gate is not None
    running_resp = gated_server.http.post(
        "/api/v2/chains",
        json={"name": "Gated Running Chain", "stages": [_stage_generate(seed=22)], "run_now": True},
    )
    assert running_resp.status_code == 201
    running_id = running_resp.json()["id"]
    try:
        assert gated_server.executor.started.wait(timeout=5.0)
        current = gated_server.http.get(f"/api/v2/chains/{running_id}").json()
        assert current["status"] == "running"

        assert gated_server.http.post(f"/api/v2/chains/{running_id}/resume").status_code == 409
        assert gated_server.http.post(f"/api/v2/chains/{running_id}/rerun", json={}).status_code == 409
        assert gated_server.http.delete(f"/api/v2/chains/{running_id}").status_code == 409
    finally:
        gated_server.gate.set()
        gated_server.wait_chain(running_id)


def test_stage_zero_failure_fails_chain_and_stops_downstream(failing_server: Any) -> None:
    create_resp = failing_server.http.post(
        "/api/v2/chains",
        json={
            "name": "Failing Chain",
            "stages": [_stage_generate(seed=33), _stage_refine(seed=44)],
            "run_now": True,
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    chain_id = create_resp.json()["id"]

    failed = failing_server.wait_chain(chain_id)
    assert failed["status"] == "failed", failed
    assert failed["error"] and "synthetic boom" in failed["error"]
    st0, st1 = failed["stages"]
    assert st0["status"] == "failed"
    assert st0["error"] and "synthetic boom" in st0["error"]
    assert st1["status"] == "pending"
    assert st1["job_id"] is None
    assert st1["output_file_ids"] == []


def test_templates_duplicate_409_and_missing_ids_404(live_server: Any) -> None:
    first = live_server.http.post(
        "/api/v2/chains/templates",
        json={"name": "Duplicate Template", "stages": [_stage_generate(seed=55)]},
    )
    assert first.status_code == 201

    dup = live_server.http.post(
        "/api/v2/chains/templates",
        json={"name": "  Duplicate Template  ", "stages": [_stage_generate(seed=56)]},
    )
    assert dup.status_code == 409

    missing_chain = "missing-chain-0000"
    assert live_server.http.get(f"/api/v2/chains/{missing_chain}").status_code == 404
    assert live_server.http.delete(f"/api/v2/chains/{missing_chain}").status_code == 404
    assert live_server.http.post(f"/api/v2/chains/{missing_chain}/rerun", json={}).status_code == 404
    assert live_server.http.post(f"/api/v2/chains/{missing_chain}/pause").status_code == 404
    assert live_server.http.post(f"/api/v2/chains/{missing_chain}/resume").status_code == 404
    assert live_server.http.post(f"/api/v2/chains/{missing_chain}/cancel").status_code == 404

    missing_tmpl = "missing-template-0000"
    assert live_server.http.get(f"/api/v2/chains/templates/{missing_tmpl}").status_code == 404
    assert live_server.http.delete(f"/api/v2/chains/templates/{missing_tmpl}").status_code == 404
