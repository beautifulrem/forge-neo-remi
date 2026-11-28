"""Opaque-box E2E tests for F7 (Gallery, Boards, Lineage & Structured Compare)
and F8 (Capability Validation, Conflicts, Advice, Schemas, Models & Styles).

Every test interacts strictly over live HTTP/WS via ``live_server``,
``make_live_server``, ``tiny_anima_server``, and ``helpers`` fixtures.
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta
from typing import Any

import pytest
from PIL import Image

# =============================================================================
# — Gallery, Boards, Lineage, Structured Compare & PNG Info (Happy)
# =============================================================================


def test_gallery_filters_sort_pagination_and_facets(live_server: Any, helpers: Any) -> None:
    """Generate outputs across tabs -> filter by q, category, since/until,
    starred, sort, cursor pagination, and verify /api/v2/gallery/facets."""

    init_upload = helpers.upload_png(live_server, filename="src_ref.png")
    job_a = helpers.submit_txt2img(live_server, "alpha mountain sunset", width=64, height=64)
    job_b = helpers.submit_txt2img(live_server, "beta neon forest", width=64, height=64)
    time.sleep(0.05)
    job_c = helpers.submit_img2img(live_server, init_upload["id"], "alpha studio portrait", width=64, height=64)

    id_a = job_a["outputs"][0]["id"]
    id_b = job_b["outputs"][0]["id"]
    id_c = job_c["outputs"][0]["id"]

    # Star one output and verify facets reflect totals and starred count
    star_resp = live_server.http.post("/api/v2/gallery/star", json={"ids": [id_b], "starred": True})
    assert star_resp.status_code == 200
    assert star_resp.json()["count"] == 1

    facets_resp = live_server.http.get("/api/v2/gallery/facets")
    assert facets_resp.status_code == 200
    facets = facets_resp.json()
    assert facets["total"] == 3
    assert facets["starred"] == 1
    assert facets["oldest"] is not None
    assert facets["newest"] is not None
    cat_counts = {entry["name"]: entry["count"] for entry in facets["categories"]}
    assert cat_counts.get("txt2img") == 2
    assert cat_counts.get("img2img") == 1

    # Single-item lookup
    one_resp = live_server.http.get(f"/api/v2/gallery/{id_a}")
    assert one_resp.status_code == 200
    assert one_resp.json()["id"] == id_a
    assert one_resp.json()["category"] == "txt2img"

    # Search q across infotext
    q_resp = live_server.http.get("/api/v2/gallery", params={"q": "alpha"})
    assert q_resp.status_code == 200
    q_ids = {item["id"] for item in q_resp.json()["items"]}
    assert q_ids == {id_a, id_c}

    # Category filter
    img2img_resp = live_server.http.get("/api/v2/gallery", params={"category": "img2img"})
    assert img2img_resp.status_code == 200
    assert [item["id"] for item in img2img_resp.json()["items"]] == [id_c]

    txt2img_resp = live_server.http.get("/api/v2/gallery", params={"category": "txt2img"})
    assert txt2img_resp.status_code == 200
    assert {item["id"] for item in txt2img_resp.json()["items"]} == {id_a, id_b}

    # Starred filter
    starred_resp = live_server.http.get("/api/v2/gallery", params={"starred": "true"})
    assert starred_resp.status_code == 200
    assert [item["id"] for item in starred_resp.json()["items"]] == [id_b]

    unstarred_resp = live_server.http.get("/api/v2/gallery", params={"starred": "false"})
    assert unstarred_resp.status_code == 200
    assert {item["id"] for item in unstarred_resp.json()["items"]} == {id_a, id_c}

    # Sort orders: newest vs oldest + name + largest
    newest = live_server.http.get("/api/v2/gallery", params={"sort": "newest"}).json()["items"]
    oldest = live_server.http.get("/api/v2/gallery", params={"sort": "oldest"}).json()["items"]
    assert [i["id"] for i in newest] == list(reversed([i["id"] for i in oldest]))
    assert oldest[0]["id"] == id_a
    assert oldest[-1]["id"] == id_c

    by_name = live_server.http.get("/api/v2/gallery", params={"sort": "name"}).json()["items"]
    names = [item["name"].lower() for item in by_name]
    assert names == sorted(names)

    # Time window: since (inclusive) & until (exclusive)
    earliest_dt = datetime.fromisoformat(oldest[0]["created_at"]) - timedelta(seconds=1)
    before_c = datetime.fromisoformat(oldest[-1]["created_at"]) - timedelta(milliseconds=10)
    window_resp = live_server.http.get("/api/v2/gallery", params={"since": earliest_dt.isoformat(), "until": before_c.isoformat()})
    assert window_resp.status_code == 200
    window_ids = {item["id"] for item in window_resp.json()["items"]}
    assert id_c not in window_ids
    assert {id_a, id_b} <= window_ids

    future_resp = live_server.http.get(
        "/api/v2/gallery",
        params={"since": (datetime.fromisoformat(oldest[-1]["created_at"]) + timedelta(seconds=1)).isoformat()},
    )
    assert future_resp.status_code == 200
    assert future_resp.json()["items"] == []

    # Cursor pagination
    page1 = live_server.http.get("/api/v2/gallery", params={"limit": 2, "sort": "oldest"}).json()
    assert len(page1["items"]) == 2
    assert page1["next_cursor"] is not None
    page2 = live_server.http.get("/api/v2/gallery", params={"limit": 2, "sort": "oldest", "cursor": page1["next_cursor"]}).json()
    assert len(page2["items"]) == 1
    all_paged_ids = [i["id"] for i in page1["items"]] + [i["id"] for i in page2["items"]]
    assert all_paged_ids == [id_a, id_b, id_c]

    # Sort="largest": replace stub 16x8 items with 3 PNGs of distinct resolutions & byte sizes
    del_resp = live_server.http.post("/api/v2/gallery/delete", json={"ids": [id_a, id_b, id_c], "delete_files": True})
    assert del_resp.status_code == 200

    txt2img_dir = live_server.outputs_dir / "txt2img-images"
    specs = [
        ("c_medium.png", 64, 64),
        ("b_large.png", 128, 128),
        ("a_small.png", 32, 32),
    ]
    for filename, width, height in specs:
        pixels = bytes((i * 37 + j * 13) % 256 for i in range(width * height) for j in range(3))
        with Image.frombytes("RGB", (width, height), pixels) as img:
            img.save(txt2img_dir / filename, format="PNG")
        time.sleep(0.02)

    refresh_resp = live_server.http.post("/api/v2/outputs/refresh")
    assert refresh_resp.status_code == 200
    assert refresh_resp.json()["added"] == 3

    by_largest = live_server.http.get("/api/v2/gallery", params={"sort": "largest"})
    assert by_largest.status_code == 200
    largest_items = by_largest.json()["items"]
    assert len(largest_items) == 3
    assert [item["name"] for item in largest_items] == ["b_large.png", "c_medium.png", "a_small.png"]

    sizes = [item["size"] for item in largest_items]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > sizes[-1]

    areas = [item["width"] * item["height"] for item in largest_items]
    assert areas == sorted(areas, reverse=True)
    assert areas[0] > areas[-1]

    largest_detail = live_server.http.get(f"/api/v2/gallery/{largest_items[0]['id']}")
    assert largest_detail.status_code == 200
    assert largest_detail.json()["size"] == sizes[0]


def test_boards_lifecycle_and_gallery_retention(live_server: Any, helpers: Any) -> None:
    """Boards CRUD: create -> add items -> verify item_count, cover_file_id &
    gallery?board_id -> update -> remove items -> delete board preserves outputs."""

    out1 = helpers.submit_txt2img(live_server, "board item one")["outputs"][0]["id"]
    out2 = helpers.submit_txt2img(live_server, "board item two")["outputs"][0]["id"]
    out3 = helpers.submit_txt2img(live_server, "unboarded item")["outputs"][0]["id"]

    # Create board
    create_resp = live_server.http.post(
        "/api/v2/boards",
        json={"name": "Concept Studies", "description": "Initial lighting tests", "color": "#4f46e5"},
    )
    assert create_resp.status_code == 201
    board = create_resp.json()
    board_id = board["id"]
    assert board["name"] == "Concept Studies"
    assert board["description"] == "Initial lighting tests"
    assert board["color"] == "#4f46e5"
    assert board["item_count"] == 0
    assert board["cover_file_id"] is None

    # List boards
    list_resp = live_server.http.get("/api/v2/boards")
    assert list_resp.status_code == 200
    assert any(b["id"] == board_id for b in list_resp.json()["items"])

    # Add items
    add_resp = live_server.http.post(f"/api/v2/boards/{board_id}/items", json={"file_ids": [out1, out2]})
    assert add_resp.status_code == 200
    assert add_resp.json() == {"board_id": board_id, "count": 2}

    # Verify item_count & cover_file_id & board_id gallery filter
    fetched = live_server.http.get(f"/api/v2/boards/{board_id}")
    assert fetched.status_code == 200
    assert fetched.json()["item_count"] == 2
    assert fetched.json()["cover_file_id"] in {out1, out2}

    filtered = live_server.http.get("/api/v2/gallery", params={"board_id": board_id})
    assert filtered.status_code == 200
    assert {i["id"] for i in filtered.json()["items"]} == {out1, out2}
    assert out3 not in {i["id"] for i in filtered.json()["items"]}

    # Update board metadata
    patch_resp = live_server.http.patch(
        f"/api/v2/boards/{board_id}",
        json={"name": "Lighting Board", "description": "Updated", "color": "#10b981"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "Lighting Board"
    assert patch_resp.json()["description"] == "Updated"
    assert patch_resp.json()["color"] == "#10b981"

    # Remove single item via DELETE
    del_item_resp = live_server.http.delete(f"/api/v2/boards/{board_id}/items/{out1}")
    assert del_item_resp.status_code == 204
    after_single = live_server.http.get(f"/api/v2/boards/{board_id}").json()
    assert after_single["item_count"] == 1
    assert after_single["cover_file_id"] == out2

    # Remove remaining item via POST /items/remove
    rem_resp = live_server.http.post(f"/api/v2/boards/{board_id}/items/remove", json={"file_ids": [out2]})
    assert rem_resp.status_code == 200
    assert rem_resp.json()["count"] == 1
    after_empty = live_server.http.get(f"/api/v2/boards/{board_id}").json()
    assert after_empty["item_count"] == 0
    assert after_empty["cover_file_id"] is None

    # Re-add out1, then delete board -> outputs stay in gallery
    assert live_server.http.post(f"/api/v2/boards/{board_id}/items", json={"file_ids": [out1]}).status_code == 200
    del_board = live_server.http.delete(f"/api/v2/boards/{board_id}")
    assert del_board.status_code == 204
    assert live_server.http.get(f"/api/v2/boards/{board_id}").status_code == 404
    assert live_server.http.get(f"/api/v2/gallery/{out1}").status_code == 200


def test_lineage_multi_depth_siblings_and_descendants(make_live_server: Any, helpers: Any) -> None:
    """txt2img (count=2) -> img2img (init_img) + sibling branch -> upscale_from:
    verify multi-depth ancestors, root descendants, same_job & same_parent siblings."""

    server = make_live_server(count=2, steps=1, step_time=0.0, realtime=False)

    root_job = helpers.submit_txt2img(server, "root lineage scene")
    root_a = root_job["outputs"][0]["id"]
    root_b = root_job["outputs"][1]["id"]

    mid_job = helpers.submit_img2img(server, root_a, "mid refine pass")
    mid_a = mid_job["outputs"][0]["id"]
    mid_b = mid_job["outputs"][1]["id"]

    # Second img2img sharing root_a as parent -> creates same_parent siblings with mid_a
    branch_job = helpers.submit_img2img(server, root_a, "branch refine pass")
    branch_a = branch_job["outputs"][0]["id"]

    # Upscale from mid_a -> depth 2 from root_a
    leaf_job = helpers.submit_txt2img(server, "leaf hires upscale", upscale_from=mid_a)
    leaf_a = leaf_job["outputs"][0]["id"]
    leaf_b = leaf_job["outputs"][1]["id"]

    # Inspect leaf lineage
    leaf_resp = server.http.get(f"/api/v2/gallery/{leaf_a}/lineage")
    assert leaf_resp.status_code == 200
    leaf_lin = leaf_resp.json()
    assert leaf_lin["file_id"] == leaf_a
    assert leaf_lin["item"]["file_id"] == leaf_a
    assert leaf_lin["item"]["exists"] is True

    anc_by_id = {node["file_id"]: node for node in leaf_lin["ancestors"]}
    assert mid_a in anc_by_id
    assert anc_by_id[mid_a]["relation"] == "upscale_from"
    assert anc_by_id[mid_a]["depth"] == 1
    assert root_a in anc_by_id
    assert anc_by_id[root_a]["relation"] == "init_img"
    assert anc_by_id[root_a]["depth"] == 2

    leaf_sibs = {node["file_id"]: node for node in leaf_lin["siblings"]}
    assert leaf_b in leaf_sibs
    assert leaf_sibs[leaf_b]["relation"] == "same_job"

    # Inspect middle node lineage -> same_job + same_parent siblings + descendants
    mid_resp = server.http.get(f"/api/v2/gallery/{mid_a}/lineage")
    assert mid_resp.status_code == 200
    mid_lin = mid_resp.json()
    mid_sibs = {node["file_id"]: node for node in mid_lin["siblings"]}
    assert mid_b in mid_sibs and mid_sibs[mid_b]["relation"] == "same_job"
    assert branch_a in mid_sibs and mid_sibs[branch_a]["relation"] == "same_parent"
    mid_desc = {node["file_id"]: node for node in mid_lin["descendants"]}
    assert leaf_a in mid_desc and mid_desc[leaf_a]["depth"] == 1

    # Inspect root node lineage -> multi-depth descendants + same_job sibling
    root_resp = server.http.get(f"/api/v2/gallery/{root_a}/lineage")
    assert root_resp.status_code == 200
    root_lin = root_resp.json()
    root_sibs = {node["file_id"]: node for node in root_lin["siblings"]}
    assert root_b in root_sibs and root_sibs[root_b]["relation"] == "same_job"
    root_desc = {node["file_id"]: node for node in root_lin["descendants"]}
    assert mid_a in root_desc and root_desc[mid_a]["depth"] == 1
    assert leaf_a in root_desc and root_desc[leaf_a]["depth"] == 2


def test_compare_infotexts_and_generated_outputs(live_server: Any, helpers: Any) -> None:
    """POST /api/v2/gallery/compare returns prompt_diff, negative_diff, core_diffs,
    lora_diffs, extension_diffs, and adopt_a_patch / adopt_b_patch for infotexts and outputs."""

    infotext_a = (
        "1girl, blue eyes, <lora:style_a:0.8>\n"
        "Negative prompt: blurry, lowres\n"
        'Steps: 20, Sampler: Euler, CFG scale: 7, Size: 512x512, ADetailer model: face_yolov8n.pt'
    )
    infotext_b = (
        "1girl, red eyes, <lora:style_a:0.5>, <lora:detail_b:1.0>\n"
        "Negative prompt: blurry, watermark\n"
        'Steps: 30, Sampler: DPM++ 2M, CFG scale: 5.5, Size: 768x512, ControlNet 0: "Module: canny"'
    )

    resp = live_server.http.post(
        "/api/v2/gallery/compare",
        json={"infotext_a": infotext_a, "infotext_b": infotext_b},
    )
    assert resp.status_code == 200
    diff = resp.json()
    assert diff["identical"] is False
    assert diff["prompt_diff"]["unchanged"] == ["1girl"]
    assert diff["prompt_diff"]["removed"] == ["blue eyes", "<lora:style_a:0.8>"]
    assert diff["prompt_diff"]["added"] == ["red eyes", "<lora:style_a:0.5>", "<lora:detail_b:1.0>"]
    assert diff["negative_diff"]["unchanged"] == ["blurry"]
    assert diff["negative_diff"]["removed"] == ["lowres"]
    assert diff["negative_diff"]["added"] == ["watermark"]

    core_map = {item["key"]: item for item in diff["core_diffs"]}
    assert core_map["Steps"] == {"key": "Steps", "field": "scripts.sampler.steps", "a": 20, "b": 30, "kind": "changed"}
    assert core_map["Sampler"]["a"] == "Euler" and core_map["Sampler"]["b"] == "DPM++ 2M"
    assert core_map["CFG scale"]["a"] == 7 and core_map["CFG scale"]["b"] == 5.5
    assert core_map["Size"]["a"] == "512x512" and core_map["Size"]["b"] == "768x512"

    lora_map = {item["name"]: item for item in diff["lora_diffs"]}
    assert lora_map["style_a"] == {"name": "style_a", "a": 0.8, "b": 0.5, "kind": "changed"}
    assert lora_map["detail_b"] == {"name": "detail_b", "a": None, "b": 1.0, "kind": "added"}

    ext_map = {item["key"]: item for item in diff["extension_diffs"]}
    assert ext_map["ADetailer model"]["group"] == "ADetailer"
    assert ext_map["ADetailer model"]["kind"] == "removed"
    assert ext_map["ControlNet 0"]["group"] == "ControlNet"
    assert ext_map["ControlNet 0"]["kind"] == "added"

    assert diff["adopt_b_patch"]["core"]["width"] == 768
    assert diff["adopt_b_patch"]["core"]["height"] == 512
    assert diff["adopt_b_patch"]["scripts"]["sampler"]["steps"] == 30
    assert diff["adopt_a_patch"]["core"]["width"] == 512
    assert diff["adopt_a_patch"]["scripts"]["sampler"]["steps"] == 20

    # Also compare two real generated outputs by file id
    out_a = helpers.submit_txt2img(live_server, "cat, sunny meadow")["outputs"][0]["id"]
    out_b = helpers.submit_txt2img(live_server, "cat, rainy alley")["outputs"][0]["id"]

    by_id = live_server.http.post("/api/v2/gallery/compare", json={"id_a": out_a, "id_b": out_b})
    assert by_id.status_code == 200
    body = by_id.json()
    assert body["id_a"] == out_a
    assert body["id_b"] == out_b
    assert body["identical"] is False
    assert body["prompt_diff"]["unchanged"] == ["cat"]
    assert body["prompt_diff"]["removed"] == ["sunny meadow"]
    assert body["prompt_diff"]["added"] == ["rainy alley"]

    same = live_server.http.post("/api/v2/gallery/compare", json={"id_a": out_a, "id_b": out_a})
    assert same.status_code == 200
    assert same.json()["identical"] is True


def test_pnginfo_zip_export_and_gallery_delete(live_server: Any, helpers: Any) -> None:
    """POST /api/v2/pnginfo on generated output + uploaded iTXt PNG,
    zip export download, and POST /api/v2/gallery/delete."""

    out_id = helpers.submit_txt2img(live_server, "crystal cathedral, sharp focus")["outputs"][0]["id"]

    png_resp = live_server.http.post("/api/v2/pnginfo", json={"file_id": out_id})
    assert png_resp.status_code == 200
    meta = png_resp.json()
    assert meta["format"] == "PNG"
    assert meta["width"] > 0 and meta["height"] > 0 and meta["size"] > 0
    assert "crystal cathedral" in (meta["infotext"] or "")
    assert isinstance(meta["params"], dict) and len(meta["params"]) >= 1
    assert len(meta["entries"]) >= 1
    chunk_types = {chunk["type"] for chunk in meta["chunks"]}
    assert {"IHDR", "IDAT", "IEND"} <= chunk_types
    assert chunk_types & {"tEXt", "iTXt", "zTXt"}

    # Uploaded PNG with full A1111 parameters + custom iTXt chunk
    up = helpers.upload_png(
        live_server,
        filename="meta.png",
        text="custom prompt\nNegative prompt: lowres\nSteps: 12, Sampler: Euler, CFG scale: 7, Seed: 7, Size: 32x32",
        itxt={"Comment": "opaque-itxt"},
    )
    up_meta = live_server.http.post("/api/v2/pnginfo", json={"file_id": up["id"]})
    assert up_meta.status_code == 200
    up_json = up_meta.json()
    assert up_json["params"].get("Steps") == "12"
    assert up_json["params"].get("Seed") == "7"
    assert any(entry["key"] == "Comment" and entry["value"] == "opaque-itxt" for entry in up_json["entries"])

    # Zip export & download
    zip_resp = live_server.http.post("/api/v2/gallery/zip", json={"ids": [out_id]})
    assert zip_resp.status_code == 200
    zip_info = zip_resp.json()
    assert zip_info["count"] == 1
    dl_resp = live_server.http.get(f"/api/v2/files/{zip_info['id']}", params={"download": "1"})
    assert dl_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(dl_resp.content)) as zf:
        assert len(zf.namelist()) == 1

    # Delete from gallery
    del_resp = live_server.http.post("/api/v2/gallery/delete", json={"ids": [out_id], "delete_files": True})
    assert del_resp.status_code == 200
    assert del_resp.json()["count"] == 1
    assert live_server.http.get(f"/api/v2/gallery/{out_id}").status_code == 404


# =============================================================================
# — Gallery, Boards, Lineage, Compare & PNG Info (Boundary/Negative)
# =============================================================================


def test_gallery_invalid_dates_and_empty_ids(live_server: Any) -> None:
    """Invalid ISO dates and empty ids lists return 422."""

    assert live_server.http.get("/api/v2/gallery", params={"since": "not-a-date"}).status_code == 422
    assert live_server.http.get("/api/v2/gallery", params={"until": "2026-99-99"}).status_code == 422
    assert live_server.http.post("/api/v2/gallery/star", json={"ids": [], "starred": True}).status_code == 422
    assert live_server.http.post("/api/v2/gallery/delete", json={"ids": []}).status_code == 422
    assert live_server.http.post("/api/v2/gallery/zip", json={"ids": []}).status_code == 422


def test_boards_name_validation_duplicate_and_missing(live_server: Any) -> None:
    """Blank/whitespace or >120-char board names return 422; duplicate names return 409; missing boards return 404."""

    for bad_name in ("", "   \t\n ", "x" * 121):
        resp = live_server.http.post("/api/v2/boards", json={"name": bad_name})
        assert resp.status_code == 422, f"expected 422 for {bad_name!r}, got {resp.status_code}"
        body = resp.json()
        assert any(err.get("param") == "name" for err in body.get("errors", []))

    created = live_server.http.post("/api/v2/boards", json={"name": "Unique Board"})
    assert created.status_code == 201
    board_id = created.json()["id"]

    dup = live_server.http.post("/api/v2/boards", json={"name": "Unique Board"})
    assert dup.status_code == 409

    dup_ws = live_server.http.post("/api/v2/boards", json={"name": "  Unique Board  "})
    assert dup_ws.status_code == 409

    # PATCH blank / >120 also 422
    assert live_server.http.patch(f"/api/v2/boards/{board_id}", json={"name": "   "}).status_code == 422
    assert live_server.http.patch(f"/api/v2/boards/{board_id}", json={"name": "y" * 121}).status_code == 422

    # Missing board 404s
    assert live_server.http.get("/api/v2/boards/no_such_board_999").status_code == 404
    assert live_server.http.patch("/api/v2/boards/no_such_board_999", json={"name": "Other"}).status_code == 404
    assert live_server.http.delete("/api/v2/boards/no_such_board_999").status_code == 404
    assert live_server.http.post("/api/v2/boards/no_such_board_999/items", json={"file_ids": ["abc"]}).status_code == 404


def test_compare_empty_and_missing_id(live_server: Any) -> None:
    """POST /api/v2/gallery/compare with empty {} -> 422; nonexistent id_a/id_b -> 404."""

    assert live_server.http.post("/api/v2/gallery/compare", json={}).status_code == 422
    assert live_server.http.post("/api/v2/gallery/compare", json={"id_a": "missing_output_1234", "infotext_b": "cat"}).status_code == 404
    assert live_server.http.post("/api/v2/gallery/compare", json={"infotext_a": "cat", "id_b": "missing_output_5678"}).status_code == 404


def test_compare_scalar_coercion_preserves_zero_and_false(live_server: Any) -> None:
    """Regression guard: 0 and false remain int 0 and bool False, while missing sides are None."""

    infotext_a = "cat\nSteps: 0, CFG scale: 0.0, Hires enabled: false, Only A: 0, Flag A: false"
    infotext_b = "cat\nSteps: 10, CFG scale: 7.5, Hires enabled: true, Only B: false"

    resp = live_server.http.post("/api/v2/gallery/compare", json={"infotext_a": infotext_a, "infotext_b": infotext_b})
    assert resp.status_code == 200
    data = resp.json()

    core_map = {d["key"]: d for d in data["core_diffs"]}
    assert core_map["Steps"]["a"] == 0 and type(core_map["Steps"]["a"]) is int
    assert core_map["Steps"]["b"] == 10 and type(core_map["Steps"]["b"]) is int
    assert core_map["CFG scale"]["a"] == 0.0 and core_map["CFG scale"]["a"] is not None
    assert core_map["CFG scale"]["b"] == 7.5

    ext_map = {d["key"]: d for d in data["extension_diffs"]}
    assert ext_map["Hires enabled"]["a"] is False
    assert ext_map["Hires enabled"]["b"] is True
    assert ext_map["Only A"]["a"] == 0 and type(ext_map["Only A"]["a"]) is int
    assert ext_map["Only A"]["b"] is None
    assert ext_map["Flag A"]["a"] is False
    assert ext_map["Flag A"]["b"] is None
    assert ext_map["Only B"]["a"] is None
    assert ext_map["Only B"]["b"] is False


def test_lineage_pnginfo_and_zip_negative_paths(live_server: Any, helpers: Any) -> None:
    """Unknown lineage id -> 404; pnginfo on uploaded JSON/text -> 415; unknown pnginfo/zip ids -> 404."""

    assert live_server.http.get("/api/v2/gallery/unknown_id_1234/lineage").status_code == 404

    json_upload = helpers.upload_bytes(live_server, b'{"hello": "world"}', filename="notes.json", content_type="application/json")
    non_img = live_server.http.post("/api/v2/pnginfo", json={"file_id": json_upload["id"]})
    assert non_img.status_code == 415

    assert live_server.http.post("/api/v2/pnginfo", json={"file_id": "unknown_id_1234"}).status_code == 404
    assert live_server.http.post("/api/v2/gallery/zip", json={"ids": ["unknown_id_1234"]}).status_code == 404


# =============================================================================
# — Capability Schemas, Actions, Conflicts, Advice, Models & Styles
# =============================================================================


def test_tab_schemas_and_pages(live_server: Any, tiny_anima_server: Any) -> None:
    """GET /api/v2/schema/{txt2img,img2img,extras} return valid TabSchema with
    capabilities, core.required, scripts, conflict_params, and advice."""

    for tab in ("txt2img", "img2img", "extras"):
        resp = live_server.http.get(f"/api/v2/schema/{tab}")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["version"] == 1
        assert schema["tab"] == tab
        assert schema["core"]["owner"] == "core"
        assert isinstance(schema["core"]["layout"], dict)
        assert isinstance(schema["core"]["defaults"], dict)
        assert isinstance(schema["core"]["required"], list)

        caps = schema["capabilities"]
        assert isinstance(caps, dict)
        assert isinstance(caps["preset"], str) and caps["preset"]
        assert isinstance(caps["capabilities"], list)
        assert isinstance(caps["attention_backends"], list)
        assert isinstance(caps["quant_formats"], list)

        assert isinstance(schema["scripts"], list)
        for script in schema["scripts"]:
            assert {"id", "title", "kind", "layout", "defaults", "args", "requires_capabilities", "applicable"} <= set(script)

        if tab in ("txt2img", "img2img"):
            assert isinstance(schema["conflict_params"], list) and len(schema["conflict_params"]) > 0
            assert isinstance(schema["advice_params"], list)
            assert isinstance(schema["advice"], list)
        if tab == "img2img":
            assert any(rule.get("param") == "init_img" for rule in schema["core"]["required"])

    pages_resp = live_server.http.get("/api/v2/pages")
    assert pages_resp.status_code == 200
    assert isinstance(pages_resp.json(), list)

    ext_pages_resp = tiny_anima_server.http.get("/api/v2/pages")
    assert ext_pages_resp.status_code == 200
    ext_pages = ext_pages_resp.json()
    assert isinstance(ext_pages, list) and len(ext_pages) >= 1
    for page in ext_pages:
        assert {"id", "title"} <= set(page)
        assert isinstance(page["id"], str) and page["id"]
        assert isinstance(page["title"], str) and page["title"]
        detail_resp = tiny_anima_server.http.get(f"/api/v2/pages/{page['id']}")
        assert detail_resp.status_code == 200
        page_schema = detail_resp.json()
        assert {"version", "id", "title", "owner"} <= set(page_schema)
        assert page_schema["id"] == page["id"]
        assert page_schema["title"] == page["title"]
        assert page_schema["owner"] == f"page:{page['id']}"


def test_core_actions_swap_dimensions_and_hr_resolution(live_server: Any) -> None:
    """swap_dimensions swaps width/height; hr_resolution computes __outputs__.hr_final_resolution."""

    swap_resp = live_server.http.post(
        "/api/v2/actions/core/swap_dimensions",
        json={"tab": "txt2img", "values": {"width": 512, "height": 768}},
    )
    assert swap_resp.status_code == 200
    swap_patch = swap_resp.json()
    assert swap_patch["width"]["value"] == 768
    assert swap_patch["height"]["value"] == 512

    hr_on = live_server.http.post(
        "/api/v2/actions/core/hr_resolution",
        json={
            "tab": "txt2img",
            "values": {"width": 512, "height": 768, "enable_hr": True, "hr_scale": 2.0, "hr_resize_x": 0, "hr_resize_y": 0},
        },
    )
    assert hr_on.status_code == 200
    hr_msg = hr_on.json()["__outputs__"]["hr_final_resolution"]
    assert hr_msg
    assert "1024" in str(hr_msg) and "1536" in str(hr_msg)

    hr_off = live_server.http.post(
        "/api/v2/actions/core/hr_resolution",
        json={
            "tab": "txt2img",
            "values": {"width": 512, "height": 768, "enable_hr": False, "hr_scale": 2.0, "hr_resize_x": 0, "hr_resize_y": 0},
        },
    )
    assert hr_off.status_code == 200
    assert hr_off.json()["__outputs__"]["hr_final_resolution"] == ""


def test_core_advice_and_capabilities(live_server: Any) -> None:
    """POST /api/v2/actions/core/__advice__ returns __advice__ and __capabilities__."""

    base_resp = live_server.http.post(
        "/api/v2/actions/core/__advice__",
        json={"tab": "txt2img", "values": {"prompt": "1girl", "cfg_scale": 7.0}},
    )
    assert base_resp.status_code == 200
    base_data = base_resp.json()
    assert isinstance(base_data["__advice__"], list)
    assert isinstance(base_data["__capabilities__"], dict)
    assert "preset" in base_data["__capabilities__"]
    assert "capabilities" in base_data["__capabilities__"]

    # Preset override triggering deterministic structured advice (Qwen 2.1 Turbo steps/CFG mismatch)
    qwen_resp = live_server.http.post(
        "/api/v2/actions/core/__advice__",
        json={
            "tab": "txt2img",
            "preset": "qwen21",
            "values": {"prompt": "1girl", "cfg_scale": 2.5, "checkpoint": "Qwen-Image-2.1-Turbo-FP8.safetensors"},
            "scripts": {"sampler": {"steps": 32}},
        },
    )
    assert qwen_resp.status_code == 200
    qwen_data = qwen_resp.json()
    assert qwen_data["__capabilities__"]["preset"] == "qwen21"
    assert len(qwen_data["__advice__"]) >= 1
    for item in qwen_data["__advice__"]:
        assert {"code", "severity", "message"} <= set(item)


def test_conflicts_and_rules_derive(live_server: Any, tiny_anima_server: Any) -> None:
    """POST /api/v2/actions/core/__conflicts__ and POST /api/v2/rules/derive return
    consistent conflict resolution patches and __advice__."""

    # Baseline check on fast live_server
    c_resp = live_server.http.post("/api/v2/actions/core/__conflicts__", json={"tab": "txt2img", "values": {"prompt": "1girl"}})
    r_resp = live_server.http.post("/api/v2/rules/derive", json={"tab": "txt2img", "core": {"prompt": "1girl"}, "scripts": {}})
    assert c_resp.status_code == 200 and r_resp.status_code == 200
    assert isinstance(c_resp.json()["__advice__"], list)
    assert isinstance(r_resp.json()["__advice__"], list)

    # Full Anima extension conflict check on tiny_anima_server (FreeFuse vs Refiner)
    schema = tiny_anima_server.http.get("/api/v2/schema/txt2img").json()
    freefuse_script = next((s for s in schema["scripts"] if "freefuse" in s["id"].lower()), None)
    assert freefuse_script is not None, "expected Anima FreeFuse script on tiny_anima_server"
    freefuse_id = freefuse_script["id"]
    freefuse_param = next(
        p.split(".", 1)[1] for p in schema["conflict_params"] if p.startswith(f"{freefuse_id}.")
    )
    models = tiny_anima_server.http.get("/api/v2/models").json()
    ckpt_name = models["checkpoints"][0]["name"]

    scripts_payload = {
        freefuse_id: {freefuse_param: True},
        "refiner": {"enable": True, "checkpoint": ckpt_name},
    }
    act_resp = tiny_anima_server.http.post(
        "/api/v2/actions/core/__conflicts__",
        json={"tab": "txt2img", "values": {"prompt": "1girl"}, "scripts": scripts_payload, "selected": f"{freefuse_id}.{freefuse_param}"},
    )
    rule_resp = tiny_anima_server.http.post(
        "/api/v2/rules/derive",
        json={"tab": "txt2img", "core": {"prompt": "1girl"}, "scripts": scripts_payload, "selected": f"{freefuse_id}.{freefuse_param}"},
    )
    assert act_resp.status_code == 200
    assert rule_resp.status_code == 200
    act_patch = act_resp.json()
    rule_patch = rule_resp.json()
    assert act_patch == rule_patch
    assert "__advice__" in act_patch
    assert "__toast__" in act_patch
    assert act_patch[f"{freefuse_id}.{freefuse_param}"]["value"] is True
    assert act_patch["refiner.enable"]["value"] is False


def test_models_presets_and_styles_lifecycle(live_server: Any) -> None:
    """GET /models, GET /models/loaded, GET /presets + /presets/{name}, and
    GET/PUT/POST/DELETE /api/v2/styles/* lifecycle."""

    models_resp = live_server.http.get("/api/v2/models")
    assert models_resp.status_code == 200
    models = models_resp.json()
    assert {"checkpoints", "vae", "text_encoders", "modules", "samplers", "schedulers", "upscalers", "styles", "unet_dtypes", "presets"} <= set(models)
    assert len(models["samplers"]) >= 1
    assert len(models["presets"]) >= 1

    loaded_resp = live_server.http.get("/api/v2/models/loaded")
    assert loaded_resp.status_code == 200
    assert {"checkpoint", "items"} <= set(loaded_resp.json())

    presets_resp = live_server.http.get("/api/v2/presets")
    assert presets_resp.status_code == 200
    presets = presets_resp.json()
    assert len(presets["anima"]) >= 1
    assert len(presets["forge"]) >= 1

    patch_resp = live_server.http.get(f"/api/v2/presets/{presets['anima'][0]}", params={"tab": "txt2img"})
    assert patch_resp.status_code == 200
    assert "__toast__" in patch_resp.json()

    # Styles CRUD + apply
    put_resp = live_server.http.put(
        "/api/v2/styles/StudioStyle",
        json={"prompt": "{prompt}, studio lighting", "negative_prompt": "blurry"},
    )
    assert put_resp.status_code == 200
    assert put_resp.json() == {"name": "StudioStyle", "prompt": "{prompt}, studio lighting", "negative_prompt": "blurry"}

    list_resp = live_server.http.get("/api/v2/styles")
    assert list_resp.status_code == 200
    assert any(s["name"] == "StudioStyle" for s in list_resp.json())

    apply_resp = live_server.http.post(
        "/api/v2/styles/apply",
        json={"prompt": "1girl", "negative_prompt": "lowres", "styles": ["StudioStyle"]},
    )
    assert apply_resp.status_code == 200
    assert "studio lighting" in apply_resp.json()["prompt"]
    assert "blurry" in apply_resp.json()["negative_prompt"]

    del_resp = live_server.http.delete("/api/v2/styles/StudioStyle")
    assert del_resp.status_code == 204
    after_del = live_server.http.get("/api/v2/styles").json()
    assert not any(s["name"] == "StudioStyle" for s in after_del)


def test_infotext_parse_prompt_tokens_and_i18n_etag(live_server: Any, tiny_anima_server: Any, helpers: Any) -> None:
    """POST /infotext/parse, POST /prompt/tokens (stub + real tokenizer), and GET /i18n/{locale} with If-None-Match -> 304."""

    parse_resp = live_server.http.post(
        "/api/v2/infotext/parse",
        json={
            "text": "masterpiece, 1girl\nNegative prompt: lowres\nSteps: 25, Sampler: Euler, CFG scale: 6.5, Seed: 42, Size: 640x832",
            "tab": "txt2img",
        },
    )
    assert parse_resp.status_code == 200
    parsed = parse_resp.json()
    assert parsed["tab"] == "txt2img"
    assert parsed["core"]["prompt"] == "masterpiece, 1girl"
    assert parsed["core"]["negative_prompt"] == "lowres"
    assert parsed["core"]["cfg_scale"] == 6.5
    assert parsed["core"]["width"] == 640
    assert parsed["core"]["height"] == 832
    assert parsed["scripts"]["sampler"]["steps"] == 25
    assert parsed["scripts"]["seed"]["seed"] == 42

    stub_tok_resp = live_server.http.post("/api/v2/prompt/tokens", json={"text": "1girl, solo", "steps": 20})
    assert stub_tok_resp.status_code == 200
    stub_tok = stub_tok_resp.json()
    assert set(stub_tok) == {"tokens", "max_length"}
    assert isinstance(stub_tok["tokens"], int) and stub_tok["tokens"] > 0
    assert isinstance(stub_tok["max_length"], int) and stub_tok["max_length"] > 0

    # Ensure TinyAnima model & tokenizer are loaded, then verify real token counts
    loaded_resp = tiny_anima_server.http.get("/api/v2/models/loaded")
    assert loaded_resp.status_code == 200
    if loaded_resp.json().get("checkpoint") is None:
        warmup = helpers.submit_txt2img(tiny_anima_server, "1girl, solo", width=64, height=64, steps=1, timeout=60.0)
        assert warmup["status"] == "completed"

    tok_resp = tiny_anima_server.http.post("/api/v2/prompt/tokens", json={"text": "1girl, solo", "steps": 20})
    assert tok_resp.status_code == 200
    tok_data = tok_resp.json()
    assert set(tok_data) == {"tokens", "max_length"}
    assert isinstance(tok_data["tokens"], int) and tok_data["tokens"] > 0
    assert isinstance(tok_data["max_length"], int) and tok_data["max_length"] > 0

    long_resp = tiny_anima_server.http.post(
        "/api/v2/prompt/tokens",
        json={"text": "1girl, solo, blue eyes, silver hair, standing in a sunlit meadow, detailed background, masterpiece", "steps": 20},
    )
    assert long_resp.status_code == 200
    long_data = long_resp.json()
    assert isinstance(long_data["tokens"], int) and long_data["tokens"] > tok_data["tokens"]
    assert long_data["max_length"] == tok_data["max_length"]

    # Discover available locales via /api/v2/app (opaque-box)
    app_info = live_server.http.get("/api/v2/app").json()
    locales = [loc["id"] for loc in app_info["locales"]]
    assert len(locales) >= 1
    locale_id = locales[0]

    cat_resp = live_server.http.get(f"/api/v2/i18n/{locale_id}")
    assert cat_resp.status_code == 200
    etag = cat_resp.headers.get("etag")
    assert etag
    catalog = cat_resp.json()
    assert isinstance(catalog, dict) and len(catalog) > 0

    not_mod = live_server.http.get(f"/api/v2/i18n/{locale_id}", headers={"If-None-Match": etag})
    assert not_mod.status_code == 304


# =============================================================================
# — Capability Schemas, Actions, Presets, Models & Styles (Boundary)
# =============================================================================


def test_schema_and_presets_negative(live_server: Any) -> None:
    """GET /schema/invalid -> 422; GET /presets/nonexistent -> 404; GET /presets/<valid>?tab=extras -> 422."""

    assert live_server.http.get("/api/v2/schema/invalid").status_code == 422
    assert live_server.http.get("/api/v2/presets/nonexistent_preset_xyz").status_code == 404

    valid_anima = live_server.http.get("/api/v2/presets").json()["anima"][0]
    extras_resp = live_server.http.get(f"/api/v2/presets/{valid_anima}", params={"tab": "extras"})
    assert extras_resp.status_code == 422
    assert any(err.get("param") == "tab" for err in extras_resp.json().get("errors", []))


def test_forge_preset_unknown_and_frozen(live_server: Any, make_live_server: Any) -> None:
    """__forge_preset__ with unknown preset or frozen forge_preset -> 422."""

    bad_preset = live_server.http.post(
        "/api/v2/actions/core/__forge_preset__",
        json={"tab": "txt2img", "preset": "not_a_real_preset_xyz"},
    )
    assert bad_preset.status_code == 422
    assert any(err.get("param") == "preset" for err in bad_preset.json().get("errors", []))

    frozen_server = make_live_server(steps=1, step_time=0.0, realtime=False, cmd_opts_overrides={"freeze_settings": True})
    frozen_resp = frozen_server.http.post(
        "/api/v2/actions/core/__forge_preset__",
        json={"tab": "txt2img", "preset": "xl"},
    )
    assert frozen_resp.status_code == 422
    errors = frozen_resp.json().get("errors", [])
    assert any(err.get("owner") == "settings" and err.get("param") == "forge_preset" for err in errors)


@pytest.mark.parametrize("bad_token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_floats_return_422_never_500(live_server: Any, bad_token: str) -> None:
    """Non-finite JSON floats on /jobs, /actions/*, and /rules/derive return 422, never 200 or 500."""

    job_body = f'{{"tab": "txt2img", "core": {{"prompt": "1girl", "cfg_scale": {bad_token}}}}}'.encode()
    job_resp = live_server.http.post("/api/v2/jobs", content=job_body, headers={"Content-Type": "application/json"})
    assert job_resp.status_code == 422

    swap_body = f'{{"tab": "txt2img", "values": {{"width": {bad_token}, "height": 512}}}}'.encode()
    swap_resp = live_server.http.post("/api/v2/actions/core/swap_dimensions", content=swap_body, headers={"Content-Type": "application/json"})
    assert swap_resp.status_code == 422

    bad_tab_body = f'{{"tab": {bad_token}, "values": {{"prompt": "1girl"}}}}'.encode()
    bad_tab_resp = live_server.http.post("/api/v2/actions/core/__advice__", content=bad_tab_body, headers={"Content-Type": "application/json"})
    assert bad_tab_resp.status_code == 422

    raw_resp = live_server.http.post("/api/v2/actions/core/__advice__", content=bad_token.encode(), headers={"Content-Type": "application/json"})
    assert raw_resp.status_code == 422

    advice_body = f'{{"tab": "txt2img", "values": {{"prompt": "1girl", "cfg_scale": {bad_token}}}}}'.encode()
    advice_resp = live_server.http.post("/api/v2/actions/core/__advice__", content=advice_body, headers={"Content-Type": "application/json"})
    assert advice_resp.status_code == 422

    conflicts_body = f'{{"tab": "txt2img", "values": {{"prompt": "1girl", "cfg_scale": {bad_token}}}}}'.encode()
    conflicts_resp = live_server.http.post("/api/v2/actions/core/__conflicts__", content=conflicts_body, headers={"Content-Type": "application/json"})
    assert conflicts_resp.status_code == 422

    rules_body = f'{{"tab": "txt2img", "core": {{"prompt": "1girl", "cfg_scale": {bad_token}}}, "scripts": {{}}}}'.encode()
    rules_resp = live_server.http.post("/api/v2/rules/derive", content=rules_body, headers={"Content-Type": "application/json"})
    assert rules_resp.status_code == 422


def test_selectable_scripts_capabilities_and_unknown_actions(live_server: Any, tiny_anima_server: Any) -> None:
    """Selectable scripts in TabSchema expose requires_capabilities & applicable /
    inapplicable_reason properly; unknown actions/owners return 404."""

    # Unknown action/owner 404s on fast live_server
    assert live_server.http.post("/api/v2/actions/core/unknown_action", json={"tab": "txt2img"}).status_code == 404
    assert live_server.http.post("/api/v2/actions/unknown_owner/act", json={"tab": "txt2img"}).status_code == 404

    # Real built-in scripts (selectable + capability-gated) on tiny_anima_server
    app_before = tiny_anima_server.http.get("/api/v2/app").json()
    original_preset = app_before["options"].get("forge_preset") or "sd"
    try:
        # Ensure we start on non-Anima preset ("sd") so anima_hooks scripts are inapplicable
        if original_preset != "sd":
            assert tiny_anima_server.http.post("/api/v2/actions/core/__forge_preset__", json={"tab": "txt2img", "preset": "sd"}).status_code == 200

        sd_schema = tiny_anima_server.http.get("/api/v2/schema/txt2img").json()
        selectable = [s for s in sd_schema["scripts"] if s["kind"] == "selectable"]
        assert len(selectable) >= 1
        for script in selectable:
            assert isinstance(script["requires_capabilities"], list)
            assert isinstance(script["applicable"], bool)
            assert script["visible_if"] == {"param": "core.script", "eq": script["title"]}

        anima_gated = [s for s in sd_schema["scripts"] if "anima_hooks" in s["requires_capabilities"]]
        assert len(anima_gated) >= 1
        for script in anima_gated:
            assert script["applicable"] is False
            assert isinstance(script.get("inapplicable_reason"), str) and script["inapplicable_reason"]
            assert isinstance(script.get("inapplicable_template"), str)
            assert isinstance(script.get("inapplicable_params"), dict)

        # Switch preset to "anima" -> gated scripts become applicable
        switch_resp = tiny_anima_server.http.post("/api/v2/actions/core/__forge_preset__", json={"tab": "txt2img", "preset": "anima"})
        assert switch_resp.status_code == 200
        anima_schema = tiny_anima_server.http.get("/api/v2/schema/txt2img").json()
        anima_by_id = {s["id"]: s for s in anima_schema["scripts"]}
        for script in anima_gated:
            updated = anima_by_id[script["id"]]
            assert updated["applicable"] is True
            assert "inapplicable_reason" not in updated
    finally:
        tiny_anima_server.http.post("/api/v2/actions/core/__forge_preset__", json={"tab": "txt2img", "preset": original_preset})


def test_styles_i18n_and_pages_negative(live_server: Any) -> None:
    """PUT /styles/None or blank name -> 422; DELETE missing style -> 404;
    GET /i18n/zz-ZZ -> 404; GET /pages/nonexistent -> 404."""

    none_style = live_server.http.put("/api/v2/styles/None", json={"prompt": "1girl", "negative_prompt": ""})
    assert none_style.status_code == 422

    blank_style = live_server.http.put("/api/v2/styles/%20%20%20", json={"prompt": "1girl", "negative_prompt": ""})
    assert blank_style.status_code == 422

    assert live_server.http.delete("/api/v2/styles/nonexistent_style_xyz").status_code == 404
    assert live_server.http.get("/api/v2/i18n/zz-ZZ").status_code == 404
    assert live_server.http.get("/api/v2/pages/nonexistent_page_xyz").status_code == 404
