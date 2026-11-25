"""Unit and route tests for Gallery Boards, Asset Lineage, and Parameter Diff."""

from __future__ import annotations

from PIL import Image

from remi.files import FileStore
from remi.jobs import Job, JobStore
from remi.library import BoardError, GalleryIndex, GalleryQuery, compare_outputs, diff_parameters
from remi.server.testing import server_app
from remi.testing import webui_state


def _save_img(files: FileStore, color: tuple[int, int, int], infotext: str, *, job_id: str | None = None, tab: str = "txt2img"):
    return files.save_output_image(Image.new("RGB", (32, 32), color), infotext, job_id=job_id, tab=tab)


def test_boards_crud_and_gallery_filter(tmp_path):
    files = FileStore(tmp_path / "files")
    index = GalleryIndex(files, folders=list)
    try:
        out1 = _save_img(files, (255, 0, 0), "red rose\nSteps: 20, Model: anima_v2")
        out2 = _save_img(files, (0, 255, 0), "green leaf\nSteps: 20, Model: anima_v2")
        out3 = _save_img(files, (0, 0, 255), "blue sky\nSteps: 20, Model: flux3")

        board = index.create_board("Character Concepts", description="Main cast", color="#e07a5f")
        assert board["item_count"] == 0 and board["cover_file_id"] is None

        try:
            index.create_board("Character Concepts")
            raise AssertionError("expected duplicate board error")
        except BoardError as err:
            assert err.status == 409

        added = index.add_to_board(board["id"], [out1.id, out2.id, out1.id])
        assert added == 2
        fetched = index.get_board(board["id"])
        assert fetched["item_count"] == 2
        assert fetched["cover_file_id"] in (out1.id, out2.id)

        # Gallery filter by board_id returns only board items
        items, _cursor, total = index.list(GalleryQuery(board_id=board["id"]))
        assert total == 2
        assert {item.record.id for item in items} == {out1.id, out2.id}
        assert out3.id not in {item.record.id for item in items}

        # Rename & remove one item
        renamed = index.update_board(board["id"], name="Hero Sheet", color="#3d405b")
        assert renamed["name"] == "Hero Sheet" and renamed["color"] == "#3d405b"
        assert index.remove_from_board(board["id"], [out1.id]) == 1
        assert index.get_board(board["id"])["item_count"] == 1

        # Deleting board preserves gallery outputs
        assert index.delete_board(board["id"]) is True
        assert index.get(out2.id) is not None
    finally:
        index.close()
        files.close()


def test_deleting_a_board_is_one_transaction(tmp_path):
    """delete_board removed the board's items, then the board, each committed on its own: a
    failure between them (a locked or full database) left an empty board behind."""
    import sqlite3

    import pytest

    files = FileStore(tmp_path / "files")
    index = GalleryIndex(files, folders=list)
    try:
        out = _save_img(files, (255, 0, 0), "red rose\nSteps: 20")
        board = index.create_board("Kept")
        assert index.add_to_board(board["id"], [out.id]) == 1
        index._db.execute("CREATE TEMP TRIGGER refuse BEFORE DELETE ON boards BEGIN SELECT RAISE(ABORT, 'refused'); END")
        with pytest.raises(sqlite3.IntegrityError):
            index.delete_board(board["id"])
        assert index.get_board(board["id"])["item_count"] == 1
        index._db.execute("DROP TRIGGER refuse")
        assert index.delete_board(board["id"]) is True and index.get_board(board["id"]) is None
    finally:
        index.close()
        files.close()


def test_asset_lineage_ancestors_siblings_descendants_and_cycles(tmp_path):
    files = FileStore(tmp_path / "files")
    job_store = JobStore(tmp_path / "jobs.db")
    index = GalleryIndex(files, folders=list)
    try:
        ref_upload = files.add_upload_bytes(b"\x89PNG\r\n\x1a\n" + _png_bytes(), "ref.png")
        # Job 1 produces two siblings (base_a, base_b) using ref_upload via ImageStitch
        j1 = job_store.add("txt2img", {"prompt": "hero"}, {"image_stitch": {"enable": True, "input_images": [ref_upload.id]}})
        base_a = _save_img(files, (10, 20, 30), "hero\nSteps: 20, Seed: 1", job_id=j1.id)
        base_b = _save_img(files, (15, 25, 35), "hero\nSteps: 20, Seed: 2", job_id=j1.id)
        j1 = job_store.finish(j1.id, "completed", outputs=[{"id": base_a.id, "kind": "image"}, {"id": base_b.id, "kind": "image"}])
        index.index_job_lineage(j1, [base_a.id, base_b.id], job_store=job_store)

        # Job 2 refines base_a via img2img init_img
        j2 = job_store.add("img2img", {"prompt": "hero detailed", "init_img": base_a.id}, {}, meta={"chain_id": "chain1", "stage_index": 1})
        refined = _save_img(files, (40, 50, 60), "hero detailed\nSteps: 20, Seed: 3", job_id=j2.id, tab="img2img")
        j2 = job_store.finish(j2.id, "completed", outputs=[{"id": refined.id, "kind": "image"}])
        index.index_job_lineage(j2, [refined.id], job_store=job_store)

        # Job 3 upscales refined via upscale_from
        j3 = job_store.add("txt2img", {"prompt": "hero 4k"}, {}, meta={"upscale_from": refined.id, "chain_id": "chain1", "stage_index": 2})
        upscaled = _save_img(files, (70, 80, 90), "hero 4k\nSteps: 12, Seed: 4", job_id=j3.id)
        j3 = job_store.finish(j3.id, "completed", outputs=[{"id": upscaled.id, "kind": "image"}])
        index.index_job_lineage(j3, [upscaled.id], job_store=job_store)

        # Job 4 retries j2
        j4 = job_store.add("img2img", {"prompt": "hero retry", "init_img": base_a.id}, {}, retry_of=j2.id)
        retried = _save_img(files, (99, 88, 77), "hero retry\nSteps: 20, Seed: 5", job_id=j4.id, tab="img2img")
        j4 = job_store.finish(j4.id, "completed", outputs=[{"id": retried.id, "kind": "image"}])
        index.index_job_lineage(j4, [retried.id], job_store=job_store)

        # Lineage of `refined`: ancestors = [base_a (depth 1), ref_upload (depth 2)], descendants = [upscaled, retried], siblings = [retried (same_parent: base_a)]
        lin = index.lineage_for(refined.id, job_store=job_store)
        ancestor_ids = {n["file_id"]: (n["relation"], n["depth"]) for n in lin["ancestors"]}
        assert ancestor_ids[base_a.id] == ("init_img", 1)
        assert ancestor_ids[ref_upload.id] == ("reference", 2)

        desc_ids = {n["file_id"]: n["relation"] for n in lin["descendants"]}
        assert desc_ids[upscaled.id] == "upscale_from"
        assert desc_ids[retried.id] == "retry_of"

        # Siblings of base_a include base_b (same_job)
        lin_base = index.lineage_for(base_a.id, job_store=job_store)
        assert any(s["file_id"] == base_b.id and s["relation"] == "same_job" for s in lin_base["siblings"])

        # Cycle safety: introducing a cycle does not infinite-loop
        index.record_lineage(base_a.id, upscaled.id, "reference")
        lin_cycle = index.lineage_for(upscaled.id, job_store=job_store)
        assert lin_cycle["file_id"] == upscaled.id
        assert len(lin_cycle["ancestors"]) <= 10
    finally:
        index.close()
        job_store.close()
        files.close()


def _png_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 120, 120)).save(buf, format="PNG")
    # return raw bytes after PNG signature for helper compatibility -> actually return full PNG
    return buf.getvalue()[8:]


def test_structured_parameter_diff_and_adopt_patch():
    info_a = (
        "1girl, silver hair, snow, <lora:style_a:0.7>, <lora:detail:0.5>\n"
        "Negative prompt: blurry, lowres\n"
        "Steps: 20, Sampler: Euler, Schedule type: Simple, CFG scale: 5.0, Seed: 1001, Size: 832x1216, "
        "Model: anima_2b, DAVE strength: 0.3, ADetailer model: face_yolov8n.pt"
    )
    info_b = (
        "1girl, golden hair, snow, sunset, <lora:style_a:0.9>, <lora:cinematic:0.4>\n"
        "Negative prompt: blurry, watermark\n"
        "Steps: 28, Sampler: Euler a, Schedule type: Simple, CFG scale: 4.0, Seed: 2002, Size: 1024x1024, "
        "Model: anima_3.8b, DAVE strength: 0.5, ControlNet 0: \"weight: 0.8\""
    )

    diff = diff_parameters(info_a, info_b)
    assert diff["identical"] is False
    assert "golden hair" in diff["prompt_diff"]["added"] and "sunset" in diff["prompt_diff"]["added"]
    assert "silver hair" in diff["prompt_diff"]["removed"]
    assert "1girl" in diff["prompt_diff"]["unchanged"] and "snow" in diff["prompt_diff"]["unchanged"]
    assert diff["negative_diff"]["added"] == ["watermark"] and diff["negative_diff"]["removed"] == ["lowres"]

    core_by_key = {item["key"]: item for item in diff["core_diffs"]}
    assert core_by_key["Steps"]["a"] == 20 and core_by_key["Steps"]["b"] == 28
    assert core_by_key["CFG scale"]["a"] == 5.0 and core_by_key["CFG scale"]["b"] == 4.0
    assert core_by_key["Seed"]["a"] == 1001 and core_by_key["Seed"]["b"] == 2002
    assert core_by_key["Sampler"]["b"] == "Euler a"
    assert core_by_key["Model"]["b"] == "anima_3.8b"
    assert "Schedule type" not in core_by_key  # unchanged

    lora_by_name = {item["name"]: item for item in diff["lora_diffs"]}
    assert lora_by_name["style_a"] == {"name": "style_a", "a": 0.7, "b": 0.9, "kind": "changed"}
    assert lora_by_name["detail"] == {"name": "detail", "a": 0.5, "b": None, "kind": "removed"}
    assert lora_by_name["cinematic"] == {"name": "cinematic", "a": None, "b": 0.4, "kind": "added"}

    ext_by_key = {item["key"]: item for item in diff["extension_diffs"]}
    assert ext_by_key["DAVE strength"]["group"] == "DAVE" and ext_by_key["DAVE strength"]["b"] == 0.5
    assert ext_by_key["ADetailer model"]["kind"] == "removed"
    assert ext_by_key["ControlNet 0"]["group"] == "ControlNet" and ext_by_key["ControlNet 0"]["kind"] == "added"

    # One-click adopt B patch carries exact form overrides
    patch_b = diff["adopt_b_patch"]
    assert patch_b["core"]["cfg_scale"] == 4.0
    assert patch_b["core"]["width"] == 1024 and patch_b["core"]["height"] == 1024
    assert patch_b["scripts"]["sampler"]["steps"] == 28
    assert patch_b["scripts"]["sampler"]["sampler_name"] == "Euler a"
    assert patch_b["scripts"]["seed"]["seed"] == 2002

    # Identical infotexts report identical=True
    same = diff_parameters(info_a, info_a)
    assert same["identical"] is True and not same["core_diffs"] and not same["lora_diffs"] and not same["extension_diffs"]


def test_boards_lineage_compare_http_routes(tmp_path):
    with webui_state(), server_app(tmp_path) as server:
        files = server.services.files
        out_a = _save_img(files, (200, 10, 10), "cat\nSteps: 15, CFG scale: 5, Seed: 1, Size: 512x512")
        out_b = _save_img(files, (10, 200, 10), "cat, hat\nSteps: 25, CFG scale: 4, Seed: 2, Size: 768x768")

        # Create board, assign out_a, filter gallery by board_id
        board = server.client.post("/api/v2/boards", json={"name": "Favorites", "color": "#ffaa00"}).json()
        bid = board["id"]
        assert server.client.post(f"/api/v2/boards/{bid}/items", json={"file_ids": [out_a.id]}).json()["count"] == 1

        gal = server.client.get("/api/v2/gallery", params={"board_id": bid}).json()
        assert gal["total"] == 1 and gal["items"][0]["id"] == out_a.id

        # Compare via file IDs
        cmp_res = server.client.post("/api/v2/gallery/compare", json={"id_a": out_a.id, "id_b": out_b.id})
        assert cmp_res.status_code == 200
        cmp = cmp_res.json()
        assert cmp["identical"] is False
        assert cmp["prompt_diff"]["added"] == ["hat"]
        assert cmp["adopt_b_patch"]["scripts"]["sampler"]["steps"] == 25

        # Record lineage and query via route
        from remi.library import gallery_for

        gallery_for(files).record_lineage(out_b.id, out_a.id, "init_img", chain_id="c1", stage_index=1)
        lin = server.client.get(f"/api/v2/gallery/{out_b.id}/lineage")
        assert lin.status_code == 200
        assert lin.json()["ancestors"][0]["file_id"] == out_a.id
        assert lin.json()["ancestors"][0]["chain_id"] == "c1"

        # Remove item and delete board
        assert server.client.delete(f"/api/v2/boards/{bid}/items/{out_a.id}").status_code == 204
        assert server.client.delete(f"/api/v2/boards/{bid}").status_code == 204

        # Verify compare_outputs direct helper raises 404 on missing id
        try:
            compare_outputs(files, id_a="missingmissing", id_b=out_b.id)
            raise AssertionError("expected 404")
        except BoardError as err:
            assert err.status == 404

        # Additional HTTP route branches (list, get, patch, duplicate 409, bulk remove, 404s, 422 compare)
        b2 = server.client.post("/api/v2/boards", json={"name": "Batch Board", "description": "d1"}).json()
        b2_id = b2["id"]
        assert server.client.post("/api/v2/boards", json={"name": "Batch Board"}).status_code == 409
        assert any(b["id"] == b2_id for b in server.client.get("/api/v2/boards").json()["items"])
        assert server.client.get(f"/api/v2/boards/{b2_id}").status_code == 200
        assert server.client.get("/api/v2/boards/no-such-board").status_code == 404
        patched = server.client.patch(f"/api/v2/boards/{b2_id}", json={"name": "Batch v2", "description": "d2", "color": "#112233"})
        assert patched.status_code == 200 and patched.json()["name"] == "Batch v2" and patched.json()["color"] == "#112233"
        assert server.client.post(f"/api/v2/boards/{b2_id}/items", json={"file_ids": [out_a.id, out_b.id]}).json()["count"] == 2
        assert server.client.post(f"/api/v2/boards/{b2_id}/items/remove", json={"file_ids": [out_a.id]}).json()["count"] == 1
        assert server.client.delete(f"/api/v2/boards/{b2_id}").status_code == 204
        assert server.client.delete("/api/v2/boards/no-such-board").status_code == 404
        assert server.client.get("/api/v2/gallery/no-such-output/lineage").status_code == 404
        assert server.client.post("/api/v2/gallery/compare", json={}).status_code == 422
        # Keep unused import clean
        assert Job.__name__ == "Job"


def test_viewing_lineage_keeps_edge_times_and_deleting_drops_edges(tmp_path):
    files = FileStore(tmp_path / "files")
    job_store = JobStore(tmp_path / "jobs.db")
    index = GalleryIndex(files, folders=list)
    try:
        j1 = job_store.add("txt2img", {"prompt": "base"}, {})
        base = _save_img(files, (1, 2, 3), "base\nSteps: 20, Seed: 1", job_id=j1.id)
        j1 = job_store.finish(j1.id, "completed", outputs=[{"id": base.id, "kind": "image"}])
        j2 = job_store.add("img2img", {"prompt": "refined", "init_img": base.id}, {})
        refined = _save_img(files, (4, 5, 6), "refined\nSteps: 20, Seed: 2", job_id=j2.id, tab="img2img")
        j2 = job_store.finish(j2.id, "completed", outputs=[{"id": refined.id, "kind": "image"}])
        index.index_job_lineage(j2, [refined.id], job_store=job_store)

        def created():
            with index._lock:
                return index._db.execute("SELECT created FROM asset_lineage WHERE child_id = ?", (refined.id,)).fetchall()

        before = created()
        assert len(before) == 1
        index.lineage_for(refined.id, job_store=job_store)  # indexes the job again
        assert created() == before  # the sort key of the lists is not rewritten by a view

        assert index.delete([base.id]) == 1
        assert created() == []
    finally:
        index.close()
        job_store.close()
        files.close()


def test_viewing_an_indexed_lineage_writes_nothing(tmp_path):
    files = FileStore(tmp_path / "files")
    job_store = JobStore(tmp_path / "jobs.db")
    index = GalleryIndex(files, folders=list)
    try:
        j1 = job_store.add("txt2img", {"prompt": "base"}, {})
        base = _save_img(files, (1, 2, 3), "base\nSteps: 20, Seed: 1", job_id=j1.id)
        j1 = job_store.finish(j1.id, "completed", outputs=[{"id": base.id, "kind": "image"}])
        j2 = job_store.add("img2img", {"prompt": "refined", "init_img": base.id}, {})
        refined = _save_img(files, (4, 5, 6), "refined\nSteps: 20, Seed: 2", job_id=j2.id, tab="img2img")
        j2 = job_store.finish(j2.id, "completed", outputs=[{"id": refined.id, "kind": "image"}])
        # a job finished before lineage was indexed is indexed by its first view
        assert [node["file_id"] for node in index.lineage_for(refined.id, job_store=job_store)["ancestors"]] == [base.id]

        statements: list[str] = []
        index._db.set_trace_callback(statements.append)
        lineage = index.lineage_for(refined.id, job_store=job_store)
        index.lineage_for(base.id, job_store=job_store)
        index._db.set_trace_callback(None)
        assert [node["file_id"] for node in lineage["ancestors"]] == [base.id]
        assert not [sql for sql in statements if "INSERT" in sql.upper() or "UPDATE" in sql.upper()]
    finally:
        index.close()
        job_store.close()
        files.close()


def test_a_lineage_walk_reads_each_level_at_once(tmp_path):
    files = FileStore(tmp_path / "files")
    job_store = JobStore(tmp_path / "jobs.db")
    index = GalleryIndex(files, folders=list)
    try:

        def run(tab, core, meta=None, *, colors, index_now=True):
            job = job_store.add(tab, core, {}, meta=meta or {})
            outputs = [_save_img(files, color, f"{core['prompt']}\nSteps: 20, Seed: {color[0]}", job_id=job.id, tab=tab) for color in colors]
            job = job_store.finish(job.id, "completed", outputs=[{"id": output.id, "kind": "image"} for output in outputs])
            if index_now:
                index.index_job_lineage(job, [output.id for output in outputs], job_store=job_store)
            return outputs

        reference = files.add_upload_bytes(b"\x89PNG\r\n\x1a\n" + _png_bytes(), "ref.png")
        [a1] = run("img2img", {"prompt": "a1", "init_img": reference.id}, colors=[(1, 1, 1)])
        chain = [a1]
        for step in range(2, 5):
            [node] = run("img2img", {"prompt": f"a{step}", "init_img": chain[-1].id}, colors=[(step, step, step)], index_now=step != 3)
            chain.append(node)
        a5, a5_twin = run("img2img", {"prompt": "a5", "init_img": chain[-1].id, "mask": reference.id}, colors=[(5, 5, 5), (6, 6, 6)])
        [cousin] = run("img2img", {"prompt": "cousin", "init_img": chain[-1].id}, colors=[(7, 7, 7)])
        [d1] = run("img2img", {"prompt": "d1", "init_img": a5.id}, colors=[(8, 8, 8)])
        [d2] = run("txt2img", {"prompt": "d2"}, {"upscale_from": d1.id}, colors=[(9, 9, 9)])

        def shape(lineage):
            return {key: [(node["file_id"], node["relation"], node["depth"], node["job_id"], node["exists"]) for node in lineage[key]] for key in ("ancestors", "siblings", "descendants")}

        a2, a3, a4 = chain[1:]
        first = index.lineage_for(a5.id, job_store=job_store)  # indexes a3, whose job was not indexed
        expected = {
            "ancestors": [
                (a4.id, "init_img", 1, a4.job_id, True),
                (reference.id, "mask", 1, a5.job_id, True),
                (a3.id, "init_img", 2, a3.job_id, True),
                (a2.id, "init_img", 3, a2.job_id, True),
                (a1.id, "init_img", 4, a1.job_id, True),
            ],
            "descendants": [(d1.id, "init_img", 1, d1.job_id, True), (d2.id, "upscale_from", 2, d2.job_id, True)],
        }
        found = shape(first)
        siblings = found.pop("siblings")
        assert found == expected
        # the same job's outputs first, then those sharing a parent (a4, or the reference a1 also used)
        assert siblings[0] == (a5_twin.id, "same_job", 0, a5.job_id, True)
        assert sorted(siblings[1:]) == sorted([(cousin.id, "same_parent", 0, cousin.job_id, True), (a1.id, "same_parent", 0, a1.job_id, True)])
        assert first["item"]["file_id"] == a5.id and first["item"]["job_id"] == a5.job_id

        statements: list[str] = []
        index._db.set_trace_callback(statements.append)
        again = index.lineage_for(a5.id, job_store=job_store)
        index._db.set_trace_callback(None)
        assert again == first
        selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
        depth = 4  # a5 → a4 → a3 → a2 → a1 (→ the reference upload)
        assert len(selects) <= 2 * depth + 4, selects
    finally:
        index.close()
        job_store.close()
        files.close()
