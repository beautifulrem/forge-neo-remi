"""Uploads, files and thumbnails by id, the outputs gallery and networks."""

import dataclasses
import io
import os
import sqlite3
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from remi.files import FileError, FileStore, clean_name, connect, snap_width, sniff, transaction
from remi.networks import NetworkEntry, NetworkIndex
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path) as server:
        yield server


def png(size=(64, 48), color=(10, 20, 30), text=None):
    buffer = io.BytesIO()
    info = None
    if text:
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", text)
    Image.new("RGB", size, color).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def upload(client, data, name="image.png", mime="image/png"):
    return client.post("/api/v2/uploads", files={"file": (name, data, mime)})


# region Uploads


def test_an_image_upload_is_sniffed_sized_and_deduplicated(server):
    response = upload(server.client, png(), name="../../evil/cat.png", mime="text/plain")
    assert response.status_code == 201
    data = response.json()
    assert data["width"] == 64 and data["height"] == 48 and data["mime"] == "image/png" and data["name"] == "cat.png"
    assert upload(server.client, png(), name="again.png").json()["id"] == data["id"]
    record = server.services.files.get(data["id"])
    assert Path(record.path).is_relative_to(server.services.files.uploads_dir)
    served = server.client.get(f"/api/v2/files/{data['id']}")
    assert served.status_code == 200 and served.content == png() and served.headers["x-content-type-options"] == "nosniff"


def test_content_decides_the_type_not_the_name(server):
    assert upload(server.client, b"MZ\x90\x00 not an image", name="photo.png").status_code == 415
    assert upload(server.client, b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, name="broken.png").status_code == 415
    assert upload(server.client, b"<script>alert(1)</script>", name="x.html", mime="text/html").status_code == 415
    text = upload(server.client, b"a, b\n1, 2\n", name="list.csv", mime="text/csv")
    assert text.status_code == 201 and text.json()["mime"] == "text/plain"
    assert server.client.get(f"/api/v2/files/{text.json()['id']}").headers["content-disposition"].startswith("attachment")
    assert upload(server.client, b"", name="empty.png").status_code == 400


def test_uploads_are_size_limited(server):
    server.services.files.max_upload_bytes = 1000
    response = upload(server.client, b"\x89PNG\r\n\x1a\n" + os.urandom(4000))
    assert response.status_code == 413
    assert upload(server.client, os.urandom(100_000)).status_code == 413  # refused by Content-Length
    assert not any(path.name.startswith(".incoming") for path in server.services.files.uploads_dir.iterdir())


def test_a_chunked_upload_without_content_length_is_cut_off(server):
    server.services.files.max_upload_bytes = 1000
    boundary = "remiboundary"
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\nContent-Type: image/png\r\n\r\n'.encode()

    def body():
        yield head
        for _ in range(20):
            yield os.urandom(64 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    response = server.client.post("/api/v2/uploads", content=body(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413


def test_a_cancelled_upload_is_a_client_error_and_closes_its_spooled_file(server, monkeypatch):
    import asyncio

    from fastapi import HTTPException
    from starlette.formparsers import MultiPartParser
    from starlette.requests import Request

    from remi.server.routes.files import upload as upload_route

    boundary = "remiboundary"
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
    messages = [{"type": "http.request", "body": head + os.urandom(2 * 1024 * 1024), "more_body": True}, {"type": "http.disconnect"}]

    async def receive():
        return messages.pop(0)

    parsers = []
    real_init = MultiPartParser.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        parsers.append(self)

    monkeypatch.setattr(MultiPartParser, "__init__", init)
    scope = {"type": "http", "method": "POST", "path": "/api/v2/uploads", "headers": [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())]}
    with pytest.raises(HTTPException) as cancelled:
        asyncio.run(upload_route(Request(scope, receive), server.services))
    assert cancelled.value.status_code == 400
    [parser] = parsers
    assert parser._files_to_close_on_error and all(spooled.closed for spooled in parser._files_to_close_on_error)


def test_the_upload_stream_limit_stops_reading_early():
    import asyncio

    from remi.server.routes.files import BodyTooLarge, limited

    read = []

    async def stream():
        for index in range(100):
            read.append(index)
            yield b"x" * 1000

    async def consume():
        async for _chunk in limited(stream(), 5000):
            pass

    with pytest.raises(BodyTooLarge):
        asyncio.run(consume())
    assert len(read) == 6


def test_a_reupload_counts_as_new_for_pruning_and_concurrent_uploads_share_one_record(tmp_path):
    import threading
    import time

    store = FileStore(tmp_path)
    first = store.add_upload_bytes(png(color=(5, 5, 5)))
    store._db.execute("UPDATE files SET created = ? WHERE id = ?", (time.time() - 30 * 86400, first.id))
    again = store.add_upload_bytes(png(color=(5, 5, 5)))
    assert again.id == first.id and store.prune_uploads(86400) == 0 and store.get(first.id) is not None

    results = []
    data = png(color=(6, 6, 6))
    threads = [threading.Thread(target=lambda: results.append(store.add_upload_bytes(data).id)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1
    assert store._db.execute("SELECT COUNT(*) FROM files WHERE kind = 'upload'").fetchone()[0] == 2
    store.close()


def test_a_bad_form_is_rejected(server):
    assert server.client.post("/api/v2/uploads", data={"other": "x"}).status_code == 422
    assert server.client.post("/api/v2/uploads", content=b"nope", headers={"Content-Type": "application/json"}).status_code in (400, 422)


def test_sniffing_and_names():
    assert sniff(b"\xff\xd8\xff\xe0").mime == "image/jpeg"
    assert sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ").mime == "image/webp"
    assert sniff(b"\x00\x00\x00\x18ftypisom").kind == "video"
    assert sniff(b"\x00\x00\x00\x18ftypavif").mime == "image/avif"
    assert sniff(b"\x1aE\xdf\xa3").mime == "video/webm"
    assert sniff((20).to_bytes(8, "little") + b'{"a":1}').ext == ".safetensors"
    assert sniff(b"hello", "notes.txt").kind == "text" and sniff(b"hello", "notes.exe") is None
    assert clean_name("C:\\x\\y\\pic.png") == "pic.png" and clean_name("") == "upload" and clean_name("a\x00b.png") == "ab.png"
    assert snap_width(None) == 256 and snap_width(200) == 256 and snap_width(5000) == 1024


# endregion

# region Files and thumbnails


def test_files_are_served_by_id_only(server):
    secret = server.services.root / "secret.txt"
    secret.write_text("secret")
    for path in ("/api/v2/files/..%2Fsecret.txt", "/api/v2/files/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd", "/api/v2/files/short", "/api/v2/files/" + "a" * 70, "/api/v2/thumbs/..%2F..%2Fsecret.txt"):
        response = server.client.get(path)
        assert response.status_code == 404, path
        assert "secret" not in response.text


def test_a_file_whose_disk_copy_vanished_is_gone(server):
    file_id = upload(server.client, png()).json()["id"]
    os.remove(server.services.files.get(file_id).path)
    assert server.client.get(f"/api/v2/files/{file_id}").status_code == 404


def test_thumbnails_are_cached_webp_with_etags(server):
    file_id = upload(server.client, png(size=(1000, 500))).json()["id"]
    response = server.client.get(f"/api/v2/thumbs/{file_id}", params={"w": 200})
    assert response.status_code == 200 and response.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(response.content)).size == (256, 128)
    assert "immutable" in response.headers["cache-control"]
    etag = response.headers["etag"]
    assert server.client.get(f"/api/v2/thumbs/{file_id}", params={"w": 200}, headers={"If-None-Match": etag}).status_code == 304
    assert server.client.get(f"/api/v2/thumbs/{file_id}", params={"w": 512}).headers["etag"] != etag
    full = server.client.get(f"/api/v2/files/{file_id}")
    assert server.client.get(f"/api/v2/files/{file_id}", headers={"If-None-Match": full.headers["etag"]}).status_code == 304
    small = upload(server.client, png(size=(40, 20), color=(1, 2, 3))).json()["id"]
    assert Image.open(io.BytesIO(server.client.get(f"/api/v2/thumbs/{small}", params={"w": 1024}).content)).size == (40, 20)


def test_text_files_have_no_thumbnail(server):
    file_id = upload(server.client, b"x, y\n", name="a.csv").json()["id"]
    assert server.client.get(f"/api/v2/thumbs/{file_id}").status_code == 415


def test_the_upload_resolver_gives_pil_images_with_their_names(server):
    from remi.ui import ParamError, nodes

    file_id = upload(server.client, png(), name="source.png").json()["id"]
    image = server.services.files.resolve_upload(nodes.Image("init_img"), file_id)
    assert image.size == (64, 48) and image.filename == "source.png"
    assert server.services.files.resolve_upload(nodes.File("f"), file_id).endswith(".png")
    with pytest.raises(ParamError):
        server.services.files.resolve_upload(nodes.Image("x"), "unknownunknown")


def test_old_uploads_are_pruned_unless_a_queued_job_needs_them(tmp_path):
    store = FileStore(tmp_path)
    keep = store.add_upload_bytes(png(color=(1, 1, 1)))
    drop = store.add_upload_bytes(png(color=(2, 2, 2)))
    assert store.prune_uploads(-1, keep={keep.id}) == 1
    assert store.get(keep.id) is not None and store.get(drop.id) is None and not Path(drop.path).exists()
    store.close()


def test_a_prune_removes_partial_uploads_a_killed_process_left(tmp_path):
    import time

    store = FileStore(tmp_path)
    store.uploads_dir.mkdir(parents=True, exist_ok=True)
    stale = store.uploads_dir / ".incoming-0123456789abcdef"
    fresh = store.uploads_dir / ".incoming-fedcba9876543210"
    stale.write_bytes(b"half an upload")
    fresh.write_bytes(b"an upload streaming in")
    old = time.time() - 2 * 3600
    os.utime(stale, (old, old))
    store.prune_uploads(3600)
    assert not stale.exists() and fresh.exists()
    store.close()


# endregion

# region Outputs gallery


def test_outputs_are_paged_and_searchable(server, tmp_path):
    files = server.services.files
    folder = tmp_path / "out"
    folder.mkdir()
    ids = []
    for index, prompt in enumerate(["red fox in snow", "blue whale", "red panda", "50% off_sale"]):
        path = folder / f"{index}.png"
        path.write_bytes(png(text=f"{prompt}\nSteps: 20"))
        os.utime(path, (1000 + index, 1000 + index))
        ids.append(files.register_output(path, tab="txt2img", infotext=f"{prompt}\nSteps: 20").id)

    page = server.client.get("/api/v2/outputs", params={"limit": 3}).json()
    assert [item["id"] for item in page["items"]] == ids[:0:-1] and page["total"] == 4
    rest = server.client.get("/api/v2/outputs", params={"cursor": page["next_cursor"]}).json()
    assert [item["id"] for item in rest["items"]] == [ids[0]] and rest["next_cursor"] is None

    assert [item["id"] for item in server.client.get("/api/v2/outputs", params={"q": "RED"}).json()["items"]] == [ids[2], ids[0]]
    assert [item["id"] for item in server.client.get("/api/v2/outputs", params={"q": "red snow"}).json()["items"]] == [ids[0]]
    assert server.client.get("/api/v2/outputs", params={"q": "%"}).json()["items"][0]["id"] == ids[3]
    assert server.client.get("/api/v2/outputs", params={"q": "_"}).json()["total"] == 1
    assert server.client.get("/api/v2/outputs", params={"cursor": "!!bad!!"}).status_code == 200

    assert server.client.delete(f"/api/v2/outputs/{ids[1]}").status_code == 204
    assert (folder / "1.png").exists() and server.client.get("/api/v2/outputs").json()["total"] == 3
    assert server.client.delete(f"/api/v2/outputs/{ids[2]}", params={"delete_file": True}).status_code == 204
    assert not (folder / "2.png").exists()
    # removed (but kept on disk) stays out of the gallery after a re-scan
    assert server.services.files.scan_outputs([str(folder)]) == 0
    assert server.client.get("/api/v2/outputs").json()["total"] == 2


def test_removing_an_output_drops_its_star_board_entry_and_lineage(server):
    """DELETE /outputs/{id} forgot the file but left its gallery rows: the board kept a
    member that no longer exists."""

    from remi.library import gallery_for

    files = server.services.files
    parent, child = (files.save_output_image(Image.new("RGB", (8, 8)), None) for _ in range(2))
    index = gallery_for(files)
    index.set_starred([child.id], True)
    board = index.create_board("keep")
    index.add_to_board(board["id"], [child.id])
    index.record_lineage(child.id, parent.id, "img2img")
    assert server.client.delete(f"/api/v2/outputs/{child.id}").status_code == 204
    for table, column in (("gallery_meta", "file_id"), ("board_items", "file_id"), ("asset_lineage", "child_id")):
        assert index._db.execute(f"SELECT count(*) FROM {table} WHERE {column} = ?", (child.id,)).fetchone() == (0,), table


def test_unsaved_outputs_are_temporary(tmp_path):
    import time

    store = FileStore(tmp_path)
    unsaved = store.save_output_image(Image.new("RGB", (8, 8)), "a cat\nSteps: 1")
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "kept.png").write_bytes(png())
    saved = store.register_output(folder / "kept.png")
    assert store.prune_unsaved(3600) == 0
    store._db.execute("UPDATE files SET created = ?", (time.time() - 7200,))
    assert store.prune_unsaved(3600) == 1
    assert store.get(unsaved.id) is None and not Path(unsaved.path).exists()
    assert store.get(saved.id) is not None
    store.close()


def test_unsaved_outputs_a_queued_job_reads_are_kept(tmp_path):
    import time

    store = FileStore(tmp_path)
    needed = store.save_output_image(Image.new("RGB", (8, 8)), None)
    other = store.save_output_image(Image.new("RGB", (8, 8)), None)
    store._db.execute("UPDATE files SET created = ?", (time.time() - 7200,))
    assert store.prune_unsaved(3600, keep={needed.id}) == 1
    assert store.get(needed.id) is not None and store.get(other.id) is None
    store.close()


def test_unsaved_outputs_of_a_starred_job_are_kept(tmp_path):
    import time

    from remi.jobs import JobStore
    from remi.jobs.store import COMPLETED

    store = FileStore(tmp_path)
    jobs = JobStore(tmp_path / "jobs.db")
    kept, pruned = (store.save_output_image(Image.new("RGB", (8, 8)), None) for _ in range(2))
    starred, plain = (jobs.add("txt2img", {}, {}) for _ in range(2))
    jobs.finish(starred.id, COMPLETED, outputs=[{"id": kept.id, "kind": "image"}, "not an output"])
    jobs.finish(plain.id, COMPLETED, outputs=[{"id": pruned.id, "kind": "image"}])
    jobs.set_starred([starred.id], True)
    assert jobs.starred_output_ids() == {kept.id}
    store._db.execute("UPDATE files SET created = ?", (time.time() - 7200,))
    assert store.prune_unsaved(3600, keep=jobs.starred_output_ids()) == 1
    assert store.get(kept.id) is not None and store.get(pruned.id) is None
    jobs.set_starred([starred.id], False)
    assert jobs.starred_output_ids() == set()
    jobs.close()
    store.close()


def test_a_data_folder_reached_through_a_symlink_still_owns_its_unsaved_outputs(tmp_path):
    import time

    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    store = FileStore(tmp_path / "link")
    unsaved = store.save_output_image(Image.new("RGB", (8, 8)), None)
    store._db.execute("UPDATE files SET created = ?", (time.time() - 7200,))
    assert store.prune_unsaved(3600) == 1 and not Path(unsaved.path).exists()
    store.close()


def test_an_outputs_folder_moved_elsewhere_through_a_symlink_still_holds_unsaved_outputs(tmp_path):
    """Outputs are stored under their real path: with ``outputs/`` a link to another disk,
    their folder was never ``outputs_dir``, so they were never pruned and a gallery
    removal kept the file and marked it removed."""

    import time

    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "outputs").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    store = FileStore(tmp_path / "data")
    try:
        old, removed = (store.save_output_image(Image.new("RGB", (8, 8)), None) for _ in range(2))
        store._db.execute("UPDATE files SET created = ? WHERE id = ?", (time.time() - 7200, old.id))
        assert store.prune_unsaved(3600) == 1 and not Path(old.path).exists()
        assert store.delete(removed.id) and not Path(removed.path).exists()
        assert store._db.execute("SELECT count(*) FROM removed").fetchone() == (0,)
    finally:
        store.close()


def test_deleting_a_file_removes_its_thumbnails(tmp_path):
    store = FileStore(tmp_path)
    record = store.save_output_image(Image.new("RGB", (300, 300)), None)
    thumbs = [store.thumbnail(record, width) for width in (64, 300)]
    other = store.save_output_image(Image.new("RGB", (300, 300)), None)
    kept = store.thumbnail(other, 64)
    assert all(thumb.exists() for thumb in thumbs)
    store.delete(record.id)
    assert not any(thumb.exists() for thumb in thumbs) and kept.exists()
    store.close()


def test_an_output_the_scan_indexed_during_its_job_gets_the_job(tmp_path):
    from remi.library import GalleryIndex

    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "1.png").write_bytes(png())
    store = FileStore(tmp_path / "data")
    index = GalleryIndex(store, folders=list)
    scanned = store.register_output(folder / "1.png", infotext="from the file")
    index.sync()
    assert index._db.execute("SELECT category FROM gallery_meta WHERE file_id = ?", (scanned.id,)).fetchone() == ("other",)
    claimed = store.register_output(folder / "1.png", job_id="job1", tab="img2img", infotext="from the job")
    assert claimed.id == scanned.id and (claimed.job_id, claimed.tab, claimed.infotext) == ("job1", "img2img", "from the job")
    assert store.get(scanned.id).job_id == "job1"
    index.sync()
    assert index._db.execute("SELECT category FROM gallery_meta WHERE file_id = ?", (scanned.id,)).fetchone() == ("img2img",)
    # a later scan does not take it over again; a later job registers the path only after writing
    # over it ("Override"), so that image gets a record of its own
    assert store.register_output(folder / "1.png").job_id == "job1"
    rewritten = store.register_output(folder / "1.png", job_id="job2", tab="txt2img")
    assert rewritten.job_id == "job2" and rewritten.id != scanned.id
    index.close()
    store.close()


def _ignores_case(folder: Path) -> bool:
    probe = folder / "CaseProbe"
    probe.write_bytes(b"")
    try:
        return (folder / "caseprobe").exists()
    finally:
        probe.unlink()


def test_one_output_under_two_spellings_is_one_record(tmp_path):
    """macOS keeps the spelling it is given on a volume that ignores case: two settings
    naming one folder differently made two gallery records of one image, and removing
    one from the gallery left the other (and the next scan brought it back)."""

    if not _ignores_case(tmp_path):
        pytest.skip("this volume tells letter case apart")
    folder = tmp_path / "Saved" / "txt2img"
    folder.mkdir(parents=True)
    (folder / "0001.png").write_bytes(png())
    alias = tmp_path / "saved" / "TXT2IMG"
    store = FileStore(tmp_path / "data")
    try:
        first = store.register_output(folder / "0001.png", job_id="job1", tab="txt2img")
        assert store.register_output(alias / "0001.png").id == first.id
        assert store.scan_outputs([alias]) == 0
        assert store._db.execute("SELECT count(*) FROM files WHERE kind = 'output'").fetchone() == (1,)
        store.delete(first.id)  # out of the gallery, kept on disk
        assert store.is_removed(alias / "0001.png")
        assert store.scan_outputs([alias, folder]) == 0
        # a job writing the image again brings it back, whichever spelling it uses
        again = store.register_output(alias / "0001.png", job_id="job2")
        assert not store.is_removed(folder / "0001.png") and store.scan_outputs([folder]) == 0
        assert store.get(again.id) is not None
    finally:
        store.close()


def test_two_spellings_registered_at_once_make_one_record(tmp_path, monkeypatch):
    """The lookup and the insert were two steps: a job saving one spelling while the
    folder scan (or the worker process, with a store of its own) indexed the other
    each found nothing and inserted a row of its own."""

    import threading

    if not _ignores_case(tmp_path):
        pytest.skip("this volume tells letter case apart")
    folder = tmp_path / "Saved"
    folder.mkdir()
    (folder / "0001.png").write_bytes(png())
    job_store, worker_store = FileStore(tmp_path / "data"), FileStore(tmp_path / "data")
    paused, go = threading.Event(), threading.Event()
    insert = job_store._insert

    def slow_insert(record, *args):  # between the job's lookup and its insert
        paused.set()
        go.wait(0.5)
        return insert(record, *args)

    monkeypatch.setattr(job_store, "_insert", slow_insert)
    found = {}
    job = threading.Thread(target=lambda: found.setdefault("job", job_store.register_output(folder / "0001.png", job_id="job1")))
    try:
        job.start()
        assert paused.wait(5)
        found["scan"] = worker_store.register_output(tmp_path / "saved" / "0001.png")
        go.set()
        job.join(5)
        assert found["job"].id == found["scan"].id
        assert worker_store._db.execute("SELECT count(*) FROM files WHERE kind = 'output'").fetchone() == (1,)
    finally:
        go.set()
        job.join(5)
        job_store.close()
        worker_store.close()


def test_an_insert_that_fails_is_retried_not_returned_as_a_record(tmp_path, monkeypatch):
    import sqlite3

    (tmp_path / "1.png").write_bytes(png())
    store = FileStore(tmp_path / "data")
    insert, failures = store._insert, []

    def fail_once(record, *args):
        if not failures:
            failures.append(record.id)
            raise sqlite3.IntegrityError("UNIQUE constraint failed: files.id")
        return insert(record, *args)

    monkeypatch.setattr(store, "_insert", fail_once)
    try:
        record = store.register_output(tmp_path / "1.png", job_id="job1")
        assert failures and store.get(record.id) is not None
    finally:
        store.close()


def test_the_output_scan_walks_each_folder_once(tmp_path):
    """Two settings naming one folder in different case, or a folder inside another
    configured one, were walked twice on every scan (realpath keeps the case it is given)."""

    from remi.files import _walk_images

    folder = tmp_path / "Saved"
    (folder / "sub").mkdir(parents=True)
    (folder / "1.png").write_bytes(png())
    (folder / "sub" / "2.png").write_bytes(png())
    (tmp_path / "Saved-link").symlink_to(folder / "sub", target_is_directory=True)  # not followed, not marked seen
    for directories in ([folder, folder / "sub"], [folder / "sub", folder], [tmp_path / "Saved-link", folder]):
        found = list(_walk_images(directories))
        assert sorted(os.path.basename(path) for path in found) == ["1.png", "2.png"], directories
    if _ignores_case(tmp_path):
        found = list(_walk_images([folder, tmp_path / "saved"]))
        assert sorted(os.path.basename(path) for path in found) == ["1.png", "2.png"]


def test_two_files_that_differ_only_in_case_stay_two_records(tmp_path):
    if _ignores_case(tmp_path):
        pytest.skip("this volume ignores letter case")
    (tmp_path / "A.png").write_bytes(png())
    (tmp_path / "a.png").write_bytes(png(color=(1, 2, 3)))
    store = FileStore(tmp_path / "data")
    try:
        upper = store.register_output(tmp_path / "A.png")
        assert store.register_output(tmp_path / "a.png").id != upper.id
        store.delete(upper.id)
        assert store.is_removed(tmp_path / "A.png") and not store.is_removed(tmp_path / "a.png")
    finally:
        store.close()


def test_the_category_follows_its_folder_under_another_spelling(tmp_path):
    from remi.library import category_of

    if not _ignores_case(tmp_path):
        pytest.skip("this volume tells letter case apart")
    grids = tmp_path / "Outputs" / "txt2img-grids"
    grids.mkdir(parents=True)
    (grids / "grid.png").write_bytes(png())
    folders = [("grids", os.path.realpath(tmp_path / "outputs" / "txt2img-grids"))]
    assert category_of(str(grids / "grid.png"), "txt2img", folders) == "grids"
    assert category_of(str(tmp_path / "Outputs" / "other.png"), "txt2img", folders) == "txt2img"


def test_media_files_are_not_gzipped(server):
    file_id = upload(server.client, png(size=(512, 512))).json()["id"]
    served = server.client.get(f"/api/v2/files/{file_id}", headers={"Accept-Encoding": "gzip"})
    assert served.status_code == 200 and "content-encoding" not in served.headers
    listing = server.client.get("/api/v2/openapi-is-not-here", headers={"Accept-Encoding": "gzip"})
    assert listing.status_code == 404


def test_gallery_save_writes_copies_a_log_and_a_zip(server, webui, tmp_path, monkeypatch):
    from modules import shared

    folder = tmp_path / "gen"
    folder.mkdir()
    (folder / "a.png").write_bytes(png(text="a red fox\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 42, Size: 64x48"))
    record = server.services.files.register_output(folder / "a.png", tab="txt2img", infotext="a red fox\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 42, Size: 64x48")
    save_dir = tmp_path / "saved"
    monkeypatch.setattr(shared.opts, "outdir_save", str(save_dir), raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    response = server.client.post("/api/v2/outputs/save", json={"ids": [record.id], "zip": True})
    assert response.status_code == 200, response.text
    result = response.json()
    [name] = result["saved"]
    assert (save_dir / name).is_file()
    log = (save_dir / "log.csv").read_text().splitlines()
    assert log[0].startswith("prompt,seed") and log[1].startswith("a red fox,42")
    archive = server.client.get(f"/api/v2/files/{result['zip']}", params={"download": True})
    assert archive.status_code == 200 and archive.content[:2] == b"PK"
    assert server.client.post("/api/v2/outputs/save", json={"ids": ["missingmissing"]}).status_code == 404


# The Save folder setting can be cleared on the settings page: Save then answered 500 (no folder named "")
def test_gallery_save_with_the_folder_cleared_uses_its_default(server, webui, tmp_path, monkeypatch):
    from modules import shared

    folder = tmp_path / "gen"
    folder.mkdir()
    (folder / "a.png").write_bytes(png(text="a red fox\nSteps: 20, Seed: 42, Size: 64x48"))
    record = server.services.files.register_output(folder / "a.png", tab="txt2img", infotext="a red fox\nSteps: 20, Seed: 42, Size: 64x48")
    default = tmp_path / "default-save"
    monkeypatch.setattr(shared.opts.data_labels["outdir_save"], "default", str(default))
    monkeypatch.setattr(shared.opts, "outdir_save", "", raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    response = server.client.post("/api/v2/outputs/save", json={"ids": [record.id]})
    assert response.status_code == 200, response.text
    [name] = response.json()["saved"]
    assert (default / name).is_file()


def test_open_folder_respects_hide_ui_dir_config(server, webui, monkeypatch):
    from modules import shared
    from remi.server import gallery

    opened = []
    monkeypatch.setattr(gallery, "open_folder", opened.append)
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    assert server.client.post("/api/v2/outputs/open-folder", json={"folder": "txt2img"}).status_code == 403
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    assert server.client.post("/api/v2/outputs/open-folder", json={"folder": "txt2img"}).json()["ok"] is True
    assert opened == [gallery.output_folder("txt2img")]
    assert server.client.post("/api/v2/outputs/open-folder", json={"folder": "/etc"}).status_code == 422


def test_output_folders_are_indexed(server, tmp_path, monkeypatch):
    from remi.server import services

    folder = tmp_path / "older"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.png").write_bytes(png(text="old cat\nSteps: 3, Seed: 1, Size: 64x48"))
    (folder / "notes.txt").write_text("x")
    monkeypatch.setattr(services, "output_dirs", lambda: [str(folder), str(folder)])
    assert server.client.post("/api/v2/outputs/refresh").json() == {"added": 1}
    assert server.client.post("/api/v2/outputs/refresh").json() == {"added": 0}
    [item] = server.client.get("/api/v2/outputs", params={"q": "cat"}).json()["items"]
    assert item["name"] == "a.png" and (item["width"], item["height"]) == (64, 48)


# endregion

# region Networks


# An image's own text can hold an unpaired surrogate (a NovelAI comment is JSON with escapes), and on
# Linux a file name that is not UTF-8 reads as one: the scan stopped at that file (Refresh answered 500,
# housekeeping logged an error every hour, and no image after it was ever indexed)
def test_output_scan_goes_on_past_text_utf8_cannot_hold(server, tmp_path, monkeypatch):
    from remi import files as files_module
    from remi.server import services

    folder = tmp_path / "outputs"
    folder.mkdir()
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "NovelAI")
    info.add_text("Description", "a cat")
    info.add_text("Comment", '{"uc": "bad \\udc80", "sampler": "k_euler", "steps": 20, "scale": 7, "seed": 1}')
    Image.new("RGB", (16, 16)).save(folder / "a-novelai.png", pnginfo=info)
    Image.new("RGB", (16, 16)).save(folder / "b.png")
    walk = files_module._walk_images
    monkeypatch.setattr(files_module, "_walk_images", lambda directories: iter([str(folder / "0-\udcff.png"), *walk(directories)]))
    monkeypatch.setattr(services, "output_dirs", lambda: [str(folder)])
    assert server.client.post("/api/v2/outputs/refresh").json() == {"added": 2}
    records = {record.name: record for record in server.services.files.list_outputs()[0]}
    assert sorted(records) == ["a-novelai.png", "b.png"]
    assert "bad \ufffd" in records["a-novelai.png"].infotext


# On Linux a file name that is not UTF-8 reads as text with surrogates: no row can hold it, and the
# network listing that registers each preview raised for all of them
def test_a_file_name_that_is_not_utf8_is_not_registered(tmp_path, monkeypatch):
    from remi import files as files_module

    store = FileStore(tmp_path / "data")
    try:
        preview = tmp_path / "preview.png"
        preview.write_bytes(png())
        found = os.stat(preview)
        stat = os.stat
        monkeypatch.setattr(files_module.os.path, "realpath", lambda path: str(path))
        monkeypatch.setattr(files_module.os, "stat", lambda path, *args, **kwargs: found if "\udcff" in str(path) else stat(path, *args, **kwargs))
        assert store.register_path(str(tmp_path / "pre\udcffview.png")) is None
        assert store.register_path(str(preview)) is not None
    finally:
        store.close()


class FakeProvider:
    def __init__(self, root):
        self.root = root
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1

    def list(self):
        return [NetworkEntry("lora", path.stem, str(path), title=path.stem, root=str(self.root), hash=path.stem[:4], prompt=f"<lora:{path.stem}:1>") for path in sorted(self.root.rglob("*.safetensors"))]


def safetensors(path, metadata):
    import json

    header = json.dumps({"__metadata__": metadata}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header)


@pytest.fixture
def networks(server, tmp_path):
    root = tmp_path / "Lora"
    for name in ("anime_style", "zeta_light", "alpha"):
        safetensors(root / ("sub" if name == "zeta_light" else "") / f"{name}.safetensors", {"ss_output_name": name, "ss_network_dim": "16"})
    (root / "alpha.preview.png").write_bytes(png(color=(9, 9, 9)))
    (root / "alpha.txt").write_text("the first letter")
    provider = FakeProvider(root)
    server.services.networks = NetworkIndex(server.services.files, lambda: {"lora": provider})
    return provider


def test_networks_are_searched_paged_and_never_expose_absolute_paths(server, networks):
    client = server.client
    assert [kind["id"] for kind in client.get("/api/v2/networks").json()] == ["lora"]
    page = client.get("/api/v2/networks/lora", params={"limit": 2}).json()
    assert [item["name"] for item in page["items"]] == ["alpha", "anime_style"] and page["total"] == 3
    rest = client.get("/api/v2/networks/lora", params={"cursor": page["next_cursor"]}).json()
    assert [item["name"] for item in rest["items"]] == ["zeta_light"] and rest["items"][0]["path"] == "sub/zeta_light.safetensors"
    assert [item["name"] for item in client.get("/api/v2/networks/lora", params={"q": "SUB light"}).json()["items"]] == ["zeta_light"]
    assert client.get("/api/v2/networks/lora", params={"sort": "name", "order": "desc"}).json()["items"][0]["name"] == "zeta_light"
    alpha = page["items"][0]
    assert alpha["description"] == "the first letter" and alpha["prompt"] == "<lora:alpha:1>"
    assert client.get(f"/api/v2/thumbs/{alpha['preview']}").status_code == 200
    assert str(networks.root) not in client.get("/api/v2/networks/lora").text
    assert client.get("/api/v2/networks/nope").status_code == 404


# A model's header or its sidecar can hold an unpaired surrogate (JSON escapes, written by another tool):
# its details answered 500, saving its notes wrote the sidecar and then answered 500, and a merge was refused
def test_network_metadata_with_text_utf8_cannot_hold_is_shown_and_edited(server, networks):
    import json

    root = networks.root
    safetensors(root / "broken.safetensors", {"ss_output_name": "x\udc80", "ss_tag_frequency": json.dumps({"set": {"t\udc80g": 3}})})
    (root / "alpha.json").write_text('{"notes": "n\\udc80"}', encoding="utf8")
    server.services.networks.refresh("lora")
    client = server.client
    meta = client.get("/api/v2/networks/lora/broken/meta")
    assert meta.status_code == 200, meta.text
    assert meta.json()["metadata"]["ss_output_name"] == "x\ufffd"
    edited = client.patch("/api/v2/networks/lora/broken/meta", json={"user_metadata": {"description": "d"}})
    assert edited.status_code == 200, edited.text
    merged = client.patch("/api/v2/networks/lora/alpha/meta", json={"user_metadata": {"description": "d"}})
    assert merged.status_code == 200, merged.text
    assert merged.json()["user_metadata"] == {"notes": "n\ufffd", "description": "d"}
    assert client.get("/api/v2/networks/lora").status_code == 200


def test_network_metadata_and_user_metadata_editing(server, networks):
    client = server.client
    meta = client.get("/api/v2/networks/lora/alpha/meta").json()
    assert meta["metadata"] == {"ss_output_name": "alpha", "ss_network_dim": "16"} and meta["user_metadata"] == {}
    edited = client.patch("/api/v2/networks/lora/alpha/meta", json={"user_metadata": {"description": "edited", "activation text": "abc", "preferred weight": 0.7}}).json()
    assert edited["user_metadata"]["activation text"] == "abc" and edited["description"] == "edited"
    removed = client.patch("/api/v2/networks/lora/alpha/meta", json={"user_metadata": {"activation text": None}}).json()
    assert "activation text" not in removed["user_metadata"] and removed["user_metadata"]["description"] == "edited"
    assert (networks.root / "alpha.json").is_file()
    assert client.get("/api/v2/networks/lora/missing/meta").status_code == 404
    too_big = client.patch("/api/v2/networks/lora/alpha/meta", json={"user_metadata": {"notes": "x" * 70000}})
    assert too_big.status_code == 422


def test_network_previews_and_refresh(server, networks):
    client = server.client
    file_id = upload(client, png(color=(200, 0, 0))).json()["id"]
    item = client.put("/api/v2/networks/lora/anime_style/preview", json={"file_id": file_id}).json()
    assert item["preview"] and (networks.root / "anime_style.preview.png").is_file()
    # a new preview gets a new id (browsers cache files by id), revalidated
    other = upload(client, png(color=(0, 200, 0))).json()["id"]
    replaced = client.put("/api/v2/networks/lora/anime_style/preview", json={"file_id": other}).json()
    assert replaced["preview"] != item["preview"]
    assert client.get("/api/v2/networks/lora", params={"q": "anime"}).json()["items"][0]["preview"] == replaced["preview"]
    assert client.get(f"/api/v2/files/{replaced['preview']}").headers["cache-control"] == "private, no-cache"
    assert client.put("/api/v2/networks/lora/anime_style/preview", json={"file_id": "missingmissing"}).status_code == 404
    assert client.post("/api/v2/networks/lora/refresh").json()["ok"] is True and networks.refreshed == 1


def test_previews_are_converted_and_downscaled(tmp_path):
    from remi.networks import PREVIEW_SIZE, preview_image

    cmyk = preview_image(Image.new("CMYK", (3000, 1500)))
    assert cmyk.mode == "RGB" and cmyk.size == (PREVIEW_SIZE, PREVIEW_SIZE // 2)
    palette = Image.new("P", (10, 10))
    palette.info["transparency"] = 0
    assert preview_image(palette).mode == "RGBA"
    assert preview_image(Image.new("I;16", (10, 10))).mode == "RGB"


def test_network_items_are_cached_until_refreshed(server, networks, monkeypatch):
    from remi import networks as module

    reads = []
    original = module.read_user_metadata
    monkeypatch.setattr(module, "read_user_metadata", lambda entry: reads.append(entry.name) or original(entry))
    server.client.get("/api/v2/networks/lora")
    count = len(reads)
    server.client.get("/api/v2/networks/lora")
    assert len(reads) == count
    server.client.post("/api/v2/networks/lora/refresh")
    server.client.get("/api/v2/networks/lora")
    assert len(reads) > count


def test_file_store_rejects_unreadable_images(tmp_path):
    store = FileStore(tmp_path)
    with pytest.raises(FileError) as error:
        store.add_upload_bytes(b"GIF89a" + b"\x00" * 10)
    assert error.value.status == 415
    store.close()


# endregion


def test_a_prune_never_deletes_an_upload_re_uploaded_meanwhile(tmp_path):
    # The hourly prune lists the old uploads, then deletes them one by one: a
    # re-upload of the same content in between keeps its record and file.
    import time

    store = FileStore(tmp_path)
    first = store.add_upload_bytes(png(color=(7, 7, 7)))
    store._db.execute("UPDATE files SET created = ? WHERE id = ?", (time.time() - 30 * 86400, first.id))
    delete = store.delete

    def reupload_then_delete(file_id, **kwargs):
        assert store.add_upload_bytes(png(color=(7, 7, 7))).id == first.id
        return delete(file_id, **kwargs)

    store.delete = reupload_then_delete
    assert store.prune_uploads(86400) == 0
    assert store.get(first.id) is not None and Path(first.path).exists()
    store.close()


def test_an_output_a_scan_indexes_during_the_insert_is_claimed_by_its_job(tmp_path, monkeypatch):
    folder = tmp_path / "outputs"
    folder.mkdir()
    (folder / "1.png").write_bytes(png())
    store = FileStore(tmp_path / "data")
    insert = store._insert

    def scan_first(record, *args):  # the folder scan wins the insert race
        insert(dataclasses.replace(record, id="scannedScannedScanned", job_id=None, tab=None, infotext=None), *args)
        return insert(record, *args)

    monkeypatch.setattr(store, "_insert", scan_first)
    claimed = store.register_output(folder / "1.png", job_id="job1", tab="txt2img", infotext="from the job")
    assert (claimed.id, claimed.job_id, claimed.tab, claimed.infotext) == ("scannedScannedScanned", "job1", "txt2img", "from the job")
    assert store.get("scannedScannedScanned").job_id == "job1"
    store.close()


def test_a_job_writing_over_another_jobs_output_gets_a_record_of_its_own(tmp_path):
    # A fixed file name ([seed], no number) with "Override": the second job's image replaces the
    # first's on disk. It must not come back as the first job's record (old size, infotext, date),
    # nor share its id, or deleting the old job with its files would delete the new image.
    folder = tmp_path / "outputs"
    folder.mkdir()
    path = folder / "123.png"
    path.write_bytes(png(size=(64, 64)))
    store = FileStore(tmp_path / "data")
    old = store.register_output(path, job_id="job1", infotext="seed 123 old", created=1_000_000.0)

    path.write_bytes(png(size=(128, 96)))
    new = store.register_output(path, job_id="job2", infotext="seed 123 new")

    assert new.id != old.id and store.get(old.id) is None
    assert (new.job_id, new.infotext, new.width, new.height) == ("job2", "seed 123 new", 128, 96)
    assert new.created > 1_000_000.0 and path.exists()
    assert store.register_output(path, job_id="job2").id == new.id  # the same job again: idempotent
    store.close()


def test_a_claimed_output_takes_its_files_time(tmp_path):
    # The scan indexed an older file at this path (its time); the job then wrote over it
    folder = tmp_path / "outputs"
    folder.mkdir()
    path = folder / "1.png"
    path.write_bytes(png())
    store = FileStore(tmp_path / "data")
    scanned = store.register_output(path, created=1_000_000.0)
    assert scanned.job_id is None and scanned.created == 1_000_000.0

    claimed = store.register_output(path, job_id="job3")
    assert claimed.id == scanned.id and claimed.created == os.stat(path).st_mtime
    assert store.get(scanned.id).created == claimed.created
    store.close()


def test_a_cursor_too_long_for_int_starts_at_the_first_page():
    from remi.files import decode_offset, encode_cursor

    assert decode_offset(encode_cursor("1" * 5000)) == 0


def test_starred_and_boarded_unsaved_outputs_survive_housekeeping(tmp_path):
    import time

    from remi.jobs import run_inline
    from remi.library import GalleryIndex
    from remi.server.services import Services

    services = Services.create(tmp_path, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        store = services.files
        starred, boarded, plain = (store.save_output_image(Image.new("RGB", (8, 8)), None) for _ in range(3))
        index = GalleryIndex(store)
        try:
            index.set_starred([starred.id], True)
            board = index.create_board("keep")
            index.add_to_board(board["id"], [boarded.id])
        finally:
            index.close()
        store._db.execute("UPDATE files SET created = ?", (time.time() - 30 * 86400,))
        services.housekeeping()
        assert store.get(starred.id) is not None and Path(starred.path).exists()
        assert store.get(boarded.id) is not None and Path(boarded.path).exists()
        assert store.get(plain.id) is None
    finally:
        services.close()


def test_a_file_deleted_from_disk_answers_404_not_500(tmp_path):
    with server_app(tmp_path) as server:
        record = server.services.files.save_output_image(Image.new("RGB", (8, 8)), None)
        os.remove(record.path)
        assert server.client.get(f"/api/v2/files/{record.id}").status_code == 404
        assert server.client.get(f"/api/v2/thumbs/{record.id}").status_code == 404


def test_a_jobs_outputs_are_found_through_an_index(tmp_path):
    store = FileStore(tmp_path)
    try:
        plan = store._db.execute("EXPLAIN QUERY PLAN SELECT id FROM files WHERE job_id = ? AND kind = 'output' AND id != ? ORDER BY created", ("j", "f")).fetchall()
        assert "files_job" in " ".join(str(row[-1]) for row in plan)
    finally:
        store.close()


class _FailingConnection:
    """A connection whose listed statements fail (``COMMIT``: a disk that filled up)."""

    def __init__(self, db, *fail):
        self.db = db
        self.fail = {sql.upper() for sql in fail}

    def execute(self, sql, *args):
        if sql.strip().upper() in self.fail:
            raise sqlite3.OperationalError(f"{sql} failed")
        return self.db.execute(sql, *args)

    @property
    def in_transaction(self):
        return self.db.in_transaction


def test_a_failed_commit_rolls_back_and_leaves_the_connection_usable(tmp_path):
    db = connect(tmp_path / "t.db")
    try:
        db.execute("CREATE TABLE t (x INTEGER)")
        proxy = _FailingConnection(db, "COMMIT")
        with pytest.raises(sqlite3.OperationalError, match="COMMIT"), transaction(proxy):
            proxy.execute("INSERT INTO t VALUES (1)")
        assert not db.in_transaction
        with transaction(db):
            db.execute("INSERT INTO t VALUES (2)")
        assert [row[0] for row in db.execute("SELECT x FROM t")] == [2]
    finally:
        db.close()


def test_a_failed_rollback_does_not_hide_the_error_that_caused_it(tmp_path):
    db = connect(tmp_path / "t.db")
    try:
        proxy = _FailingConnection(db, "ROLLBACK")
        with pytest.raises(ValueError, match="the body"), transaction(proxy, "BEGIN"):
            raise ValueError("the body failed")
        db.execute("ROLLBACK")
    finally:
        db.close()


@pytest.mark.parametrize("which", ["recipes", "prompt_tools", "chains"])
def test_a_store_whose_commit_failed_takes_the_next_transaction(tmp_path, which):
    """The stores' multi-row writes: a COMMIT that fails must not leave the
    shared connection inside the transaction (every later BEGIN would fail)."""

    from remi.jobs.chains import ChainStore, StageSpec
    from remi.prompt_tools.store import PromptToolsStore
    from remi.recipes import RecipeStore

    if which == "recipes":
        store = RecipeStore(tmp_path / "recipes.db")
        write = lambda: store.import_recipes({"recipes": [{"name": f"r{len(store.list())}"}]})  # noqa: E731
    elif which == "prompt_tools":
        store = PromptToolsStore(tmp_path / "prompt-tools.db")
        write = lambda: store.set_meta("k", "v")  # noqa: E731
    else:
        store = ChainStore(tmp_path / "chains.db")
        write = lambda: store.create_chain([StageSpec(kind="generate")])  # noqa: E731
    try:
        real = store._db
        store._db = _FailingConnection(real, "COMMIT")
        with pytest.raises(sqlite3.OperationalError):
            write()
        store._db = real
        assert not real.in_transaction
        write()
    finally:
        store.close()
