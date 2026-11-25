"""remi.pages: the page registry, schema export (schema-format.md §8), page
actions and tasks through the server, and the script life cycle."""

import io
import threading
import time

import pytest
from PIL import Image

from remi import pages, ui
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture(autouse=True)
def registry():
    pages.clear()
    yield
    pages.clear()


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path) as server:
        yield server


class Echo(pages.Page):
    """A small schema page with an image, actions and a task."""

    gate: threading.Event

    def params(self):
        return ui.Column(
            ui.Image("image", label="Image"),
            ui.Text("text", "hello", label="Text"),
            ui.Choice("mode", ["a", "b"], "a", label="Mode"),
            ui.Button("Run", "run"),
            ui.Output("result", kind="text", label="Result"),
        )

    @ui.action("size")
    def size(self, ctx, values):
        resolved = self.values(ctx, values)
        image = resolved["image"]
        return {"__outputs__": {"result": f"{image.width}x{image.height}" if image else "none"}, "text": {"value": resolved["text"].upper()}}

    @ui.action("run")
    def run(self, ctx, values):
        resolved = self.values(ctx, values, uploads=False)

        def work(task):
            task.progress(0.5, "Step {{n}}", n=1)
            self.gate.wait(5)
            task.check()
            return {"__outputs__": {"result": resolved["text"] * 2}}

        return self.start_task(ctx, "run", work, gpu=True)


def _png(width=24, height=16):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, "PNG")
    return buffer.getvalue()


def test_page_validation():
    with pytest.raises(ui.SchemaError):
        pages.Page("Bad Id", "x")
    with pytest.raises(ui.SchemaError):
        pages.Page("x", "")
    with pytest.raises(ui.SchemaError):
        pages.Page("x", "X", params=ui.Text("a"), widget="w")
    with pytest.raises(ui.SchemaError):
        pages.Page("x", "X", params=ui.Text("a"), send_to=[pages.SendTarget("a", "X")]).layout()
    with pytest.raises(ui.SchemaError):
        pages.register(object())


def test_a_page_is_attributed_to_the_file_that_builds_it_past_a_base_class_elsewhere():
    """Extensions → Diagnostics counts a page's failures against the extension that built it.
    A page class subclassing another extension's ``Page`` subclass was counted against the base
    class's extension: its ``__init__`` was taken for the code building the page."""

    base_file, own_file = "/x/extensions/base-ext/scripts/base.py", "/x/extensions/own-ext/scripts/own.py"
    base = {"__file__": base_file, "__name__": "base_ext_script"}
    exec(compile("from remi import pages\nclass BasePage(pages.Page):\n    def __init__(self, id, title, **kw):\n        super().__init__(id, title, **kw)\ndef make(id):\n    return BasePage(id, 'Made')\n", base_file, "exec"), base)
    own = {"__file__": own_file, "__name__": "own_ext_script", "BasePage": base["BasePage"], "make": base["make"]}
    exec(compile("class MyPage(BasePage):\n    def __init__(me, id):\n        super().__init__(id, 'Mine')\nclass Plain(BasePage):\n    pass\nmine, plain, made = MyPage('mine'), Plain('plain', 'Plain'), make('made')\n", own_file, "exec"), own)
    assert (own["mine"].filename, own["plain"].filename) == (own_file, own_file)
    assert own["made"].filename == base_file  # built by the other extension's code: that one's


def test_schema_export():
    page = pages.register(Echo("echo", "Echo", "wrench", send_to=[pages.SendTarget("image", "Echo")]))
    data = pages.page_schema("echo")
    assert data["version"] == 1 and data["owner"] == "page:echo" and data["icon"] == "wrench" and data["widget"] is None
    assert data["defaults"] == {"image": None, "text": "hello", "mode": "a"}
    assert data["layout"]["type"] == "column" and [child["type"] for child in data["layout"]["children"]] == ["image", "text", "choice", "button", "output"]
    assert data["send_to"] == [{"kind": "image", "param": "image", "label": "Echo"}]
    assert pages.registered() == [page] and pages.get("echo") is page and pages.get("nope") is None
    with pytest.raises(KeyError):
        pages.page_schema("nope")
    widget = pages.Page("view", "View", widget="view").schema()
    assert widget["layout"] is None and widget["defaults"] == {} and widget["widget"] == "view"


def test_callable_params_are_built_per_schema():
    calls = []

    def build():
        calls.append(1)
        return ui.Choice("pick", [f"c{len(calls)}"], None)

    page = pages.Page("dyn", "Dyn", params=build)
    assert page.schema()["defaults"] == {"pick": "c1"}
    assert page.schema()["defaults"] == {"pick": "c2"}


def test_actions_from_an_object_and_a_module():
    import types

    class Handlers:
        prefix = ">"

        @ui.action("hello")
        def hello(self, ctx, values):
            return ui.toast(self.prefix + values.get("name", ""))

    module = types.ModuleType("handlers")

    @ui.action("ping")
    def ping(ctx, values):
        return {"name": {"value": "pong"}}

    module.ping = ping
    page = pages.Page("acts", "Acts", params=ui.Text("name"), actions=Handlers())
    other = pages.Page("acts2", "Acts", params=ui.Text("name"), actions=module)
    ctx = ui.ActionContext(owner="page:acts")
    assert ui.call_action(page, "hello", ctx, {"name": "x"}) == {"__toast__": {"level": "info", "message": ">x"}}
    assert ui.call_action(other, "ping", ctx, {}) == {"name": {"value": "pong"}}
    assert "hello" not in ui.actions_of(other) and {"__task__", "__cancel__"} <= set(ui.actions_of(page))
    assert isinstance(page, pages.Page) and type(page) is not type(other)

    class Clash:
        @ui.action("__task__")
        def nope(self, ctx, values):
            return {}

    with pytest.raises(ui.SchemaError):
        pages.Page("clash", "Clash", actions=Clash())


def test_pages_follow_the_script_life_cycle(webui):
    from modules import script_callbacks

    page = pages.register(pages.Page("cycle", "Cycle", params=ui.Text("a")))
    assert pages.get("cycle") is page
    script_callbacks.script_unloaded_callback()
    assert pages.get("cycle") is None
    # a stale unload hook never removes a newer registration
    first = pages.register(pages.Page("cycle", "One", params=ui.Text("a")))
    second = pages.register(pages.Page("cycle", "Two", params=ui.Text("a")))
    pages.unregister("cycle", first)
    assert pages.get("cycle") is second


def test_server_lists_and_serves_pages(server):
    pages.register(Echo("echo", "Echo", "wrench", send_to=[pages.SendTarget("image", "Echo")]))
    pages.register(pages.Page("view", "View", widget="view"))
    app = server.client.get("/api/v2/app").json()
    assert app["pages"] == [{"id": "echo", "title": "Echo", "icon": "wrench", "widget": None}, {"id": "view", "title": "View", "icon": None, "widget": "view"}]
    assert "pages" in app["features"]
    assert server.client.get("/api/v2/pages").json() == app["pages"]
    schema = server.client.get("/api/v2/pages/echo").json()
    assert schema["owner"] == "page:echo" and schema["send_to"] == [{"kind": "image", "param": "image", "label": "Echo"}]
    assert server.client.get("/api/v2/pages/nope").status_code == 404


def test_page_actions_resolve_uploads(server):
    pages.register(Echo("echo", "Echo"))
    upload = server.client.post("/api/v2/uploads", files={"file": ("a.png", _png(), "image/png")}).json()
    answer = server.client.post("/api/v2/actions/page:echo/size", json={"tab": "page:echo", "values": {"image": upload["id"], "text": "abc", "stale": 1}})
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"__outputs__": {"result": "24x16"}, "text": {"value": "ABC"}}
    bad = server.client.post("/api/v2/actions/page:echo/size", json={"tab": "page:echo", "values": {"image": "missingmissing"}})
    assert bad.status_code == 422 and bad.json()["errors"][0]["param"] == "image"
    assert server.client.post("/api/v2/actions/page:echo/nope", json={"values": {}}).status_code == 404
    assert server.client.post("/api/v2/actions/page:nope/size", json={"values": {}}).status_code == 404


def test_a_gpu_action_with_a_timeout_runs_without_the_limit_and_keeps_its_isolation(server):
    class Handlers:
        @ui.action("slow", gpu=True, timeout=0.05)
        def slow(self, ctx, values):
            time.sleep(0.2)  # past the limit: a GPU action is not cut off (its thread would keep the GPU)
            return {"__outputs__": {"result": "done"}}

        @ui.action("broken", gpu=True, timeout=5)
        def broken(self, ctx, values):
            raise RuntimeError("boom")

    pages.register(pages.Page("gpu", "GPU", params=ui.Text("a"), actions=Handlers()))
    slow = server.client.post("/api/v2/actions/page:gpu/slow", json={"tab": "page:gpu", "values": {}})
    assert slow.status_code == 200 and slow.json() == {"__outputs__": {"result": "done"}}
    broken = server.client.post("/api/v2/actions/page:gpu/broken", json={"tab": "page:gpu", "values": {}})
    assert broken.status_code == 200 and broken.json()["__toast__"]["level"] == "error"


def test_gpu_requests_past_the_waiting_limit_are_refused(server):
    """Each GPU action waits for the generation in a worker thread of the
    server; past a few, more are refused instead of taking every thread."""

    class Handlers:
        @ui.action("run", gpu=True)
        def run(self, ctx, values):
            return {"__outputs__": {"result": "ran"}}

    pages.register(pages.Page("gpu", "GPU", params=ui.Text("a"), actions=Handlers()))
    slots = server.services.gpu_requests
    taken = 0
    while slots.acquire(blocking=False):
        taken += 1
    try:
        busy = server.client.post("/api/v2/actions/page:gpu/run", json={"tab": "page:gpu", "values": {}})
        assert busy.status_code == 503 and taken >= 1
    finally:
        for _ in range(taken):
            slots.release()
    ran = server.client.post("/api/v2/actions/page:gpu/run", json={"tab": "page:gpu", "values": {}})
    assert ran.status_code == 200 and ran.json() == {"__outputs__": {"result": "ran"}}
    assert slots.acquire(blocking=False)  # the request gave its slot back
    slots.release()


def _poll(server, body, until):
    deadline = time.time() + 5
    while time.time() < deadline:
        answer = server.client.post("/api/v2/actions/page:echo/__task__", json=body).json()
        status = answer["__outputs__"]["task:run"]
        if until(status):
            return answer
        time.sleep(0.02)
    raise AssertionError(f"task stuck: {status}")


def test_tasks_report_progress_and_their_result_once(server):
    page = pages.register(Echo("echo", "Echo"))
    page.gate = threading.Event()
    started = server.client.post("/api/v2/actions/page:echo/run", json={"tab": "page:echo", "values": {"text": "ab"}}).json()
    status = started["__outputs__"]["task:run"]
    assert status["state"] in ("queued", "running") and not status["done"]
    running = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["progress"] == 0.5)
    assert running["__outputs__"]["task:run"]["text"] == {"text": "Step {{n}}", "params": {"n": 1}}
    again = server.client.post("/api/v2/actions/page:echo/run", json={"values": {"text": "zz"}}).json()
    assert again["__toast__"]["level"] == "warning" and again["__outputs__"]["task:run"]["id"] == status["id"]
    page.gate.set()
    done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
    assert done["__outputs__"]["result"] == "abab" and done["__outputs__"]["task:run"]["state"] == "completed"
    resumed = server.client.post("/api/v2/actions/page:echo/__task__", json={"values": {}, "task": "run", "id": status["id"], "resume": True}).json()
    assert "result" not in resumed["__outputs__"]
    unknown = server.client.post("/api/v2/actions/page:echo/__task__", json={"values": {}, "task": "other"}).json()
    assert unknown == {"__outputs__": {"task:other": None}}
    assert server.client.post("/api/v2/actions/page:echo/__task__", json={"values": {}}).status_code == 422


def test_a_view_mounting_after_the_task_ended_gets_its_result_once(server):
    """A reload (or another browser) resumes: a result no view has received yet comes with it."""

    page = pages.register(Echo("echo", "Echo"))
    page.gate = threading.Event()
    page.gate.set()
    status = server.client.post("/api/v2/actions/page:echo/run", json={"values": {"text": "xy"}}).json()["__outputs__"]["task:run"]
    task = page.task(ui.ActionContext(owner=page.owner), "run")
    deadline = time.time() + 5
    while not task.done and time.time() < deadline:
        time.sleep(0.01)
    assert page.busy() is False
    body = {"values": {}, "task": "run", "resume": True}
    resumed = server.client.post("/api/v2/actions/page:echo/__task__", json=body).json()
    assert resumed["__outputs__"]["result"] == "xyxy" and resumed["__outputs__"]["task:run"]["state"] == "completed"
    again = server.client.post("/api/v2/actions/page:echo/__task__", json=body).json()
    assert "result" not in again["__outputs__"]  # delivered: a later reload does not apply it again
    asked = server.client.post("/api/v2/actions/page:echo/__task__", json={"values": {}, "task": "run", "id": status["id"]}).json()
    assert asked["__outputs__"]["result"] == "xyxy"  # the follower of that task always gets it


def test_busy_reports_queued_and_running_tasks(server):
    page = pages.register(Echo("echo", "Echo"))
    page.gate = threading.Event()
    assert not page.busy()
    status = server.client.post("/api/v2/actions/page:echo/run", json={"values": {}}).json()["__outputs__"]["task:run"]
    assert page.busy() and page.busy("run") and not page.busy("other")
    page.gate.set()
    _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
    assert not page.busy()


def test_pages_follow_the_tab_order_and_hidden_tabs_settings(server, webui, monkeypatch):
    from remi.server import runtime

    for page_id, title in (("one", "One"), ("two", "Two"), ("three", "Three")):
        pages.register(pages.Page(page_id, title, params=ui.Text("a")))
    assert runtime._page_titles() == ["One", "Two", "Three"]  # choices of both settings
    monkeypatch.setattr(webui.opts, "ui_tab_order", ["txt2img", "Three", "One"], raising=False)
    monkeypatch.setattr(webui.opts, "hidden_tabs", ["Two"], raising=False)
    assert [page.id for page in pages.registered()] == ["three", "one"]
    assert [page.id for page in pages.registered(include_hidden=True)] == ["one", "two", "three"]
    assert [page["id"] for page in server.client.get("/api/v2/app").json()["pages"]] == ["three", "one"]
    assert pages.get("two") is not None  # hidden from the navigation only


def test_tasks_wait_for_generation_and_can_be_cancelled(server):
    page = pages.register(Echo("echo", "Echo"))
    page.gate = threading.Event()
    with server.services.queue.lock:
        status = server.client.post("/api/v2/actions/page:echo/run", json={"values": {}}).json()["__outputs__"]["task:run"]
        waiting = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["text"] is not None)
        assert waiting["__outputs__"]["task:run"]["state"] == "queued"
        cancel = server.client.post("/api/v2/actions/page:echo/__cancel__", json={"values": {}, "task": "run"}).json()
        assert cancel["__outputs__"]["task:run"]["cancelling"] is True
        done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
    assert done["__outputs__"]["task:run"]["state"] == "cancelled" and done["__toast__"]["message"] == "Cancelled"


def test_tasks_wait_behind_the_webui_generation_lock(webui, tmp_path):
    """The real generation lock (``modules.call_queue.queue_lock``) is a
    FIFOLock: a waiting task runs once it is released, or stops when cancelled."""

    from modules.fifo_lock import FIFOLock

    lock = FIFOLock()
    with server_app(tmp_path, lock=lock) as server:
        page = pages.register(Echo("echo", "Echo"))
        page.gate = threading.Event()
        page.gate.set()
        body = {"values": {"text": "go"}}
        with lock:
            status = server.client.post("/api/v2/actions/page:echo/run", json=body).json()["__outputs__"]["task:run"]
            waiting = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["text"] is not None)
            assert waiting["__outputs__"]["task:run"]["state"] == "queued"
        done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
        assert done["__outputs__"]["task:run"]["state"] == "completed" and done["__outputs__"]["result"] == "gogo"
        with lock:
            status = server.client.post("/api/v2/actions/page:echo/run", json=body).json()["__outputs__"]["task:run"]
            _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["text"] is not None)
            server.client.post("/api/v2/actions/page:echo/__cancel__", json={"values": {}, "task": "run"})
            done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
        assert done["__outputs__"]["task:run"]["state"] == "cancelled"
        assert lock.acquire(blocking=False)  # the cancelled waiter left the queue
        lock.release()


def test_fifo_lock_timeout():
    from modules.fifo_lock import FIFOLock

    lock = FIFOLock()
    assert lock.acquire()
    started = time.monotonic()
    assert lock.acquire(timeout=0.05) is False and time.monotonic() - started >= 0.04
    assert not lock._pending_threads
    got = []
    waiter = threading.Thread(target=lambda: got.append(lock.acquire(timeout=5)))
    waiter.start()
    time.sleep(0.05)
    lock.release()
    waiter.join(5)
    assert got == [True]
    lock.release()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # re-raised on its thread
def test_a_task_ended_by_a_base_exception_fails_and_can_start_again(server):
    """Only ``Exception`` was caught: a task ended by ``SystemExit`` (an extension calling
    ``sys.exit``) stayed "running", and starting it again only answered "already running"."""

    class Exiting(Echo):
        @ui.action("run")
        def run(self, ctx, values):
            def work(task):
                if values.get("exit"):
                    raise SystemExit(2)
                return {"__outputs__": {"result": "ran"}}

            return self.start_task(ctx, "run", work)

    pages.register(Exiting("echo", "Echo"))
    status = server.client.post("/api/v2/actions/page:echo/run", json={"values": {"exit": True}}).json()["__outputs__"]["task:run"]
    done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
    assert done["__outputs__"]["task:run"]["state"] == "failed" and done["__outputs__"]["task:run"]["error"] == "SystemExit"
    again = server.client.post("/api/v2/actions/page:echo/run", json={"values": {}}).json()
    assert "__toast__" not in again and again["__outputs__"]["task:run"]["id"] != status["id"]
    assert _poll(server, {"values": {}, "task": "run", "id": again["__outputs__"]["task:run"]["id"]}, lambda s: s["done"])["__outputs__"]["result"] == "ran"


def test_failed_tasks_toast_their_error(server):
    class Broken(Echo):
        @ui.action("run")
        def run(self, ctx, values):
            return self.start_task(ctx, "run", lambda task: 1 / 0)

    pages.register(Broken("echo", "Echo"))
    status = server.client.post("/api/v2/actions/page:echo/run", json={"values": {}}).json()["__outputs__"]["task:run"]
    done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
    assert done["__outputs__"]["task:run"]["state"] == "failed" and "division" in done["__outputs__"]["task:run"]["error"]
    assert done["__toast__"]["level"] == "error" and done["__toast__"]["message"]["text"] == "Failed: {{error}}"


def test_a_task_refusing_a_value_keeps_the_sentence_for_the_ui(server):
    """A task whose work raises ``ValidationError``: its messages went into one English
    string, so a ``Sentence`` (a message that names the value) lost its template. The
    status carries the refused values as jobs do (``errors``), and a single one is the toast."""

    reason = ui.Sentence("unknown model {{name}}", name="'x'")

    class Refusing(Echo):
        @ui.action("run")
        def run(self, ctx, values):
            def work(task):
                raise ui.ValidationError([ui.ParamError(reason, param="model", owner=self.owner)])

            return self.start_task(ctx, "run", work)

    pages.register(Refusing("echo", "Echo"))
    status = server.client.post("/api/v2/actions/page:echo/run", json={"values": {}}).json()["__outputs__"]["task:run"]
    done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
    ended = done["__outputs__"]["task:run"]
    assert ended["state"] == "failed" and ended["error"] == "unknown model 'x'"
    assert ended["errors"] == [{"owner": "page:echo", "param": "model", "message": "unknown model 'x'", "template": "unknown model {{name}}", "params": {"name": "'x'"}}]
    assert done["__toast__"] == {"level": "error", "message": {"text": "unknown model {{name}}", "params": {"name": "'x'"}}}


def test_a_task_refusing_several_values_toasts_them_as_failed(server):
    """Several refused values make the toast "Failed: {{error}}" with their English messages
    joined, and the status keeps each value's sentence: the UI recognises this exact text and
    fills the parameter from the status's values in its own language (task-runner follow.ts)."""

    model = ui.Sentence("unknown model {{name}}", name="'x'")
    style = ui.Sentence("unknown style {{name}}", name="'y'")

    class Refusing(Echo):
        @ui.action("run")
        def run(self, ctx, values):
            def work(task):
                raise ui.ValidationError([ui.ParamError(model, param="model", owner=self.owner), ui.ParamError(style, param="style", owner=self.owner)])

            return self.start_task(ctx, "run", work)

    pages.register(Refusing("echo", "Echo"))
    status = server.client.post("/api/v2/actions/page:echo/run", json={"values": {}}).json()["__outputs__"]["task:run"]
    done = _poll(server, {"values": {}, "task": "run", "id": status["id"]}, lambda s: s["done"])
    ended = done["__outputs__"]["task:run"]
    assert ended["state"] == "failed" and ended["error"] == "unknown model 'x'; unknown style 'y'"
    assert ended["errors"] == [
        {"owner": "page:echo", "param": "model", "message": "unknown model 'x'", "template": "unknown model {{name}}", "params": {"name": "'x'"}},
        {"owner": "page:echo", "param": "style", "message": "unknown style 'y'", "template": "unknown style {{name}}", "params": {"name": "'y'"}},
    ]
    assert done["__toast__"] == {"level": "error", "message": {"text": "Failed: {{error}}", "params": {"error": "unknown model 'x'; unknown style 'y'"}}}
