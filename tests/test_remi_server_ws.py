"""The WebSocket hub, binary preview frames and the realtime threads."""

import asyncio
import io
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from remi.realtime import FORMAT_JPEG, FORMAT_WEBP, HEADER_SIZE, EventBus, JobRef, PreviewEncoder, PreviewFrame, ProgressMonitor, encode_preview, pack_frame, progress_snapshot, unpack_frame
from remi.server.testing import WS_URL, FakeState, StubExecutor, server_app
from remi.server.ws import MAX_EVENTS, Client
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


# region Frames


def test_frame_header_is_little_endian_type_format_seq_step():
    frame = pack_frame(FORMAT_WEBP, 0x01020304, 7, b"IMG")
    assert frame[:HEADER_SIZE] == bytes([1, 2, 4, 3, 2, 1, 7, 0, 0, 0]) and frame[HEADER_SIZE:] == b"IMG"
    assert unpack_frame(frame) == PreviewFrame(FORMAT_WEBP, 0x01020304, 7, b"IMG")
    with pytest.raises(ValueError):
        unpack_frame(b"\x02\x01" + bytes(8))
    with pytest.raises(ValueError):
        unpack_frame(b"\x01")


@pytest.mark.parametrize(("fmt", "magic"), [(FORMAT_JPEG, b"\xff\xd8"), (FORMAT_WEBP, b"RIFF")])
def test_previews_encode_as_jpeg_or_webp(fmt, magic):
    data = encode_preview(Image.new("RGBA", (40, 30), (255, 0, 0, 128)), fmt)
    assert data.startswith(magic)
    assert Image.open(io.BytesIO(data)).size == (40, 30)


# endregion

# region Realtime threads


def test_progress_snapshot_matches_internal_progress():
    state = FakeState()
    state.job_count, state.job_no, state.sampling_steps, state.sampling_step = 2, 1, 10, 5
    state.time_start, state.textinfo = 100.0, "Hires"
    snapshot = progress_snapshot(state, now=175.0)
    assert snapshot == {"step": 5, "steps": 10, "job_no": 1, "job_count": 2, "progress": 0.75, "eta": 25.0, "text": "Hires"}


def recorder(bus):
    seen = []
    bus.subscribe(SimpleNamespace(send_event=seen.append, send_frame=seen.append))
    return seen


def test_progress_monitor_publishes_changes_only():
    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    ref = JobRef("j1", 4, "txt2img")
    ticks = []
    monitor = ProgressMonitor(bus, lambda: ref, state=lambda: state, on_tick=lambda: ticks.append(1))
    state.job_count, state.sampling_steps, state.sampling_step, state.time_start = 1, 8, 1, time.time()
    last = monitor.tick(None)
    last = monitor.tick(last)
    state.sampling_step = 2
    monitor.tick(last)
    assert [(event["type"], event["step"], event["seq"]) for event in seen] == [("job.progress", 1, 4), ("job.progress", 2, 4)]
    assert len(ticks) == 3
    assert ProgressMonitor(bus, lambda: None, state=lambda: state).tick("x") is None
    # a job not begun yet: the state still shows the previous job's steps
    assert ProgressMonitor(bus, lambda: JobRef("j2", 5, "txt2img", begun=False), state=lambda: state).tick("x") == "x" and len(seen) == 2


def test_progress_monitor_skips_the_finishing_state():
    """A poll between the end of sampling (counters reset) and job.completed must not
    send the step again at 0 % (it raced under load and flickered the progress bar)."""

    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    monitor = ProgressMonitor(bus, lambda: JobRef("j1", 4, "txt2img"), state=lambda: state)
    state.job_count, state.sampling_steps, state.sampling_step, state.time_start = 1, 4, 4, time.time()
    last = monitor.tick(None)
    state.job_count = 0  # as the executor / shared.state does when the job ends
    assert monitor.tick(last) == last
    assert [(event["step"], event["progress"]) for event in seen] == [(4, 1.0)]
    # a new job starting from a reset state still publishes
    other = ProgressMonitor(bus, lambda: JobRef("j2", 5, "txt2img"), state=lambda: state)
    state.job_count, state.sampling_step = 1, 1
    other.tick(last)
    assert len(seen) == 2 and (seen[-1]["job_id"], seen[-1]["step"]) == ("j2", 1)


def test_preview_encoder_sends_only_the_newest_frame_in_the_configured_format():
    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="png", live_preview_fps=6)
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 9, "txt2img"), state=lambda: state, opts=lambda: opts)
    assert encoder.tick(None) is None  # no image yet
    state.current_image, state.id_live_preview, state.current_image_sampling_step = Image.new("RGB", (8, 8)), 1, 3
    last = encoder.tick(None)
    assert encoder.tick(last) == last  # same preview id: nothing new
    state.id_live_preview = 5  # several previews were produced meanwhile: only the newest is sent
    encoder.tick(last)
    frames = [item for item in seen if isinstance(item, PreviewFrame)]
    assert [(frame.fmt, frame.job_seq, frame.step) for frame in frames] == [(FORMAT_WEBP, 9, 3), (FORMAT_WEBP, 9, 3)]
    opts.live_previews_image_format = "jpeg"
    state.id_live_preview = 6
    encoder.tick(None)
    assert seen[-1].fmt == FORMAT_JPEG
    opts.live_previews_enable = False
    state.id_live_preview = 7
    assert encoder.tick(None) is None and len([item for item in seen if isinstance(item, PreviewFrame)]) == 3
    opts.live_preview_fps = 1000
    assert encoder.interval() == pytest.approx(1 / 30)


def test_a_new_job_never_streams_the_previous_jobs_last_preview():
    """Between a job becoming current and its ``begin()``, the state still holds the
    previous job's last preview: nothing goes out tagged with the new job until it begun."""

    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="png", live_preview_fps=6)
    current = {"ref": JobRef("a", 1, "txt2img")}
    encoder = PreviewEncoder(bus, lambda: current["ref"], state=lambda: state, opts=lambda: opts)
    state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 4
    last = encoder.tick(None)
    current["ref"] = JobRef("b", 2, "img2img", begun=False)  # the next job, before its begin()
    last = encoder.tick(last)
    assert [frame.job_seq for frame in seen if isinstance(frame, PreviewFrame)] == [1]
    current["ref"] = JobRef("b", 2, "img2img")  # begun: the state is its own
    state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 1
    encoder.tick(last)
    assert [frame.job_seq for frame in seen if isinstance(frame, PreviewFrame)] == [1, 2]


def _frame_times(bus):
    times = []
    bus.subscribe(SimpleNamespace(send_event=lambda event: None, send_frame=lambda frame: times.append((time.monotonic(), frame.step))))
    return times


def test_preview_encoder_wakes_on_a_new_preview_and_throttles_to_the_fps():
    bus, state = EventBus(), FakeState()
    times = _frame_times(bus)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=2)  # 0.5 s interval
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts)
    encoder.start()
    try:
        deadline = time.monotonic() + 2
        while not state.live_preview_listeners and time.monotonic() < deadline:
            time.sleep(0.005)
        assert state.live_preview_listeners == [encoder.notify]
        state.sampling_step = 1
        assigned = time.monotonic()
        state.assign_current_image(Image.new("RGB", (8, 8)))
        while not times and time.monotonic() < deadline:
            time.sleep(0.002)
        assert times and times[0][0] - assigned < 0.1  # not the 0.5 s polling interval
        # two more previews right away: only the newest goes out, one interval after the first
        state.sampling_step = 2
        state.assign_current_image(Image.new("RGB", (8, 8)))
        state.sampling_step = 3
        state.assign_current_image(Image.new("RGB", (8, 8)))
        while len(times) < 2 and time.monotonic() < deadline + 1:
            time.sleep(0.005)
        assert [step for _, step in times] == [1, 3]
        assert times[1][0] - times[0][0] >= 0.45
    finally:
        encoder.stop()
        encoder.join(2)
    assert state.live_preview_listeners == []


def test_a_small_preview_is_encoded_on_the_assigning_thread():
    """No hand-over to the encoder thread for the usual tiny previews (the sampler
    would wait for that thread to get the GIL); large ones and throttled ones
    wake the thread instead."""

    bus, state = EventBus(), FakeState()
    published = []
    bus.subscribe(SimpleNamespace(send_event=lambda event: None, send_frame=lambda frame: published.append((threading.get_ident(), frame.step, frame.fmt))))
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=2)  # 0.5 s interval
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts)
    state.live_preview_listeners.append(encoder.notify)  # as run() does; the thread is not started
    state.sampling_step = 1
    state.assign_current_image(Image.new("RGB", (96, 96)))
    assert published == [(threading.get_ident(), 1, FORMAT_JPEG)]
    assert not encoder._wake.is_set()
    # within the throttle interval: left to the thread (woken), nothing encoded here
    state.sampling_step = 2
    state.assign_current_image(Image.new("RGB", (96, 96)))
    assert len(published) == 1 and encoder._wake.is_set()
    encoder._wake.clear()
    encoder._ready_at = 0.0
    # too large for the generation thread: the encoder thread does it
    state.sampling_step = 3
    state.assign_current_image(Image.new("RGB", (512, 512)))
    assert len(published) == 1 and encoder._wake.is_set()
    encoder._wake.clear()
    # WebP costs more per pixel: 128x128 is encoded inline, larger previews wake the encoder thread
    opts.live_previews_image_format = "webp"
    state.sampling_step = 4
    state.assign_current_image(Image.new("RGB", (128, 128)))
    assert not encoder._wake.is_set()
    state.sampling_step = 5
    encoder._ready_at = 0.0
    state.assign_current_image(Image.new("RGB", (160, 160)))
    assert [(step, fmt) for _, step, fmt in published] == [(1, FORMAT_JPEG), (4, FORMAT_WEBP)] and encoder._wake.is_set()


def test_an_inline_encoding_error_never_reaches_the_sampler(caplog):
    bus, state = EventBus(), FakeState()
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=6)

    def broken():
        raise RuntimeError("no job")

    encoder = PreviewEncoder(bus, broken, state=lambda: state, opts=lambda: opts)
    state.live_preview_listeners.append(encoder.notify)
    state.assign_current_image(Image.new("RGB", (8, 8)))  # does not raise
    assert "preview encoder" in caplog.text


def test_preview_encoder_polls_a_state_without_listeners():
    bus = EventBus()
    times = _frame_times(bus)
    state = SimpleNamespace(current_image=None, id_live_preview=0, current_image_sampling_step=4, sampling_step=4)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=20)
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts)
    encoder.start()
    try:
        state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 1
        deadline = time.monotonic() + 2
        while not times and time.monotonic() < deadline:
            time.sleep(0.005)
        assert [step for _, step in times] == [4]
    finally:
        encoder.stop()
        encoder.join(2)


def test_a_steps_progress_goes_out_right_before_its_preview():
    """The encoder publishes the step's progress just before the frame (one wake-up of the
    hub for both); the monitor's next poll then has nothing new to publish."""

    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    ref = JobRef("j", 3, "txt2img")
    monitor = ProgressMonitor(bus, lambda: ref, state=lambda: state)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=30)
    encoder = PreviewEncoder(bus, lambda: ref, state=lambda: state, opts=lambda: opts, before_frame=monitor.publish_now)
    state.job_count, state.sampling_steps, state.time_start = 1, 8, time.time()
    for step in (1, 2):
        state.sampling_step = state.current_image_sampling_step = step
        state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), step
        encoder.tick(None)
        monitor.publish_now()  # nothing new: no second event for the step
    kinds = [("frame", item.step) if isinstance(item, PreviewFrame) else (item["type"], item["step"]) for item in seen]
    assert kinds == [("job.progress", 1), ("frame", 1), ("job.progress", 2), ("frame", 2)]
    # the polling thread shares the dedup: the step is not published again
    with monitor._lock:
        monitor._last = monitor._publish(monitor._last)
    assert len(seen) == 4
    # publish_now never waits for the polling thread
    with monitor._lock:
        state.sampling_step = 3
        monitor.publish_now()
    assert len(seen) == 4


def test_a_failing_before_frame_hook_still_sends_the_frame(caplog):
    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=30)

    def broken():
        raise RuntimeError("monitor")

    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts, before_frame=broken)
    state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 1
    encoder.tick(None)
    assert [type(item) for item in seen] == [PreviewFrame] and "before_frame" in caplog.text


def test_previews_are_not_encoded_when_nobody_watches():
    bus, state = EventBus(), FakeState()

    class Hub:
        watching = False
        frames = 0

        def send_event(self, event):
            pass

        def send_frame(self, frame):
            self.frames += 1

        def wants_frames(self):
            return self.watching

    hub = Hub()
    bus.subscribe(hub)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=6)
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts)
    state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 1
    last = encoder.tick(None)
    assert hub.frames == 0
    hub.watching = True
    encoder.tick(last)
    assert hub.frames == 1


# endregion

# region Hub


def test_a_slow_client_drops_the_oldest_events_and_keeps_one_preview():
    async def scenario():
        client = Client(SimpleNamespace(), max_events=3)
        for index in range(5):
            client.push_event({"type": "toast", "message": str(index)})
        for step in range(3):
            client.push_frame(PreviewFrame(FORMAT_WEBP, 1, step, b"x"))
        return client

    client = asyncio.run(scenario())
    assert [event["message"] for event in client.events] == ["2", "3", "4"] and client.dropped == 2
    assert client.preview.step == 2
    assert MAX_EVENTS >= 64


class FakeSocket:
    def __init__(self, fail_after=None):
        self.sent, self.closed, self.fail_after = [], None, fail_after

    async def send_text(self, text):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise RuntimeError("socket gone")
        self.sent.append(text)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self, code=1000):
        self.closed = code


def test_a_message_nested_past_the_parser_is_ignored_like_any_bad_message():
    """Only ``ValueError`` was ignored: ``RecursionError`` ended the receiver, and with it the
    connection."""

    from starlette.websockets import WebSocketDisconnect

    from remi.server.ws import Hub

    async def scenario():
        frames = iter(["[" * 100_000 + "]" * 100_000, "not json", '{"type": "ping"}'])
        socket = FakeSocket()

        async def receive():
            text = next(frames, None)
            return {"type": "websocket.receive", "text": text} if text is not None else {"type": "websocket.disconnect", "code": 1000}

        socket.receive = receive
        client = Client(socket)
        with pytest.raises(WebSocketDisconnect):  # the client left: the messages before it were all read
            await asyncio.wait_for(Hub()._receive(client), 5)
        return list(client.events)

    assert asyncio.run(scenario()) == [{"type": "pong"}]


def test_an_event_with_text_utf8_cannot_hold_is_sent_escaped_not_a_closed_socket():
    """A job's error or an extension's progress text can hold an unpaired surrogate: the
    socket's UTF-8 encoding failed and the send loop closed the connection (1011)."""

    import json

    class Utf8Socket(FakeSocket):
        async def send_text(self, text):
            text.encode("utf-8")  # as the server's WebSocket does
            await super().send_text(text)

    async def scenario():
        socket = Utf8Socket()
        client = Client(socket)
        client.push_event({"type": "job.failed", "error": "x\udc80 猫"})
        client.push_event({"type": "pong"})
        task = asyncio.create_task(client.send_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        return socket

    socket = asyncio.run(scenario())
    assert socket.closed is None and [json.loads(text) for text in socket.sent] == [{"type": "job.failed", "error": "x\udc80 猫"}, {"type": "pong"}]
    assert "\\udc80" in socket.sent[0]


def test_a_client_that_fell_behind_gets_a_resync_and_a_fresh_snapshot():
    import json

    async def scenario():
        socket = FakeSocket()
        client = Client(socket, max_events=3, snapshot=lambda: [{"type": "queue.changed", "size": 0}])
        for index in range(5):
            client.push_event({"type": "toast", "message": str(index)})
        task = asyncio.create_task(client.send_loop())
        await asyncio.sleep(0.05)
        client.push_event({"type": "toast", "message": "later"})
        await asyncio.sleep(0.05)
        task.cancel()
        return [json.loads(text) for text in socket.sent]

    sent = asyncio.run(scenario())
    assert sent == [{"type": "resync", "dropped": 2}, {"type": "queue.changed", "size": 0}, {"type": "toast", "message": "later"}]


def test_a_failed_send_closes_the_socket():
    async def scenario():
        socket = FakeSocket(fail_after=1)
        client = Client(socket)
        client.push_event({"type": "a"})
        client.push_event({"type": "b"})
        await asyncio.wait_for(client.send_loop(), 1)  # returns instead of dying silently
        return socket

    socket = asyncio.run(scenario())
    assert socket.closed == 1011


def test_clients_only_get_preview_formats_they_announced():
    async def scenario():
        client = Client(SimpleNamespace())
        client.features = {"events.v1", "preview.binary", "preview.jpeg"}
        client.push_frame(PreviewFrame(FORMAT_WEBP, 1, 1, b"w"))
        first = client.preview
        client.push_frame(PreviewFrame(FORMAT_JPEG, 1, 2, b"j"))
        return first, client.preview

    first, second = asyncio.run(scenario())
    assert first is None and second.fmt == FORMAT_JPEG


def test_hidden_pages_get_no_previews_and_none_are_encoded_for_them(webui, tmp_path):
    """A page reports when it is hidden (``visibility``): it gets no frames, and while no
    client is visible the hub does not want any (the encoder skips the work)."""

    with server_app(tmp_path) as server, server.client.websocket_connect(WS_URL) as socket:
        socket.receive_json(), socket.receive_json()  # hello, queue
        bus = server.services.bus
        socket.send_json({"type": "hello", "features": ["events.v1", "preview.binary", "preview.webp"], "visible": False})
        socket.send_json({"type": "ping"})
        assert socket.receive_json() == {"type": "pong"}  # the hello was handled
        assert not bus.wants_frames()
        bus.publish_frame(PreviewFrame(FORMAT_WEBP, 1, 1, b"hidden"))  # not sent
        socket.send_json({"type": "visibility", "visible": True})
        socket.send_json({"type": "ping"})
        assert socket.receive_json() == {"type": "pong"}
        assert bus.wants_frames()
        bus.publish_frame(PreviewFrame(FORMAT_WEBP, 1, 2, b"shown"))
        assert unpack_frame(socket.receive_bytes()) == PreviewFrame(FORMAT_WEBP, 1, 2, b"shown")
        socket.send_json({"type": "visibility", "visible": False})
        socket.send_json({"type": "ping"})
        assert socket.receive_json() == {"type": "pong"}
        assert not bus.wants_frames()


def test_websocket_hello_events_and_binary_previews(webui, tmp_path):
    with server_app(tmp_path) as server, server.client.websocket_connect(WS_URL) as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello" and "preview.binary" in hello["features"]
        assert socket.receive_json() == {"type": "queue.changed", "size": 0, "paused": False, "running": None}
        socket.send_json({"type": "hello", "features": ["events.v1", "preview.binary", "preview.webp"]})
        socket.send_json({"type": "ping"})
        assert socket.receive_json() == {"type": "pong"}

        server.services.bus.publish("options.changed", keys=["sd_model_checkpoint"])
        assert socket.receive_json() == {"type": "options.changed", "keys": ["sd_model_checkpoint"]}
        server.services.bus.publish_frame(PreviewFrame(FORMAT_JPEG, 3, 1, b"jpeg"))  # not announced: skipped
        server.services.bus.publish_frame(PreviewFrame(FORMAT_WEBP, 3, 2, b"webp"))
        assert unpack_frame(socket.receive_bytes()) == PreviewFrame(FORMAT_WEBP, 3, 2, b"webp")


def test_a_job_streams_progress_and_previews_over_the_socket(webui, tmp_path, monkeypatch):
    state = FakeState()
    monkeypatch.setitem(webui.opts.data, "live_preview_fps", 30)
    monkeypatch.setitem(webui.opts.data, "live_previews_enable", True)
    executor = StubExecutor(state, steps=6, step_time=0.1)
    with server_app(tmp_path, state=state, executor=executor, realtime=True) as server, server.client.websocket_connect(WS_URL) as socket:
        socket.receive_json(), socket.receive_json()  # hello, queue
        job = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "x"}}).json()
        types, frames = [], []
        while "job.completed" not in types:
            message = socket.receive()
            if message.get("bytes") is not None:
                frames.append(unpack_frame(message["bytes"]))
            elif message.get("text"):
                import json

                event = json.loads(message["text"])
                types.append(event["type"])
                if event["type"] == "job.progress":
                    assert event["job_id"] == job["job_id"] and event["seq"] == job["seq"] and event["steps"] == 6
        assert types.index("job.queued") < types.index("job.started") < types.index("job.completed")
        assert "job.progress" in types and frames and all(frame.job_seq == job["seq"] for frame in frames)


def test_each_steps_progress_arrives_right_before_its_preview(webui, tmp_path, monkeypatch):
    """Served, a step's ``job.progress`` goes out with its frame (one wake-up of the hub per
    step): the client sees the step's progress just before the step's preview, even with steps
    longer than the progress poll."""

    import json

    state = FakeState()
    monkeypatch.setitem(webui.opts.data, "live_preview_fps", 30)
    monkeypatch.setitem(webui.opts.data, "live_previews_enable", True)
    executor = StubExecutor(state, steps=4, step_time=0.35)
    with server_app(tmp_path, state=state, executor=executor, realtime=True) as server, server.client.websocket_connect(WS_URL) as socket:
        socket.receive_json(), socket.receive_json()  # hello, queue
        server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "x"}})
        stream = []
        while not stream or stream[-1] != ("job.completed", None):
            message = socket.receive()
            if message.get("bytes") is not None:
                stream.append(("frame", unpack_frame(message["bytes"]).step))
            elif message.get("text"):
                event = json.loads(message["text"])
                if event["type"] in ("job.progress", "job.completed"):
                    stream.append((event["type"], event.get("step")))
        frames = [index for index, item in enumerate(stream) if item[0] == "frame"]
        assert [stream[index][1] for index in frames] == [1, 2, 3, 4]
        assert all(stream[index - 1] == ("job.progress", stream[index][1]) for index in frames)
        # and no second progress event for a step that already went out with its frame
        steps = [step for kind, step in stream if kind == "job.progress"]
        assert len(steps) == len(set(steps))


def test_hub_wants_frames_requires_visible_and_preview_capabilities(webui, tmp_path):
    """clients that only subscribe to events.v1 (or omit preview format features) must not trigger preview encoding."""

    from remi.server.ws import Hub

    hub = Hub()
    c_events_only = Client(FakeSocket())
    c_events_only.features = {"events.v1"}
    c_events_only.visible = True
    hub.clients.add(c_events_only)
    assert hub.wants_frames() is False

    c_binary_no_fmt = Client(FakeSocket())
    c_binary_no_fmt.features = {"events.v1", "preview.binary"}
    c_binary_no_fmt.visible = True
    hub.clients.add(c_binary_no_fmt)
    assert hub.wants_frames() is False

    c_webp = Client(FakeSocket())
    c_webp.features = {"events.v1", "preview.binary", "preview.webp"}
    c_webp.visible = False
    hub.clients.add(c_webp)
    assert hub.wants_frames() is False

    c_webp.visible = True
    assert hub.wants_frames() is True

    with server_app(tmp_path) as server, server.client.websocket_connect(WS_URL) as socket:
        socket.receive_json(), socket.receive_json()  # hello, queue
        socket.send_json({"type": "hello", "features": ["events.v1"], "visible": True})
        socket.send_json({"type": "ping"})
        assert socket.receive_json() == {"type": "pong"}
        assert not server.services.bus.wants_frames()


def test_stalled_ws_client_times_out_and_stops_pinning_wants_frames():
    """a stalled peer triggers send timeout, closes, and leaves hub.clients."""

    from remi.server.ws import Hub

    class StalledSocket(FakeSocket):
        def __init__(self):
            super().__init__()
            self.cookies = {}

        async def accept(self):
            return None

        async def receive(self):
            await asyncio.sleep(10)
            return {"type": "websocket.disconnect", "code": 1000}

        async def send_text(self, text):
            if '"hello"' in text:
                await super().send_text(text)
                return
            await asyncio.Future()  # hangs forever

    async def scenario():
        hub = Hub(send_timeout=0.05)
        socket = StalledSocket()
        serve_task = asyncio.create_task(hub.serve(socket))
        await asyncio.sleep(0.02)
        assert hub.wants_frames() is True
        hub.send_event({"type": "toast", "message": "stalled"})
        await asyncio.wait_for(serve_task, timeout=2.0)
        assert socket.closed == 1011
        assert hub.wants_frames() is False
        assert len(hub.clients) == 0

    asyncio.run(scenario())


def test_hub_serve_closes_accept_vs_logout_revocation_race():
    """disconnect(session) arriving while Hub.serve() awaits accept() must close with 1008 and never register the client."""

    from remi.server.auth import COOKIE_NAME, Auth
    from remi.server.ws import Hub

    async def scenario():
        auth = Auth({"alice": "secret"})
        token = auth.issue("alice")
        accept_entered = asyncio.Event()
        allow_accept = asyncio.Event()

        class RacingSocket(FakeSocket):
            def __init__(self):
                super().__init__()
                self.cookies = {COOKIE_NAME: token}
                self.headers = {}
                self.app = SimpleNamespace(state=SimpleNamespace(auth=auth))

            async def accept(self):
                accept_entered.set()
                await allow_accept.wait()

            async def receive(self):
                return {"type": "websocket.disconnect", "code": 1000}

        hub = Hub(send_timeout=0.2)
        hub.bind(asyncio.get_running_loop())
        socket = RacingSocket()
        serve_task = asyncio.create_task(hub.serve(socket))
        await accept_entered.wait()
        # Simulate concurrent /api/v2/auth/logout during accept()
        auth.revoke(token)
        hub.disconnect(token)
        await asyncio.sleep(0.01)
        allow_accept.set()
        await asyncio.wait_for(serve_task, timeout=2.0)
        assert socket.closed == 1008
        assert len(hub.clients) == 0

    asyncio.run(scenario())


def test_client_coalesces_job_progress_so_terminal_events_are_not_dropped():
    """high-frequency job.progress events coalesce per job_id without overflowing MAX_EVENTS or dropping terminal events."""

    client = Client(SimpleNamespace(), max_events=8)
    client.push_event({"type": "job.started", "job_id": "j1"})
    for step in range(1, 301):
        client.push_event({"type": "job.progress", "job_id": "j1", "step": step, "steps": 300})
    client.push_event({"type": "job.progress", "job_id": "j2", "step": 1, "steps": 10})
    client.push_event({"type": "job.progress", "job_id": "j2", "step": 2, "steps": 10})
    client.push_event({"type": "job.completed", "job_id": "j1"})
    client.push_event({"type": "chain.stage_completed", "chain_id": "c1", "stage_index": 0})

    assert client.dropped == 0
    events = list(client.events)
    assert [(e["type"], e.get("job_id"), e.get("step")) for e in events] == [
        ("job.started", "j1", None),
        ("job.progress", "j1", 300),
        ("job.progress", "j2", 2),
        ("job.completed", "j1", None),
        ("chain.stage_completed", None, None),
    ]


def test_snapshot_preview_and_animated_image_lock_prevent_tearing_and_seek_races(webui, tmp_path, monkeypatch):
    """snapshot_preview avoids torn (id, image) reads and encode_preview/progressapi synchronize animated seek(0)."""

    from modules import progress, shared
    from remi.realtime import preview_image_lock, snapshot_preview

    # 1. Torn-read retry check: mutate id_live_preview during getattr(state, "current_image")
    class TearingState:
        def __init__(self):
            self.id_live_preview = 1
            self.sampling_step = 1
            self.current_image_sampling_step = 1
            self._reads = 0

        @property
        def current_image(self):
            self._reads += 1
            if self._reads == 1:
                self.id_live_preview = 2
                return "stale-1"
            return "fresh-2"

    snap_id, img, _step = snapshot_preview(TearingState())
    assert (snap_id, img) == (2, "fresh-2")

    # 2. Animated GIF concurrent encode_preview() vs progressapi()
    gif_buf = io.BytesIO()
    frames = [Image.new("RGB", (16, 16), (i * 50, 10, 20)) for i in range(4)]
    frames[0].save(gif_buf, format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
    gif_buf.seek(0)
    anim = Image.open(gif_buf)
    assert getattr(anim, "is_animated", False)

    state = FakeState()
    state.current_image = anim
    state.id_live_preview = 1
    state.job_count = 1
    state.sampling_steps = 4
    state.sampling_step = 2
    state.time_start = time.time()

    monkeypatch.setattr(shared, "state", state)
    monkeypatch.setattr(shared.opts, "live_previews_enable", True)
    monkeypatch.setattr(progress, "current_task", "task-anim")
    errors: list[BaseException] = []

    def run_encoder():
        try:
            for _ in range(15):
                data = encode_preview(anim, FORMAT_JPEG)
                assert data.startswith(b"\xff\xd8")
        except BaseException as err:
            errors.append(err)

    def run_progress():
        try:
            for _ in range(15):
                resp = progress.progressapi(progress.ProgressRequest(id_task="task-anim", id_live_preview=-1, live_preview=True))
                assert resp.live_preview and resp.live_preview.startswith("data:image/gif;base64,")
                assert resp.id_live_preview == 1
        except BaseException as err:
            errors.append(err)

    with preview_image_lock():
        pass
    t1 = threading.Thread(target=run_encoder)
    t2 = threading.Thread(target=run_progress)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert errors == []
    anim.close()


# endregion
