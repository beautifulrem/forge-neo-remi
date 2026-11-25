"""A model picked in the settings while a job runs (PATCH /api/v2/options,
POST /sdapi/v1/options) does not reach the running job: it keeps the model it
started with across its batches and hires/refiner switches, the settings show
the pick at once, and the next job loads it."""

import contextlib
import re
import threading
import time
from types import SimpleNamespace

import pytest

from remi import model_selection
from remi.server.testing import FakeState, StubExecutor, server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def applied(webui, monkeypatch, tmp_path):
    """The model manager, faked: an applied pick is stored in the options (as main_entry does)."""

    from modules import sd_models, shared

    done = []

    def apply(key, value):
        done.append((key, value))
        shared.opts.data[key] = value
        return True

    monkeypatch.setattr(model_selection, "_apply_now", apply)
    monkeypatch.setattr(sd_models, "get_closet_checkpoint_match", lambda name: object() if name.endswith(".safetensors") else None)
    monkeypatch.setitem(shared.opts.data, "sd_model_checkpoint", "first.safetensors")
    monkeypatch.setattr(shared, "config_filename", str(tmp_path / "config.json"))
    yield done
    model_selection.apply_held(apply=lambda key, value: True)  # nothing left for the next test


def wait_until(predicate, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_pick_made_while_a_job_runs_waits_for_the_next_job(applied, tmp_path, monkeypatch):
    from modules import shared
    from modules.call_queue import queue_lock

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)  # only the queue applies it here
    state = FakeState()
    stub = StubExecutor(state, gate=threading.Event())
    seen = []

    def executor(job, ctx):
        seen.append(shared.opts.data["sd_model_checkpoint"])
        result = stub(job, ctx)
        seen.append(shared.opts.data["sd_model_checkpoint"])  # what its later batches would load
        return result

    with server_app(tmp_path, executor=executor, state=state, lock=queue_lock) as server:
        first = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "a"}, "scripts": {}})
        assert first.status_code == 201 and stub.started.wait(5)

        response = server.client.patch("/api/v2/options", json={"sd_model_checkpoint": "second.safetensors"})
        assert response.status_code == 200
        assert response.json()["values"]["sd_model_checkpoint"] == "second.safetensors"  # the settings show the pick
        assert server.client.get("/api/v2/app").json()["models"]["checkpoint"] == "second.safetensors"
        assert shared.opts.data["sd_model_checkpoint"] == "first.safetensors" and applied == []

        second = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "b"}, "scripts": {}})
        assert second.status_code == 201
        stub.gate.set()
        assert server.services.queue.wait_idle()

    assert seen == ["first.safetensors", "first.safetensors", "second.safetensors", "second.safetensors"]
    assert applied == [("sd_model_checkpoint", "second.safetensors")]


def test_a_pick_while_idle_applies_at_once(applied, tmp_path):
    from modules import shared

    with server_app(tmp_path) as server:
        assert server.client.patch("/api/v2/options", json={"sd_model_checkpoint": "second.safetensors"}).status_code == 200
    assert applied == [("sd_model_checkpoint", "second.safetensors")] and model_selection.held() == {}
    assert shared.opts.data["sd_model_checkpoint"] == "second.safetensors"


def test_a_held_pick_applies_when_the_generation_ends_with_nothing_queued(applied):
    from modules.call_queue import queue_lock

    with queue_lock:  # a generation (an /sdapi request, a GPU action) holds the lock
        assert model_selection.select("sd_model_checkpoint", "second.safetensors") is True
        assert model_selection.select("forge_unet_storage_dtype", "bnb-nf4") is True
        assert applied == []
    assert wait_until(lambda: len(applied) == 2)
    assert applied == [("sd_model_checkpoint", "second.safetensors"), ("forge_unet_storage_dtype", "bnb-nf4")]
    assert model_selection.held() == {}


def test_a_pick_the_settings_already_show_is_no_change_while_a_job_runs(applied, monkeypatch):
    """Held while a generation runs, every pick counted as a change, so posting back the model
    the settings show (a client sending what GET answered) announced options.changed for
    nothing. The pick the settings show (the stored one, or the one already held) is none."""
    from modules.call_queue import queue_lock

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    with queue_lock:
        assert model_selection.select("sd_model_checkpoint", "first.safetensors") is False  # the stored one
        assert model_selection.held() == {}
        assert model_selection.select("sd_model_checkpoint", "second.safetensors") is True
        assert model_selection.select("sd_model_checkpoint", "second.safetensors") is False  # already held
        assert model_selection.select("sd_model_checkpoint", "first.safetensors") is True  # back to the stored one
    assert applied == []


def test_sdapi_options_announce_no_change_for_the_model_shown(applied, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from modules import sd_models
    from modules.call_queue import queue_lock
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {"first.safetensors": object(), "second.safetensors": object()})
    services = Services.create(tmp_path, dispatch=run_inline, realtime=False, housekeeping=False)
    events: list[dict] = []
    unsubscribe = services.bus.subscribe(type("R", (), {"send_event": lambda self, event: events.append(event), "send_frame": lambda self, frame: None})())
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860") as client, queue_lock:
            assert client.post("/sdapi/v1/options", json={"sd_model_checkpoint": "first.safetensors"}).status_code == 200
            assert client.post("/sdapi/v1/options", json={"sd_model_checkpoint": "second.safetensors"}).status_code == 200
    finally:
        unsubscribe()
        services.close()
    assert [event["keys"] for event in events if event["type"] == "options.changed"] == [["sd_model_checkpoint"]]


def test_other_settings_of_the_same_request_apply_at_once(applied, tmp_path, monkeypatch):
    from modules import shared
    from modules.call_queue import queue_lock

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    monkeypatch.setitem(shared.opts.data, "samples_format", "png")
    with server_app(tmp_path) as server, queue_lock:
        response = server.client.patch("/api/v2/options", json={"sd_model_checkpoint": "second.safetensors", "samples_format": "webp"})
        assert response.status_code == 200
        assert shared.opts.data["samples_format"] == "webp" and applied == []
        assert model_selection.held() == {"sd_model_checkpoint": "second.safetensors"}


def test_sdapi_options_hold_the_pick_too(applied, monkeypatch):
    from fastapi import FastAPI

    from modules import sd_models, shared
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {"second.safetensors": object()})
    monkeypatch.setitem(shared.opts.data, "samples_format", "png")
    api = Api(FastAPI(), queue_lock)
    with queue_lock:
        api.set_config({"sd_model_checkpoint": "second.safetensors", "samples_format": "webp"})
    assert shared.opts.data["sd_model_checkpoint"] == "first.safetensors" and shared.opts.data["samples_format"] == "webp"
    assert model_selection.held() == {"sd_model_checkpoint": "second.safetensors"}
    with pytest.raises(Exception, match="unknown checkpoint"):
        api.set_config({"sd_model_checkpoint": "missing.safetensors"})


@pytest.mark.parametrize(("flag", "value"), [("freeze_settings", True), ("freeze_specific_settings", "sd_model_checkpoint"), ("freeze_settings_in_sections", "sd")])
def test_a_frozen_model_setting_is_refused_also_while_a_job_runs(applied, monkeypatch, flag, value):
    """A pick made during a generation was held without the freeze check: the request
    answered as if it had worked, and the waiter's apply failed later, only logged."""

    from fastapi import FastAPI

    from modules import sd_models, shared
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {"first.safetensors": object(), "second.safetensors": object()})
    monkeypatch.setitem(shared.opts.data, "samples_format", "png")
    monkeypatch.setattr(shared.cmd_opts, flag, value)
    assert shared.opts.data_labels["sd_model_checkpoint"].section[0] == "sd"
    with queue_lock, pytest.raises(AssertionError, match=r"frozen|disabled"):
        model_selection.select("sd_model_checkpoint", "second.safetensors")
    assert model_selection.held() == {} and applied == []
    with pytest.raises(AssertionError):  # and with the lock free, as Options.set refuses it
        model_selection.select("sd_model_checkpoint", "second.safetensors")
    assert applied == []
    # the stored pick posted back unchanged (a client sending what GET answered) is no change
    with queue_lock:
        model_selection.select("sd_model_checkpoint", "first.safetensors")
    model_selection.apply_held(apply=lambda key, value: True)
    model_selection.select("sd_model_checkpoint", "first.safetensors")
    if flag != "freeze_settings":  # (--freeze-settings refuses saving the other settings too, as upstream does)
        Api(FastAPI(), queue_lock).set_config({"sd_model_checkpoint": "first.safetensors", "samples_format": "webp"})
        assert shared.opts.data["samples_format"] == "webp"


@pytest.mark.parametrize(("flag", "value"), [("freeze_settings", True), ("freeze_specific_settings", "sd_model_checkpoint"), ("freeze_settings_in_sections", "sd")])
@pytest.mark.parametrize("running", [True, False])
def test_sdapi_refuses_a_frozen_model_pick_with_the_other_bad_values(applied, monkeypatch, flag, value, running):
    """select's AssertionError reached the client as a 500, after the request's other
    settings were already saved: the pick is now one of the refused values (422)."""

    import contextlib

    from fastapi import FastAPI, HTTPException

    from modules import sd_models, shared
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {"first.safetensors": object(), "second.safetensors": object()})
    monkeypatch.setitem(shared.opts.data, "samples_format", "png")
    monkeypatch.setattr(shared.cmd_opts, flag, value)
    with queue_lock if running else contextlib.nullcontext(), pytest.raises(HTTPException) as refused:
        Api(FastAPI(), queue_lock).set_config({"samples_format": "webp", "sd_model_checkpoint": "second.safetensors"})
    assert refused.value.status_code == 422 and "settings.sd_model_checkpoint: " in refused.value.detail
    assert re.search(r"frozen|disabled", refused.value.detail)
    assert shared.opts.data["samples_format"] == "png" and shared.opts.data["sd_model_checkpoint"] == "first.safetensors"
    assert model_selection.held() == {} and applied == []


@contextlib.contextmanager
def options_client(root):
    """A client of both settings routes (``POST /sdapi/v1/options``, ``PATCH /api/v2/options``)
    and the ``options.changed`` events they announce."""
    from fastapi.testclient import TestClient

    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    services = Services.create(root, dispatch=run_inline, realtime=False, housekeeping=False)
    events: list[dict] = []
    unsubscribe = services.bus.subscribe(SimpleNamespace(send_event=events.append, send_frame=lambda frame: None, wants_frames=lambda: False))
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860", raise_server_exceptions=False) as client:

            def send(route, body):
                return client.post("/sdapi/v1/options", json=body) if route == "sdapi" else client.patch("/api/v2/options", json=body)

            yield SimpleNamespace(send=send, changed=lambda: [key for event in events if event["type"] == "options.changed" for key in event["keys"]])
    finally:
        unsubscribe()
        services.close()


MODULES = {"vae.safetensors": "/models/VAE/vae.safetensors", "te.safetensors": "/models/text_encoder/te.safetensors"}


@pytest.fixture
def model_files(monkeypatch, tmp_path):
    """Checkpoints and modules the WebUI knows, the model manager's own functions with its
    reload of the loading parameters recorded, and the "sd" UI preset."""
    from modules import sd_models, shared
    from modules_forge import main_entry
    from remi import networks

    infos = {name: object() for name in ("first.safetensors", "second.safetensors")}
    monkeypatch.setattr(sd_models, "get_closet_checkpoint_match", lambda name: infos.get(name))
    monkeypatch.setattr(sd_models, "checkpoint_aliases", infos)
    monkeypatch.setattr(main_entry, "module_list", dict(MODULES))
    monkeypatch.setattr(networks, "module_files", lambda kind: [(name, path, path.rpartition("/")[0]) for name, path in MODULES.items() if (kind == "vae") == ("VAE" in path)])
    refreshed = []
    monkeypatch.setattr(main_entry, "refresh_model_loading_parameters", lambda refresh=True, **kwargs: refreshed.append(refresh))
    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    monkeypatch.setattr(shared, "config_filename", str(tmp_path / "config.json"))
    for key, value in (("forge_preset", "sd"), ("sd_model_checkpoint", "first.safetensors"), ("forge_additional_modules", [MODULES["vae.safetensors"]]), ("forge_unet_storage_dtype", "Automatic"), ("samples_format", "png")):
        monkeypatch.setitem(shared.opts.data, key, value)
    assert "forge_unet_storage_dtype_sd" in shared.opts.data_labels
    return refreshed


@pytest.mark.parametrize("route", ["sdapi", "patch"])
def test_an_unchanged_model_pick_does_nothing(webui, model_files, monkeypatch, tmp_path, route):
    """A pick equal to the stored one still went to the model manager while no generation ran:
    dtype_change sets the dtype again, saves the settings, reloads the loading parameters and
    says it changed, so posting back what the settings show saved and announced
    ``options.changed`` for nothing. An unchanged pick (modules by name or path, in any order)
    now does nothing, on either route."""
    from modules import shared

    saves = []
    monkeypatch.setattr(shared.opts, "save", saves.append)
    stored = {key: shared.opts.data[key] for key in model_selection.MODEL_OPTIONS}
    bodies = [dict(stored), {**stored, "forge_additional_modules": ["vae.safetensors"]}]
    with options_client(tmp_path / "data") as client:
        for body in bodies:
            response = client.send(route, body)
            assert response.status_code == 200, (body, response.text)
    assert client.changed() == [] and model_files == []
    assert {key: shared.opts.data[key] for key in model_selection.MODEL_OPTIONS} == stored
    assert len(saves) == (len(bodies) if route == "sdapi" else 0)  # upstream's set_config saves each request's settings, changed or not


@pytest.mark.parametrize("route", ["sdapi", "patch"])
@pytest.mark.parametrize(
    ("key", "stored", "copy", "sent"),
    [
        ("forge_unet_storage_dtype", "float8-e4m3fn", "Automatic", "float8-e5m2"),
        ("sd_model_checkpoint", "first.safetensors", "first.safetensors", "second.safetensors"),
        ("forge_additional_modules", [MODULES["vae.safetensors"]], [MODULES["vae.safetensors"]], ["te.safetensors"]),
    ],
    ids=["dtype", "checkpoint", "modules"],
)
def test_a_pick_the_presets_frozen_copy_keeps_is_refused(webui, model_files, monkeypatch, tmp_path, route, key, stored, copy, sent):
    """The model manager also writes a pick into the UI preset's own copy of it
    (``forge_unet_storage_dtype_<preset>``, ...). With that copy frozen
    (``--freeze-specific-settings``) and other than the pick, ``Options.set``'s assertion came
    after the setting itself was set: a 500 on ``/sdapi/v1/options`` (the request's other
    settings set and saved), a 422 with the assertion's text on ``PATCH /api/v2/options``, the
    pick half applied either way; a dtype posted back unchanged did it too. A change is refused
    naming the frozen copy (422, nothing set); an unchanged pick does nothing."""
    from modules import shared

    preset_copy = {"sd_model_checkpoint": "forge_checkpoint_sd", "forge_additional_modules": "forge_additional_modules_sd", "forge_unet_storage_dtype": "forge_unet_storage_dtype_sd"}[key]
    monkeypatch.setitem(shared.opts.data, key, stored)
    monkeypatch.setitem(shared.opts.data, preset_copy, copy)
    monkeypatch.setattr(shared.cmd_opts, "freeze_specific_settings", preset_copy)
    before = dict(shared.opts.data)
    with options_client(tmp_path / "data") as client:
        response = client.send(route, {key: stored})
        assert response.status_code == 200, response.text
        assert shared.opts.data == before and model_files == [] and client.changed() == []
        response = client.send(route, {"samples_format": "webp", key: sent})
        assert response.status_code == 422, response.text
        if route == "sdapi":
            assert f"settings.{preset_copy}: this setting is frozen" in response.json()["detail"], response.text
        else:
            assert [(error["param"], error["message"]) for error in response.json()["errors"]] == [(preset_copy, "this setting is frozen")]
        assert shared.opts.data == before and model_files == [] and client.changed() == []
        assert model_selection.held() == {}


def test_held_picks_reach_the_model_manager_as_it_stores_them(webui, monkeypatch, tmp_path):
    """The real main_entry functions (not a faked ``_apply_now``): a modules pick held during a
    generation is shown as the sorted paths modules_change stores, and applying the held picks
    goes through checkpoint_change / modules_change / dtype_change and refreshes the loading
    parameters once each."""
    from modules import sd_models, shared
    from modules.call_queue import queue_lock
    from modules_forge import main_entry

    monkeypatch.setattr(model_selection, "_start_waiter", lambda: None)
    monkeypatch.setattr(shared, "config_filename", str(tmp_path / "config.json"))
    infos = {name: object() for name in ("first.safetensors", "second.safetensors")}
    monkeypatch.setattr(sd_models, "get_closet_checkpoint_match", lambda name: infos.get(name))
    monkeypatch.setattr(main_entry, "module_list", {"vae.safetensors": "/models/VAE/vae.safetensors", "te.safetensors": "/models/text_encoder/te.safetensors"})
    refreshed = []
    monkeypatch.setattr(main_entry, "refresh_model_loading_parameters", lambda refresh=True: refreshed.append(refresh))
    saved = dict(shared.opts.data)
    try:
        shared.opts.data.update(sd_model_checkpoint="first.safetensors", forge_additional_modules=[], forge_unet_storage_dtype="Automatic")
        with queue_lock:
            model_selection.select("sd_model_checkpoint", "second.safetensors")
            model_selection.select("forge_additional_modules", ["te.safetensors", "/elsewhere/vae.safetensors", "missing.safetensors"])
            model_selection.select("forge_unet_storage_dtype", "bnb-nf4")
            assert model_selection.held()["forge_additional_modules"] == ["/models/VAE/vae.safetensors", "/models/text_encoder/te.safetensors"]
            assert refreshed == [] and shared.opts.sd_model_checkpoint == "first.safetensors"
            applied = model_selection.apply_held()
        assert applied == ["sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype"]
        assert shared.opts.sd_model_checkpoint == "second.safetensors"
        assert shared.opts.forge_additional_modules == ["/models/VAE/vae.safetensors", "/models/text_encoder/te.safetensors"]
        assert shared.opts.forge_unet_storage_dtype == "bnb-nf4"
        assert refreshed == [True, True, True] and model_selection.held() == {}
    finally:
        shared.opts.data.clear()
        shared.opts.data.update(saved)


# A preset without its own model settings (a stored name --disable-preset removed, or one of another
# kind written through /sdapi): main_entry set the global setting, then raised KeyError on the preset's
# own one: the change answered 422 while half applied (shown, never loaded nor saved)
@pytest.mark.parametrize("preset", ["gone", 5])
def test_a_model_change_under_a_preset_without_settings_names_no_preset(webui, monkeypatch, preset):
    from modules import shared

    seen = []

    class MainEntry:
        def dtype_change(self, value, preset):
            seen.append(preset)
            return True

    monkeypatch.setitem(shared.opts.data, "forge_preset", preset)
    monkeypatch.setattr(model_selection, "_main_entry", lambda: MainEntry())
    assert model_selection._apply_now("forge_unet_storage_dtype", "Automatic") is True
    assert seen == [None]
    monkeypatch.setitem(shared.opts.data, "forge_preset", "sd")
    model_selection._apply_now("forge_unet_storage_dtype", "Automatic")
    assert seen[-1] == ("sd" if "forge_unet_storage_dtype_sd" in shared.opts.data_labels else None)


def test_the_options_refuse_a_preset_that_is_not_offered(webui, tmp_path, monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared, "config_filename", str(tmp_path / "config.json"))
    with server_app(tmp_path / "data") as server:
        response = server.client.patch("/api/v2/options", json={"forge_preset": "nosuch"})
        assert response.status_code == 422, response.text
        assert [error["param"] for error in response.json()["errors"]] == ["forge_preset"]
        assert shared.opts.forge_preset != "nosuch"
