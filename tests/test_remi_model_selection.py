"""A model picked in the settings while a job runs (PATCH /api/v2/options,
POST /sdapi/v1/options) does not reach the running job: it keeps the model it
started with across its batches and hires/refiner switches, the settings show
the pick at once, and the next job loads it."""

import threading
import time

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
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {"second.safetensors": object()})
    monkeypatch.setattr(shared.cmd_opts, flag, value)
    assert shared.opts.data_labels["sd_model_checkpoint"].section[0] == "sd"
    with queue_lock:
        with pytest.raises(AssertionError, match=r"frozen|disabled"):
            model_selection.select("sd_model_checkpoint", "second.safetensors")
        with pytest.raises(AssertionError):
            Api(FastAPI(), queue_lock).set_config({"sd_model_checkpoint": "second.safetensors"})
    assert model_selection.held() == {} and applied == []
    with pytest.raises(AssertionError):  # and with the lock free, as Options.set refuses it
        model_selection.select("sd_model_checkpoint", "second.safetensors")
    assert applied == []


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
