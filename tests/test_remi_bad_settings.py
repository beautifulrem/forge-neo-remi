"""A setting holding a value of the wrong kind (a hand-edited config.json, an
/sdapi/v1/options write, which checks no types) reads as its default: the
pages and the API keep working, and the settings page can set it right.
``remi_upload_max_mb = "x"`` made every request a 500, ``ui_tab_order = 5``
the app's bootstrap and the settings page."""

import math

import pytest

from remi.server.testing import server_app
from remi.setting_values import fit, repair
from remi.testing import webui_state

# Routes that stream, download or start work are not reads to probe here.
SKIP = {"/internal/sysinfo-download", "/internal/profile-startup", "/api/v2/recipes/export", "/api/v2/prompt-tools/favorites-export", "/docs", "/docs/oauth2-redirect", "/redoc"}
# 10**400: an int no float holds (math.isfinite raises on it); "x\udc80": text UTF-8 cannot write
GARBAGE = ["x", math.inf, math.nan, -math.inf, 5, ["x"], {"a": 1}, 10**400, "x\udc80"]
# numbers of the right kind beyond any slider: 1e303 MB overflowed into bytes, 1e-9 MB made no upload limit
EXTREMES = [1e303, -1e303, 1e-9]


def _id(value):
    return repr(value)[:12]
# Read by upstream code on these routes (modules/scripts.py, scripts_postprocessing.py, infotext_utils.py),
# which Remi does not change: a value of the wrong kind written while the server runs still breaks them
# until the next start repairs it (test_the_server_repairs_settings_when_it_starts).
UPSTREAM_READS = {"ui_reorder_list", "postprocessing_disable_in_extras", "sd_model_checkpoint"}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from remi.prompt_tools import options as prompt_tool_options
    from remi.server.settings import register_options

    with webui_state() as state, server_app(tmp_path_factory.mktemp("bad-settings")) as server:
        labels = state.opts.data_labels
        before = set(labels)
        # registered as at the start (runtime.prepare): their sliders bound what Remi reads
        register_options(state.opts)
        prompt_tool_options.register_tool_options()
        server.opts = state.opts
        try:
            yield server
        finally:
            # the labels are the shared options_templates: later tests' webui_state would see them
            for key in set(labels) - before:
                del labels[key]


@pytest.fixture
def garbage(server):
    opts = server.opts
    saved = dict(opts.data)

    def fill(value, keys):
        for key in keys:
            opts.data[key] = value

    yield fill
    opts.data.clear()
    opts.data.update(saved)


def _failures(client) -> dict[str, object]:
    """Every read that fails, not only the first traceback."""

    paths = [route.path for route in client.app.routes if "GET" in (getattr(route, "methods", None) or ()) and "{" not in route.path]
    requests = [("GET", path, None) for path in paths if path not in SKIP]
    requests += [("GET", f"/api/v2/schema/{tab}", None) for tab in ("txt2img", "img2img", "extras")]
    requests += [("POST", "/api/v2/infotext/parse", {"text": "a cat\nSteps: 20, Size: 512x512"}), ("POST", "/api/v2/upload", None)]
    failed = {}
    for method, path, body in requests:
        try:
            status = client.request(method, path, json=body).status_code
        except Exception as error:  # the test client re-raises what the server answered with a 500
            status = f"{type(error).__name__}: {error}"
        if not isinstance(status, int) or status >= 500:
            failed[f"{method} {path}"] = status
    return failed


@pytest.mark.parametrize("value", GARBAGE + EXTREMES, ids=_id)
def test_settings_of_the_wrong_kind_written_while_running_break_nothing(server, garbage, value):
    garbage(value, [key for key in server.opts.data_labels if key not in UPSTREAM_READS])
    assert _failures(server.client) == {}


# Only the numbers changed, every switch left as it is (the paths they turn on read them): the preset
# defaults of every tab's form read 10**400 steps as a float and failed the schema and the infotext parse.
@pytest.mark.parametrize("value", [*EXTREMES, 10**400, 2**63], ids=_id)
def test_numbers_of_any_size_in_every_number_setting_break_nothing(server, garbage, value):
    labels = server.opts.data_labels
    garbage(value, [key for key, info in labels.items() if type(info.default) in (int, float) and key not in UPSTREAM_READS])
    assert _failures(server.client) == {}


# runtime.prepare repairs them at every start and reload (test_remi_server_runtime: the WebUI's start)
@pytest.mark.parametrize("value", GARBAGE + EXTREMES, ids=_id)
def test_repaired_settings_break_nothing(server, garbage, value):
    opts = server.opts
    garbage(value, opts.data_labels)
    repair(opts)
    for key, info in opts.data_labels.items():
        stored = opts.data.get(key)
        assert stored is None or fit(info.default, stored) is stored, key
    if value == "x":
        assert opts.data["ui_tab_order"] == [] and opts.data["res_step"] == 64
    assert _failures(server.client) == {}


def test_a_value_reads_as_the_kind_of_its_default():
    assert fit(64, "x") == 64 and fit(64, math.inf) == 64 and fit(64, True) == 64
    assert fit(64, 32.0) == 32.0 and fit(0.5, 1) == 1
    assert fit(True, 0) is False and fit(False, 1) is True and fit(True, 2) is True and fit(False, "yes") is False
    assert fit([], 5) == [] and fit([], ["a"]) == ["a"] and fit([], [5]) == [] and fit([1, 2], [3]) == [3]
    assert fit("png", 5) == "png" and fit("png", "jpg") == "jpg"
    assert fit(None, 5) == 5 and fit("png", None) == "png"
    assert fit(64, 10**400) == 64 and fit("png", "x\udc80") == "png" and fit([], ["a", "\udc80"]) == [] and fit(None, "\ud83d") is None
    assert fit(64, 1e303, (1, 2048)) == 2048 and fit(64, 1e-9, (1, 2048)) == 1 and fit(64, 100, (1, 2048)) == 100


def test_repair_keeps_values_of_their_kind_and_reports_the_others():
    from types import SimpleNamespace

    labels = {"steps": SimpleNamespace(default=20), "tabs": SimpleNamespace(default=[]), "show": SimpleNamespace(default=True), "free": SimpleNamespace(default=None)}
    opts = SimpleNamespace(data_labels=labels, data={"steps": 30, "tabs": 5, "show": 0, "free": {"a": 1}, "unknown": "x"})
    assert repair(opts) == ["tabs", "show"]
    assert opts.data == {"steps": 30, "tabs": [], "show": False, "free": {"a": 1}, "unknown": "x"}


# The upload limit is read on every request (the body limit): "x" failed them all, 0 refused every upload.
def test_the_upload_limit_reads_as_a_size_above_zero(server, garbage):
    from remi.server.services import configured_upload_limit

    for value in ("x", math.inf, math.nan, ["x"], True, 10**400):
        garbage(value, ["remi_upload_max_mb"])
        assert configured_upload_limit() == 64 * 1024 * 1024, _id(value)
    # within the slider (1-2048 MB): 1e303 overflowed into bytes on every request, 1e-9 refused every upload
    for value, mb in ((1e303, 2048), (1e-9, 1), (0.5, 1), (0, 1), (-5, 1), (100, 100)):
        garbage(value, ["remi_upload_max_mb"])
        assert configured_upload_limit() == mb * 1024 * 1024, value


def test_the_resolution_step_reads_as_a_step_above_zero(server, garbage):
    from modules.processing import sRound
    from remi.core.forms import resolution_step

    for value in ("x", math.inf, 0, -8, 0.5, 10**400, ["x"]):
        garbage(value, ["res_step"])
        assert resolution_step() == 64 and sRound(100) == 128, _id(value)  # sRound divided by 0.5 as int 0
    garbage(32, ["res_step"])
    assert resolution_step() == 32 and sRound(100) == 96


# Read before the start repairs the settings: a temp folder of the wrong kind stopped the server thread.
def test_the_temp_folder_cleanup_reads_a_folder(server, garbage, tmp_path):
    from remi.server.launch import cleanup_temp_dir

    garbage(True, ["clean_temp_dir_at_start"])
    for value in (["x"], 1.5, {"a": 1}, "x\udc80"):
        garbage(value, ["temp_dir"])
        cleanup_temp_dir()
    (tmp_path / "left.png").write_bytes(b"")
    garbage(str(tmp_path), ["temp_dir"])
    cleanup_temp_dir()
    assert not (tmp_path / "left.png").exists()


def test_the_preview_rate_reads_as_a_rate(server, garbage):
    from remi.realtime import DEFAULT_PREVIEW_FPS, PreviewEncoder

    encoder = PreviewEncoder(server.services.bus, lambda: None, state=lambda: None, opts=lambda: server.opts)
    for value in ("x", math.nan, 10**400, ["x"], 0):
        garbage(value, ["live_preview_fps"])
        assert encoder.interval() == 1.0 / DEFAULT_PREVIEW_FPS, _id(value)  # NaN turned the throttle off
    garbage(1e303, ["live_preview_fps"])
    assert encoder.interval() == 1.0 / 30.0


# 1e303 entries overflowed SQLite's LIMIT: recording a prompt failed
def test_the_prompt_history_limit_reads_within_its_slider(server, garbage):
    from remi import prompt_tools as pt
    from remi.prompt_tools import options

    try:
        for value in (1e303, 10**400, -5):
            garbage(value, ["prompt_tools_history_limit"])
            assert 100 <= options.option("prompt_tools_history_limit") <= 10000, _id(value)
            response = server.client.post("/api/v2/prompt-tools/history", json={"prompt": f"a cat {_id(value)}"})
            assert response.status_code == 201, response.text
    finally:
        pt.forget(server.services)
