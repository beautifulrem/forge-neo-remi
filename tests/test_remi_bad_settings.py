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
GARBAGE = ["x", math.inf, math.nan, -math.inf, 5, ["x"], {"a": 1}]
# Read by upstream code on these routes (modules/scripts.py, scripts_postprocessing.py, infotext_utils.py),
# which Remi does not change: a value of the wrong kind written while the server runs still breaks them
# until the next start repairs it (test_the_server_repairs_settings_when_it_starts).
UPSTREAM_READS = {"ui_reorder_list", "postprocessing_disable_in_extras", "sd_model_checkpoint"}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with webui_state() as state, server_app(tmp_path_factory.mktemp("bad-settings")) as server:
        server.opts = state.opts
        yield server


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


@pytest.mark.parametrize("value", GARBAGE, ids=repr)
def test_settings_of_the_wrong_kind_written_while_running_break_nothing(server, garbage, value):
    garbage(value, [key for key in server.opts.data_labels if key not in UPSTREAM_READS])
    assert _failures(server.client) == {}


# runtime.prepare repairs them at every start and reload (test_remi_server_runtime: the WebUI's start)
@pytest.mark.parametrize("value", GARBAGE, ids=repr)
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


def test_repair_keeps_values_of_their_kind_and_reports_the_others():
    from types import SimpleNamespace

    labels = {"steps": SimpleNamespace(default=20), "tabs": SimpleNamespace(default=[]), "show": SimpleNamespace(default=True), "free": SimpleNamespace(default=None)}
    opts = SimpleNamespace(data_labels=labels, data={"steps": 30, "tabs": 5, "show": 0, "free": {"a": 1}, "unknown": "x"})
    assert repair(opts) == ["tabs", "show"]
    assert opts.data == {"steps": 30, "tabs": [], "show": False, "free": {"a": 1}, "unknown": "x"}


# The upload limit is read on every request (the body limit): "x" failed them all, 0 refused every upload.
def test_the_upload_limit_reads_as_a_size_above_zero(server, garbage):
    from remi.server.services import configured_upload_limit

    for value in ("x", math.inf, math.nan, ["x"], True, 0, -5):
        garbage(value, ["remi_upload_max_mb"])
        assert configured_upload_limit() == 64 * 1024 * 1024, value
    garbage(0.5, ["remi_upload_max_mb"])
    assert configured_upload_limit() == 512 * 1024


def test_the_resolution_step_reads_as_a_step_above_zero(server, garbage):
    from remi.core.forms import resolution_step

    for value in ("x", math.inf, 0, -8, 0.5):
        garbage(value, ["res_step"])
        assert resolution_step() == 64, value
    garbage(32, ["res_step"])
    assert resolution_step() == 32
