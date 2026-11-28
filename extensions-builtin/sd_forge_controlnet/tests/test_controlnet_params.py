"""The ported ControlNet panel: ``params()`` declares the processing
arguments the Gradio ``ui()`` did (one ``ControlNetUnit`` per unit), the
unit JSON of the ``controlnet-unit`` widget becomes the same ``ControlNetUnit``
the Gradio controls built, infotext still round-trips, and the widget's
actions (preview, pose, region canvas, model refresh) work without Gradio.

Legacy record: ``tests/fixtures/legacy_ui_<tab>.json`` (``args``: the
positional arguments; ``unit_controls``: the controls of one unit), taken from
the Gradio ``ui()`` before the port (and before the Gradio UI was removed).

Intentional differences from the Gradio panel (the widget's behaviour,
webui/src/widgets/controlnet-unit, tested there):

- a blank canvas ("Create New Canvas") is the shared image editor's: white
  (it was black, ``np.zeros``) and the generation size (no size sliders);
- any new control image switches the unit on, a blank canvas included (the
  old JS clicked "Enable" on a file upload only); "Generate Canvas" does not;
- "Preview as Input" is a visible switch whenever there is a preview (it was
  hidden and only set by applying a pose, which still sets it);
- switching the control type to "Region" keeps the control image (it was
  cleared; "Generate Canvas" replaces it);
- the enabled units show as an "Nx Unit" summary in the section header (the
  old accordion badge) and as a line of buttons inside the unit (the old
  highlighted tab headers);
- the batch directory is ignored under ``--hide-ui-dir-config`` (it was read
  from a hidden textbox).

Not a difference: the Forge panel had no webcam. ``controlnet_ui_group.py``
kept the tooltips "Enable webcam" / "Mirror webcam", ``camera_symbol``,
``reverse_symbol`` and ``webcam_enabled`` / ``webcam_mirrored`` from
sd-webui-controlnet, but no control used them (its image input was a
ForgeCanvas, which has no camera source); the recorded unit controls have no
webcam toggle (:func:`test_the_legacy_panel_had_no_webcam`).
"""

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import remi_port_harness as harness
import torch
from PIL import Image

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "controlnet.py"
CONTRACT = harness.REPO / "webui" / "src" / "widgets" / "controlnet-unit" / "contract.json"
LEGACY_PREPROCESSORS = harness.REPO / "extensions-builtin" / "forge_legacy_preprocessors"
MODELS = ["None", "control_v11p_sd15_canny", "diffusers_xl_depth_full", "t2i_union_promax"]


@pytest.fixture(scope="module")
def cn():
    from remi.testing import webui_state

    if str(LEGACY_PREPROCESSORS) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(LEGACY_PREPROCESSORS))  # annotator (pose drawing)
    with webui_state():
        import modules_forge.supported_preprocessor  # noqa: F401 - None and canny

        module = harness.load_script(SCRIPT, "controlnet_params_under_test")
        yield module


@pytest.fixture
def registry(cn, monkeypatch):
    """A known preprocessor registry and model list (restored afterwards)."""

    from lib_controlnet import global_state

    from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter

    class NoModel(Preprocessor):
        def __init__(self):
            super().__init__()
            self.name = "reference_test"
            self.tags = ["Reference"]
            self.do_not_need_model = True
            self.show_control_mode = False
            self.slider_resolution = PreprocessorParameter(visible=False)
            self.slider_1 = PreprocessorParameter(label="Style Fidelity", value=0.5, minimum=0.0, maximum=1.0, step=0.01, visible=True)

    saved = dict(global_state.supported_preprocessors)
    base = {name: saved[name] for name in ("None", "canny")}
    global_state.supported_preprocessors.clear()
    global_state.supported_preprocessors.update(base, reference_test=NoModel())
    global_state.get_sorted_preprocessors.cache_clear()
    monkeypatch.setattr(global_state, "controlnet_names", list(MODELS))
    monkeypatch.setattr(global_state, "controlnet_filename_dict", {name: None if name == "None" else f"/models/{name}.safetensors" for name in MODELS})
    yield global_state
    global_state.supported_preprocessors.clear()
    global_state.supported_preprocessors.update(saved)
    global_state.get_sorted_preprocessors.cache_clear()


def layout(cn, tab="txt2img"):
    return harness.script_layout(cn.ControlNetForForgeOfficial(), tab)


def plain(value):
    return value.value if hasattr(value, "value") and not isinstance(value, (bool, int, float, str)) else value


def settings_of(unit):
    from lib_controlnet.unit_param import SETTINGS

    return {name: plain(getattr(unit, name)) for name in SETTINGS}


def png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def files(tmp_path):
    from remi.files import FileStore

    store = FileStore(tmp_path / "files")
    yield store
    store.close()


def resolver(files):
    return lambda node, file_id: files.resolve_upload(node, file_id)


def call(script, action, values, files, tab="txt2img", core=None, **extra):
    from remi import ui

    ctx = ui.ActionContext(tab=tab, owner="controlnet", core=core or {"width": 512, "height": 512}, files=ui.ActionFiles(files), extra=extra)
    return ui.call_action(script, action, ctx, values)


# region Parity with the Gradio ui()


@pytest.mark.parametrize("tab", harness.TABS)
def test_arguments_match_the_legacy_ui(cn, tab):
    """Each unit is still one argument; it is now the unit JSON (a custom
    ``controlnet-unit`` param, read from its infotext entry ``ControlNet
    <i>``) instead of a ``gr.State``. img2img adds the batch directory,
    formerly a Gradio textbox outside the script's arguments."""

    from lib_controlnet.unit_param import default_value

    snapshot = harness.load_snapshot(SCRIPT, tab)
    count = len(snapshot["args"])
    changes = {i: {"kind": "custom", "default": default_value(tab == "img2img"), "infotext": f"ControlNet {i}", "widget": "controlnet-unit", "props": ...} for i in range(count)}
    if tab == "img2img":
        changes[count] = {
            "kind": "text",
            "label": "Controlnet input directory",
            "default": "",
            "infotext": None,
            "paste": False,
            "preset": None,
            "exclusive": None,
            "visible_if": {"param": "core.mode", "eq": "Batch"},
            "enabled_if": None,
        }
    harness.assert_matches_snapshot(layout(cn, tab), snapshot["args"], changes)

    # The unit's settings start where the Gradio unit's did (images: none)
    for legacy in snapshot["args"]:
        default = default_value(tab == "img2img")
        assert {key: default[key] for key in legacy["default"] if key in default} == {key: value for key, value in legacy["default"].items() if key != "_idx"}


LEGACY_FIELDS = {
    "upload_independent_img_in_img2img": "upload_independent",
    "region_num": "region_subjects",
    "region_height": "region_vertical",
}
# Dropped: the preview shows whenever there is one ("Allow Preview"); a blank
# canvas takes the generation size (the "New Canvas" sliders); the hidden
# detect-map switch stays a unit field (save_detected_map, default on).
DROPPED = {"preprocessor_preview", "new_canvas_width", "new_canvas_height", "save_detected_map"}
# Bounds that follow the preprocessor (catalog sliders), not the control.
DYNAMIC = {"processor_res", "threshold_a", "threshold_b"}
# Choices that follow the registry and model folders (catalog).
CATALOG_CHOICES = {"type_filter", "module", "model"}


@pytest.mark.parametrize("tab", harness.TABS)
def test_unit_controls_match_the_legacy_ui(cn, tab):
    from lib_controlnet.external_code import ControlNetUnit
    from lib_controlnet.unit_param import CONTROLS, default_value

    snapshot = harness.load_snapshot(SCRIPT, tab)
    default = default_value(tab == "img2img")
    seen = set()
    for legacy in snapshot["unit_controls"]:
        field = legacy["field"]
        if field in DROPPED:
            continue
        key = LEGACY_FIELDS.get(field, field)
        control = CONTROLS[key]
        seen.add(key)
        assert control["label"] == legacy["label"], field
        assert control.get("info") == legacy.get("info"), field
        if "minimum" in legacy and key not in DYNAMIC:
            assert (control["minimum"], control["maximum"], control["step"]) == (legacy["minimum"], legacy["maximum"], legacy["step"]), field
        if "choices" in legacy and key not in CATALOG_CHOICES:
            assert control["choices"] == legacy["choices"], field
        if key in default:
            assert default[key] == legacy["default"], field
        elif key == "timestep_range":
            assert [default["guidance_start"], default["guidance_end"]] == legacy["default"]
        elif key.startswith("region_"):
            assert control["default"] == legacy["default"], field
    expected = {key for key in CONTROLS if tab == "img2img" or key != "upload_independent"}
    assert seen == expected
    # The infotext fields the Gradio controls were registered under are what a paste reads
    fields = {key.split(" ", 2)[2] for key in snapshot["infotext_keys"]}
    assert fields == set(ControlNetUnit.infotext_fields())


@pytest.mark.parametrize("tab", harness.TABS)
def test_the_legacy_panel_had_no_webcam(tab):
    """The "Enable webcam" / "Mirror webcam" tooltips of the Gradio panel had
    no controls: nothing to port (see the module docstring)."""

    snapshot = harness.load_snapshot(SCRIPT, tab)
    described = json.dumps([snapshot["unit_controls"], snapshot["args"], snapshot["unit_args"]]).lower()
    assert "webcam" not in described and "camera" not in described and "mirror" not in described
    assert len(snapshot["unit_controls"]) == (24 if tab == "img2img" else 23)


@pytest.mark.parametrize(("count", "tab", "names"), [(1, "txt2img", ["unit_0"]), (5, "img2img", ["unit_0", "unit_1", "unit_2", "unit_3", "unit_4", "batch_input_dir"])])
def test_the_unit_count_follows_the_setting(cn, monkeypatch, count, tab, names):
    from modules import shared

    monkeypatch.setitem(shared.opts.data, "control_net_unit_count", count)
    assert layout(cn, tab).names == names


def test_default_arguments_are_the_legacy_default_units(cn):
    from lib_controlnet.external_code import ControlNetUnit

    legacy = ControlNetUnit(enabled=False, module="None", model="None")
    args = layout(cn, "img2img").values_to_args({})
    assert [settings_of(unit) for unit in args[:3]] == [settings_of(legacy)] * 3
    assert all(unit.image is None and unit.generated_image is None for unit in args[:3])
    assert args[3] == ""


def test_schema_export(cn, registry):
    schema = layout(cn, "img2img").schema()
    json.dumps(schema, allow_nan=False)
    assert schema["type"] == "section" and schema["title"] == "ControlNet Integrated" and schema["open"] is False
    tabs, batch_dir = schema["children"]
    assert [tab["title"] for tab in tabs["children"]] == ["ControlNet Unit 1", "ControlNet Unit 2", "ControlNet Unit 3"]
    first, unit = (tabs["children"][i]["children"][0] for i in (0, 1))
    assert unit["type"] == "custom" and unit["widget"] == "controlnet-unit" and unit["name"] == "unit_1" and unit["label"] == ""
    assert unit["infotext"] == "ControlNet 1"
    assert unit["props"]["unit"] == 1 and unit["props"]["img2img"] is True and unit["props"]["tabs"] == [tab["title"] for tab in tabs["children"]]
    # The catalog ships once per tab, with the first unit (the widget of every unit reads it there)
    assert "catalog" not in unit["props"] and unit["props"]["controls"] == first["props"]["controls"]
    props = first["props"]
    assert [p["name"] for p in props["catalog"]["preprocessors"]] == ["None", "canny", "reference_test"]
    assert props["catalog"]["models"] == MODELS
    types = {entry["name"]: entry for entry in props["catalog"]["types"]}
    assert list(types) == ["All", "Canny", "Reference", "Region"]
    # The old "Control Type" switch: the second entry of the filtered lists
    # Models by name: the type, its preprocessors' filters, and union models
    assert types["Canny"] == {"name": "Canny", "modules": ["None", "canny"], "models": ["None", "control_v11p_sd15_canny", "t2i_union_promax"], "default_module": "canny", "default_model": "control_v11p_sd15_canny"}
    assert types["All"]["default_module"] == "None" and types["All"]["default_model"] == "None"
    canny = props["catalog"]["preprocessors"][1]
    assert canny["slider_1"] == {"label": "Low Threshold", "minimum": 0, "maximum": 256, "step": 1, "value": 100, "visible": True}
    assert canny["model"] is True and props["catalog"]["preprocessors"][2]["model"] is False
    assert batch_dir["name"] == "batch_input_dir" and batch_dir["visible_if"] == {"param": "core.mode", "eq": "Batch"}


def test_the_widget_contract_is_current(cn, registry):
    """webui/src/widgets/controlnet-unit/contract.json is what the server
    exports (the widget's tests read it). ``REMI_UPDATE_CONTRACT=1`` rewrites it."""

    from lib_controlnet.unit_param import CONTROLS, PASTE_MARKER, default_value

    schema = layout(cn, "txt2img").schema()
    props = schema["children"][0]["children"][0]["children"][0]["props"]
    contract = {
        "paste_marker": PASTE_MARKER,
        "default": {"txt2img": default_value(False), "img2img": default_value(True)},
        "controls": CONTROLS,
        "props": props,
    }
    text = json.dumps(contract, indent=1, ensure_ascii=False) + "\n"
    if os.environ.get("REMI_UPDATE_CONTRACT"):
        CONTRACT.write_text(text, encoding="utf8")
    assert CONTRACT.read_text(encoding="utf8") == text, "stale widget contract: rerun with REMI_UPDATE_CONTRACT=1"


# endregion

# region Values


def test_the_unit_value_is_normalized(cn):
    from lib_controlnet.unit_param import normalize

    from remi import ui

    value = normalize({"enabled": "true", "weight": 3, "guidance_start": 0.9, "guidance_end": 0.2, "processor_res": 511.6, "threshold_a": -5, "resize_mode": "Inner Fit (Scale to Fit)", "control_mode": 2, "hr_option": "HiResFixOption.LOW_RES_ONLY", "__paste__": True, "unknown": 1}, False)
    assert value["enabled"] is True and value["weight"] == 2.0
    assert (value["guidance_start"], value["guidance_end"]) == (0.2, 0.9)
    assert value["processor_res"] == 512 and value["threshold_a"] == -1
    assert (value["resize_mode"], value["control_mode"], value["hr_option"]) == ("Crop and Resize", "ControlNet is more important", "Low res only")
    assert "__paste__" not in value and "unknown" not in value and "upload_independent" not in value
    assert normalize({}, True)["upload_independent"] is False
    for bad in ({"weight": "heavy"}, {"enabled": "yes"}, {"module": 3}, {"image": ""}, {"control_mode": "Loud"}, {"pose": "x"}, {"weight": float("nan")}, []):
        with pytest.raises(ui.ParamError):
            normalize(bad, False)


def test_a_refused_field_keeps_its_reason_for_the_ui(cn):
    """A field's error was made ``f"{key}: {error}"``: English text the UI could not translate."""
    from lib_controlnet.unit_param import normalize

    from remi import ui

    for bad, reason in (({"weight": "heavy"}, "expected a number, got {{value}}"), ({"pose": "x"}, "a pose is an OpenPose JSON object")):
        with pytest.raises(ui.ParamError) as caught:
            normalize(bad, False)
        error = caught.value.to_json()
        field = next(iter(bad))
        assert error["template"] == "{{field}}: {{reason}}" and error["params"]["field"] == field
        assert error["params"]["reason.template"] == reason and error["message"].startswith(f"{field}: ")


def test_slider_values_stay_inside_the_preprocessor_sliders(cn, registry):
    """The resolution and thresholds were only floored at -1 (unset): an API unit could send
    ``processor_res: 1e308`` to a preprocessor that resizes to it. They are clamped to the
    unit's preprocessor's sliders, as a Slider clamps; -1 stays unset, and a slider the
    preprocessor hides (it does not read it) keeps its value."""
    from lib_controlnet.unit_param import normalize

    canny = normalize({"module": "canny", "processor_res": 1e308, "threshold_a": 1e9, "threshold_b": -0.5}, False)
    assert (canny["processor_res"], canny["threshold_a"], canny["threshold_b"]) == (2048, 256, -1)
    low = normalize({"module": "canny", "processor_res": 3, "threshold_a": 0.5, "threshold_b": 300}, False)
    assert (low["processor_res"], low["threshold_a"], low["threshold_b"]) == (128, 0.5, 256)
    assert type(low["processor_res"]) is int
    reference = normalize({"module": "reference_test", "processor_res": 4096, "threshold_a": 7, "threshold_b": 9}, False)
    assert (reference["processor_res"], reference["threshold_a"], reference["threshold_b"]) == (4096, 1.0, 9)
    assert normalize({"module": "unknown", "processor_res": 4096}, False)["processor_res"] == 4096  # refused when it runs (_check)


def test_detect_keeps_its_values_inside_the_preprocessor_sliders(cn, registry):
    """``POST /controlnet/detect`` passed its resolution and thresholds on as sent
    (``controlnet_processor_res: 100000`` resized to it): they are clamped as a unit's are.
    A module it does not know was a KeyError (500) instead of its own 422."""
    import base64

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from lib_controlnet.api import controlnet_api

    from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter

    seen = []

    class Recorder(Preprocessor):
        def __init__(self):
            super().__init__()
            self.name = "detect_test"
            self.slider_resolution = PreprocessorParameter(label="Resolution", value=512, minimum=128, maximum=2048, step=8, visible=True)
            self.slider_1 = PreprocessorParameter(label="Low Threshold", value=100, minimum=1, maximum=255, step=1, visible=True)
            self.slider_2 = PreprocessorParameter(visible=False)

        def __call__(self, image, resolution, slider_1=None, slider_2=None, **kwargs):
            seen.append((resolution, slider_1, slider_2))
            return image

    registry.supported_preprocessors["detect_test"] = Recorder()
    app = FastAPI()
    controlnet_api(None, app)
    image = base64.b64encode(png(Image.new("RGB", (8, 8)))).decode()
    body = {"controlnet_module": "detect_test", "controlnet_input_images": [image], "controlnet_processor_res": 100000, "controlnet_threshold_a": 1e9, "controlnet_threshold_b": 1e9}
    with TestClient(app) as client:
        assert client.post("/controlnet/detect", json=body).status_code == 200
        assert client.post("/controlnet/detect", json={**body, "controlnet_processor_res": 3, "controlnet_threshold_a": 0.5}).status_code == 200
        assert client.post("/controlnet/detect", json={**body, "controlnet_processor_res": -1}).status_code == 200  # unset: the preprocessor's default
        assert client.post("/controlnet/detect", json={**body, "controlnet_module": "no_such_module"}).status_code == 422
    assert seen == [(2048, 255, 1e9), (128, 1, 1e9), (-1, 255, 1e9)]


def test_processing_args_are_what_the_legacy_controls_built(cn, registry, files):
    """The same settings and pictures, once as the Gradio unit group passed
    them (``UiControlNetUnit(*unit_args)``, the canvases' RGBA arrays) and
    once as the widget's JSON with uploads, give the same ``ControlNetUnit``.
    Painted masks only carry their alpha (the brush colour was never read)."""

    from lib_controlnet.external_code import UiControlNetUnit

    rng = np.random.default_rng(1)
    picture = Image.fromarray(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8))
    alpha = np.zeros((12, 16), dtype=np.uint8)
    alpha[2:6, 3:9] = 255
    painted = np.zeros((12, 16, 4), dtype=np.uint8)
    painted[..., 3] = alpha
    mask_picture = Image.fromarray(np.where(rng.random((12, 16)) > 0.5, 255, 0).astype(np.uint8))
    preview = Image.fromarray(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8))

    def upload(image):
        return files.add_upload_bytes(png(image), "x.png").id

    value = {
        "enabled": True,
        "module": "canny",
        "model": "control_v11p_sd15_canny",
        "weight": 0.85,
        "resize_mode": "Resize and Fill",
        "processor_res": 640,
        "threshold_a": 90,
        "threshold_b": 210,
        "guidance_start": 0.1,
        "guidance_end": 0.8,
        "pixel_perfect": True,
        "control_mode": "My prompt is more important",
        "hr_option": "Low res only",
        "image": upload(picture),
        "image_fg": upload(Image.fromarray(np.dstack([np.full_like(alpha, 255)] * 3 + [alpha]))),
        "mask_upload": True,
        "mask_image": upload(mask_picture),
        "mask_image_fg": upload(Image.fromarray(np.dstack([np.full_like(alpha, 255)] * 3 + [alpha]))),
        "generated_image": upload(preview),
        "use_preview_as_input": True,
    }
    from remi import ui

    new = layout(cn).values_to_args({"unit_0": value}, ui.Context(resolve_upload=resolver(files)))[0]

    rgba = lambda image: np.asarray(image.convert("RGBA"))  # noqa: E731 - what LogicalImage gave
    unit_args = (
        True,  # use_preview_as_input
        rgba(preview),  # generated_image.background
        rgba(mask_picture),  # mask_image.background
        painted,  # mask_image.foreground
        "Low res only",  # hr_option
        True,  # enabled
        "canny",
        "control_v11p_sd15_canny",
        0.85,
        rgba(picture),  # image.background
        painted,  # image.foreground
        "Resize and Fill",
        640,
        90,
        210,
        0.1,
        0.8,
        True,  # pixel_perfect
        "My prompt is more important",
    )
    old = UiControlNetUnit(*unit_args)

    assert settings_of(new) == settings_of(old)
    for name in ("image", "generated_image", "mask_image"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name), err_msg=name)
    for name in ("image_fg", "mask_image_fg"):
        np.testing.assert_array_equal(getattr(new, name)[:, :, 3], getattr(old, name)[:, :, 3], err_msg=name)


def test_a_sketch_is_painted_into_the_control_image(cn, registry, files):
    from remi import ui

    base = files.add_upload_bytes(png(Image.new("RGB", (4, 4), "white")), "base.png").id
    stroke = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    stroke.putpixel((1, 2), (255, 0, 0, 255))
    sketch = files.add_upload_bytes(png(stroke), "sketch.png").id
    value = {"enabled": True, "module": "None", "model": "control_v11p_sd15_canny", "image": base, "sketch": sketch}
    unit = layout(cn).values_to_args({"unit_0": value}, ui.Context(resolve_upload=resolver(files)))[0]
    assert tuple(unit.image[2, 1]) == (255, 0, 0, 255) and tuple(unit.image[0, 0]) == (255, 255, 255, 255)


def test_img2img_units_read_the_img2img_image_unless_independent(cn, registry, files):
    from remi import ui

    image = files.add_upload_bytes(png(Image.new("RGB", (4, 4))), "a.png").id
    ctx = ui.Context(resolve_upload=resolver(files))
    value = {"enabled": True, "module": "canny", "model": "control_v11p_sd15_canny", "image": image, "generated_image": image, "use_preview_as_input": True}
    shared_input = layout(cn, "img2img").values_to_args({"unit_0": value}, ctx)[0]
    assert shared_input.image is None and shared_input.generated_image is None and shared_input.use_preview_as_input is False
    own = layout(cn, "img2img").values_to_args({"unit_0": {**value, "upload_independent": True}}, ctx)[0]
    assert own.image.shape == (4, 4, 4) and own.use_preview_as_input is True


def test_enabled_units_are_checked_before_the_job(cn, registry, files):
    from remi import ui

    lay = layout(cn)
    ctx = ui.Context(resolve_upload=files.check_upload)
    image = files.add_upload_bytes(png(Image.new("RGB", (4, 4))), "a.png").id
    good = {"enabled": True, "module": "canny", "model": "control_v11p_sd15_canny", "image": image}
    lay.values_to_args({"unit_0": good}, ctx)
    for change, message in (
        ({"model": "None"}, "select a ControlNet model"),
        ({"model": "gone"}, "was not found"),
        ({"module": "nope"}, "unknown preprocessor"),
        ({"image": None}, "upload a control image"),
        ({"image": "missing-upload-id"}, "unknown upload"),
        # the colour layer painted over the image: read with it, so checked with it
        ({"sketch": "missing-upload-id"}, "unknown upload"),
    ):
        with pytest.raises(ui.ValidationError) as error:
            lay.values_to_args({"unit_0": {**good, **change}}, ctx)
        assert message in str(error.value), change
    # A model-free preprocessor, a disabled unit with stale uploads, a preview used as input
    lay.values_to_args({"unit_0": {**good, "module": "reference_test", "model": "None"}}, ctx)
    lay.values_to_args({"unit_0": {**good, "enabled": False, "image": "missing-upload-id", "sketch": "missing-upload-id", "model": "gone"}}, ctx)
    lay.values_to_args({"unit_0": {**good, "image": None, "generated_image": image, "use_preview_as_input": True}}, ctx)


def test_api_units_and_the_batch_directory_reach_processing(cn, monkeypatch):
    """``/sdapi`` sends unit dicts: one per unit slot, then (img2img) the
    batch directory; arguments past the slots are ignored, as before."""

    from lib_controlnet.external_code import ControlNetUnit

    from modules import shared

    script = cn.ControlNetForForgeOfficial()
    layout(cn, "img2img")  # three units
    on = {"enabled": True, "module": "canny", "weight": 0.5}
    args = [on, ControlNetUnit(enabled=False), "not a unit", "/data/control", {**on, "weight": 0.25}]
    units = script.get_enabled_units(args)
    assert len(units) == 1 and isinstance(units[0], ControlNetUnit) and units[0].weight == 0.5
    assert script.get_batch_input_dir(args) == "/data/control"
    assert script.get_batch_input_dir(args[:3]) == "" and script.get_batch_input_dir([on, on, on, on]) == ""
    # ADetailer runs ControlNet with one unit argument
    assert len(script.get_enabled_units([ControlNetUnit(enabled=True)])) == 1
    # With directory settings hidden no request may point ControlNet at a server directory
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    assert script.get_batch_input_dir(args) == ""


def test_api_units_decode_the_image_mask_pair():
    """The A1111 API also sends ``"image": {"image": b64, "mask": b64}``."""

    import base64

    from lib_controlnet.external_code import ControlNetUnit

    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), (200, 10, 10)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    unit = ControlNetUnit.from_dict({"enabled": True, "image": {"image": encoded, "mask": None}})
    assert unit.image["image"].shape == (4, 6, 3) and unit.image["image"][0, 0, 0] == 200 and unit.image["mask"] is None


def test_the_batch_directory_supplies_control_images(cn, registry, tmp_path):
    from lib_controlnet.external_code import ControlNetUnit

    from modules.processing import StableDiffusionProcessingImg2Img

    for name, colour in (("a", (10, 0, 0)), ("b", (20, 0, 0))):
        Image.new("RGB", (8, 8), colour).save(tmp_path / f"{name}.png")
    source = Image.new("RGB", (8, 8))
    source.filename = str(tmp_path / "b.png")

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    p.init_images, p.image_mask, p.resize_mode, p.inpaint_full_res = [source], None, 0, False
    script = cn.ControlNetForForgeOfficial()
    script.batch_input_dir = str(tmp_path)
    unit = ControlNetUnit(enabled=True, module="canny")
    unit._idx = 0
    preprocessor = registry.get_preprocessor("canny")
    [(image, mask)], _ = script.get_input_data(p, unit, preprocessor, 8, 8)
    assert image[0, 0, 0] == 20 and mask is None  # the file of the same name
    source.filename = "elsewhere.png"
    [(image, _)], _ = script.get_input_data(p, unit, preprocessor, 8, 8)
    assert image[0, 0, 0] == 10  # else the next file


# endregion

# region Infotext


def test_infotext_round_trips_through_the_unit(cn, registry):
    from lib_controlnet.external_code import ControlNetUnit
    from lib_controlnet.infotext import Infotext

    source = ControlNetUnit(enabled=True, module="canny", model="control_v11p_sd15_canny", weight=0.75, resize_mode="Resize and Fill", processor_res=768, threshold_a=64, threshold_b=128, guidance_start=0.05, guidance_end=0.6, pixel_perfect=True, control_mode="Balanced", hr_option="High res only")
    p = SimpleNamespace(extra_generation_params={})
    Infotext.write_infotext([source], p)
    params = dict(p.extra_generation_params)

    lay = layout(cn)
    values, problems = lay.paste(params)
    assert problems == []
    pasted = values["unit_0"]
    assert pasted["__paste__"] is True and "enabled" not in pasted and "image" not in pasted
    assert set(pasted) == {"__paste__", *ControlNetUnit.infotext_fields()}
    unit = lay.values_to_args({"unit_0": {**pasted, "enabled": True, "image": "id"}})[0]
    assert {field: plain(getattr(unit, field)) for field in ControlNetUnit.infotext_fields()} == {field: plain(getattr(source, field)) for field in ControlNetUnit.infotext_fields()}


def test_api_written_infotext_and_bad_fields_paste(cn):
    """Every infotext field is pasted, as the old paste of a whole
    ``ControlNetUnit`` did: the ones the entry leaves out (the infotext
    omits an unset -1 resolution or threshold) take the unit defaults, so a
    paste does not keep the current unit's values. A field that does not
    read keeps the current value."""

    lay = layout(cn)
    values, problems = lay.paste({"ControlNet 1": "Module: canny, Weight: heavy, Resize Mode: ResizeMode.OUTER_FIT, Hr Option: HiResFixOption.BOTH, Control Mode: ControlMode.CONTROL"})
    assert problems == []
    assert values == {
        "unit_1": {
            "__paste__": True,
            "module": "canny",
            "model": "None",
            "resize_mode": "Resize and Fill",
            "processor_res": -1,
            "threshold_a": -1,
            "threshold_b": -1,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "pixel_perfect": False,
            "control_mode": "ControlNet is more important",
            "hr_option": "Both",
        }
    }
    _, problems = lay.paste({"ControlNet 0": "garbage"})
    assert [problem.param for problem in problems] == ["unit_0"]


def test_model_and_preprocessor_names_paste_as_written(cn):
    """A name that reads as a number or a boolean ("Model: 12345") stays the
    name: read as an int it was not text, and the paste dropped the field."""

    from lib_controlnet.unit_param import pasted_fields

    fields = pasted_fields("Module: 007, Model: 12345, Weight: 1")
    assert (fields["module"], fields["model"], fields["weight"]) == ("007", "12345", 1.0)
    assert pasted_fields("Module: True, Model: 1.50")["model"] == "1.50"
    assert pasted_fields("Module: True, Model: 1.50")["module"] == "True"
    values, problems = layout(cn).paste({"ControlNet 0": "Module: canny, Model: 12345"})
    assert problems == [] and values["unit_0"]["model"] == "12345"


def test_reading_units_from_infotext_can_be_switched_off(cn, monkeypatch):
    from modules import shared

    monkeypatch.setitem(shared.opts.data, "control_net_sync_field_args", False)
    assert layout(cn).infotext == {}


def test_units_from_positional_arguments(cn):
    """Old positional arguments (imported queue items, ``args_to_values``)."""

    from lib_controlnet.external_code import ControlNetUnit

    values = layout(cn).args_to_values([ControlNetUnit(enabled=True, module="canny", weight=0.4, image=np.zeros((2, 2, 3), np.uint8)), {"enabled": False, "resize_mode": 0}])
    assert values["unit_0"]["enabled"] is True and values["unit_0"]["weight"] == 0.4 and values["unit_0"]["image"] is None
    assert values["unit_1"]["resize_mode"] == "Just Resize"
    json.dumps(values)


@pytest.mark.parametrize("tab", harness.TABS)
def test_imported_units_keep_their_pictures(cn, registry, files, tab):
    """A job importer that stores images (``Context.store_upload``) gets a
    unit's pictures as uploads, and the unit then gives processing the same
    pictures: arrays, API base64 text and the old ``{image, mask}`` pair."""

    import base64

    from lib_controlnet.external_code import ControlNetUnit

    from remi import ui

    rng = np.random.default_rng(2)
    picture = rng.integers(0, 255, (6, 8, 4), dtype=np.uint8)
    alpha = np.zeros((6, 8), np.uint8)
    alpha[1:3] = 255
    painted = np.dstack([rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), alpha])  # brush colour never read
    preview = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    mask_picture = np.dstack([np.where(rng.random((6, 8)) > 0.5, 255, 0).astype(np.uint8)] * 3 + [np.full((6, 8), 255, np.uint8)])
    old = ControlNetUnit(enabled=True, module="canny", model="control_v11p_sd15_canny", image=picture, image_fg=painted, mask_image=mask_picture, mask_image_fg=painted, generated_image=preview, use_preview_as_input=True)

    lay = layout(cn, tab)
    store = lambda image: files.add_upload_bytes(png(image), "imported.png").id  # noqa: E731
    values = lay.args_to_values([old, {"enabled": True, "module": "canny", "model": "control_v11p_sd15_canny", "image": base64.b64encode(png(Image.fromarray(preview))).decode()}, ControlNetUnit(enabled=True, module="canny", model="control_v11p_sd15_canny", image={"image": preview, "mask": mask_picture})], ui.Context(store_upload=store))
    json.dumps(values)
    assert values["unit_0"]["mask_upload"] is True and values["unit_0"].get("upload_independent", True) is True
    new = lay.values_to_args(values, ui.Context(resolve_upload=resolver(files)))

    assert settings_of(new[0]) == settings_of(old)
    for name in ("image", "mask_image"):
        np.testing.assert_array_equal(getattr(new[0], name), getattr(old, name), err_msg=name)
    np.testing.assert_array_equal(new[0].generated_image[:, :, :3], preview)
    for name in ("image_fg", "mask_image_fg"):
        np.testing.assert_array_equal(getattr(new[0], name)[:, :, 3], alpha, err_msg=name)
    np.testing.assert_array_equal(new[1].image[:, :, :3], preview)
    np.testing.assert_array_equal(new[2].image[:, :, :3], preview)
    np.testing.assert_array_equal(new[2].mask_image, mask_picture)
    # Without an image store the pictures are dropped (and an enabled txt2img unit then asks for one)
    assert lay.args_to_values([old])["unit_0"]["image"] is None


# endregion

# region Actions


def test_preview_runs_the_preprocessor(cn, registry, files):
    script = cn.ControlNetForForgeOfficial()
    picture = Image.new("RGB", (64, 48), "black")
    picture.paste((255, 255, 255), (16, 8, 48, 40))
    image = files.add_upload_bytes(png(picture), "in.png").id
    values = {"unit_1": {"module": "canny", "image": image, "processor_res": 128, "threshold_a": 100, "threshold_b": 200}}
    patch = call(script, "preview", values, files, unit=1)
    preview = patch["__outputs__"]["preview"]
    assert preview["unit"] == 1 and preview["pose"] is None
    edges = np.asarray(Image.open(files.get(preview["image"]).path))
    # Canny works at the preprocessor resolution (the short side), as before
    assert edges.shape[:2] == (128, 171) and edges.max() == 255 and edges[0, 0].max() == 0

    assert call(script, "preview", {"unit_0": {"module": "canny"}}, files, unit=0)["__toast__"]["level"] == "warning"
    assert call(script, "preview", {"unit_0": {"module": "nope", "image": image}}, files, unit=0)["__toast__"]["level"] == "error"


def test_preview_passes_pixel_perfect_resolution_and_the_painted_mask(cn, registry, files, monkeypatch):
    seen = {}
    canny = registry.get_preprocessor("canny")

    def fake(input_image, resolution, slider_1=None, slider_2=None, input_mask=None, json_pose_callback=None, **kwargs):
        seen.update(resolution=resolution, slider_1=slider_1, mask=input_mask, shape=input_image.shape)
        return input_image

    monkeypatch.setattr(type(canny), "__call__", lambda self, *args, **kwargs: fake(*args, **kwargs))
    image = files.add_upload_bytes(png(Image.new("RGB", (100, 50))), "in.png").id
    alpha = np.zeros((50, 100), np.uint8)
    alpha[:10] = 255
    mask = files.add_upload_bytes(png(Image.fromarray(np.dstack([np.full_like(alpha, 255)] * 3 + [alpha]))), "m.png").id
    script = cn.ControlNetForForgeOfficial()
    call(script, "preview", {"unit_0": {"module": "canny", "image": image, "image_fg": mask, "pixel_perfect": True, "resize_mode": "Just Resize"}}, files, core={"width": 1024, "height": 768}, unit=0)
    assert seen["resolution"] == 768  # max(768/50, 1024/100) * 50
    assert seen["slider_1"] == 100  # an unset (-1) threshold is the preprocessor's default
    assert seen["mask"].shape == (50, 100, 3) and seen["mask"][0, 0, 0] == 255 and seen["mask"][20, 0, 0] == 0


def test_pose_rendering(cn, files):
    script = cn.ControlNetForForgeOfficial()
    pose = {"canvas_width": 64, "canvas_height": 32, "people": [{"pose_keypoints_2d": [10, 10, 1, 20, 20, 1] + [0, 0, 0] * 16}]}
    patch = call(script, "render_pose", {}, files, unit=2, pose=json.dumps(pose))
    out = patch["__outputs__"]["pose"]
    assert out["unit"] == 2 and out["pose"] == pose
    with Image.open(files.get(out["image"]).path) as rendered:
        assert rendered.size == (64, 32)
    assert call(script, "render_pose", {}, files, unit=0, pose="{")["__toast__"]["level"] == "error"
    assert call(script, "render_pose", {}, files, unit=0, pose={"people": []})["__toast__"]["level"] == "error"


def test_region_canvas(cn, files):
    from remi import ui

    script = cn.ControlNetForForgeOfficial()
    out = call(script, "region_canvas", {}, files, core={"width": 96, "height": 64}, unit=0, subjects=3, shape="Oval")["__outputs__"]["canvas"]
    picture = np.asarray(Image.open(files.get(out["image"]).path).convert("RGB"))
    assert picture.shape == (64, 96, 3) and (picture != 255).any()
    blank = call(script, "region_canvas", {}, files, unit=0, subjects=0)["__outputs__"]["canvas"]
    assert (np.asarray(Image.open(files.get(blank["image"]).path)) == 255).all()
    for extra in ({"unit": 9}, {"unit": 0, "shape": "Star"}, {"unit": 0, "subjects": "many"}):
        with pytest.raises(ui.ValidationError):
            call(script, "region_canvas", {}, files, **extra)


def test_refresh_models_rescans_and_returns_the_catalog(cn, registry, files, monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "update_controlnet_filenames", lambda: calls.append(1))
    catalog = call(cn.ControlNetForForgeOfficial(), "refresh_models", {}, files)["__outputs__"]["catalog"]
    assert calls == [1] and catalog["models"] == MODELS


def test_preview_reports_a_failing_preprocessor_and_keeps_a_non_image_result(cn, registry, files, monkeypatch):
    canny = registry.get_preprocessor("canny")
    image = files.add_upload_bytes(png(Image.new("RGB", (100, 50), "white")), "in.png").id
    script = cn.ControlNetForForgeOfficial()

    def broken(self, *args, **kwargs):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(type(canny), "__call__", broken)
    toast = call(script, "preview", {"unit_0": {"module": "canny", "image": image}}, files, unit=0)["__toast__"]
    assert toast["level"] == "error" and "model download failed" in json.dumps(toast)

    # a result that is no image (e.g. a preprocessor that only reports a pose) previews the input
    monkeypatch.setattr(type(canny), "__call__", lambda self, *args, **kwargs: None)
    preview = call(script, "preview", {"unit_0": {"module": "canny", "image": image}}, files, unit=0)["__outputs__"]["preview"]
    with Image.open(files.get(preview["image"]).path) as shown:
        assert shown.size == (100, 50) and np.asarray(shown.convert("RGB")).min() == 255


def test_preview_scales_a_painted_mask_to_the_image_and_reads_a_bad_size_as_512(cn, registry, files, monkeypatch):
    seen = {}
    canny = registry.get_preprocessor("canny")

    def fake(self, input_image, resolution, input_mask=None, **kwargs):
        seen.update(resolution=resolution, mask=input_mask)
        return input_image

    monkeypatch.setattr(type(canny), "__call__", fake)
    image = files.add_upload_bytes(png(Image.new("RGB", (100, 50))), "in.png").id
    alpha = np.zeros((25, 50), np.uint8)
    alpha[:5] = 255  # the top fifth, painted at half the image's size
    mask = files.add_upload_bytes(png(Image.fromarray(np.dstack([np.full_like(alpha, 255)] * 3 + [alpha]))), "m.png").id
    values = {"unit_0": {"module": "canny", "image": image, "image_fg": mask, "pixel_perfect": True, "resize_mode": "Just Resize"}}
    call(cn.ControlNetForForgeOfficial(), "preview", values, files, core={"width": "wide", "height": 512}, unit=0)
    assert seen["mask"].shape == (50, 100, 3) and seen["mask"][9, 0, 0] == 255 and seen["mask"][10, 0, 0] == 0
    assert seen["resolution"] == 512  # an unreadable width counts as 512: max(512/50, 512/100) * 50


def test_actions_need_the_file_store(cn):
    from remi import ui

    ctx = ui.ActionContext(tab="txt2img", owner="controlnet", core={"width": 512, "height": 512}, files=None, extra={"unit": 0})
    with pytest.raises(ui.ValidationError, match="file store"):
        ui.call_action(cn.ControlNetForForgeOfficial(), "region_canvas", ctx, {})


def test_pose_size_limit_and_animal_poses(cn, files):
    from lib_controlnet import unit_actions

    script = cn.ControlNetForForgeOfficial()
    huge = json.dumps({"canvas_width": 64, "canvas_height": 32, "people": [], "pad": "x" * unit_actions.MAX_POSE_BYTES})
    assert call(script, "render_pose", {}, files, unit=0, pose=huge)["__toast__"]["level"] == "error"
    # an animal (AP-10K, 17 keypoints) is drawn when the pose holds no human
    animal = [coordinate for i in range(17) for coordinate in (8 + 3 * i, 4 + i, 1)]
    out = call(script, "render_pose", {}, files, unit=1, pose={"canvas_width": 64, "canvas_height": 32, "people": [], "animals": [animal]})["__outputs__"]["pose"]
    with Image.open(files.get(out["image"]).path) as rendered:
        assert rendered.size == (64, 32) and np.asarray(rendered.convert("RGB")).any()


# Why the pose editor refuses a pose (lib_controlnet/openpose.py): the toast shows the sentence
POSE_REASONS = {
    "The pose list is empty",
    "A pose is a JSON object",
    "The pose's {{key}} must be a size between 1 and {{max}}",
    "The pose's {{key}} must be a list",
    "Not an OpenPose pose: {{error}}",
}


def test_check_pose_reads_the_openpose_json_shapes(cn):
    from lib_controlnet import openpose

    from remi import ui

    pose = {"canvas_width": 64, "canvas_height": 32, "people": []}
    assert openpose.check_pose(pose) is pose
    assert openpose.check_pose([pose, {"canvas_width": 1, "canvas_height": 1}]) is pose  # one pose per image: the first
    for bad, message in (
        ([], "The pose list is empty"),
        ("pose", "A pose is a JSON object"),
        ({"canvas_width": 64}, "The pose's canvas_height must be a size"),
        ({"canvas_width": True, "canvas_height": 32}, "The pose's canvas_width must be a size"),
        ({"canvas_width": float("nan"), "canvas_height": 32}, "The pose's canvas_width must be a size"),
        ({"canvas_width": openpose.MAX_CANVAS + 1, "canvas_height": 32}, "The pose's canvas_width must be a size"),
        ({**pose, "people": {}}, "The pose's people must be a list"),
        ({**pose, "animals": "cat"}, "The pose's animals must be a list"),
        ({**pose, "people": [123]}, "Not an OpenPose pose"),
        ({**pose, "people": [None]}, "Not an OpenPose pose"),
        ({**pose, "animals": [123]}, "Not an OpenPose pose"),
    ):
        with pytest.raises(ValueError, match=message) as refused:
            openpose.check_pose(bad)
        # the reason is a sentence the UI translates (its template is the catalog key)
        assert isinstance(refused.value.args[0], ui.Sentence) and refused.value.args[0].template in POSE_REASONS
    # keypoints that are no (x, y, confidence) triples are no OpenPose pose
    with pytest.raises(ValueError, match="Not an OpenPose pose") as refused:
        openpose.render_pose({**pose, "people": [{"pose_keypoints_2d": [1, 2]}]})
    assert refused.value.args[0].template in POSE_REASONS


def test_a_refused_pose_says_why_in_a_sentence_the_ui_translates(cn, files):
    # the reason went into "Not an OpenPose pose: {{error}}" as English text
    script = cn.ControlNetForForgeOfficial()
    toast = call(script, "render_pose", {}, files, unit=0, pose={"canvas_width": 64, "people": []})["__toast__"]
    assert toast == {"level": "error", "message": {"text": "The pose's {{key}} must be a size between 1 and {{max}}", "params": {"key": "canvas_height", "max": "4096"}}}
    assert call(script, "render_pose", {}, files, unit=0, pose=[])["__toast__"]["message"] == "The pose list is empty"
    finite = call(script, "render_pose", {}, files, unit=0, pose={"canvas_width": float("inf"), "canvas_height": 8})["__toast__"]["message"]
    assert finite == {"text": "{{path}} must be a finite number, got {{value}}", "params": {"path": "value.canvas_width", "value": "inf"}}


def test_render_pose_and_normalize_survive_deep_json_without_recursion_error(cn, files, monkeypatch):
    """deeply nested pose strings/objects return a validation toast or normalize
    without raising RecursionError."""
    from lib_controlnet import unit_actions
    from lib_controlnet.unit_param import normalize

    script = cn.ControlNetForForgeOfficial()
    deep_str = "[" * 2000 + "1" + "]" * 2000
    toast_str = call(script, "render_pose", {}, files, unit=0, pose=deep_str)["__toast__"]
    assert toast_str["level"] == "error"
    assert toast_str["message"] in ("The pose is not valid JSON", "A pose is a JSON object")

    monkeypatch.setattr(unit_actions.json, "loads", lambda _s: (_ for _ in ()).throw(RecursionError("depth")))
    toast_rec = call(script, "render_pose", {}, files, unit=0, pose="[]")["__toast__"]
    assert toast_rec == {"level": "error", "message": "The pose is not valid JSON"}

    deep_dict: dict = {"leaf": 1}
    for _ in range(1199):
        deep_dict = {"child": deep_dict}
    toast_dict = call(script, "render_pose", {}, files, unit=0, pose=deep_dict)["__toast__"]
    assert toast_dict["level"] == "error"

    normalized = normalize({"enabled": False, "pose": deep_dict}, is_img2img=False)
    assert normalized["pose"] is deep_dict


def test_render_pose_rejects_malformed_entries_and_out_of_range_coordinates(cn, files):
    """non-dict people, non-list animals, and large keypoint coordinates
    raise ValueError(Sentence(...)) in openpose.render_pose and return error toasts."""
    from lib_controlnet import openpose

    from remi import ui

    script = cn.ControlNetForForgeOfficial()
    bad_poses = [
        {"canvas_width": 512, "canvas_height": 512, "people": [123]},
        {"canvas_width": 512, "canvas_height": 512, "people": [None]},
        {"canvas_width": 512, "canvas_height": 512, "people": [{}]},
        {"canvas_width": 512, "canvas_height": 512, "people": [], "animals": [123]},
        {"canvas_width": 512, "canvas_height": 512, "people": [], "animals": [[1e15, 1e15, 1.0] * 17]},
        {"canvas_width": 512, "canvas_height": 512, "people": [{"pose_keypoints_2d": [1e200, 1e200, 1.0] + [0.0, 0.0, 1.0] * 17}]},
    ]
    for bad in bad_poses:
        with pytest.raises(ValueError) as refused:
            openpose.render_pose(bad)
        assert isinstance(refused.value.args[0], ui.Sentence)
        assert refused.value.args[0].template in POSE_REASONS

        toast = call(script, "render_pose", {}, files, unit=0, pose=bad)["__toast__"]
        assert toast["level"] == "error"

    human_img = openpose.render_pose({"canvas_width": 64, "canvas_height": 32, "people": [{"pose_keypoints_2d": [10, 10, 1, 20, 20, 1] + [0, 0, 0] * 16}]})
    assert human_img.shape == (32, 64, 3) and human_img.dtype == np.uint8 and human_img.any()
    animal = [coord for i in range(17) for coord in (8 + 3 * i, 4 + i, 1)]
    animal_img = openpose.render_pose({"canvas_width": 64, "canvas_height": 32, "people": [], "animals": [animal]})
    assert animal_img.shape == (32, 64, 3) and animal_img.dtype == np.uint8 and animal_img.any()


# endregion


def test_zh_cn_covers_the_panel(cn, registry):
    from lib_controlnet.unit_param import CONTROLS, INVERT_HINT

    from modules import shared

    cn.on_ui_settings()
    schemas = [layout(cn, tab).schema() for tab in harness.TABS]
    props = schemas[0]["children"][0]["children"][0]["children"][0]["props"]
    texts = {INVERT_HINT, *(f"ControlNet Unit {i}" for i in range(1, 6))}
    for control in CONTROLS.values():
        texts |= {control["label"], *control.get("choices", ()), *([control["info"]] if control.get("info") else [])}
    texts |= {entry["name"] for entry in props["catalog"]["types"]}
    texts |= {spec["label"] for p in props["catalog"]["preprocessors"] for spec in (p["resolution"], p["slider_1"], p["slider_2"]) if spec["visible"] and p["name"] != "reference_test"}
    texts |= {"Upload a control image first", "Unknown preprocessor {{name}}", "The preprocessor failed: {{error}}", "The pose file is too large", "The pose is not valid JSON", *POSE_REASONS}
    texts |= {info.label for key, info in shared.opts.data_labels.items() if key.startswith("control_net_")} | {"ControlNet", "e.g. training output directory"}
    harness.assert_localized(EXTENSION, schemas, extra=texts)


def test_a_generation_that_raised_has_its_models_undo_their_patches(cn):
    """postprocess_batch_list, where a model undoes what it patched (Anima ControlNet-LLLite swaps the
    DiT's own Linear forwards), is never reached by a generation that raised: cleanup does it."""
    from types import SimpleNamespace

    undone = []
    model = SimpleNamespace(process_after_every_sampling=lambda p, params: undone.append(params))
    script = cn.ControlNetForForgeOfficial()
    with_model, without_model = SimpleNamespace(model=model, preprocessor=None), SimpleNamespace(model=None, preprocessor=None)
    script.current_params = {0: with_model, 1: without_model}

    script.cleanup(SimpleNamespace())
    script.cleanup(SimpleNamespace())  # after a successful postprocess, nothing is left to undo

    assert undone == [with_model] and script.current_params == {}


def test_inpaint_only_keeps_the_hires_size_of_a_unit_that_sat_out_the_hires_pass(cn, monkeypatch):
    """An inpaint_only unit set to "Low res only" holds the first pass's image and mask when the
    hires image comes back: the blend fits them to the image instead of scaling it back down."""
    from types import SimpleNamespace

    import torch

    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_preprocessors", dict(forge_shared.supported_preprocessors))
    inpaint = harness.load_script(harness.REPO / "extensions-builtin" / "forge_preprocessor_inpaint" / "scripts" / "preprocessor_inpaint.py", "preprocessor_inpaint_under_test")
    preprocessor = inpaint.PreprocessorInpaintOnly()
    preprocessor.image = torch.zeros(1, 3, 32, 32)
    preprocessor.mask = torch.zeros(1, 1, 32, 32)
    preprocessor.mask[..., 8:24, 8:24] = 1
    batch = SimpleNamespace(images=[torch.ones(3, 64, 64)])

    preprocessor.process_after_every_sampling(SimpleNamespace(), SimpleNamespace(), batch)

    (image,) = batch.images
    assert image.shape == (3, 64, 64)
    assert image[:, 32, 32].eq(1).all() and image[:, 0, 0].eq(0).all()  # generated inside the mask, the original outside

    # The original and mask ControlNet prepared at the hires size are used when there are some
    hr_mask = torch.zeros(1, 1, 64, 64)
    hr_mask[..., 16:48, 16:48] = 1
    params = SimpleNamespace(control_cond_for_hr_fix=torch.full((1, 3, 64, 64), 0.5), control_mask_for_hr_fix=hr_mask)
    batch = SimpleNamespace(images=[torch.ones(3, 64, 64)])

    preprocessor.process_after_every_sampling(SimpleNamespace(), params, batch)

    (image,) = batch.images
    assert image.shape == (3, 64, 64)
    assert image[:, 32, 32].eq(1).all() and image[:, 0, 0].eq(0.5).all()


def test_anima_lllite_follows_a_hires_checkpoint_to_its_dit(cn, monkeypatch):
    """A hires checkpoint swaps the DiT between the passes: the hires pass builds the net over the
    DiT it runs, and the first pass's patches are undone (not left on a model that no longer runs)."""
    from types import SimpleNamespace

    import torch

    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_control_models", list(forge_shared.supported_control_models))
    lllite = harness.load_script(harness.REPO / "extensions-builtin" / "sd_forge_controlllite" / "scripts" / "forge_controllllite.py", "forge_controllllite_under_test")

    class Net:
        def __init__(self, dit, **cfg):
            self.dit, self.applied, self.restored = dit, 0, 0

        def eval(self):
            return self

        def to(self, **kwargs):
            return self

        def set_cond_image(self, image): ...
        def set_multiplier(self, strength): ...
        def set_step_range(self, **kwargs): ...

        def apply_to(self):
            self.applied += 1

        def restore(self):
            self.restored += 1

    monkeypatch.setattr(lllite, "ControlNetLLLiteDiT", Net)
    monkeypatch.setattr(lllite, "map_blocks", lambda dit, state_dict: (dit, state_dict))
    monkeypatch.setattr(lllite, "infer_anima_config", lambda state_dict: {})
    monkeypatch.setattr(lllite, "load_lllite_weights_from_dict", lambda net, state_dict: None)

    def process(dit):
        unet = SimpleNamespace(load_device="cpu", model=SimpleNamespace(computation_dtype=torch.float32, diffusion_model=dit))
        return SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)), steps=8, enable_hr=True, hr_checkpoint_info=object())

    patcher = lllite.ControlLLLiteAnimaPatcher({"lllite_dit.weight": torch.zeros(1)}, inpaint=False)
    base, hires = object(), object()
    cond = torch.zeros(1, 3, 8, 8)

    patcher.process_before_every_sampling(process(base), cond, None)
    first = patcher._lllite_net
    patcher.process_before_every_sampling(process(hires), cond, None)

    assert first.dit is base and first.restored == 1
    assert patcher._lllite_net.dit is hires and patcher._lllite_net.applied == 1
    patcher.process_before_every_sampling(process(hires), cond, None)  # the same DiT again: the same net
    second = patcher._lllite_net
    assert second.applied == 2 and second.restored == 0

    patcher.process_after_every_sampling()  # the batch ends: nothing may keep the hires model alive
    assert second.restored == 1 and patcher._lllite_net is None and patcher._dit is None
    patcher.process_before_every_sampling(process(base), cond, None)  # the next batch's first pass
    assert patcher._lllite_net.dit is base


def test_anima_lllite_skips_a_hires_checkpoint_it_does_not_fit(cn, monkeypatch):
    """A hires checkpoint that is not an Anima DiT (or has another block count): the hires pass runs
    without the control, as before; the generation is not failed after its first pass."""
    from types import SimpleNamespace

    import torch

    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_control_models", list(forge_shared.supported_control_models))
    lllite = harness.load_script(harness.REPO / "extensions-builtin" / "sd_forge_controlllite" / "scripts" / "forge_controllllite.py", "forge_controllllite_under_test_unfit")

    class Net:
        def __init__(self, dit, **cfg):
            self.dit, self.applied, self.restored = dit, 0, 0

        def eval(self):
            return self

        def to(self, **kwargs):
            return self

        def set_cond_image(self, image): ...
        def set_multiplier(self, strength): ...
        def set_step_range(self, **kwargs): ...

        def apply_to(self):
            self.applied += 1

        def restore(self):
            self.restored += 1

    def map_blocks(dit, state_dict):
        if dit == "sdxl-unet":
            raise AttributeError("'UNet' object has no attribute 'blocks'")
        return dit, state_dict

    monkeypatch.setattr(lllite, "ControlNetLLLiteDiT", Net)
    def load_weights(net, state_dict):
        if net.dit == "anima-wider":
            raise RuntimeError("size mismatch for lllite_modules.0.down.weight")

    monkeypatch.setattr(lllite, "map_blocks", map_blocks)
    monkeypatch.setattr(lllite, "infer_anima_config", lambda state_dict: {})
    monkeypatch.setattr(lllite, "load_lllite_weights_from_dict", load_weights)

    def process(dit):
        unet = SimpleNamespace(load_device="cpu", model=SimpleNamespace(computation_dtype=torch.float32, diffusion_model=dit))
        return SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)), steps=8, enable_hr=True, hr_checkpoint_info=object(), is_hr_pass=dit != "anima-dit")

    cond = torch.zeros(1, 3, 8, 8)
    hires_only = lllite.ControlLLLiteAnimaPatcher({"lllite_dit.weight": torch.zeros(1)}, inpaint=False)
    hires_only.process_before_every_sampling(process("sdxl-unet"), cond, None)  # "High res only": skipped too
    assert hires_only._lllite_net is None

    wider = lllite.ControlLLLiteAnimaPatcher({"lllite_dit.weight": torch.zeros(1)}, inpaint=False)
    wider.process_before_every_sampling(process("anima-dit"), cond, None)
    wider.process_before_every_sampling(process("anima-wider"), cond, None)  # same blocks, other widths
    assert wider._lllite_net is None

    patcher = lllite.ControlLLLiteAnimaPatcher({"lllite_dit.weight": torch.zeros(1)}, inpaint=False)
    patcher.process_before_every_sampling(process("anima-dit"), cond, None)
    first = patcher._lllite_net

    patcher.process_before_every_sampling(process("sdxl-unet"), cond, None)

    assert first.restored == 1 and patcher._lllite_net is None
    patcher.process_before_every_sampling(process("sdxl-unet"), cond, None)  # again on it: skipped again
    patcher.process_after_every_sampling()  # nothing left to undo
    patcher.process_before_every_sampling(process("anima-dit"), cond, None)  # the next batch's first pass
    assert patcher._lllite_net.dit == "anima-dit" and patcher._lllite_net.applied == 1

    first_pass_on_sdxl = process("sdxl-unet")
    first_pass_on_sdxl.is_hr_pass = False
    with pytest.raises(AttributeError):  # the first pass on a model it does not fit still fails as before
        lllite.ControlLLLiteAnimaPatcher({"lllite_dit.weight": torch.zeros(1)}, inpaint=False).process_before_every_sampling(first_pass_on_sdxl, cond, None)


def _fake_process(seed=1):
    """A processing object for the preprocessors' ``process_before_every_sampling``: a VAE that
    encodes to zeros and a UNet whose clone records the modifiers a preprocessor adds."""

    import torch

    added = {}

    class Unet:
        model = SimpleNamespace(predictor=SimpleNamespace(sigma_data=1.0, percent_to_sigma=lambda percent: 10.0 * (1 - percent)))

        def clone(self):
            return self

        def add_block_modifier(self, f):
            added["block"] = f

        def add_conditioning_modifier(self, f):
            added["conditioning"] = f

        def set_model_replace_all(self, f, name):
            added[name] = f

        def add_sampler_pre_cfg_function(self, f):
            added["pre_cfg"] = f

        def set_model_sampler_post_cfg_function(self, f):
            added["post_cfg"] = f

    vae = SimpleNamespace(encode=lambda image: torch.zeros(1, 4, 8, 8), first_stage_model=SimpleNamespace(process_in=lambda latent: latent))
    process = SimpleNamespace(sd_model=SimpleNamespace(is_sdxl=False, forge_objects=SimpleNamespace(vae=vae, unet=Unet())), seeds=[seed])
    return process, added


def test_tile_colorfix_uses_each_images_sigma_in_a_batch(cn, monkeypatch):
    """A batch of two images calls the UNet with their cond and uncond rows (4 rows) and one sigma
    per image: each row is scaled by its own image's sigma (upstream broadcast the 2 sigmas against
    the image width, failing or mixing the images up)."""
    import torch

    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_preprocessors", dict(forge_shared.supported_preprocessors))
    tile = harness.load_script(harness.REPO / "extensions-builtin" / "forge_preprocessor_tile" / "scripts" / "preprocessor_tile.py", "preprocessor_tile_under_test")
    process, added = _fake_process()
    tile.PreprocessorTileColorFix().process_before_every_sampling(process, torch.zeros(1, 3, 64, 64), None, unit=SimpleNamespace(threshold_a=3))

    sigmas = torch.tensor([1.0, 3.0])
    options = {"sigmas": sigmas, "cond_mark": torch.tensor([0.0, 0.0, 1.0, 1.0])}
    h = torch.ones(4, 4, 8, 8)
    added["block"](h, "before", {**options, "block": ("input", 0)})
    eps = added["block"](torch.zeros(4, 4, 8, 8), "after", {**options, "block": ("last", 0)})

    # the latent is zero and the prediction exact: what is left is x * sqrt(s^2 + 1) / s per row
    expected = torch.stack([(s**2 + 1) ** 0.5 / s for s in torch.cat([sigmas, sigmas])])
    assert eps.shape == h.shape
    assert torch.allclose(eps.mean(dim=(1, 2, 3)), expected, atol=1e-4)


def test_reference_recording_stops_after_an_error_and_its_records_are_released(cn, monkeypatch):
    """An error while recording the reference (out of memory) does not leave the preprocessor
    recording, which would make every later run skip the reference; the recorded tensors are
    released after sampling."""
    import torch

    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_preprocessors", dict(forge_shared.supported_preprocessors))
    reference = harness.load_script(harness.REPO / "extensions-builtin" / "forge_preprocessor_reference" / "scripts" / "forge_reference.py", "forge_reference_under_test")

    def out_of_memory(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(reference, "sampling_function_inner", out_of_memory)
    preprocessor = reference.PreprocessorReference("reference_only")
    unit = SimpleNamespace(weight=1.0, threshold_a=0.5, guidance_start=0.0, guidance_end=1.0)
    process, added = _fake_process()
    preprocessor.process_before_every_sampling(process, torch.zeros(1, 3, 64, 64), None, unit=unit)

    x = torch.zeros(1, 4, 8, 8)
    with pytest.raises(RuntimeError):
        added["conditioning"](None, x, torch.tensor([5.0]), [], [], 7.0, {}, 1)
    assert preprocessor.is_recording_style is False

    preprocessor.recorded_attn1 = {("input", 1, 0): (torch.zeros(1), torch.zeros(1))}
    preprocessor.recorded_h = {("input", 1): (torch.zeros(1), torch.zeros(1))}
    preprocessor.process_after_every_sampling(process, SimpleNamespace())
    assert preprocessor.recorded_attn1 == {} and preprocessor.recorded_h == {}


def _lllite_libs(monkeypatch):
    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_control_models", list(forge_shared.supported_control_models))
    script = harness.load_script(harness.REPO / "extensions-builtin" / "sd_forge_controlllite" / "scripts" / "forge_controllllite.py", "forge_controllllite_under_test_libs")
    import sys

    return script, sys.modules["lib_controllllite.lib_controllllite_anima"], sys.modules["lib_controllllite.lib_controllllite"]


@pytest.mark.parametrize("order", [(0, 1), (1, 0), (1, 0, 2), (0, 2, 1)])
def test_anima_lllite_units_on_the_same_linear_restore_in_any_order(cn, monkeypatch, order):
    """Two (or three) Anima LLLite units wrap the same DiT Linears; ControlNet restores them in unit
    order. Every order leaves the Linear's own forward, which still runs (upstream reinstalled the
    first unit's wrapper: the next call found its org_forward gone)."""
    import torch

    _, anima, _ = _lllite_libs(monkeypatch)
    linear = torch.nn.Linear(8, 8)
    units = [anima.LLLiteModuleDiT(f"u{i}", linear, cond_emb_dim=4, mlp_dim=4) for i in range(len(order))]
    for unit in units:
        unit.apply_to()
    for index in order:
        units[index].restore()
    assert "forward" not in vars(linear) or vars(linear)["forward"].__self__ is linear
    x = torch.ones(1, 3, 8)
    assert torch.equal(linear(x), torch.nn.functional.linear(x, linear.weight, linear.bias))
    for unit in units:  # the next batch applies them again, without wrapping themselves
        unit.apply_to()
    assert linear(x).shape == x.shape


def test_lllite_runs_in_its_window_of_sampler_steps_whatever_the_forward_calls(cn, monkeypatch):
    """The start / end window is read from the sampler's step, not counted per forward call: the
    tiles of one MultiDiffusion step (or cond and uncond called apart, second-order sub-steps)
    stay in the same step, and a Hires pass with 4 steps uses its own 4."""
    import torch

    _, anima, lib = _lllite_libs(monkeypatch)
    linear = torch.nn.Linear(8, 8)
    unit = anima.LLLiteModuleDiT("u", linear, cond_emb_dim=4, mlp_dim=4)
    torch.nn.init.constant_(unit.up.weight, 0.5)  # a control that changes the output
    unit.cond_emb = torch.ones(1, 3, 4)
    now = {"step": 0, "total": 8}
    unit.progress = lambda: (now["step"], now["total"])
    unit.start_percent, unit.end_percent = 0.0, 0.5
    unit.num_steps = 8  # upstream's counter (process.steps) must not be used
    unit.apply_to()
    x = torch.ones(1, 3, 8)
    plain = torch.nn.functional.linear(x, linear.weight, linear.bias)

    active = []
    for step in range(8):
        now["step"] = step
        active.append([not torch.equal(linear(x), plain) for _tile in range(3)])
    assert active == [[True] * 3] * 4 + [[False] * 3] * 4

    now["total"] = 4  # the hires pass: 4 steps
    active = []
    for step in range(4):
        now["step"] = step
        active.append(not torch.equal(linear(x), plain))
    assert active == [True, True, False, False]
    unit.restore()

    assert lib.step_active(None, 0.0, 0.5) is None  # no sampler: upstream's counting
    process = SimpleNamespace(sampler=SimpleNamespace(model_wrap_cfg=SimpleNamespace(step=3, total_steps=8)))
    assert lib.sampler_progress(process)() == (3, 8)
    assert lib.sampler_progress(SimpleNamespace())() is None


def test_lllite_masks_a_copy_of_the_units_control_image(cn, monkeypatch):
    """The unit's control image is used by every batch and the hires pass: masking it in place
    compounded a soft mask (the second batch got cond * mask * mask)."""
    import torch

    script, _, _ = _lllite_libs(monkeypatch)
    seen = {}
    monkeypatch.setattr(script.LLLiteLoader, "load_lllite", staticmethod(lambda **kwargs: seen.setdefault("cond", kwargs["cond_image"]) is not None and kwargs["model"]))
    patcher = script.ControlLLLitePatcher({"lllite": torch.zeros(1)})
    unet = SimpleNamespace()
    process = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)), steps=8)
    cond = torch.full((1, 3, 4, 4), 0.8)
    patcher.process_before_every_sampling(process, cond, torch.full((1, 1, 4, 4), 0.5))
    assert torch.allclose(cond, torch.full_like(cond, 0.8))
    assert torch.allclose(seen["cond"], torch.full((1, 4, 4, 3), 0.4))


def test_two_inpaint_only_units_blend_each_with_its_own_image_and_mask(cn, monkeypatch):
    """inpaint_only is one preprocessor object for every unit: each unit's blend after sampling
    uses that unit's original and mask (upstream blended every unit with the last unit's)."""
    import torch

    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_preprocessors", dict(forge_shared.supported_preprocessors))
    inpaint = harness.load_script(harness.REPO / "extensions-builtin" / "forge_preprocessor_inpaint" / "scripts" / "preprocessor_inpaint.py", "preprocessor_inpaint_under_test_units")
    preprocessor = inpaint.PreprocessorInpaintOnly()
    left, right = torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 32)
    left[..., :, :16], right[..., :, 16:] = 1, 1
    units = [SimpleNamespace(), SimpleNamespace()]
    for params, mask, value in zip(units, (left, right), (0.25, 0.75), strict=True):
        process, _ = _fake_process()
        preprocessor.process_before_every_sampling(process, torch.full((1, 3, 32, 32), value), mask, params=params)

    results = []
    for params in units:
        batch = SimpleNamespace(images=[torch.ones(3, 32, 32)])
        preprocessor.process_after_every_sampling(SimpleNamespace(), params, batch)
        results.append(batch.images[0])
    # far from its mask's edge, each unit keeps its own original outside its own mask
    assert torch.allclose(results[0][:, 16, 28], torch.full((3,), 0.25)) and torch.allclose(results[0][:, 16, 2], torch.ones(3))
    assert torch.allclose(results[1][:, 16, 2], torch.full((3,), 0.75)) and torch.allclose(results[1][:, 16, 28], torch.ones(3))

    batch = SimpleNamespace(images=[torch.ones(3, 32, 32)])  # a unit that never sampled (interrupted)
    inpaint.PreprocessorInpaintOnly().process_after_every_sampling(SimpleNamespace(), SimpleNamespace(), batch)
    assert torch.equal(batch.images[0], torch.ones(3, 32, 32))


def test_a_unit_whose_model_fails_to_load_leaves_no_infotext(cn, monkeypatch):
    """The units are written to the infotext before they run (so pixel perfect and preview-as-input
    do not change what was asked for); a unit whose control model failed to load does not run, so
    its entry is removed (upstream kept it, so the image claimed a ControlNet it never had)."""
    from lib_controlnet.external_code import ControlNetUnit

    units = [
        ControlNetUnit(enabled=True, module="canny", model="missing_model", pixel_perfect=True),
        ControlNetUnit(enabled=True, module="canny", model="control_v11p_sd15_canny", pixel_perfect=True),
    ]
    script = cn.ControlNetForForgeOfficial()
    monkeypatch.setattr(script, "get_enabled_units", lambda args: units)

    def run(p, unit, params, *args, **kwargs):
        unit.processor_res = 1024  # pixel perfect: not what the infotext records
        params.model = None if unit.model == "missing_model" else object()

    monkeypatch.setattr(script, "process_unit_after_click_generate", run)
    p = SimpleNamespace(extra_generation_params={})
    script.process(p)

    assert set(p.extra_generation_params) == {"ControlNet 1"}
    assert "1024" not in p.extra_generation_params["ControlNet 1"]


def test_a_generation_that_raised_releases_the_reference_recordings(cn, monkeypatch):
    """An error during sampling never reaches the reference preprocessor's after-sampling hook:
    ControlNet's cleanup releases its recorded attention tensors (and its recording flag)."""
    import torch

    from modules_forge import shared as forge_shared

    monkeypatch.setattr(forge_shared, "supported_preprocessors", dict(forge_shared.supported_preprocessors))
    reference = harness.load_script(harness.REPO / "extensions-builtin" / "forge_preprocessor_reference" / "scripts" / "forge_reference.py", "forge_reference_under_test")
    preprocessor = reference.PreprocessorReference("reference_only")
    preprocessor.is_recording_style = True
    preprocessor.recorded_attn1 = {("input", 1, 0): (torch.zeros(1), torch.zeros(1))}
    preprocessor.recorded_h = {("input", 1): (torch.zeros(1), torch.zeros(1))}

    script = cn.ControlNetForForgeOfficial()
    params = cn.ControlNetCachedParameters()
    params.preprocessor, params.model = preprocessor, SimpleNamespace(process_after_every_sampling=lambda p, params: None)
    script.current_params = {0: params}
    script.cleanup(SimpleNamespace())

    assert preprocessor.recorded_attn1 == {} and preprocessor.recorded_h == {} and preprocessor.is_recording_style is False



class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(width, width, bias=False)


def _dit(blocks, width=8):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList(_Block(width) for _ in range(blocks))
    return dit


def _lllite_weights(blocks, width=8):
    return {f"lllite_dit_blocks_{i}_self_attn_q_proj.down.weight": torch.zeros(4, width) for i in range(blocks)}


@pytest.mark.parametrize("trained, model", [(28, 40), (28, 52), (40, 52)])
def test_anima_lllite_goes_on_the_original_blocks_of_a_deeper_model(cn, monkeypatch, trained, model):
    from backend.nn import anima_layout

    _, lllite, _ = _lllite_libs(monkeypatch)
    dit = _dit(model)
    view, weights = lllite.map_blocks(dit, _lllite_weights(trained))
    assert weights.keys() == _lllite_weights(trained).keys()
    originals = anima_layout.original_blocks(trained, model)
    for block in range(trained):
        assert view.blocks[str(block)] is dit.blocks[originals[block]]
    # The LLLite names its modules by these keys, as its file does
    assert {name.split(".")[1] for name, _ in view.named_modules() if name.count(".") == 1} == {str(i) for i in range(trained)}


def test_anima_lllite_from_a_deeper_model_drops_the_inserted_blocks(cn, monkeypatch):
    _, lllite, _ = _lllite_libs(monkeypatch)
    dit = _dit(28)
    view, weights = lllite.map_blocks(dit, _lllite_weights(40))
    assert len(view.blocks) == 28 and len(weights) == 28
    assert "lllite_dit_blocks_2_self_attn_q_proj.down.weight" not in weights  # 2.9B block 2 is an inserted copy
    assert view.blocks["3"] is dit.blocks[2]


def test_anima_lllite_of_another_width_fails_by_name(cn, monkeypatch):
    _, lllite, _ = _lllite_libs(monkeypatch)
    with pytest.raises(NotImplementedError, match="width 16"):
        lllite.map_blocks(_dit(28, width=8), _lllite_weights(28, width=16))
    with pytest.raises(NotImplementedError, match="Cannot map"):
        lllite.map_blocks(_dit(30), _lllite_weights(28))


def test_the_model_list_only_holds_control_models_for_the_loaded_architecture(cn, monkeypatch, tmp_path):
    """The list was filtered by file name only: an SD ControlNet named "depth" was offered for
    Anima, whose DiT ignores it, and Anima's LLLite / control-LoRA for SD. The family is read from
    the tensor names."""
    from lib_controlnet import global_state
    from safetensors.torch import save_file

    from backend.args import dynamic_args

    files = {
        "control_sd15_depth": {"input_blocks.0.0.weight": torch.zeros(1)},
        "anima-lllite-depth-1": {"lllite_dit_blocks_0_self_attn_q_proj.down.weight": torch.zeros(1)},
        "anima_pose_depth": {"diffusion_model.control_embedder.proj.weight": torch.zeros(1)},
    }
    paths = {}
    for name, tensors in files.items():
        save_file(tensors, tmp_path / f"{name}.safetensors")
        paths[name] = str(tmp_path / f"{name}.safetensors")
    monkeypatch.setattr(global_state, "controlnet_names", ["None", *files])
    monkeypatch.setattr(global_state, "controlnet_filename_dict", {"None": None, **paths})

    monkeypatch.setattr(dynamic_args, "anima", True, raising=False)
    assert global_state.get_filtered_controlnet_names("depth") == ["None", "anima-lllite-depth-1", "anima_pose_depth"]
    monkeypatch.setattr(dynamic_args, "anima", False, raising=False)
    assert global_state.get_filtered_controlnet_names("depth") == ["None", "control_sd15_depth"]


def test_control_net_disabled_skips_sampling_and_postprocess_hooks(cn):
    from lib_controlnet.external_code import ControlNetUnit

    script = cn.ControlNetForForgeOfficial()
    unit = ControlNetUnit(enabled=True)
    p = SimpleNamespace(control_net_disabled=True, clear_prompt_cache=lambda: None)

    # Without current_params initialized (e.g. ADetailer inner run with ControlNet = None)
    script.process(p, unit)
    script.process_batch(p, unit)
    script.process_before_every_sampling(p, unit)
    script.postprocess_batch_list(p, SimpleNamespace(images=[]), unit)
    script.postprocess(p, SimpleNamespace(), unit)
    script.cleanup(p, unit)

    # With stale/partial current_params, control_net_disabled=True still skips and preserves outer state
    sentinel = object()
    script.current_params = {0: sentinel}
    script.process_before_every_sampling(p, unit)
    script.postprocess_batch_list(p, SimpleNamespace(images=[]), unit)
    script.postprocess(p, SimpleNamespace(), unit)
    script.cleanup(p, unit)
    assert script.current_params == {0: sentinel}

    # Even with ControlNet enabled, missing index in current_params does not raise KeyError
    enabled_p = SimpleNamespace(control_net_disabled=False)
    script.current_params = {}
    script.process_before_every_sampling(enabled_p, unit)
    script.postprocess_batch_list(enabled_p, SimpleNamespace(images=[]), unit)

