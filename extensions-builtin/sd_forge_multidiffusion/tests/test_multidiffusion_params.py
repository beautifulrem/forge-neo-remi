"""The ported panel: params() declares what the Gradio ui() did; the former
📐 button is the ``detect_size`` action, reading the img2img source image's
size from the upload index."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "forge_multidiffusion.py"
# backend.patcher.controlnet and backend.nn.cnets import each other; the WebUI's start order imports cldm first
PREIMPORT = ("backend.nn.cnets.cldm",)


@pytest.fixture(scope="module")
def multidiffusion():
    import importlib

    from remi.testing import webui_state

    for name in PREIMPORT:
        importlib.import_module(name)
    with webui_state():
        yield harness.load_script(SCRIPT, "multidiffusion_params_under_test")


def test_params_match_the_legacy_ui(multidiffusion):
    script = multidiffusion.MultiDiffusionForForge()
    assert not script.show(False) and script.show(True)
    # Documented change: the keys the run writes are the params' infotext keys, so a paste restores them
    keys = ("enabled", "method", "tile_width", "tile_height", "tile_overlap", "tile_batch_size")
    changes = {index: {"infotext": f"multidiffusion_{name}"} for index, name in enumerate(keys)}
    harness.assert_matches_snapshot(harness.script_layout(script, "img2img"), harness.load_snapshot(SCRIPT, "img2img"), changes)


def test_the_infotext_the_run_writes_pastes_back(multidiffusion):
    from remi.ui import Context

    layout = harness.script_layout(multidiffusion.MultiDiffusionForForge(), "img2img")
    # as parse_generation_parameters reads the keys process_before_every_sampling writes
    written = {"multidiffusion_enabled": "True", "multidiffusion_method": "MultiDiffusion", "multidiffusion_tile_width": "1024", "multidiffusion_tile_height": "512", "multidiffusion_tile_overlap": "32", "multidiffusion_tile_batch_size": "2"}
    ctx = Context()
    pasted, problems = layout.paste(written, ctx)
    assert not problems, [str(problem) for problem in problems]
    assert layout.values_to_args({**layout.defaults(ctx), **pasted}, ctx) == [True, "MultiDiffusion", 1024, 512, 32, 2]


def test_arguments_are_the_processing_signature(multidiffusion):
    layout = harness.script_layout(multidiffusion.MultiDiffusionForForge(), "img2img")
    hook = inspect.signature(multidiffusion.MultiDiffusionForForge.process_before_every_sampling)
    assert layout.names == [name for name in hook.parameters if name not in ("self", "p", "kwargs")]


def _record(width, height):
    return SimpleNamespace(kind="upload", is_image=True, width=width, height=height)


def _context(init_img="upload-1", record=None):
    from remi import ui

    record = record or _record(1216, 832)

    store = SimpleNamespace(get=lambda file_id: record if file_id == "upload-1" else None)
    return ui.ActionContext(tab="img2img", owner="multidiffusion_integrated", core={"init_img": init_img}, files=ui.ActionFiles(store))


def test_detect_size_sets_the_tiles_to_the_source_image(multidiffusion):
    from remi import ui

    script = multidiffusion.MultiDiffusionForForge()
    patch = ui.call_action(script, "detect_size", _context(), {"tile_width": 768, "tile_height": 768})
    assert patch == {"tile_width": {"value": 1216}, "tile_height": {"value": 832}}


def test_detect_size_keeps_the_tiles_inside_the_slider_bounds(multidiffusion):
    from remi import ui

    patch = ui.call_action(multidiffusion.MultiDiffusionForForge(), "detect_size", _context(record=_record(4096, 128)), {})
    assert patch == {"tile_width": {"value": 2048}, "tile_height": {"value": 256}}


@pytest.mark.parametrize("context", [_context(init_img=None), _context(init_img="unknown"), _context(record=_record(None, None))])
def test_detect_size_without_a_source_image_changes_nothing(multidiffusion, context):
    from remi import ui

    patch = ui.call_action(multidiffusion.MultiDiffusionForForge(), "detect_size", context, {})
    assert set(patch) == {"__toast__"} and patch["__toast__"]["level"] == "warning"


def test_detect_size_needs_the_server(multidiffusion):
    from remi import ui

    # Wiring errors surface instead of passing for a missing source image
    with pytest.raises(AttributeError):
        ui.call_action(multidiffusion.MultiDiffusionForForge(), "detect_size", ui.ActionContext(core={"init_img": "upload-1"}), {})


def test_schema_export(multidiffusion):
    schema = harness.script_layout(multidiffusion.MultiDiffusionForForge(), "img2img").schema()
    json.dumps(schema, allow_nan=False)
    sizes, button = schema["children"][1:3]
    assert [child["name"] for child in sizes["children"]] == ["tile_width", "tile_height"]
    assert button == {"type": "button", "label": "Use source image size", "action": "detect_size", "variant": "tool"}


def test_zh_cn_covers_the_panel(multidiffusion):
    schema = harness.script_layout(multidiffusion.MultiDiffusionForForge(), "img2img").schema()
    harness.assert_localized(SCRIPT.parents[1], [schema], extra=["Add a source image to read its size"])


def test_imports_standalone():
    harness.assert_imports_standalone([SCRIPT], preimport=PREIMPORT)


def test_each_tile_gets_the_anima_tensors_of_its_prompts(multidiffusion, monkeypatch):
    """Tiles are batched (x, timesteps and the context repeated per tile): Anima-3.8B's connector
    inputs and NegPiP's mask must be repeated the same way, or the DiT gets the untiled batch."""
    import torch
    from lib_multidiffusion import tiled_diffusion

    monkeypatch.setattr(tiled_diffusion, "opt_f", 8)  # set by the script when it runs
    diffusion = tiled_diffusion.MultiDiffusion()
    diffusion.tile_width = diffusion.tile_height = 16
    diffusion.tile_overlap, diffusion.tile_batch_size = 0, 4
    device = tiled_diffusion._dev  # where the tile weights live
    x = torch.randn(2, 16, 1, 32, 32, device=device)
    c = {
        "c_crossattn": torch.randn(2, 8, 4, device=device),
        "anima_ids": torch.arange(10, device=device).reshape(2, 5),
        "anima_semantic": torch.randn(2, 4, 3, 6, device=device),
        "anima_semantic_rows": [{"id": 0}, None],
        "c_negpip_mask": torch.ones(2, 8, 1, device=device),
    }
    seen = []

    def model(x_tile, t_tile, **c_tile):
        seen.append({key: len(value) if isinstance(value, list) else value.shape[0] for key, value in c_tile.items()} | {"x": x_tile.shape[0]})
        for key in ("anima_ids", "c_crossattn"):
            assert torch.equal(c_tile[key][: len(c[key])], c[key])  # the prompts' order, per tile
        assert c_tile["anima_semantic_rows"][:2] == c["anima_semantic_rows"]
        return x_tile

    diffusion(model, {"input": x, "timestep": torch.ones(2, device=device), "c": c, "cond_or_uncond": [0, 1]})

    assert seen and all(len(set(batch.values())) == 1 for batch in seen), seen


def test_apply_chains_previous_wrapper_and_uses_vae_downscale_ratio(multidiffusion, monkeypatch):
    import torch
    from lib_multidiffusion import tiled_diffusion

    from modules import processing, shared

    class FakeModel:
        def __init__(self, previous=None):
            self.model_options = {"model_function_wrapper": previous} if previous is not None else {}

        def clone(self):
            cloned = FakeModel()
            cloned.model_options = dict(self.model_options)
            return cloned

        def set_model_unet_function_wrapper(self, wrapper):
            self.model_options["model_function_wrapper"] = wrapper

    chained = []

    def previous_wrapper(model_function, args):
        chained.append(tuple(args["input"].shape))
        return model_function(args["input"], args["timestep"], **args["c"])

    monkeypatch.setattr(processing, "opt_f", 8)
    monkeypatch.setattr(shared, "sd_model", SimpleNamespace(forge_objects=SimpleNamespace(vae=SimpleNamespace(downscale_ratio=16))), raising=False)

    patched = tiled_diffusion.TiledDiffusion.apply(FakeModel(previous_wrapper), "MultiDiffusion", 1024, 512, 64, 1)
    wrapper = patched.model_options["model_function_wrapper"]
    assert tiled_diffusion.opt_f == 16
    assert (wrapper.tile_width, wrapper.tile_height, wrapper.tile_overlap) == (64, 32, 4)

    device = tiled_diffusion._dev
    x = torch.randn(1, 16, 1, 64, 64, device=device)
    out = wrapper(lambda xt, tt, **c: xt, {"input": x, "timestep": torch.ones(1, device=device), "c": {"c_crossattn": torch.randn(1, 4, 8, device=device)}, "cond_or_uncond": [0]})
    assert out.shape == x.shape and chained

    # process_controlnet scales latent dims by resolved downscale ratio (16), not hardcoded 8
    hint = torch.ones(1, 3, 64 * 16, 64 * 16, device=device)
    control = SimpleNamespace(cond_hint_original=hint, cond_hint=None, manual_cast_dtype=torch.float32, device=device, previous_controlnet=None)
    wrapper.process_controlnet(x.shape, torch.float32, {"control_model": control}, [0], [tiled_diffusion.BBox(0, 0, 32, 32)], 1, 0)
    assert control.cond_hint.shape == (1, 3, 32 * 16, 32 * 16)

    # Fallback to processing.opt_f when vae lacks downscale_ratio
    monkeypatch.setattr(shared, "sd_model", SimpleNamespace(forge_objects=SimpleNamespace(vae=None)), raising=False)
    monkeypatch.setattr(processing, "opt_f", 12)
    fallback_patched = tiled_diffusion.TiledDiffusion.apply(FakeModel(), "Mixture of Diffusers", 768, 768, 48, 1)
    assert tiled_diffusion.opt_f == 12
    assert fallback_patched.model_options["model_function_wrapper"].tile_width == 64
