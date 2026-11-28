"""The ported panel: params() declares what the Gradio ui() did, in the
processing order (which differs from the panel's)."""

import json
from pathlib import Path

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "forge_radial.py"


@pytest.fixture(scope="module")
def radial():
    from remi.testing import webui_state

    with webui_state():
        yield harness.load_script(SCRIPT, "radial_params_under_test")


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(radial, tab):
    layout = harness.script_layout(radial.RadialAttentionForForge(), tab)
    harness.assert_matches_snapshot(layout, harness.load_snapshot(SCRIPT, tab))


def test_arguments_reach_the_patch_in_its_order(radial, monkeypatch):
    seen = {}

    def patch_radial_attn(model, *args):
        seen["args"] = args
        return model

    monkeypatch.setattr(radial.PatchRadialAttn, "patch_radial_attn", staticmethod(patch_radial_attn))
    script = radial.RadialAttentionForForge()
    args = harness.script_layout(script).values_to_args(
        {"enable": True, "dense_block": 3, "decay_factor": 0.5, "dense_timestep": 4, "last_dense_timestep": 5, "block_size": 64, "allow_compile": True}
    )
    unet = object()
    p = type("P", (), {})()
    p.sd_model = type("M", (), {})()
    p.sd_model.forge_objects = type("F", (), {"unet": unet})()
    script.process_before_every_sampling(p, *args)
    # patch_radial_attn(model, dense_block, dense_timestep, last_dense_timestep, block_size, decay_factor, allow_compile)
    assert seen["args"] == (3, 4, 5, 64, 0.5, True)


def test_schema_export(radial):
    schema = harness.script_layout(radial.RadialAttentionForForge()).schema()
    json.dumps(schema, allow_nan=False)
    assert schema["enable"]["name"] == "enable"
    block_size = schema["children"][2]["children"][0]
    assert block_size["style"] == "radio" and [choice["value"] for choice in block_size["choices"]] == [64, 128]


def test_zh_cn_covers_the_panel(radial):
    harness.assert_localized(SCRIPT.parents[1], [harness.script_layout(radial.RadialAttentionForForge(), tab).schema() for tab in harness.TABS])


def test_imports_standalone():
    harness.assert_imports_standalone([SCRIPT])


def test_cleanup_puts_back_wan_attention(radial):
    """The wrapper swaps the module-global wan.attention and restores it only on a dense step: a run
    ending on a radial step, or raising, must not leave it to the next Wan generation."""
    radial.wan.attention = lambda *args, **kwargs: "radial"
    try:
        script = radial.RadialAttentionForForge()
        script.cleanup(object(), False)  # a run without Radial leaves it to whoever set it
        assert radial.wan.attention is not radial.ORIG_ATTENTION
        script.cleanup(object(), True)
        assert radial.wan.attention is radial.ORIG_ATTENTION
    finally:
        radial.wan.attention = radial.ORIG_ATTENTION


def test_patch_chains_previous_wrapper_and_restores_wan_attention(radial, monkeypatch):
    import torch

    class WanModel:
        patch_size = (1, 2, 2)

    class FakeModel:
        def __init__(self, previous):
            self.model_options = {"transformer_options": {}, "model_function_wrapper": previous}
            self.wrapper = None

        def clone(self):
            cloned = FakeModel(self.model_options["model_function_wrapper"])
            cloned.model_options = {
                "transformer_options": dict(self.model_options["transformer_options"]),
                "model_function_wrapper": self.model_options["model_function_wrapper"],
            }
            return cloned

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return WanModel()

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper
            self.model_options["model_function_wrapper"] = wrapper

    monkeypatch.setattr(radial, "get_radial_attn_func", lambda *args: (lambda *a, **kw: None))
    events = []

    def previous_wrapper(model_function, kwargs):
        events.append(("previous", radial.wan.attention is not radial.ORIG_ATTENTION))
        return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

    def model_function(x, t, **c):
        events.append(("model", radial.wan.attention is not radial.ORIG_ATTENTION))
        return x

    patched = radial.PatchRadialAttn.patch_radial_attn(
        FakeModel(previous_wrapper),
        dense_block=0,
        dense_timestep=0,
        last_dense_timestep=0,
        block_size=64,
        decay_factor=0.5,
        allow_compile=True,
    )
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    # Step index 1 is a radial step even with last_dense_timestep=0
    patched.wrapper(
        model_function,
        {
            "input": torch.zeros(1, 4, 1, 4, 4),
            "timestep": torch.tensor([0.5]),
            "c": {"transformer_options": {"sampling_sigmas": sigmas, "radial_attn": patched.model_options["transformer_options"]["radial_attn"]}},
        },
    )
    assert events == [("previous", True), ("model", True)]
    assert radial.wan.attention is radial.ORIG_ATTENTION

    def failing_model(x, t, **c):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        patched.wrapper(
            failing_model,
            {
                "input": torch.zeros(1, 4, 1, 4, 4),
                "timestep": torch.tensor([0.5]),
                "c": {"transformer_options": {"sampling_sigmas": sigmas, "radial_attn": patched.model_options["transformer_options"]["radial_attn"]}},
            },
        )
    assert radial.wan.attention is radial.ORIG_ATTENTION
