"""LoRA Control has no panel: no parameters, no legacy ui(), no Gradio."""

import sys
from pathlib import Path
from types import ModuleType

import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "control.py"
LORA = Path(__file__).resolve().parents[2] / "sd_forge_lora"


def test_the_runner_gives_it_an_empty_ported_layout(monkeypatch):
    from modules import script_callbacks, scripts
    from remi.testing import webui_state

    # control.py reads sd_forge_lora's index (the initialised WebUI): a stand-in
    networks = ModuleType("networks")
    networks.available_network_aliases = {}
    monkeypatch.setitem(sys.modules, "networks", networks)
    monkeypatch.setattr(script_callbacks, "on_cfg_denoiser", lambda callback: None)
    with webui_state():
        control = harness.load_script(SCRIPT, "lora_schedule_params_under_test")
    assert not scripts.has_params(control.LoRAControl) and not scripts.legacy_ui_only(control.LoRAControl())
    with webui_state(script_classes=[control.LoRAControl]) as webui:
        script = next(script for script in webui.txt2img.scripts if isinstance(script, control.LoRAControl))
        assert script.layout is not None and len(script.layout) == 0
        assert script.script_id in {item["id"] for item in webui.txt2img.scripts_schema()}


def test_imports_standalone():
    harness.assert_imports_standalone([SCRIPT], extra_paths=[LORA])


def test_a_schedule_holds_its_first_weight_before_its_first_point(monkeypatch):
    """``<lora:x:[0@0.5,1@1]>`` at t < 0.5: no segment contains t, and the weight was unbound
    (an error every step, or the previous LoRA's weight reused). An unknown alias, or a point that is not weight@time, is logged."""
    from types import SimpleNamespace

    from modules import script_callbacks, shared
    from remi.testing import webui_state

    networks = ModuleType("networks")
    networks.available_network_aliases = {"x": SimpleNamespace(filename="/loras/x.safetensors"), "y": SimpleNamespace(filename="/loras/y.safetensors"), "z": SimpleNamespace(filename="/loras/z.safetensors")}
    monkeypatch.setitem(sys.modules, "networks", networks)
    monkeypatch.setattr(script_callbacks, "on_cfg_denoiser", lambda callback: None)
    with webui_state():
        control = harness.load_script(SCRIPT, "lora_schedule_weights_under_test")
        monkeypatch.setattr(control.dynamic_args, "online_lora", True, raising=False)
        script = control.LoRAControl()
        script.before_process(SimpleNamespace(prompt="<lora:x:[0.2@0.5,1@1]> <lora:y:[0.7@0,0.9@1]> <lora:missing:[0@0,1@1]> <lora:z:[1@0,0.5]>"))
        assert set(control.LoRAControl.mapping) == {"/loras/x.safetensors", "/loras/y.safetensors"}

        x, y = (SimpleNamespace(name=f"/loras/{name}.safetensors", patch=[[None]]) for name in "xy")
        unet = SimpleNamespace(has_online_lora=lambda: True, weight_wrapper_patches={"w": [y, x]})
        monkeypatch.setattr(shared, "sd_model", SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)), raising=False)
        control.LoRAControl.adjust_lora(SimpleNamespace(sampling_step=0, total_sampling_steps=10))
        control.LoRAControl.mapping.clear()

    assert x.patch[0][0] == 0.2 and y.patch[0][0] == 0.7


def test_batched_all_prompts_no_unet_lora_skip_duplicate_timestamps_and_cleanup(monkeypatch):
    from types import SimpleNamespace

    from modules import script_callbacks, shared
    from remi.testing import webui_state

    networks = ModuleType("networks")
    networks.available_network_aliases = {
        "a": SimpleNamespace(filename="/loras/a.safetensors"),
        "b": SimpleNamespace(filename="/loras/b.safetensors"),
    }
    monkeypatch.setitem(sys.modules, "networks", networks)
    monkeypatch.setattr(script_callbacks, "on_cfg_denoiser", lambda callback: None)
    with webui_state():
        control = harness.load_script(SCRIPT, "lora_schedule_batched_under_test")
        monkeypatch.setattr(control.dynamic_args, "online_lora", True, raising=False)
        script = control.LoRAControl()

        # Supports batched p.all_prompts even when p.prompt is not a string
        p = SimpleNamespace(
            prompt=["<lora:a:[0.1@0,0.9@1]>"],
            all_prompts=["<lora:a:[0.1@0,0.9@1]>", "<lora:b:[0.2@0.5,0.8@0.5,1.0@1.0]>"],
        )
        script.before_process(p)
        assert set(control.LoRAControl.mapping) == {"/loras/a.safetensors", "/loras/b.safetensors"}

        # Skips cleanly when UNet has no online LoRA keys (e.g. CLIP-only LoRA)
        no_unet_lora = SimpleNamespace(has_online_lora=lambda: False, weight_wrapper_patches={})
        monkeypatch.setattr(shared, "sd_model", SimpleNamespace(forge_objects=SimpleNamespace(unet=no_unet_lora)), raising=False)
        control.LoRAControl.adjust_lora(SimpleNamespace(sampling_step=5, total_sampling_steps=10))

        # Duplicate timestamps (t1 == t2) do not raise ZeroDivisionError
        b_patch = SimpleNamespace(name="/loras/b.safetensors", patch=[[None]])
        with_lora = SimpleNamespace(has_online_lora=lambda: True, weight_wrapper_patches={"w": [b_patch]})
        monkeypatch.setattr(shared, "sd_model", SimpleNamespace(forge_objects=SimpleNamespace(unet=with_lora)), raising=False)
        control.LoRAControl.adjust_lora(SimpleNamespace(sampling_step=5, total_sampling_steps=10))
        assert b_patch.patch[0][0] == 0.8

        # postprocess and cleanup clear class-level mapping
        script.postprocess(p, SimpleNamespace())
        assert control.LoRAControl.mapping == {}
        script.before_process(p)
        assert control.LoRAControl.mapping
        script.cleanup(p)
        assert control.LoRAControl.mapping == {}
