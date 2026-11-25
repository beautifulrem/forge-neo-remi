"""Tests for full-stack ModelCapability architecture, structured AdviceItem
patches, PresetArch bridging, script ``requires_capabilities`` filtering,
and Remi tab/conflict/action capability exposure."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from backend.args import dynamic_args
from backend.nn.dit_registry import (
    AdviceItem,
    DiTCapability,
    all_dit_capabilities,
    capability_for_preset,
    collect_structured_advice,
    collect_workflow_advice,
    derive_model_advice,
    get_dit_capability,
    register_dit_capability,
    unregister_dit_capability,
)
from backend.nn.model_capabilities import (
    _reset_ext_loaded_for_tests,
    all_preset_capability_summaries,
    capability_summary_for_preset,
    check_script_applicability,
    detect_model_badges,
    ensure_ext_capabilities_loaded,
    resolve_capabilities_for_preset,
)
from remi import ui
from remi.testing import webui_state

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _preserve_dit_registry():
    """Restore global DiT registry & ext-loader flag after each test so test
    execution order never leaks extended registrations into ``test_dit_registry.py``."""

    before = {cap.name: cap for cap in all_dit_capabilities()}
    yield
    for cap in list(all_dit_capabilities()):
        if cap.name not in before:
            unregister_dit_capability(cap.name)
    for cap in before.values():
        register_dit_capability(cap)
    _reset_ext_loaded_for_tests()


# region 1. DiTCapability declarative fields & backward compatibility


def test_builtin_dit_capabilities_declare_full_stack_metadata():
    anima = get_dit_capability("anima")
    moss = get_dit_capability("moss")
    krea2 = get_dit_capability("krea2")
    flux3 = get_dit_capability("flux3")
    qwen21 = get_dit_capability("qwen21")

    assert anima is not None and "anima_hooks" in anima.capabilities and anima.supports_rgba and anima.supports_control_blocks
    assert anima.preset_arch == "anima" and anima.max_references == 4
    assert "masked_sdpa" in anima.attention_backends and "int8_convrot" in anima.quant_formats

    assert moss is not None and moss.preset_arch == "moss" and "structured_layout" in moss.capabilities and "distilled_cfg1" in moss.capabilities
    assert moss.max_references == 4 and not moss.supports_rgba

    assert krea2 is not None and krea2.preset_arch == "krea" and "multi_reference" in krea2.capabilities
    assert capability_for_preset("krea") is krea2 and capability_for_preset("krea2") is krea2

    assert flux3 is not None and flux3.preset_arch == "flux3" and flux3.max_references == 10
    assert flux3.reference_slot_format == "ref_image_{i}" and "structured_layout" in flux3.capabilities

    assert qwen21 is not None and qwen21.preset_arch == "qwen21" and qwen21.max_references == 10
    assert qwen21.reference_slot_format == "<image_{i}>" and qwen21.supports_rgba and qwen21.supports_control_blocks
    assert "prefix_kv_cache" in qwen21.capabilities

    assert capability_for_preset("unknown_arch") is None


def test_dit_capability_minimal_constructor_stays_backward_compatible():
    minimal = DiTCapability(
        name="minimal_test_dit",
        dit_cls_name="MinimalDiT",
        model_type_name="MinimalModel",
        engine_label="Minimal",
    )
    assert minimal.capabilities == frozenset()
    assert minimal.preset_arch is None
    assert minimal.max_references == 0
    assert minimal.reference_slot_format is None
    assert minimal.supports_rgba is False
    assert minimal.supports_control_blocks is False
    assert minimal.attention_backends == ()
    assert minimal.quant_formats == ()
    assert minimal.structured_advice is None
    assert minimal.form_advice is None


# endregion


# region 2. Structured AdviceItem (runtime + pre-submit)


def test_advice_item_serialization():
    item = AdviceItem(
        code="qwen21.turbo_mismatch",
        severity="warning",
        message="Use 8 steps with CFG 1.0.",
        patch={"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}},
        action_label="Set Turbo",
    )
    data = item.to_dict()
    assert json.dumps(data, allow_nan=False)
    assert data == {
        "code": "qwen21.turbo_mismatch",
        "severity": "warning",
        "message": "Use 8 steps with CFG 1.0.",
        "patch": {"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}},
        "action_label": "Set Turbo",
    }
    plain = AdviceItem(code="info.only", severity="info", message="FYI")
    assert plain.to_dict() == {"code": "info.only", "severity": "info", "message": "FYI", "patch": None}


def test_structured_advice_anima_runtime_and_form(monkeypatch):
    for flag in ("anima", "moss", "krea2", "flux3", "qwen21"):
        monkeypatch.setattr(dynamic_args, flag, flag == "anima")

    rgba_img = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    p = SimpleNamespace(
        sd_model=SimpleNamespace(
            is_anima_engine=True,
            sd_checkpoint_info=SimpleNamespace(name="anima-turbo-edit-v1.safetensors"),
            forge_objects=SimpleNamespace(vae=SimpleNamespace(output_channels=4)),
            ref_latents=[],
        ),
        prompt="1girl, transparent background",
        steps=32,
        cfg_scale=4.5,
        init_images=[rgba_img],
    )
    opts = SimpleNamespace(anima_do_reference=False, img2img_background_color="#808080", samples_format="jpg")
    items = collect_structured_advice(p, opts=opts)
    by_code = {item.code: item for item in items}
    assert {"anima.turbo_mismatch", "anima.edit_ref_disabled", "anima.rgba_non_white_bg", "anima.rgba_jpeg_format"} <= set(by_code)
    assert by_code["anima.turbo_mismatch"].patch == {
        "sampler.steps": {"value": 10},
        "cfg_scale": {"value": 1.0},
        "sampler.sampler_name": {"value": "Euler"},
    }
    assert by_code["anima.edit_ref_disabled"].patch == {"__options__": {"anima_do_reference": True}}
    assert by_code["anima.rgba_non_white_bg"].patch == {"__options__": {"img2img_background_color": "#ffffff"}}
    assert by_code["anima.rgba_jpeg_format"].patch == {"__options__": {"samples_format": "png"}}
    # Runtime collect_workflow_advice stays backward compatible (no duplicate Anima comments alongside anima_remi_presets)
    assert collect_workflow_advice(p, opts=opts) == []

    # Pre-submit form-level advice for Anima Base low steps + RGBA/JPEG
    form_opts = SimpleNamespace(forge_checkpoint_anima="anima-base-v1.0.safetensors", anima_do_reference=False, samples_format="jpeg")
    form_items = derive_model_advice(
        "anima",
        "txt2img",
        {"core": {"cfg_scale": 1.0, "prompt": "transparent sticker"}, "sampler": {"steps": 8}},
        opts=form_opts,
    )
    form_codes = {item.code: item for item in form_items}
    assert "anima.base_low_steps" in form_codes and form_codes["anima.base_low_steps"].patch == {"sampler.steps": {"value": 36}, "cfg_scale": {"value": 4.5}}
    assert "anima.rgba_jpeg_format" in form_codes and form_codes["anima.rgba_jpeg_format"].patch == {"__options__": {"samples_format": "png"}}


def test_structured_advice_moss_kromoss_dmd2(monkeypatch):
    for flag in ("anima", "moss", "krea2", "flux3", "qwen21"):
        monkeypatch.setattr(dynamic_args, flag, flag == "moss")

    p = SimpleNamespace(
        sd_model=SimpleNamespace(
            sd_checkpoint_info=SimpleNamespace(name="lodestones_Kromoss_bf16.safetensors"),
            ref_latents=[object()],
            ini_latent=None,
        ),
        prompt="editorial poster <lora:Kromoss-DMD2-8step:1.0>",
        all_prompts=["editorial poster <lora:Kromoss-DMD2-8step:1.0>"],
        steps=40,
        cfg_scale=3.0,
    )
    opts = SimpleNamespace(moss_do_reference="Disable")
    items = collect_structured_advice(p, opts=opts)
    by_code = {item.code: item for item in items}
    assert by_code["moss.ref_disabled"].patch == {"__options__": {"moss_do_reference": "Moss-Image-Edit"}}
    assert by_code["moss.dmd_turbo_mismatch"].patch == {"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}}
    assert "Turbo/DMD" in by_code["moss.dmd_turbo_mismatch"].message
    # collect_workflow_advice mirrors messages verbatim
    assert collect_workflow_advice(p, opts=opts) == [item.message for item in items]

    # Pre-submit form derivation with ImageStitch reference attached
    form_items = derive_model_advice(
        "moss",
        "txt2img",
        {
            "core": {"prompt": "poster <lora:kromoss_dmd2:1>", "cfg_scale": 3.5},
            "sampler": {"steps": 30},
            "imagestitch_integrated": {"enable": True, "references": ["upload-1"]},
        },
        opts=SimpleNamespace(forge_checkpoint_moss="kromoss.safetensors", moss_do_reference="Disable"),
    )
    form_codes = {item.code: item for item in form_items}
    assert "moss.ref_disabled" in form_codes and "moss.dmd_turbo_mismatch" in form_codes
    assert form_codes["moss.dmd_turbo_mismatch"].patch == {"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}}


def test_structured_advice_krea2_flux3_and_qwen21_pre_submit():
    # Krea 2 Turbo mismatch + ref disabled (both canonical "krea" and alias "krea2" resolve forge_checkpoint_krea!)
    krea_opts = SimpleNamespace(forge_checkpoint_krea="krea2_turbo_fp8.safetensors", krea2_do_reference=False)
    krea_values = {
        "core": {"prompt": "portrait", "cfg_scale": 4.0},
        "sampler": {"steps": 28},
        "imagestitch_integrated": {"enable": True, "references": ["u1"]},
    }
    for alias in ("krea", "krea2"):
        krea_items = derive_model_advice(alias, "txt2img", krea_values, opts=krea_opts)
        krea_codes = {i.code: i for i in krea_items}
        assert krea_codes["krea2.turbo_mismatch"].patch == {"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}}
        assert krea_codes["krea2.ref_disabled"].patch == {"__options__": {"krea2_do_reference": True}}

    # FLUX.3 distilled CFG fix + ref disabled + empty prompt does not nag short_prompt
    flux_items = derive_model_advice(
        "flux3",
        "txt2img",
        {
            "core": {"prompt": "", "cfg_scale": 3.5},
            "sampler": {"steps": 16},
            "imagestitch_integrated": {"enable": True, "references": ["u1"]},
        },
        opts=SimpleNamespace(forge_checkpoint_flux3="flux3-klein-distilled.safetensors", flux3_do_reference=False),
    )
    flux_codes = {i.code: i for i in flux_items}
    assert flux_codes["flux3.distilled_cfg"].patch == {"cfg_scale": {"value": 1.0}, "sampler.steps": {"value": 16}}
    assert flux_codes["flux3.ref_disabled"].patch == {"__options__": {"flux3_do_reference": True}}
    assert "flux3.short_prompt" not in flux_codes

    # Qwen-Image-2.1 Turbo + RGBA JPEG + too many refs + ref disabled
    qwen_items = derive_model_advice(
        "qwen21",
        "txt2img",
        {
            "core": {"prompt": "透明背景 sticker", "cfg_scale": 3.0},
            "sampler": {"steps": 32},
            "imagestitch_integrated": {"enable": True, "references": [f"u{i}" for i in range(11)]},
        },
        opts=SimpleNamespace(forge_checkpoint_qwen21="Qwen-Image-2.1-Turbo-FP8.safetensors", qwen21_do_reference=False, samples_format="jpg"),
    )
    qwen_codes = {i.code: i for i in qwen_items}
    assert qwen_codes["qwen21.turbo_mismatch"].patch == {"sampler.steps": {"value": 8}, "cfg_scale": {"value": 1.0}}
    assert qwen_codes["qwen21.rgba_jpeg_format"].patch == {"__options__": {"samples_format": "png"}}
    assert qwen_codes["qwen21.ref_disabled"].patch == {"__options__": {"qwen21_do_reference": True}}
    assert qwen_codes["qwen21.too_many_refs"].patch is None


def test_legacy_workflow_advice_only_dit_wraps_into_structured_advice():
    cap = DiTCapability(
        name="legacy_advice_dit",
        dit_cls_name="LegacyAdviceDiT",
        model_type_name="LegacyAdviceModel",
        engine_label="LegacyAdvice",
        workflow_advice=lambda p: ["Legacy string hint"] if getattr(p, "flag", False) else [],
        form_advice=lambda preset, tab, values, opts=None: [
            AdviceItem(code="legacy.form", severity="info", message="Form hint", patch={"cfg_scale": {"value": 2.0}})
        ],
    )
    register_dit_capability(cap)
    try:
        items = collect_structured_advice(SimpleNamespace(flag=True))
        legacy = [i for i in items if i.code == "legacy_advice_dit.advice"]
        assert len(legacy) == 1 and legacy[0].message == "Legacy string hint" and legacy[0].severity == "warning"

        form_items = derive_model_advice("legacy_advice_dit", "txt2img", {"core": {}})
        assert len(form_items) == 1 and form_items[0].code == "legacy.form" and form_items[0].patch == {"cfg_scale": {"value": 2.0}}
    finally:
        unregister_dit_capability("legacy_advice_dit")


# endregion


# region 3. PresetArch bridging & external capability loading


def test_capability_summary_covers_all_15_presets_as_strict_json():
    from modules_forge.presets import PresetArch

    all_summaries = all_preset_capability_summaries()
    assert set(PresetArch.choices()) <= set(all_summaries)
    json.dumps(all_summaries, allow_nan=False)

    assert "anima_hooks" in all_summaries["anima"]["capabilities"]
    assert "anima_hooks" not in all_summaries["sd"]["capabilities"]
    assert "anima_hooks" not in all_summaries["qwen21"]["capabilities"]

    qwen21 = all_summaries["qwen21"]
    assert qwen21["reference_slot_format"] == "<image_{i}>" and qwen21["reference_slot_start"] == 1
    assert qwen21["max_references"] == 10 and qwen21["supports_rgba"] is True and qwen21["supports_control_blocks"] is True
    assert "prefix_kv_cache" in qwen21["capabilities"]

    flux3 = all_summaries["flux3"]
    assert flux3["reference_slot_format"] == "ref_image_{i}" and flux3["reference_slot_start"] == 0
    assert flux3["max_references"] == 10 and "structured_layout" in flux3["capabilities"]

    wan = all_summaries["wan"]
    assert wan["is_video"] is True and wan["frames"] == 16 and "video" in wan["capabilities"]

    # Aliases normalize cleanly
    assert capability_summary_for_preset("krea2")["preset"] == "krea"
    assert capability_summary_for_preset("krea2")["dit_name"] == "krea2"
    assert "anima_hooks" in resolve_capabilities_for_preset("anima")


def test_detect_model_badges_and_live_metadata():
    assert detect_model_badges("qwen21", checkpoint="Qwen-Image-2.1-Turbo-FP8.safetensors", supports_rgba=True, supports_control_blocks=True) == (
        "FP8",
        "Turbo",
        "RGBA",
        "Control",
    )
    assert detect_model_badges("moss", checkpoint="moss-diffusers/model_index.json", unet_dtype="int8-convrot") == ("Diffusers", "INT8-ConvRot")
    assert detect_model_badges("flux", checkpoint="flux1-dev-Q4_K_M.gguf") == ("GGUF",)
    assert detect_model_badges("flux", checkpoint="flux1-nvfp4.safetensors") == ("NVFP4",)
    assert detect_model_badges("flux", checkpoint="svdq-int4-flux.safetensors") == ("SVDQuant",)


def test_ensure_ext_capabilities_loaded_bridges_external_dit_module(monkeypatch):
    fake_mod = ModuleType("backend.nn.dit_capabilities_ext")

    def _register():
        register_dit_capability(
            DiTCapability(
                name="fibo_test_ext",
                dit_cls_name="FiboTestTransformer",
                model_type_name="FiboTest",
                engine_label="Bria Fibo 2.0 Test",
                capabilities=frozenset({"structured_layout", "distilled_cfg1"}),
                preset_arch="fibo_test_ext",
                max_references=4,
                reference_slot_format="ref_image_{i}",
                supports_rgba=False,
            )
        )

    fake_mod.register_extended_dit_capabilities = _register
    monkeypatch.setitem(sys.modules, "backend.nn.dit_capabilities_ext", fake_mod)
    _reset_ext_loaded_for_tests()
    try:
        assert ensure_ext_capabilities_loaded(force=True) is True
        summary = capability_summary_for_preset("fibo_test_ext")
        assert summary["dit_name"] == "fibo_test_ext"
        assert summary["engine_label"] == "Bria Fibo 2.0 Test"
        assert "structured_layout" in summary["capabilities"]
        assert summary["reference_slot_format"] == "ref_image_{i}"
        assert summary["reference_slot_start"] == 0
    finally:
        unregister_dit_capability("fibo_test_ext")
        _reset_ext_loaded_for_tests()


# endregion


# region 4. Script requires_capabilities & Remi core integration


ANIMA_SCRIPT_FILES = (
    "extensions-builtin/anima-artist-scheduled-mixer/scripts/anima_artist_scheduled_mixer.py",
    "extensions-builtin/anima-lora-layer-weight/scripts/anima_lora_layer_weight.py",
    "extensions-builtin/anima-lora-stage-scheduler/scripts/anima_lora_stage_scheduler.py",
    "extensions-builtin/sd_forge_anima_dave/scripts/anima_dave.py",
    "extensions-builtin/sd_forge_anima_freefuse/scripts/anima_freefuse.py",
    "extensions-builtin/sd_forge_anima_guidance/scripts/anima_guidance.py",
    "extensions-builtin/sd_forge_anima_hires_guard/scripts/anima_hires_guard.py",
    "extensions-builtin/sd_forge_anima_presets/scripts/anima_remi_presets.py",
    "extensions-builtin/sd_forge_anima_regional/scripts/anima_regional.py",
    "extensions-builtin/sd_forge_anima_xattn_boost/scripts/anima_xattn_boost.py",
)


def test_all_anima_builtin_scripts_declare_anima_hooks_capability():
    spec = importlib.util.spec_from_file_location("remi_panel_harness_cap_test", REPO / "tests" / "remi_panel_harness.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    from modules import scripts

    for rel_path in ANIMA_SCRIPT_FILES:
        mod = harness.load_script(REPO / rel_path, f"cap_check_{Path(rel_path).stem}")
        found_classes = [
            obj
            for obj in vars(mod).values()
            if isinstance(obj, type) and issubclass(obj, scripts.Script) and obj is not scripts.Script
        ]
        assert found_classes, f"no Script subclass in {rel_path}"
        for cls in found_classes:
            assert getattr(cls, "requires_capabilities", ()) == ("anima_hooks",), f"{rel_path}:{cls.__name__}"


def test_scripts_schema_and_tab_schema_filter_by_capabilities_without_altering_slots():
    from modules import scripts
    from remi.core.tabs import tab_schema

    class AnimaOnlyDemo(scripts.Script):
        requires_capabilities = ("anima_hooks",)

        def title(self):
            return "Anima Only Demo"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def params(self, is_img2img):
            return ui.Section("Anima Only Demo", ui.Slider("gain", 0.0, 2.0, 1.0, 0.1), enable=ui.Toggle("enabled", False))

    class UniversalDemo(scripts.Script):
        def title(self):
            return "Universal Demo"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def params(self, is_img2img):
            return ui.Toggle("flag", False)

    with webui_state(script_classes=[AnimaOnlyDemo, UniversalDemo]) as state:
        # Check slot order is completely independent of active preset
        state.opts.data["forge_preset"] = "qwen21"
        state.opts.data["forge_checkpoint_qwen21"] = "Qwen-Image-2.1-Turbo-FP8.safetensors"
        qwen_schema = tab_schema("txt2img")
        qwen_slots = {s.script_id: (s.args_from, s.args_to) for s in state.txt2img.scripts}

        assert qwen_schema["capabilities"]["preset"] == "qwen21"
        assert "FP8" in qwen_schema["capabilities"]["badges"] and "Turbo" in qwen_schema["capabilities"]["badges"]
        # Initial schema advice catches Qwen 2.1 Turbo default 32 steps / CFG 2.5 immediately!
        assert any(item["code"] == "qwen21.turbo_mismatch" for item in qwen_schema["advice"])

        qwen_entries = {e["id"]: e for e in qwen_schema["scripts"]}
        assert qwen_entries["anima_only_demo"]["requires_capabilities"] == ["anima_hooks"]
        assert qwen_entries["anima_only_demo"]["applicable"] is False
        assert "Anima engine hooks" in qwen_entries["anima_only_demo"]["inapplicable_reason"]
        # the reason names the active model: the UI translates its template, each capability by its name
        assert qwen_entries["anima_only_demo"]["inapplicable_template"] == "Requires {{capabilities}} (active model: {{model}})"
        assert qwen_entries["anima_only_demo"]["inapplicable_params"]["capabilities"] == ["Anima engine hooks"]
        turbo = next(item for item in qwen_schema["advice"] if item["code"] == "qwen21.turbo_mismatch")
        assert "{{steps}}" in turbo["template"] and turbo["params"]["steps"] in turbo["message"]
        assert qwen_entries["universal_demo"]["requires_capabilities"] == []
        assert qwen_entries["universal_demo"]["applicable"] is True
        assert "inapplicable_reason" not in qwen_entries["universal_demo"]
        assert qwen_entries["seed"]["applicable"] is True

        # Switch preset to Anima -> Anima script becomes applicable, slots unchanged
        state.opts.data["forge_preset"] = "anima"
        anima_schema = tab_schema("txt2img")
        anima_slots = {s.script_id: (s.args_from, s.args_to) for s in state.txt2img.scripts}
        assert anima_slots == qwen_slots

        anima_entries = {e["id"]: e for e in anima_schema["scripts"]}
        assert anima_entries["anima_only_demo"]["applicable"] is True
        assert "inapplicable_reason" not in anima_entries["anima_only_demo"]
        assert json.dumps(anima_schema, allow_nan=False)


def test_conflict_patch_and_core_actions_expose_structured_advice():
    from remi.core.actions import core_actions
    from remi.core.conflicts import conflict_patch

    with webui_state() as state:
        state.opts.data["forge_preset"] = "qwen21"
        state.opts.data["forge_checkpoint_qwen21"] = "Qwen-Image-2.1-Turbo.safetensors"
        state.opts.data["samples_format"] = "jpg"

        values = {
            "core": {"prompt": "transparent icon", "cfg_scale": 3.5},
            "sampler": {"steps": 32},
        }
        patch = conflict_patch("txt2img", values)
        assert "__advice__" in patch
        codes = {item["code"]: item for item in patch["__advice__"]}
        assert "qwen21.turbo_mismatch" in codes
        assert "qwen21.rgba_jpeg_format" in codes

        # Core __advice__ action returns both __advice__ and __capabilities__
        ctx = ui.ActionContext(tab="txt2img", owner="core", scripts={"sampler": {"steps": 32}})
        advice_patch = ui.call_action(core_actions, "__advice__", ctx, {"prompt": "transparent icon", "cfg_scale": 3.5})
        assert advice_patch["__capabilities__"]["preset"] == "qwen21"
        assert {i["code"] for i in advice_patch["__advice__"]} >= {"qwen21.turbo_mismatch", "qwen21.rgba_jpeg_format"}

        # Fixing parameters clears the advice list to []
        state.opts.data["samples_format"] = "png"
        clean_ctx = ui.ActionContext(tab="txt2img", owner="core", scripts={"sampler": {"steps": 8}})
        clean_patch = ui.call_action(core_actions, "__advice__", clean_ctx, {"prompt": "transparent icon", "cfg_scale": 1.0})
        assert clean_patch["__advice__"] == []


def test_check_script_applicability_edge_cases():
    assert check_script_applicability((), set()) == (True, None)
    assert check_script_applicability(["anima_hooks"], ["anima_hooks", "spectrum"]) == (True, None)
    ok, reason = check_script_applicability(["anima_hooks", "rgba_output"], ["rgba_output"], engine_label="Qwen-Image-2.1")
    assert ok is False and reason == "Requires Anima engine hooks (active model: Qwen-Image-2.1)"


def test_remi_core_capabilities_full_branch_coverage(monkeypatch):
    from remi.core import capabilities as cap_mod

    # Explicit opts and sd_model paths + active_capabilities()
    custom_opts = SimpleNamespace(forge_preset="anima", forge_checkpoint_anima="anima-turbo.safetensors")
    fake_model = SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name="anima-turbo.safetensors"))
    assert cap_mod.active_preset(custom_opts) == "anima"
    caps = cap_mod.active_capabilities(opts=custom_opts, sd_model=fake_model)
    assert "anima_hooks" in caps

    summary = cap_mod.active_capability_summary(opts=custom_opts, preset="flux3", sd_model=SimpleNamespace(is_flux3_distilled=True))
    assert summary["preset"] == "flux3" and "Turbo" in summary["badges"]

    # script_applicability() when active_caps/summary are omitted vs provided
    anima_script = SimpleNamespace(requires_capabilities=("anima_hooks",))
    meta_on = cap_mod.script_applicability(anima_script, opts=custom_opts)
    assert meta_on == {"requires_capabilities": ["anima_hooks"], "applicable": True}

    meta_off = cap_mod.script_applicability(anima_script, preset="qwen21", opts=SimpleNamespace(forge_preset="qwen21"))
    assert meta_off["applicable"] is False and "Anima engine hooks" in meta_off["inapplicable_reason"]

    # derive_advice() with explicit opts and preset
    advice = cap_mod.derive_advice(
        "txt2img",
        {"core": {"prompt": "transparent icon", "cfg_scale": 3.5}, "sampler": {"steps": 32}},
        preset="qwen21",
        opts=SimpleNamespace(forge_checkpoint_qwen21="Qwen-Image-2.1-Turbo.safetensors", samples_format="jpg"),
    )
    assert {item["code"] for item in advice} >= {"qwen21.turbo_mismatch", "qwen21.rgba_jpeg_format"}

    # Fallback branches when modules.shared / modules.sd_models fail to import
    import modules as modules_pkg

    monkeypatch.delattr(modules_pkg, "shared", raising=False)
    monkeypatch.delattr(modules_pkg, "sd_models", raising=False)
    monkeypatch.setitem(sys.modules, "modules.shared", None)
    monkeypatch.setitem(sys.modules, "modules.sd_models", None)
    assert cap_mod._resolve_opts() is None
    assert cap_mod._resolve_sd_model() is None
    assert cap_mod.active_preset() == "sd"


def test_fibo_reference_slot_start_is_zero_indexed():
    from backend.nn.model_capabilities import _slot_start_for_format

    assert _slot_start_for_format("ref_{i}") == 0
    assert _slot_start_for_format("ref_image_{i}") == 0
    assert _slot_start_for_format("Picture {i}") == 0
    assert _slot_start_for_format("<image_{i}>") == 1

    # Verify live Bria Fibo 2.0 capability summary from dit_capabilities_ext aligns ref_{i} -> start=0 (ref_0)
    summary = capability_summary_for_preset("fibo")
    assert summary["dit_name"] == "fibo"
    assert summary["reference_slot_format"] == "ref_{i}"
    assert summary["reference_slot_start"] == 0


def test_ui_types_include_capability_and_advice_fields():
    from typing import get_type_hints

    from remi.ui.types import AdviceItemSchema, ModelCapabilitySchema, ScriptEntry, TabSchema

    tab_hints = get_type_hints(TabSchema, include_extras=True)
    assert {"capabilities", "advice"} <= set(tab_hints)
    script_hints = get_type_hints(ScriptEntry, include_extras=True)
    assert {"requires_capabilities", "applicable", "inapplicable_reason"} <= set(script_hints)
    advice_hints = get_type_hints(AdviceItemSchema, include_extras=True)
    assert {"code", "severity", "message", "patch", "action_label"} <= set(advice_hints)
    cap_hints = get_type_hints(ModelCapabilitySchema, include_extras=True)
    assert {"preset", "capabilities", "reference_slot_format", "reference_slot_start", "badges"} <= set(cap_hints)


def test_loader_split_state_dict_lazily_loads_ext_capabilities(monkeypatch):
    from backend import loader
    from backend.nn import model_capabilities

    calls: list[bool] = []
    monkeypatch.setattr(model_capabilities, "ensure_ext_capabilities_loaded", lambda: calls.append(True) or True)

    def _stop(_path):
        raise RuntimeError("stop-after-hook")

    monkeypatch.setattr("backend.nn.dit_registry.is_diffusers_checkpoint_dir", _stop)
    try:
        loader.split_state_dict("nonexistent_checkpoint.safetensors")
    except RuntimeError as exc:
        assert str(exc) == "stop-after-hook"
    assert calls == [True]


def test_spectrum_script_exposes_taylorseer_and_magcache_without_breaking_slots(tmp_path):
    import torch

    spectrum_root = REPO / "extensions-builtin" / "sd_forge_spectrum"
    if str(spectrum_root) not in sys.path:
        sys.path.insert(0, str(spectrum_root))

    spec = importlib.util.spec_from_file_location("spectrum_wave2_test", spectrum_root / "scripts" / "spectrum.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    script = mod.SpectrumForForge()
    layout = ui.ScriptLayout(script.params(False), owner="spectrum_integrated", order=script.param_order)
    # Positional argument slots stay untouched (16 args); forecast_mode is a tool control (arg=False)
    assert layout.names == ["enable", *mod.PARAM_NAMES, "preset_name"]
    assert "forecast_mode" in layout.by_name and not layout.by_name["forecast_mode"].arg
    assert layout.by_name["forecast_mode"].values(ui.Context()) == ["Chebyshev", "TaylorSeer", "MagCache"]

    try:
        patch = ui.call_action(script, "set_forecast_mode", ui.ActionContext(tab="txt2img", owner="spectrum_integrated"), {"forecast_mode": "TaylorSeer"})
        assert patch == {"forecast_mode": {"value": "TaylorSeer"}}
        assert mod.active_forecast_mode() == "TaylorSeer"

        class _FakePatcher:
            def __init__(self):
                self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
                self.model_options = {}

            def clone(self):
                return self

            def set_model_unet_function_wrapper(self, wrapper):
                self.model_options["model_function_wrapper"] = wrapper

        def _make_p():
            return SimpleNamespace(
                sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=_FakePatcher())),
                extra_generation_params={},
                is_hr_pass=False,
                steps=16,
                sampler_name="Euler",
                scheduler="Simple",
                cfg_scale=4.0,
                distilled_cfg_scale=3.0,
            )

        defaults = [mod.DEFAULT[k] for k in mod.INFOTEXT_KEYS]
        p_ts = _make_p()
        script.process_before_every_sampling(p_ts, True, *defaults, x=torch.zeros(2, 16, 8, 8))
        assert p_ts._spectrum_states[0].forecast_mode == "TaylorSeer"
        assert p_ts.extra_generation_params["spec_forecast_mode"] == "TaylorSeer"

        # MagCache override via p.spectrum_forecast_mode or action
        ui.call_action(script, "set_forecast_mode", ui.ActionContext(tab="txt2img", owner="spectrum_integrated"), {"forecast_mode": "MagCache"})
        p_mc = _make_p()
        script.process_before_every_sampling(p_mc, True, *defaults, x=torch.zeros(2, 16, 8, 8))
        assert p_mc._spectrum_states[0].forecast_mode == "MagCache"
        assert p_mc.extra_generation_params["spec_forecast_mode"] == "MagCache"

        # Default Chebyshev leaves spec_forecast_mode absent for exact infotext compatibility
        ui.call_action(script, "set_forecast_mode", ui.ActionContext(tab="txt2img", owner="spectrum_integrated"), {"forecast_mode": "Chebyshev"})
        p_ch = _make_p()
        script.process_before_every_sampling(p_ch, True, *defaults, x=torch.zeros(2, 16, 8, 8))
        assert p_ch._spectrum_states[0].forecast_mode == "Chebyshev"
        assert "spec_forecast_mode" not in p_ch.extra_generation_params
    finally:
        mod.set_active_forecast_mode("Chebyshev")
