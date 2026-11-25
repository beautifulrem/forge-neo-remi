"""The core Simplified Chinese catalog (``localizations/zh_CN.json``) covers
every core UI string: the generation forms and built-in processing scripts,
every setting (categories, sections, labels, infos, notes, links, choices) and
the server's translatable messages (tests/remi_core_strings.py)."""

import ast
import json
import re
from pathlib import Path

import pytest

from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "localizations" / "zh_CN.json"


def _load(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strings_module = _load("remi_core_strings")


@pytest.fixture(scope="module")
def strings():
    with webui_state() as state:
        yield strings_module.core_strings(state.opts)


@pytest.fixture(scope="module")
def catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _placeholders(text: str) -> list[str]:
    return sorted(re.findall(r"\{\{(\w+)\}\}", text))


def test_the_collector_sees_the_whole_ui(strings):
    # forms, settings and messages all contribute
    assert {"Hires. fix", "Denoising strength", "Stable Diffusion", "Saving Images/Grids", "This task is already running"} <= strings
    assert len(strings) > 500
    # data is left out: sampler and setting names, the choices of DATA_CHOICES
    assert "Euler a" not in strings and "samples_format" not in strings and not {"png", "veryfast"} & strings
    # lowercase choice labels elsewhere are UI text ("Masked content")
    assert {"fill", "original"} <= strings
    # settings help markup (keyed by its exact text, as the SPA looks it up) and the Callbacks section
    assert any(text.startswith("These settings allow you to enable PyTorch profiler") for text in strings)
    assert {"Callbacks", "For categories below, callbacks added to dropdowns happen before others, in order listed."} <= strings
    assert not any(text.startswith(("prioritized_callbacks_", "base/")) for text in strings)


def test_the_sentences_a_job_ends_with_are_collected(strings, catalog):
    """The history and the failure toasts show them translated (the job's
    own exception text stays as it is)."""

    from remi.jobs.texts import JOB_TEXTS

    assert set(JOB_TEXTS) <= strings
    assert {"the server stopped while the job ran", "Out of memory: the model was unloaded", "the model was not loaded"} <= set(JOB_TEXTS)
    assert all((catalog.get(text) or "").strip() for text in JOB_TEXTS)


def test_the_model_advice_is_collected(strings, catalog):
    """The advice bar above Generate and the reason a script does not apply:
    the server formatted their sentences with the values in them, which the
    UI could never translate. A ``Sentence`` keeps its template for it."""

    from remi_core_strings import advice_strings, unlisted_action_labels

    advice = advice_strings()
    assert advice <= strings
    # the SPA translates a fix button's label as a catalog key: a label built from values has none
    assert unlisted_action_labels() == []
    assert {
        "Anima-Turbo checkpoint with CFG {{cfg}} / {{steps}} steps: it is distilled for CFG 1 and 8-12 steps (see the Official Turbo preset); higher values over-cook it.",
        "Requires {{capabilities}} (active model: {{model}})",
        "Set Turbo (10 steps · CFG 1.0)",
        "Set 4 steps · CFG 1.0",
        "Set 16 steps · CFG 1.0",
        "Z-Image: reference images are attached while Z-Image Omni/Edit reference mode is disabled.",
    } <= advice
    missing = sorted(text for text in advice if not (catalog.get(text) or "").strip())
    assert missing == []
    assert all(_placeholders(catalog[text]) == _placeholders(text) for text in advice)


def test_the_callbacks_section_keeps_the_vae_explanation():
    """The Callbacks section's help once reused the key ``sd_vae_explanation``
    and replaced the VAE section's explanation."""

    with webui_state():  # imported first, shared_items would import itself half-way through
        from modules import shared_items, shared_options

        callbacks = set(shared_items.callbacks_order_settings())
    # The server adds this section to the same templates once it starts (another test may
    # have), so compare with the other sections' keys and with the keys shared_options.py defines.
    source = ast.parse(Path(shared_options.__file__).read_text(encoding="utf-8"))
    defined = {key.value for node in ast.walk(source) if isinstance(node, ast.Dict) for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    others = {key for key, info in shared_options.options_templates.items() if (getattr(info, "section", None) or (None,))[0] != "callbacks"}
    assert "sd_vae_explanation" in defined
    assert not callbacks & (defined | others)


def test_every_core_string_is_translated(strings, catalog):
    spa = strings_module.spa_catalog()
    missing = sorted(text for text in strings if not (catalog.get(text) or "").strip() and text not in spa)
    assert missing == [], f"localizations/zh_CN.json lacks {len(missing)} core strings: {missing[:40]}"


def test_the_forms_under_hidden_directories_are_translated(catalog, monkeypatch):
    """``--hide-ui-dir-config`` appended a sentence to the img2img Batch help, so
    the whole paragraph was a key no catalog had and showed in English."""

    from modules import shared
    from remi.core.tabs import tab_schema

    found: set[str] = set()
    with webui_state():
        monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
        for tab in strings_module.TABS:
            strings_module._text_strings(tab_schema(tab)["core"]["layout"], found)
        dynamic = strings_module._dynamic_names()
    spa = strings_module.spa_catalog()
    missing = sorted(text for text in found if text not in dynamic and not strings_module._is_data(text) and not (catalog.get(text) or "").strip() and text not in spa)
    assert missing == []


def _formatted_toasts() -> list[str]:
    """``toast(...)`` calls in Remi's own Python whose text is formatted on the server
    (an f-string, ``+``, ``%`` or ``.format``): no catalog can have that sentence."""

    import ast

    roots = [ROOT / "remi", *sorted((ROOT / "extensions-builtin").glob("*/")), ROOT / "scripts", ROOT / "modules" / "processing_scripts"]
    found = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.relative_to(ROOT).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "toast" and node.args):
                    continue
                text = node.args[0]
                formatted = isinstance(text, (ast.JoinedStr, ast.BinOp)) or (isinstance(text, ast.Call) and getattr(text.func, "attr", None) == "format")
                if formatted:
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def test_toasts_are_catalog_sentences():
    """An action's failure toast was ``f"Action {name} failed: {error}"``: English in
    every language. A sentence with variable parts is a ``message(text, **params)``."""

    assert _formatted_toasts() == []


def test_the_action_failure_toasts_are_translated(catalog):
    from remi import ui

    class Owner:
        @ui.action(timeout=0.05)
        def slow(self, ctx, values):
            import time

            time.sleep(0.3)

        @ui.action(timeout=5)
        def boom(self, ctx, values):
            raise RuntimeError("disk full")

    ctx = ui.ActionContext(tab="txt2img")
    for name in ("slow", "boom"):
        message = ui.call_action_isolated(Owner(), name, ctx, {})["__toast__"]["message"]
        assert catalog.get(message["text"]), message
        assert set(message["params"]) == set(_placeholders(message["text"]))
    assert catalog.get("the setting rejected the value")


def test_entries_are_well_formed(catalog):
    assert catalog and all(isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in catalog.items())
    broken = [key for key, value in catalog.items() if _placeholders(key) != _placeholders(value)]
    assert broken == []


def test_terms_follow_the_glossary(catalog):
    """Consistent terminology for the core concepts."""

    glossary = {
        "Seed": "种子",
        "Sampler": "采样器",
        "Sampling Method": "采样器",
        "Schedule type": "调度器",
        "CFG Scale": "提示词相关性 CFG",
        "Hires. fix": "高分辨率修复",
        "Denoising strength": "重绘幅度",
        "Batch Count": "生成批次",
        "Batch Size": "每批数量",
        "Extra Networks": "附加网络",
        "Infotext": "生成信息",
    }
    assert {key: catalog.get(key) for key in glossary} == glossary
    # no half-translated core terms left behind
    leftovers = [key for key, value in catalog.items() if re.search(r"Denoising 强度|负向提示词|Hires 步数|Batch 大小", value)]
    assert leftovers == []


# Everyday UI words that have an established Chinese term (步数, 卸载, 分块,
# 批次, 种子, 提示词, 模型, 局部重绘, ...): left in English inside a Chinese
# sentence they make a mixed-language label, not a technical name. Syntax
# (``blocks=``), identifiers (``positive_prompt``), paths and `code` are exempt.
UNTRANSLATED_WORDS = frozenset(
    {
        *("step", "steps", "offload", "offloaded", "tiled", "tile", "tiles", "batch", "batches", "seed", "seeds"),
        *("prompt", "prompts", "checkpoint", "checkpoints", "sampler", "samplers", "scheduler", "schedulers"),
        *("inpaint", "inpainted", "inpainting", "sketch", "hires", "detect", "detected", "detector", "detectors"),
        *("detection", "encoder", "encoders", "outputs", "profiler", "epoch", "epochs", "workflow", "block"),
        *("blocks", "mask", "masks", "strength", "upscaler", "denoise", "padding", "noise", "multiplier"),
        *("enabled", "disabled", "image", "images", "width", "height", "infotext", "refiner"),
    }
)
_WORD = re.compile(r"(?<![\w=./\\\-])([A-Za-z][a-z]*)(?![\w=./\\\-])")
# Not labels: placeholders, URLs, `code`, markup, regular expressions (``^...$``), names of other
# products and models that contain a common word ("Prompt Queue", "Wan 2.2 High Noise"), a name
# kept with its gloss ("Tile (分块)", ControlNet's control type) and the English term given after
# its translation ("生成信息（Infotext）").
_EXEMPT = re.compile(
    r"\{\{?\w+\}?\}|https?://\S+|`[^`]*`|<[^>]+>|\^\S*\$"
    r"|Agent Scheduler|Prompt (?:Queue|Workshop)|Qwen Image|Image Browser|(?:High|Low) Noise"
    r"|\b[A-Z][a-z]+ \([一-鿿]+\)|(?<=[一-鿿])（[A-Za-z][A-Za-z .]*）"
)


def _mixed_words(value: str) -> set[str]:
    """The common English words (also capitalised: "Step", "Tile") left in a Chinese text."""

    return {word for word in _WORD.findall(_EXEMPT.sub(" ", value)) if word.lower() in UNTRANSLATED_WORDS}


def _zh_catalogs():
    yield "localizations/zh_CN.json", CATALOG
    yield from ((str(path.relative_to(ROOT)), path) for path in sorted((ROOT / "extensions-builtin").glob("*/localizations/zh_CN.json")))
    yield from ((str(path.relative_to(ROOT)), path) for path in sorted((ROOT / "webui" / "src").rglob("zh-CN.json")))


def test_no_mixed_language_labels_in_any_catalog():
    """zh-CN keeps English only for technical names (LoRA, VAE, UNet, sigma,
    bbox, ...), never for common words such as step, offload or tiled."""

    mixed = []
    for name, path in _zh_catalogs():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if not isinstance(value, str) or not re.search(r"[一-鿿]", value):
                continue
            words = _mixed_words(value)
            if words:
                mixed.append(f"{name}: {key[:60]!r} -> {sorted(words)}")
    assert mixed == [], "\n".join(mixed)


# Kept in English on purpose: technical names that read the same in the
# Chinese UI (parameter names of a paper or library, prompt examples).
KEPT_IN_ENGLISH = frozenset({"Dense Block", "sigma noise", "retro artstyle, sketch, watercolor, flat color, 1990s (style)"})


def test_untranslated_entries_are_technical_names():
    """An entry left in English ("Block {{index}}": "Block {{index}}") is
    either a technical name listed above or contains no common word."""

    left = []
    for name, path in _zh_catalogs():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if isinstance(value, str) and not re.search(r"[一-鿿]", value) and value not in KEPT_IN_ENGLISH and _mixed_words(value):
                left.append(f"{name}: {key[:60]!r} -> {value[:60]!r}")
    assert left == [], "\n".join(left)


# One Chinese term per concept in every catalog (core, extensions, SPA, widgets).
SHARED_TERMS = {"Hires. fix": ("高分辨率修复", "高清修复"), "Refiner": ("精炼器", "Refiner")}


def test_every_catalog_uses_the_shared_terms():
    stray = []
    for name, path in _zh_catalogs():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if not isinstance(value, str) or not re.search(r"[一-鿿]", value):
                continue
            for _concept, (_term, wrong) in SHARED_TERMS.items():
                if wrong in _EXEMPT.sub(" ", value):
                    stray.append(f"{name}: {key[:60]!r} uses {wrong!r}")
    assert stray == [], "\n".join(stray)


def test_the_glossary_check_catches_mixed_labels():
    flagged = _mixed_words
    assert flagged("停止缓存 step") == {"step"}
    assert flagged("为 UNet 启用(始终 offload)") == {"offload"}
    assert flagged("为 VAE 启用(始终 tiled)") == {"tiled"}
    # capitalised common words are no technical names either
    assert flagged("绕过 RadialAttention 的最后几个 Step 的数量") == {"Step"}
    assert flagged("Tile Batch 大小") == {"Tile", "Batch"}
    assert flagged("将 Lora 哈希值添加到 infotext") == {"infotext"}
    assert flagged("生成信息（Infotext）是 WebUI 对包含生成参数的文本的称呼") == set()
    # syntax, identifiers, code, placeholders, regular expressions and product names are not labels
    assert flagged("每行一个预设：名称=blocks=...;modules=...") == set()
    assert flagged("格式：classname: positive_prompt | negative_prompt") == set()
    assert flagged("示例 `<lora:name:blocks=0-9=0.8>`，共 {{steps}} 步") == set()
    assert flagged("regex - 例如 ^(?!.*Hires).*$") == set()
    assert flagged("从 Agent Scheduler 或 Prompt Queue 导入…") == set()
    assert flagged("Tile (分块)") == set()


def test_the_spa_catalog_is_not_contradicted(catalog):
    """The SPA's own catalog wins at runtime; the core one must agree with it."""

    spa = strings_module.spa_catalog()
    assert sorted(key for key, value in catalog.items() if key in spa and spa[key] != value) == []


def test_the_language_pack_does_not_shadow_the_core_catalog(catalog):
    """Extension catalogs are merged after the core one: the language pack
    keeps only texts the core catalog does not translate."""

    pack = json.loads((ROOT / "extensions-builtin" / "sd-webui-language-diffusion" / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    assert sorted(set(pack) & set(catalog)) == []


def test_extension_catalogs_agree_with_the_core_catalog(catalog):
    """Extension catalogs are merged after the core one and win: a different
    translation of a core text would rename that text everywhere."""

    conflicts = {}
    for path in sorted((ROOT / "extensions-builtin").glob("*/localizations/zh_CN.json")):
        own = json.loads(path.read_text(encoding="utf-8"))
        conflicts |= {f"{path.parts[-3]}: {key}": (value, catalog[key]) for key, value in own.items() if key in catalog and catalog[key] != value}
    assert conflicts == {}


# Messages of a rejected value that are no catalog key, each with why. The
# rest are constants or ``Sentence`` templates the catalogs translate.
FREE_TEXT_REJECTIONS = {
    ("remi/core/forms.py", "message"): "a form's `required` message (REQUIRED_INPUTS, collected with the form)",
    ("remi/server/options.py", "message"): "the parameter of `_error`, the helper its callers use",
    ("remi/server/options.py", "reason"): "frozen_reason()'s sentences, in the SPA catalog (a locked setting's reason)",
    ("remi/server/options.py", 'f"{type(error).__name__}: {error}"'): "a setting's own code raised: its exception, as it is",
    ("remi/ui/nodes.py", "reason"): "a param's own `validate` raised: the extension's text, or its Sentence",
    ("extensions-builtin/sd_forge_controlnet/lib_controlnet/unit_actions.py", "message"): "the parameter of `bad_input`, the helper its callers use",
    ("extensions-builtin/sd_forge_controlnet/lib_controlnet/unit_param.py", "str(error)"): "the infotext parser's own error",
    ("extensions-builtin/sd_forge_controlnet/lib_controlnet/unit_param.py", 'f"{key}: {error}"'): "a unit field's name before that field's own message",
}
EXTENSION_REJECTIONS = {
    "extensions-builtin/sd_forge_controlnet/lib_controlnet": "extensions-builtin/sd_forge_controlnet",
    "extensions-builtin/sd-postprocess-suite": "extensions-builtin/sd-postprocess-suite",
}


def test_every_rejected_value_has_a_catalog_key(catalog):
    """A field shows the server's message under it in the UI language: the
    message is a catalog key, or a ``Sentence`` whose template is one (a
    message formatted with the value, ``f"must be at most {maximum}"``, could
    never be translated)."""

    texts, unchecked = strings_module.param_error_messages(strings_module.PARAM_ERROR_SOURCES + tuple(EXTENSION_REJECTIONS))
    assert sorted(set(unchecked) - set(FREE_TEXT_REJECTIONS)) == []
    assert sorted(set(FREE_TEXT_REJECTIONS) - set(unchecked)) == []  # an entry that no longer applies
    assert {"must be at most {{maximum}}, got {{value}}", "unknown upload {{id}}", "a choice is required", "upload a control image"} <= texts

    spa = strings_module.spa_catalog()
    core = strings_module.param_error_strings()
    missing = sorted(text for text in core if not (catalog.get(text) or "").strip() and text not in spa)
    assert missing == []
    for folder, extension in EXTENSION_REJECTIONS.items():
        own = json.loads((ROOT / extension / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
        texts = strings_module.param_error_messages((folder,))[0]
        assert sorted(text for text in texts if not (own.get(text) or catalog.get(text) or "").strip()) == [], extension
        assert all(_placeholders(own.get(text) or catalog[text]) == _placeholders(text) for text in texts)
    assert all(_placeholders(catalog.get(text) or spa[text]) == _placeholders(text) for text in core)


def test_a_rejected_value_keeps_its_template():
    """The JSON of a rejected value carries the catalog key and the values
    next to the English message."""

    from remi import ui

    with pytest.raises(ui.ParamError) as caught:
        ui.Number("steps", minimum=1, maximum=150).coerce(999, ui.Context())
    assert caught.value.to_json() | {"owner": "core", "param": "steps"} == {
        "owner": "core",
        "param": "steps",
        "message": "must be at most 150, got 999",
        "template": "must be at most {{maximum}}, got {{value}}",
        "params": {"maximum": "150", "value": "999"},
    }
    # a constant message is its own key
    assert ui.ParamError("a choice is required").to_json() == {"owner": None, "param": None, "message": "a choice is required"}
