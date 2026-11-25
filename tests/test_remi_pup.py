"""Unit tests for Multimodal & Local Prompt Upsampler (``remi/prompt_tools/pup.py``
and multimodal extensions in ``remi/prompt_tools/translate.py``).
"""

from __future__ import annotations

import base64
import io
import json
import sys
import types

import pytest
import torch
from PIL import Image

from remi.prompt_tools import pup
from remi.prompt_tools.translate import (
    DEFAULT_INSTRUCTIONS,
    MAX_CHARS,
    MAX_DESCRIPTION,
    NEGATIVE_INSTRUCTIONS,
    ProviderConfig,
    TranslationError,
    _deepl,
    _http_post,
    _libretranslate,
    _openai,
    generate_prompt,
    has_cjk,
    translate,
)


def _openai_cfg() -> ProviderConfig:
    return ProviderConfig(provider="openai", url="http://127.0.0.1:8080/v1", key="test-key", model="qwen3.5-9b-pup")


def _png_bytes(size: tuple[int, int] = (32, 32), mode: str = "RGB", color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    with Image.new(mode, size, color) as im:
        im.save(buf, format="PNG")
    return buf.getvalue()


def test_generate_prompt_supports_multimodal_images_and_preserve_newlines():
    captured = {}

    def fake_post(url, headers, body):
        captured["url"] = url
        captured["body"] = body
        return {"choices": [{"message": {"content": "Line one caption\nLine two detail"}}]}

    out = generate_prompt(
        _openai_cfg(),
        "a lighthouse at dusk",
        images=["data:image/jpeg;base64,AAAA"],
        temperature=0.3,
        max_tokens=512,
        preserve_newlines=True,
        post=fake_post,
    )
    assert out == "Line one caption\nLine two detail"
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["max_tokens"] == 512
    assert captured["body"]["temperature"] == 0.3
    user_msg = captured["body"]["messages"][1]["content"]
    assert isinstance(user_msg, list)
    assert user_msg[0]["type"] == "text"
    assert user_msg[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}

    # List-of-content-parts VLM response & malformed response handling
    list_out = generate_prompt(
        _openai_cfg(),
        "a lighthouse",
        preserve_newlines=True,
        post=lambda _u, _h, _b: {"choices": [{"message": {"content": [{"type": "text", "text": "Part A "}, {"type": "text", "text": "Part B"}]}}]},
    )
    assert list_out == "Part A Part B"

    with pytest.raises(TranslationError, match="unexpected chat completion answer"):
        generate_prompt(_openai_cfg(), "a lighthouse", post=lambda _u, _h, _b: {"unexpected": True})


def test_upsample_prompt_flux3_multimodal_aspect_ratio_and_bbox_reconciliation(tmp_path):
    img_path = tmp_path / "ref1.png"
    with Image.new("RGB", (1200, 800), (120, 180, 220)) as im:
        im.save(img_path)

    captured = {}

    def fake_post(url, headers, body):
        captured["body"] = body
        # Return a caption citing <cat_1> plus a slightly drifted box to verify reconcile_boxes re-imposes locked coords
        rows = [
            {"id": "cat_1", "bbox": [105, 102, 505, 498], "desc": "An orange tabby cat sitting upright."},
            {"id": "lamp_1", "bbox": [50, 650, 350, 850], "desc": "A warm brass table lamp."},
        ]
        caption = f"Full-color indoor photograph of a tabby cat <cat_1> beside a warm brass lamp <lamp_1>. {json.dumps(rows)}"
        return {"choices": [{"message": {"content": caption}}]}

    res = pup.upsample_prompt(
        "cat next to a lamp",
        model_family="flux3",
        reference_images=[img_path],
        aspect_ratio="16:9",
        bbox_draft=[{"id": "cat_1", "bbox": [0.1, 0.1, 0.5, 0.5], "desc": "orange cat"}],
        mode="auto",
        config=_openai_cfg(),
        post=fake_post,
    )

    assert res.model_family == "flux3"
    assert res.mode == "i2i_bbox"
    assert res.backend == "openai_vlm"
    assert res.used_reference_count == 1
    assert res.aspect_ratio == "16:9"
    assert res.layout_rows is not None
    # Locked coordinates [100, 100, 500, 500] MUST be preserved exactly after reconciliation
    cat_row = next(r for r in res.layout_rows if r["id"] == "cat_1")
    assert cat_row["bbox"] == [100, 100, 500, 500]
    assert "orange tabby" in cat_row["desc"].lower()

    payload = res.to_dict()
    assert payload["model_family"] == "flux3"
    assert payload["warnings"] == []
    assert payload["layout_rows"] == res.layout_rows

    # Verify multimodal payload carried resized image + TARGET CANVAS: 16:9
    user_parts = captured["body"]["messages"][1]["content"]
    assert isinstance(user_parts, list) and len(user_parts) == 2
    assert "TARGET CANVAS: 16:9" in user_parts[0]["text"]
    assert user_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_upsample_prompt_qwen21_moss_and_fibo_vgl_templates():
    seen_systems = {}

    def fake_post(url, headers, body):
        sys_msg = body["messages"][0]["content"]
        if "VGL 2.0" in sys_msg:
            seen_systems["fibo"] = sys_msg
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "short_description": "Glass perfume bottle on white marble",
                                    "objects": [{"description": "amber glass bottle", "location": "center"}],
                                    "background_setting": "clean studio sweep",
                                    "lighting": {"conditions": "softbox", "direction": "top-left", "shadows": "crisp"},
                                    "aesthetics": {"composition": "centered", "color_scheme": "warm amber", "mood_atmosphere": "luxury"},
                                    "photographic_characteristics": {"depth_of_field": "shallow", "focus": "sharp", "camera_angle": "eye-level", "lens_focal_length": "85mm"},
                                    "style_medium": "studio photograph",
                                }
                            )
                        }
                    }
                ]
            }
        if "Qwen-Image-2.1" in sys_msg:
            seen_systems["qwen21"] = sys_msg
            return {"choices": [{"message": {"content": "Replace the jacket in <image_1> with a red trench coat."}}]}
        seen_systems["moss"] = sys_msg
        return {"choices": [{"message": {"content": 'Editorial poster reading "REMI 2026" in bold serif typography.'}}]}

    with Image.new("RGBA", (64, 64), (255, 0, 0, 128)) as rgba_ref:
        qwen_res = pup.upsample_prompt("change jacket to red", model_family="qwen21", reference_images=[rgba_ref], config=_openai_cfg(), post=fake_post)
    assert qwen_res.mode == "i2i" and "<image_1>" in qwen_res.prompt and qwen_res.used_reference_count == 1
    assert "qwen21" in seen_systems

    fibo_res = pup.upsample_prompt("perfume bottle", model_family="fibo", config=_openai_cfg(), post=fake_post)
    assert fibo_res.mode == "vgl_json"
    assert fibo_res.vgl_json is not None and fibo_res.vgl_json["style_medium"] == "studio photograph"
    assert fibo_res.warnings == ()

    moss_res = pup.upsample_prompt("poster", model_family="moss", config=_openai_cfg(), post=fake_post)
    assert moss_res.mode == "t2i" and "REMI 2026" in moss_res.prompt
    assert "moss" in seen_systems


def test_pup_system_prompts_and_active_model_family_detection(monkeypatch):
    assert pup.get_pup_system_prompt("flux3", "t2i", negative=True) == NEGATIVE_INSTRUCTIONS
    assert pup.get_pup_system_prompt("zimage", "t2i") == pup.ZIMAGE_T2I_PROMPT
    assert pup.get_pup_system_prompt("ernie", "t2i") == pup.ERNIE_T2I_PROMPT
    assert pup.get_pup_system_prompt("anima", "t2i") == DEFAULT_INSTRUCTIONS
    assert pup.get_pup_system_prompt("moss", "i2i") == pup.MOSS_I2I_PROMPT
    assert pup.get_pup_system_prompt("qwen21", "t2i") == pup.QWEN21_T2I_PROMPT
    assert "JSON" in pup.get_pup_system_prompt("generic", "t2i_bbox")
    assert "VGL 2.0" in pup.get_pup_system_prompt("generic", "vgl_json")

    from backend.args import dynamic_args

    # 1. dynamic_args flag wins when active
    for flag in ("flux3", "qwen21", "moss", "anima"):
        monkeypatch.setattr(dynamic_args, flag, False, raising=False)
    monkeypatch.setattr(dynamic_args, "qwen21", True, raising=False)
    assert pup.detect_active_model_family() == "qwen21"
    monkeypatch.setattr(dynamic_args, "qwen21", False, raising=False)

    # 2. sd_model engine flag detection
    fake_sd_models = types.ModuleType("modules.sd_models")
    fake_sd_models.model_data = types.SimpleNamespace(sd_model=types.SimpleNamespace(is_fibo_engine=True))
    monkeypatch.setitem(sys.modules, "modules.sd_models", fake_sd_models)
    assert pup.detect_active_model_family() == "fibo"

    # 3. sd_model class-name fallback
    class _ErnieImageDummy:
        pass

    fake_sd_models.model_data = types.SimpleNamespace(sd_model=_ErnieImageDummy())
    assert pup.detect_active_model_family() == "ernie"

    # 4. Unknown sd_model falls back to generic
    fake_sd_models.model_data = types.SimpleNamespace(sd_model=object())
    assert pup.detect_active_model_family() == "generic"


def test_aspect_ratio_and_bbox_normalization_edge_cases():
    assert pup.normalize_aspect_ratio(None) is None
    assert pup.normalize_aspect_ratio("   ") is None
    assert pup.normalize_aspect_ratio("1024x768") == "4:3"
    assert pup.normalize_aspect_ratio([1920, 1080]) == "16:9"
    # Non-standard ratio whose 100x-scaled gcd exceeds 64 falls back to rounded integer ratio
    assert pup.normalize_aspect_ratio("1001.3:703.2") == "1001:703"

    for bad in ([1, 2, 3], 123, "0:9", "-4:3", (float("inf"), 1.0)):
        with pytest.raises(TranslationError) as err:
            pup.normalize_aspect_ratio(bad)  # type: ignore[arg-type]
        assert err.value.status == 422

    # Canvas [0, 1000] boxes and src_bbox/tgt_bbox edit boxes
    canvas_rows = pup._normalize_bbox_rows(
        [
            {"id": "obj_1", "bbox": [100, 150, 400, 650], "desc": "tree"},
            {"id": "move_1", "from": "ref_0", "src_bbox": [10, 20, 200, 300], "tgt_bbox": [50, 60, 400, 500], "desc": "moved mug"},
        ]
    )
    assert canvas_rows[0] == {"id": "obj_1", "bbox": [100, 150, 400, 650], "desc": "tree"}
    assert canvas_rows[1]["from"] == "ref_0"
    assert canvas_rows[1]["src_bbox"] == [10, 20, 200, 300]
    assert canvas_rows[1]["tgt_bbox"] == [50, 60, 400, 500]

    for bad_draft in (
        ["not-a-dict"],
        [{"id": "missing_boxes"}],
        [{"id": "bad_unit", "src_bbox": [0.1, 0.1, 0.5, 0.5], "tgt_bbox": [0.1, 0.2]}],
        [{"id": "bad_canvas", "bbox": [900, 900, 100, 100]}],
        [{"id": "bad_src", "src_bbox": [900, 900, 100, 100], "tgt_bbox": [10, 10, 200, 200]}],
    ):
        with pytest.raises(TranslationError) as err:
            pup._normalize_bbox_rows(bad_draft)  # type: ignore[arg-type]
        assert err.value.status == 422


def test_prepare_reference_images_all_input_types_and_limits(tmp_path):
    raw_png = _png_bytes((40, 40), "RGB", (200, 100, 50))
    b64_uri = "data:image/png;base64," + base64.b64encode(raw_png).decode("ascii")
    corrupt_b64_uri = "data:image/png;base64,!!!not-valid-b64!!!"
    non_b64_uri = "data:image/svg+xml,<svg/>"
    http_url = "https://example.com/ref.png"

    with Image.new("LA", (32, 32), (180, 128)) as la_img, Image.new("L", (32, 32), 120) as gray_img:
        uris, pils, warnings = pup.prepare_reference_images(
            [la_img, gray_img, raw_png, b64_uri, corrupt_b64_uri, non_b64_uri, http_url],
            max_references=7,
        )
    try:
        assert warnings == []
        assert len(uris) == 7
        assert uris[0].startswith("data:image/png;base64,")  # LA converted to RGBA PNG
        assert uris[1].startswith("data:image/jpeg;base64,")  # L converted to RGB JPEG
        assert uris[2].startswith("data:image/jpeg;base64,")  # bytes decoded & encoded
        assert uris[3].startswith("data:image/jpeg;base64,")  # valid base64 URI re-encoded
        assert uris[4] == corrupt_b64_uri  # corrupt base64 passed through
        assert uris[5] == non_b64_uri
        assert uris[6] == http_url
        assert len(pils) == 4
    finally:
        for im in pils:
            im.close()

    # Over max_references trims and warns
    uris_trim, pils_trim, trim_warnings = pup.prepare_reference_images([http_url, http_url, http_url], max_references=2)
    try:
        assert len(uris_trim) == 2
        assert any("Trimmed reference images" in w for w in trim_warnings)
    finally:
        for im in pils_trim:
            im.close()

    with pytest.raises(TranslationError) as missing_err:
        pup.prepare_reference_images([str(tmp_path / "does_not_exist.png")])
    assert missing_err.value.status == 422

    with pytest.raises(TranslationError) as type_err:
        pup.prepare_reference_images([12345])
    assert type_err.value.status == 422


def test_local_qwen35_pup_runner_always_releases_vram_on_success_and_error(tmp_path, monkeypatch):
    fake_gguf = tmp_path / "qwen35-9b-pup-Q4_K_M.gguf"
    fake_gguf.write_bytes(b"GGUF")

    release_calls = []
    monkeypatch.setattr(pup.LocalQwen35PUPRunner, "release_vram", classmethod(lambda cls: release_calls.append(1)))

    # 1. Success path via injectable local_runner
    res = pup.upsample_prompt(
        "sunset over alpine lake",
        model_family="flux3",
        aspect_ratio=(1920, 1080),
        local_model_path=fake_gguf,
        local_runner=lambda **kw: "Full-color outdoor photograph of a serene alpine lake at golden sunset.",
    )
    assert res.backend == "local_qwen35"
    assert res.aspect_ratio == "16:9"
    assert "alpine lake" in res.prompt
    assert len(release_calls) == 1

    # 2. Failure path MUST STILL call release_vram in finally!
    def failing_runner(**_kw):
        raise TranslationError("synthetic local OOM", 502)

    with pytest.raises(TranslationError, match="synthetic local OOM"):
        pup.upsample_prompt("boom", model_family="flux3", local_model_path=fake_gguf, local_runner=failing_runner)
    assert len(release_calls) == 2

    # 3. Empty answer raises 502
    with pytest.raises(TranslationError) as empty_local:
        pup.upsample_prompt("empty", model_family="flux3", local_model_path=fake_gguf, local_runner=lambda **_kw: "   ")
    assert empty_local.value.status == 502


def test_local_qwen35_runner_gguf_transformers_and_release_vram(tmp_path, monkeypatch):
    # Real release_vram execution across CUDA and MPS branches
    cuda_cleared = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cuda_cleared.append(1))
    pup.LocalQwen35PUPRunner.release_vram()
    assert len(cuda_cleared) == 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pup.LocalQwen35PUPRunner.release_vram()

    # Path validation in generate()
    with pytest.raises(TranslationError) as missing_err:
        pup.LocalQwen35PUPRunner(tmp_path / "missing.gguf").generate("sys", "usr")
    assert missing_err.value.status == 422

    bad_file = tmp_path / "weights.bin"
    bad_file.write_bytes(b"BIN")
    with pytest.raises(TranslationError) as fmt_err:
        pup.LocalQwen35PUPRunner(bad_file).generate("sys", "usr")
    assert fmt_err.value.status == 422

    # GGUF path: missing llama_cpp -> 502
    gguf_file = tmp_path / "pup.gguf"
    gguf_file.write_bytes(b"GGUF")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(TranslationError, match="llama-cpp-python") as no_llama:
        pup.LocalQwen35PUPRunner(gguf_file).generate("sys", "usr")
    assert no_llama.value.status == 502

    # GGUF path: mocked llama_cpp success + error + upsample_prompt default runner path
    closed = []
    seen_msgs = []
    mode = {"behavior": "ok"}

    class FakeLlama:
        def __init__(self, **_kw):
            pass

        def create_chat_completion(self, *, messages, **_kw):
            seen_msgs.append(messages)
            if mode["behavior"] == "boom":
                raise RuntimeError("gguf crash")
            if mode["behavior"] == "terr":
                raise TranslationError("explicit terr", 503)
            return {"choices": [{"message": {"content": "Upsampled via GGUF"}}]}

        def close(self):
            closed.append(1)

    fake_llama_mod = types.ModuleType("llama_cpp")
    fake_llama_mod.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_mod)

    gguf_res = pup.upsample_prompt("a cozy cabin", model_family="zimage", reference_images=["https://example.com/x.png"], local_model_path=gguf_file)
    assert gguf_res.backend == "local_qwen35"
    assert gguf_res.prompt == "Upsampled via GGUF"
    assert isinstance(seen_msgs[-1][1]["content"], list)
    assert len(closed) == 1

    mode["behavior"] = "boom"
    with pytest.raises(TranslationError, match="gguf crash"):
        pup.LocalQwen35PUPRunner(gguf_file).generate("sys", "usr")
    assert len(closed) == 2

    mode["behavior"] = "terr"
    with pytest.raises(TranslationError) as terr:
        pup.LocalQwen35PUPRunner(gguf_file).generate("sys", "usr")
    assert terr.value.status == 503
    assert len(closed) == 3

    # Transformers path: mocked transformers module covering text-only, VLM multimodal, fallback & error
    hf_dir = tmp_path / "qwen35-hf"
    hf_dir.mkdir()
    moved_to_cpu = []
    tf_state = {"vlm_fail": False, "gen_fail": False, "has_template": True}

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _path, **_kw):
            return cls()

        def apply_chat_template(self, _messages, **_kw):
            return "templated prompt"

        def __call__(self, _text, **_kw):
            return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

        def decode(self, _tokens, **_kw):
            return "Upsampled via Transformers text"

    class FakeVLMProcessor:
        image_processor = object()

        @classmethod
        def from_pretrained(cls, _path, **_kw):
            if tf_state["vlm_fail"]:
                raise RuntimeError("no vlm processor")
            return cls()

        def __call__(self, *, text, images, **_kw):
            assert len(text) == 1 and len(images) == 1
            return FakeInputs(input_ids=torch.tensor([[10, 20]]))

        def decode(self, _tokens, **_kw):
            return "Upsampled via Transformers VLM"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path, **_kw):
            return cls()

        def to(self, device):
            if device == "cpu":
                moved_to_cpu.append(1)
            return self

        def eval(self):
            return self

        def generate(self, **inputs):
            if tf_state["gen_fail"]:
                raise RuntimeError("cuda oom")
            return torch.cat([inputs["input_ids"], torch.tensor([[99, 100]])], dim=-1)

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = FakeTokenizer
    fake_tf.AutoProcessor = FakeVLMProcessor
    fake_tf.AutoModelForCausalLM = FakeModel
    fake_tf.AutoModelForImageTextToText = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    # Text-only
    out_text = pup.LocalQwen35PUPRunner(hf_dir).generate("sys", "user prompt")
    assert out_text == "Upsampled via Transformers text"
    assert len(moved_to_cpu) >= 1

    # Multimodal VLM (FakeVLMProcessor has no apply_chat_template -> exercises ChatML fallback!)
    with Image.new("RGB", (32, 32), (1, 2, 3)) as ref_im:
        out_vlm = pup.LocalQwen35PUPRunner(hf_dir).generate("sys", "edit this", pil_images=[ref_im])
        assert out_vlm == "Upsampled via Transformers VLM"

        # VLM load error falls back to AutoTokenizer + AutoModelForCausalLM
        tf_state["vlm_fail"] = True
        out_fb = pup.LocalQwen35PUPRunner(hf_dir).generate("sys", "edit fallback", pil_images=[ref_im])
        assert out_fb == "Upsampled via Transformers text"

    # Generate failure raises 502 and still cleans up
    tf_state["gen_fail"] = True
    before_cpu = len(moved_to_cpu)
    with pytest.raises(TranslationError, match="cuda oom"):
        pup.LocalQwen35PUPRunner(hf_dir).generate("sys", "fail now")
    assert len(moved_to_cpu) > before_cpu


def test_upsample_prompt_warnings_modes_and_validation_errors():
    # Unknown family falls back to generic; explicit mode="t2i"; current prompt included; moss with ref
    seen = []

    def echo_post(_url, _headers, body):
        seen.append(body)
        return {"choices": [{"message": {"content": "Clean upsampled prompt"}}]}

    r_generic = pup.upsample_prompt(
        "sunset",
        model_family="unknown_family_xyz",
        mode="t2i",
        current="old prompt",
        config=_openai_cfg(),
        post=echo_post,
    )
    assert r_generic.model_family == "generic"
    assert r_generic.mode == "t2i"
    assert "Current prompt: old prompt" in seen[-1]["messages"][1]["content"]

    r_moss_ref = pup.upsample_prompt(
        "add snow",
        model_family="moss",
        reference_images=["https://example.com/a.png"],
        config=_openai_cfg(),
        post=echo_post,
    )
    assert r_moss_ref.mode == "i2i"
    assert "Reference images follow in order" in seen[-1]["messages"][1]["content"][0]["text"]

    # mode="auto" with bbox_draft only -> t2i_bbox, plus Layout warning when <cat_1> omitted from prose
    r_bbox_warn = pup.upsample_prompt(
        "cat on chair",
        model_family="flux3",
        bbox_draft=[{"id": "cat_1", "bbox": [0.1, 0.1, 0.5, 0.5], "desc": "cat"}],
        config=_openai_cfg(),
        post=lambda _u, _h, _b: {"choices": [{"message": {"content": 'A tabby cat sitting on a chair. [{"id": "cat_1", "bbox": [100, 100, 500, 500], "desc": "cat"}]'}}]},
    )
    assert r_bbox_warn.mode == "t2i_bbox"
    assert any("Layout warning:" in w for w in r_bbox_warn.warnings)

    # VGL warning on partial JSON object, and raw-text warning when non-JSON returned
    r_vgl_partial = pup.upsample_prompt(
        "product shot",
        model_family="fibo",
        config=_openai_cfg(),
        post=lambda _u, _h, _b: {"choices": [{"message": {"content": '{"short_description": "only one key"}'}}]},
    )
    assert r_vgl_partial.vgl_json == {"short_description": "only one key"}
    assert any("VGL warning:" in w for w in r_vgl_partial.warnings)

    r_vgl_raw = pup.upsample_prompt(
        "product shot",
        model_family="fibo",
        mode="vgl_json",
        config=_openai_cfg(),
        post=lambda _u, _h, _b: {"choices": [{"message": {"content": "Not a JSON response at all"}}]},
    )
    assert r_vgl_raw.vgl_json is None
    assert any("VGL JSON mode expected a JSON object" in w for w in r_vgl_raw.warnings)

    # Truncation warning when user_message exceeds MAX_DESCRIPTION
    long_prompt = "a" * (MAX_DESCRIPTION + 200)
    r_trunc = pup.upsample_prompt(long_prompt, model_family="zimage", config=_openai_cfg(), post=echo_post)
    assert any("Truncated upsampler request" in w for w in r_trunc.warnings)

    # Validation errors
    with pytest.raises(TranslationError) as empty_err:
        pup.upsample_prompt("   ", config=_openai_cfg())
    assert empty_err.value.status == 422

    with pytest.raises(TranslationError) as too_long_err:
        pup.upsample_prompt("x" * (pup._MAX_PUP_CHARS + 1), config=_openai_cfg())
    assert too_long_err.value.status == 413

    with pytest.raises(TranslationError) as unconf_err:
        pup.upsample_prompt("a dog", config=ProviderConfig(provider="none"))
    assert unconf_err.value.status == 409

    with pytest.raises(TranslationError) as bad_aspect:
        pup.upsample_prompt("a dog", aspect_ratio="not-a-ratio", config=_openai_cfg())
    assert bad_aspect.value.status == 422

    with pytest.raises(TranslationError) as bad_mode:
        pup.upsample_prompt("a dog", mode="nonexistent", config=_openai_cfg())
    assert bad_mode.value.status == 422


def test_translate_http_post_and_provider_error_paths(monkeypatch):
    cfg = _openai_cfg()
    assert cfg.cache_key == "openai:http://127.0.0.1:8080/v1:qwen3.5-9b-pup"
    assert has_cjk("一只猫") is True
    assert has_cjk("a cat") is False

    # MAX_CHARS guard
    with pytest.raises(TranslationError) as too_big:
        translate(["a" * (MAX_CHARS + 1)], "en", ProviderConfig(provider="dictionary"), {})
    assert too_big.value.status == 413

    # Provider response edge cases
    lt_cfg = ProviderConfig(provider="libretranslate", url="http://lt.local")
    assert _libretranslate(lt_cfg, ["cat"], "zh_CN", lambda _u, _h, _b: {"translatedText": "猫"}) == ["猫"]
    with pytest.raises(TranslationError, match="unexpected LibreTranslate"):
        _libretranslate(lt_cfg, ["cat"], "zh_CN", lambda _u, _h, _b: {"bad": 1})

    with pytest.raises(TranslationError, match="unexpected chat completion"):
        _openai(cfg, ["cat"], "zh_CN", lambda _u, _h, _b: {"bad": 1})
    with pytest.raises(TranslationError, match="list of the same length"):
        _openai(cfg, ["cat"], "zh_CN", lambda _u, _h, _b: {"choices": [{"message": {"content": "[not json]"}}]})

    dl_cfg = ProviderConfig(provider="deepl", key="abc:fx")
    with pytest.raises(TranslationError, match="unexpected DeepL"):
        _deepl(dl_cfg, ["cat"], "zh_CN", lambda _u, _h, _b: {"bad": 1})

    # _http_post success, RequestException, HTTP 500, and non-JSON
    import requests

    def answer(status_code: int, body: bytes):
        response = requests.models.Response()
        response.status_code = status_code
        response.raw = io.BytesIO(body)
        return response

    monkeypatch.setattr(requests, "post", lambda *_a, **_kw: answer(200, b'{"ok": true}'))
    assert _http_post("http://x", {}, {}) == {"ok": True}

    def raise_req(*_a, **_kw):
        raise requests.RequestException("offline")

    monkeypatch.setattr(requests, "post", raise_req)
    with pytest.raises(TranslationError, match="unreachable"):
        _http_post("http://x", {}, {})

    monkeypatch.setattr(requests, "post", lambda *_a, **_kw: answer(500, b"server error"))
    with pytest.raises(TranslationError, match="answered 500: server error"):
        _http_post("http://x", {}, {})

    monkeypatch.setattr(requests, "post", lambda *_a, **_kw: answer(200, b"<html>"))
    with pytest.raises(TranslationError, match="sent no JSON"):
        _http_post("http://x", {}, {})


def test_a_provider_answer_is_read_up_to_a_limit(monkeypatch):
    """A misbehaving endpoint must not fill the server's memory: the answer is
    streamed and refused past 4 MB, and an error page is read only as far as it is shown."""

    import requests

    read: list[int] = []
    sent: dict[str, object] = {}

    class Body(io.BytesIO):
        def read(self, size=-1):
            data = super().read(size)
            read.append(len(data))
            return data

    def post(*_args, **kwargs):
        sent.update(kwargs)
        response = requests.models.Response()
        response.status_code = post.status
        response.raw = Body(b'{"a": "' + b"x" * (5 * 1024 * 1024) + b'"}')
        post.responses.append(response)
        return response

    post.responses = []
    post.status = 200
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(TranslationError, match="larger than 4 MB") as too_large:
        _http_post("http://x", {}, {})
    assert too_large.value.status == 502 and sent["stream"] is True
    assert sum(read) <= 4 * 1024 * 1024 + 64 * 1024 and post.responses[-1].raw.closed
    read.clear()
    post.status = 500
    with pytest.raises(TranslationError, match="answered 500"):
        _http_post("http://x", {}, {})
    assert sum(read) <= 64 * 1024


def test_a_reference_image_over_the_pixel_limit_is_refused_before_decoding(monkeypatch):
    import remi.files

    monkeypatch.setattr(remi.files, "MAX_UPLOAD_PIXELS", 40 * 40 - 1)
    uri = "data:image/png;base64," + base64.b64encode(_png_bytes((40, 40), "RGB", (1, 2, 3))).decode("ascii")
    for item in (uri, _png_bytes((40, 40), "RGB", (1, 2, 3))):
        with pytest.raises(TranslationError) as err:
            pup.prepare_reference_images([item])
        assert err.value.status == 413
