"""Multimodal & local Prompt Upsampler (PUP) for Remi WebUI.

Supports:
- Multimodal OpenAI-compatible VLM endpoints (llama.cpp, Ollama, MLX-LM, vLLM,
  or cloud providers) receiving resized reference images, target aspect ratio,
  and optional bounding-box layout drafts;
- Official / model-tailored system prompt templates for FLUX.3 Image Klein
  (t2i / t2i_bbox / i2i / i2i_bbox with exact box reconciliation),
  Qwen-Image-2.1 (with ``<image_1>`` reference binding & RGBA transparency
  guidance), MOSS-Image / Kromoss, Bria Fibo 2.0 (VGL JSON schema output),
  Z-Image, ERNIE-Image, and Anima;
- An optional in-process local quantized Qwen3.5 PUP runner
  (:class:`LocalQwen35PUPRunner`) that loads the checkpoint on demand,
  stops between tokens when Interrupt is pressed (:class:`GenerationInterrupted`),
  and always unloads and frees GPU/MPS memory in a ``finally`` block after
  upsampling.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from backend.nn import flux3_layout
from remi.prompt_tools.translate import (
    DEFAULT_INSTRUCTIONS,
    NEGATIVE_INSTRUCTIONS,
    GenerationInterrupted,
    Post,
    ProviderConfig,
    TranslationError,
    _clean_prompt,
    _http_post,
    generate_prompt,
)
from remi.ui.sentence import Sentence, sentence_fields

log = logging.getLogger(__name__)

SUPPORTED_PUP_FAMILIES: tuple[str, ...] = (
    "flux3",
    "qwen21",
    "moss",
    "fibo",
    "zimage",
    "ernie",
    "anima",
    "generic",
)

SUPPORTED_PUP_MODES: tuple[str, ...] = (
    "auto",
    "t2i",
    "t2i_bbox",
    "i2i",
    "i2i_bbox",
    "vgl_json",
)

_MAX_PUP_CHARS = 12_000
_ASPECT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[:xX/]\s*(\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class UpsampleResult:
    """Structured result returned by :func:`upsample_prompt`."""

    prompt: str
    original_prompt: str
    model_family: str
    mode: str
    backend: Literal["openai_vlm", "local_qwen35"]
    used_reference_count: int = 0
    aspect_ratio: str | None = None
    layout_rows: list[dict[str, Any]] | None = None
    vgl_json: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "original_prompt": self.original_prompt,
            "model_family": self.model_family,
            "mode": self.mode,
            "backend": self.backend,
            "used_reference_count": self.used_reference_count,
            "aspect_ratio": self.aspect_ratio,
            "layout_rows": self.layout_rows,
            "vgl_json": self.vgl_json,
            # as model advice: English, with the template and params the UI translates
            "warnings": [{"message": str(warning), **sentence_fields(warning)} for warning in self.warnings],
        }


# region Model-family prompt templates

QWEN21_T2I_PROMPT = (
    "You are the prompt upsampler for Qwen-Image-2.1. Expand the user's request into one vivid, "
    "concrete, self-contained image generation prompt. Preserve every explicit subject, count, spatial "
    "relationship, style cue, and any quoted text verbatim in its original language/script (including "
    "Chinese or English typography, layout, font style, color, and placement). If transparent or RGBA "
    "output is requested, explicitly specify an isolated subject on a clean transparent RGBA alpha "
    "background with crisp edges and no background fill. Output only the expanded prompt with no "
    "commentary, Markdown fences, or surrounding quotes."
)

QWEN21_I2I_PROMPT = (
    "You are the multimodal editing prompt upsampler for Qwen-Image-2.1. Reference images are attached "
    "in order and MUST be referenced as <image_1>, <image_2>, ... inside the rewritten editing prompt. "
    "Inspect the attached reference image(s), preserve identity/details not meant to change, and write "
    "one clear, concrete editing instruction describing the target image, lighting, composition, and "
    "any verbatim quoted text. Output only the final editing prompt without commentary or quotes."
)

MOSS_T2I_PROMPT = (
    "You are the prompt upsampler for MOSS-Image / Kromoss. Rewrite the user's brief request into one "
    "dense, cohesive, publication-grade caption covering medium, composition, subjects, materials, "
    "lighting, camera perspective, and exact quoted typography (preserving line breaks and language). "
    "Avoid filler adjectives or meta-phrases like 'this image shows'. Output only the final prompt."
)

MOSS_I2I_PROMPT = (
    "You are the multimodal prompt upsampler for MOSS-Image Edit. Reference images are attached in "
    "order. Describe the exact target image after applying the user's edit while preserving unchanged "
    "subjects, structure, and typography from the reference(s). Output only the final prompt."
)

FIBO_VGL_PROMPT = (
    "You are the structured Visual Generation Language (VGL 2.0) prompt compiler for Bria Fibo 2.0. "
    "Output ONLY a single valid JSON object (no Markdown fences, no prose outside the JSON) matching "
    "the VGL 2.0 schema with top-level keys:\n"
    '  "short_description": concise 1-2 sentence overview of the full scene,\n'
    '  "objects": array of objects, each with "description", "location", "relationship", '
    '"relative_size", and "shape_and_color",\n'
    '  "background_setting": detailed environment/backdrop description,\n'
    '  "lighting": {"conditions", "direction", "shadows"},\n'
    '  "aesthetics": {"composition", "color_scheme", "mood_atmosphere"},\n'
    '  "photographic_characteristics": {"depth_of_field", "focus", "camera_angle", "lens_focal_length"},\n'
    '  "style_medium": exact medium (e.g., "photograph", "digital illustration", "3D render").\n'
    'When reference images are supplied, also include "edit_instruction" and "reference_binding". '
    "Ensure strict JSON validity."
)

ZIMAGE_T2I_PROMPT = (
    "You are the prompt upsampler for Z-Image (NextDiT). Expand the user's idea into one crisp, "
    "natural-language caption with concrete subject appearance, pose, setting, lighting, color palette, "
    "and exact quoted text if any. Keep it focused (60-180 words), direct, and free of quality-tag "
    "spam. Output only the caption."
)

ERNIE_T2I_PROMPT = (
    "You are the prompt upsampler for ERNIE-Image. Expand the user's request into a clear, richly "
    "structured visual description specifying medium, layout, foreground/midground/background elements, "
    "lighting, material textures, and any verbatim bilingual text with typography and placement. "
    "Output only the final caption."
)

ANIMA_T2I_PROMPT = DEFAULT_INSTRUCTIONS


def get_pup_system_prompt(model_family: str, mode: str, *, negative: bool = False) -> str:
    """Select the official or model-family-tailored system prompt."""

    if negative:
        return NEGATIVE_INSTRUCTIONS

    family = (model_family or "generic").strip().lower()
    if family == "flux3":
        return flux3_layout.SYSTEM_PROMPTS["t2i_bbox"] if mode in ("t2i_bbox", "i2i_bbox") else flux3_layout.SYSTEM_PROMPTS["t2i"]
    if family == "qwen21":
        return QWEN21_I2I_PROMPT if mode in ("i2i", "i2i_bbox") else QWEN21_T2I_PROMPT
    if family == "moss":
        return MOSS_I2I_PROMPT if mode in ("i2i", "i2i_bbox") else MOSS_T2I_PROMPT
    if family == "fibo" or mode == "vgl_json":
        return FIBO_VGL_PROMPT
    if family == "zimage":
        return ZIMAGE_T2I_PROMPT
    if family == "ernie":
        return ERNIE_T2I_PROMPT
    if family == "anima":
        return ANIMA_T2I_PROMPT
    return flux3_layout.SYSTEM_PROMPTS["t2i_bbox"] if mode in ("t2i_bbox", "i2i_bbox") else flux3_layout.SYSTEM_PROMPTS["t2i"]


# endregion


# region Active family, aspect ratio, BBox & multimodal reference preparation


def detect_active_model_family() -> str:
    """Detect the currently loaded DiT family without triggering model loading."""

    try:
        from backend.args import dynamic_args
    except Exception:
        return "generic"

    for flag, family in (
        ("flux3", "flux3"),
        ("qwen21", "qwen21"),
        ("moss", "moss"),
        ("anima", "anima"),
    ):
        if bool(getattr(dynamic_args, flag, False)):
            return family

    try:
        import sys

        sd_models = sys.modules.get("modules.sd_models")
        sd_model = getattr(getattr(sd_models, "model_data", None), "sd_model", None) if sd_models else None
        if sd_model is not None:
            for attr, family in (
                ("is_flux3_engine", "flux3"),
                ("is_qwen21_engine", "qwen21"),
                ("is_moss_engine", "moss"),
                ("is_fibo_engine", "fibo"),
                ("is_zimage_engine", "zimage"),
                ("is_ernie_engine", "ernie"),
                ("is_anima_engine", "anima"),
            ):
                if getattr(sd_model, attr, False):
                    return family
            cls_name = type(sd_model).__name__.lower()
            for token, family in (
                ("flux3", "flux3"),
                ("qwenimage21", "qwen21"),
                ("moss", "moss"),
                ("fibo", "fibo"),
                ("zimage", "zimage"),
                ("ernie", "ernie"),
                ("anima", "anima"),
            ):
                if token in cls_name:
                    return family
    except Exception:
        pass
    return "generic"


def normalize_aspect_ratio(aspect_ratio: str | tuple[int, int] | list[int] | None) -> str | None:
    """Normalize ``"16:9"``, ``"1024x768"``, or ``(1024, 768)`` to a reduced ``"W:H"`` string."""

    if aspect_ratio is None:
        return None
    if isinstance(aspect_ratio, (tuple, list)):
        if len(aspect_ratio) != 2:
            raise TranslationError(Sentence("aspect_ratio must have 2 elements, got {{value}}", value=repr(aspect_ratio)), 422, param="aspect_ratio")
        w_val, h_val = float(aspect_ratio[0]), float(aspect_ratio[1])
    elif isinstance(aspect_ratio, str):
        text = aspect_ratio.strip()
        if not text:
            return None
        match = _ASPECT_RE.match(text)
        if not match:
            raise TranslationError(Sentence("invalid aspect_ratio {{value}}; expected 'W:H' or 'WxH'", value=repr(aspect_ratio)), 422, param="aspect_ratio")
        w_val, h_val = float(match.group(1)), float(match.group(2))
    else:
        raise TranslationError(Sentence("unsupported aspect_ratio type: {{type}}", type=type(aspect_ratio).__name__), 422, param="aspect_ratio")

    if w_val <= 0 or h_val <= 0 or not math.isfinite(w_val) or not math.isfinite(h_val):
        raise TranslationError("aspect_ratio dimensions must be positive finite numbers", 422, param="aspect_ratio")

    wi, hi = round(w_val * 100), round(h_val * 100)
    divisor = math.gcd(wi, hi) or 1
    rw, rh = wi // divisor, hi // divisor
    if rw > 64 or rh > 64:
        # Keep human-readable integer ratio for non-standard pixel sizes.
        wi, hi = max(1, round(w_val)), max(1, round(h_val))
        divisor = math.gcd(wi, hi) or 1
        rw, rh = wi // divisor, hi // divisor
    return f"{rw}:{rh}"


def _normalize_bbox_rows(bbox_draft: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept either normalized [0, 1] user boxes or [0, 1000] canvas boxes."""

    def _is_unit_box(box: Any) -> bool:
        return (
            isinstance(box, (list, tuple))
            and len(box) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)
            and max(float(v) for v in box) <= 1.0
        )

    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(bbox_draft):
        if not isinstance(raw, dict):
            raise TranslationError(Sentence("bbox_draft[{{index}}] must be an object", index=idx), 422, param="bbox_draft")
        candidate_boxes = [raw[k] for k in ("bbox", "src_bbox", "tgt_bbox") if raw.get(k) is not None]
        if not candidate_boxes:
            raise TranslationError(Sentence("bbox_draft[{{index}}] needs 'bbox' or 'src_bbox'/'tgt_bbox'", index=idx), 422, param="bbox_draft")
        if any(_is_unit_box(b) for b in candidate_boxes):
            try:
                rows.extend(flux3_layout.boxes_to_canvas([raw]))
            except (TypeError, ValueError) as err:  # a box that is not a list of four numbers
                raise TranslationError(Sentence("bbox_draft[{{index}}] has an invalid box: {{reason}}", index=idx, reason=err), 422, param="bbox_draft") from err
        else:
            row_id = str(raw.get("id") or f"obj_{idx + 1}")
            desc = str(raw.get("desc") or "")
            if raw.get("bbox") is not None:
                valid = flux3_layout.valid_canvas_box(raw["bbox"])
                if valid is None:
                    raise TranslationError(Sentence("bbox_draft[{{index}}] has invalid canvas box: {{box}}", index=idx, box=repr(raw["bbox"])), 422, param="bbox_draft")
                rows.append({"id": row_id, "bbox": [round(v) for v in valid], "desc": desc})
            else:
                src_v = flux3_layout.valid_canvas_box(raw["src_bbox"]) if raw.get("src_bbox") is not None else None
                tgt_v = flux3_layout.valid_canvas_box(raw["tgt_bbox"]) if raw.get("tgt_bbox") is not None else None
                if (raw.get("src_bbox") is not None and src_v is None) or (raw.get("tgt_bbox") is not None and tgt_v is None):
                    raise TranslationError(Sentence("bbox_draft[{{index}}] has invalid src_bbox/tgt_bbox", index=idx), 422, param="bbox_draft")
                rows.append(
                    {
                        "id": row_id,
                        "from": str(raw.get("from") or ""),
                        "src_bbox": [round(v) for v in src_v] if src_v is not None else None,
                        "tgt_bbox": [round(v) for v in tgt_v] if tgt_v is not None else None,
                        "desc": desc,
                    }
                )
    return rows


def _encode_pil_to_data_uri(img: Image.Image, num_references: int) -> tuple[str, Image.Image]:
    """Resize ``img`` to the FLUX.3/VLM pixel budget and encode as a compact data URI."""

    w, h = img.size
    target_w, target_h = flux3_layout.reference_size(max(1, w), max(1, h), max(1, num_references))
    resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS) if (w, h) != (target_w, target_h) else img.copy()
    has_alpha = resized.mode in ("RGBA", "LA") or (resized.mode == "P" and "transparency" in resized.info)
    buf = io.BytesIO()
    if has_alpha:
        if resized.mode != "RGBA":
            resized = resized.convert("RGBA")
        resized.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        resized.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}", resized


def prepare_reference_images(
    reference_images: Sequence[str | os.PathLike[str] | bytes | Any] | None,
    *,
    max_references: int = flux3_layout.MAX_REFERENCES,
) -> tuple[list[str], list[Image.Image], list[str]]:
    """Normalize reference images into ``(data_uris, pil_images, warnings)``."""

    if not reference_images:
        return [], [], []

    warnings: list[str] = []
    items = list(reference_images)
    if len(items) > max_references:
        warnings.append(Sentence("Trimmed reference images from {{count}} to {{limit}} (model limit).", count=len(items), limit=max_references))
        items = items[:max_references]

    num_refs = len(items)
    data_uris: list[str] = []
    pil_images: list[Image.Image] = []

    def checked(opened: Image.Image, idx: int) -> Image.Image:
        # before load(): the header is read, the pixels are not (X/Y/Z plot turns Pillow's own
        # decompression-bomb limit off for the whole process)
        from remi.files import MAX_UPLOAD_PIXELS

        if opened.width * opened.height > MAX_UPLOAD_PIXELS:
            raise TranslationError(Sentence("reference_images[{{index}}] is too large ({{width}}×{{height}})", index=idx, width=opened.width, height=opened.height), 413)
        opened.load()
        return opened

    for idx, item in enumerate(items):
        if isinstance(item, Image.Image):
            uri, resized = _encode_pil_to_data_uri(item, num_refs)
            data_uris.append(uri)
            pil_images.append(resized)
        elif isinstance(item, (bytes, bytearray)):
            with Image.open(io.BytesIO(bytes(item))) as opened:
                uri, resized = _encode_pil_to_data_uri(checked(opened, idx), num_refs)
            data_uris.append(uri)
            pil_images.append(resized)
        elif isinstance(item, (str, os.PathLike)):
            text = os.fspath(item)
            if text.startswith("data:image/"):
                if ";base64," in text:
                    raw_b64 = text.split(";base64,", 1)[1]
                    try:
                        raw_bytes = base64.b64decode(raw_b64)
                        with Image.open(io.BytesIO(raw_bytes)) as opened:
                            uri, resized = _encode_pil_to_data_uri(checked(opened, idx), num_refs)
                        data_uris.append(uri)
                        pil_images.append(resized)
                    except TranslationError:
                        raise
                    except Exception:
                        data_uris.append(text)
                else:
                    data_uris.append(text)
            elif text.startswith(("http://", "https://")):
                data_uris.append(text)
            elif os.path.isfile(text):
                with Image.open(text) as opened:
                    uri, resized = _encode_pil_to_data_uri(checked(opened, idx), num_refs)
                data_uris.append(uri)
                pil_images.append(resized)
            else:
                raise TranslationError(Sentence("reference_images[{{index}}] file not found: {{path}}", index=idx, path=text), 422, param="reference_images")
        else:
            raise TranslationError(Sentence("unsupported reference image type at index {{index}}: {{type}}", index=idx, type=type(item).__name__), 422, param="reference_images")

    return data_uris, pil_images, warnings


def _build_user_message(
    prompt: str,
    *,
    model_family: str,
    mode: str,
    aspect_ratio: str | None,
    locked_rows: list[dict[str, Any]] | None,
    num_refs: int,
    current: str = "",
) -> str:
    sections: list[str] = []
    if aspect_ratio:
        sections.append(f"TARGET CANVAS: {aspect_ratio}")

    if current.strip():
        body = f"Current prompt: {current.strip()}\n\nUser request: {prompt}"
    else:
        body = prompt

    if locked_rows:
        body = flux3_layout.format_layout_request(body, locked_rows)

    if num_refs > 0:
        if model_family == "qwen21":
            slot_tags = ", ".join(f"<image_{i + 1}>" for i in range(num_refs))
            body = f"Reference images follow in attachment order ({slot_tags}).\nUser editing instruction: {body}\nOutput the editing prompt."
        elif model_family == "flux3":
            scaffold = flux3_layout.I2I_SCAFFOLD[mode == "i2i_bbox"]
            body = scaffold.format(instruction=body)
        else:
            body = f"Reference images follow in order ({num_refs} image(s)).\nUser request: {body}"

    sections.append(body)
    return "\n".join(sections)


# endregion


# region Optional in-process local quantized Qwen3.5 PUP runner


def webui_interrupted() -> bool:
    """Whether Interrupt was pressed: ``shared.state.interrupted``, which the generation queue's
    Interrupt (``POST /api/v2/interrupt``, when no job holds it) and ``/sdapi/v1/interrupt`` set.
    False before the WebUI made its state."""

    try:
        from modules import shared
    except Exception:
        return False
    return bool(getattr(getattr(shared, "state", None), "interrupted", False))


class LocalQwen35PUPRunner:
    """Optional in-process local quantized Qwen3.5 Prompt Upsampler runner.

    Loads either a GGUF checkpoint (via ``llama_cpp`` when installed) or a local
    HuggingFace / Unsloth quantized directory (via ``transformers``), runs one
    generation call in ``torch.inference_mode()``, and unconditionally unloads
    weights and clears VRAM/MPS cache in a ``finally`` block.

    ``should_stop`` is asked between tokens (default :func:`webui_interrupted`):
    transformers through a stopping criterion, llama-cpp by streaming the
    completion and closing the stream. A run it stopped raises
    :class:`GenerationInterrupted` instead of returning what was written so far.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        n_ctx: int = 4096,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.model_path = os.fspath(model_path)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.n_ctx = int(n_ctx)
        self.should_stop = should_stop or webui_interrupted

    @staticmethod
    def release_vram() -> None:
        """Aggressively free GPU/MPS/CPU memory after local PUP execution."""

        gc.collect()
        try:
            from backend import memory_management

            memory_management.soft_empty_cache()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _run_gguf(self, system: str, user_text: str, image_uris: Sequence[str]) -> str:
        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise TranslationError(
                "local GGUF PUP requires 'llama-cpp-python' (or serve the GGUF via llama.cpp server and configure the OpenAI-compatible provider)",
                502,
            ) from err

        llm = None
        try:
            llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_gpu_layers=-1, verbose=False)
            user_content: Any = user_text
            if image_uris:
                user_content = [{"type": "text", "text": user_text}, *[{"type": "image_url", "image_url": {"url": u}} for u in image_uris]]
            # streamed, so that Interrupt is seen between tokens: closing the stream ends
            # llama.cpp's token loop (a plain call returns only once it has written everything)
            stream = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            pieces: list[str] = []
            try:
                for chunk in stream:
                    if self.should_stop():
                        raise GenerationInterrupted()
                    choices = chunk.get("choices") or []
                    content = (choices[0].get("delta") or {}).get("content") if choices else None
                    if content:
                        pieces.append(str(content))
            finally:
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    close_stream()
            if self.should_stop():
                raise GenerationInterrupted()
            return "".join(pieces)
        except TranslationError:
            raise
        except Exception as err:
            raise TranslationError(f"local GGUF PUP execution failed: {err}", 502) from err
        finally:
            if llm is not None:
                close_fn = getattr(llm, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                del llm
            self.release_vram()

    def _run_transformers(self, system: str, user_text: str, pil_images: Sequence[Image.Image]) -> str:
        import torch

        model = None
        processor = None
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

            should_stop = self.should_stop

            class InterruptCriteria(StoppingCriteria):
                """Stops ``generate`` at the next token once Interrupt is pressed."""

                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                    return torch.full((input_ids.shape[0],), bool(should_stop()), device=input_ids.device, dtype=torch.bool)

            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            dtype = torch.bfloat16 if device != "cpu" else torch.float32
            if pil_images:
                try:
                    from transformers import AutoModelForImageTextToText

                    processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
                    model = AutoModelForImageTextToText.from_pretrained(self.model_path, torch_dtype=dtype, local_files_only=True).to(device).eval()
                except Exception:
                    processor = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                    model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype, local_files_only=True).to(device).eval()
            else:
                processor = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype, local_files_only=True).to(device).eval()

            messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
            if hasattr(processor, "apply_chat_template"):
                prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

            if pil_images and hasattr(processor, "image_processor"):
                inputs = processor(text=[prompt_text], images=list(pil_images), return_tensors="pt").to(device)
            else:
                inputs = processor(prompt_text, return_tensors="pt").to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0.0,
                    temperature=max(self.temperature, 1e-5),
                    stopping_criteria=StoppingCriteriaList([InterruptCriteria()]),
                )
            if self.should_stop():
                raise GenerationInterrupted()
            prompt_len = int(inputs["input_ids"].shape[-1])
            generated = output_ids[0, prompt_len:]
            decode = getattr(processor, "decode", None) or processor.tokenizer.decode
            return str(decode(generated, skip_special_tokens=True))
        except TranslationError:
            raise
        except Exception as err:
            raise TranslationError(f"local Qwen3.5 PUP execution failed: {err}", 502) from err
        finally:
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
                del model
            if processor is not None:
                del processor
            self.release_vram()

    def generate(self, system: str, user_text: str, *, image_uris: Sequence[str] = (), pil_images: Sequence[Image.Image] = ()) -> str:
        if not os.path.exists(self.model_path):
            raise TranslationError(Sentence("local_model_path does not exist: {{path}}", path=self.model_path), 422, param="local_model_path")
        if os.path.isfile(self.model_path) and self.model_path.lower().endswith(".gguf"):
            return self._run_gguf(system, user_text, image_uris)
        if os.path.isdir(self.model_path):
            return self._run_transformers(system, user_text, pil_images)
        raise TranslationError(
            Sentence("unsupported local Qwen3.5 PUP checkpoint format: {{path}} (expected .gguf file or HuggingFace model directory)", path=self.model_path),
            422,
            param="local_model_path",
        )


# endregion


def upsample_prompt(
    prompt: str,
    *,
    model_family: str | None = None,
    reference_images: Sequence[str | os.PathLike[str] | bytes | Any] | None = None,
    aspect_ratio: str | tuple[int, int] | None = None,
    bbox_draft: Sequence[dict[str, Any]] | None = None,
    mode: str = "auto",
    local_model_path: str | os.PathLike[str] | None = None,
    config: ProviderConfig | None = None,
    current: str = "",
    negative: bool = False,
    instructions: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    post: Post = _http_post,
    local_runner: Callable[..., str] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> UpsampleResult:
    """Upsample a user prompt using either a multimodal OpenAI-compatible VLM
    endpoint or an optional local quantized Qwen3.5 PUP runner.

    A local model stops between tokens once ``should_stop()`` says so (default
    :func:`webui_interrupted`) and raises :class:`GenerationInterrupted`. It runs
    behind generation, and the caller clears the interrupt flag around it
    (``JobQueue.interruptible``), as a job's start and end do.

    Raises :class:`remi.prompt_tools.translate.TranslationError` with an HTTP
    status code on validation, configuration, or execution errors.
    """

    original = (prompt or "").strip()
    if not original:
        raise TranslationError("describe the image first", 422, param="prompt")
    if len(original) + len(current or "") > _MAX_PUP_CHARS:
        raise TranslationError(Sentence("the prompt is too long for upsampling"), 413)

    family = (model_family or "auto").strip().lower()
    if family in ("", "auto"):
        family = detect_active_model_family()
    if family not in SUPPORTED_PUP_FAMILIES:
        family = "generic"

    req_mode = (mode or "auto").strip().lower()
    if req_mode not in SUPPORTED_PUP_MODES:
        raise TranslationError(Sentence("unsupported upsample mode {{mode}}; expected one of {{modes}}", mode=repr(mode), modes=", ".join(SUPPORTED_PUP_MODES)), 422, param="mode")

    norm_aspect = normalize_aspect_ratio(aspect_ratio)
    locked_rows = _normalize_bbox_rows(bbox_draft) if bbox_draft else None
    image_uris, pil_images, warnings = prepare_reference_images(reference_images)

    if req_mode == "auto":
        if family == "fibo":
            resolved_mode = "vgl_json"
        elif image_uris and locked_rows:
            resolved_mode = "i2i_bbox"
        elif image_uris:
            resolved_mode = "i2i"
        elif locked_rows:
            resolved_mode = "t2i_bbox"
        else:
            resolved_mode = "t2i"
    else:
        resolved_mode = req_mode

    system = instructions.strip() or get_pup_system_prompt(family, resolved_mode, negative=negative)
    user_message = _build_user_message(
        original,
        model_family=family,
        mode=resolved_mode,
        aspect_ratio=norm_aspect,
        locked_rows=locked_rows,
        num_refs=len(image_uris),
        current=current,
    )

    use_local = bool(local_model_path and str(local_model_path).strip())
    preserve_newlines = resolved_mode in ("t2i_bbox", "i2i_bbox", "vgl_json") or family in ("flux3", "qwen21", "moss", "fibo", "zimage", "ernie")

    try:
        if use_local:
            backend: Literal["openai_vlm", "local_qwen35"] = "local_qwen35"
            stop = should_stop or webui_interrupted
            if local_runner is not None:
                try:
                    raw_output = local_runner(
                        local_model_path=os.fspath(local_model_path),
                        system=system,
                        user_message=user_message,
                        image_uris=image_uris,
                        pil_images=pil_images,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        should_stop=stop,
                    )
                finally:
                    LocalQwen35PUPRunner.release_vram()
                if stop():
                    raise GenerationInterrupted()
            else:
                runner = LocalQwen35PUPRunner(local_model_path, max_tokens=max_tokens, temperature=temperature, should_stop=stop)
                raw_output = runner.generate(system, user_message, image_uris=image_uris, pil_images=pil_images)
            upsampled = _clean_prompt(str(raw_output or ""), preserve_newlines=preserve_newlines)
            if not upsampled:
                raise TranslationError("local Qwen3.5 PUP answered with an empty prompt", 502)
        else:
            backend = "openai_vlm"
            if config is None or config.provider != "openai" or not config.enabled:
                raise TranslationError(
                    Sentence("prompt upsampling needs either an OpenAI-compatible VLM endpoint (llama.cpp / Ollama / MLX-LM / vLLM / cloud) or local_model_path"),
                    409,
                )
            # the whole request: its prompt is already held to _MAX_PUP_CHARS above (a cut request
            # would lose the end of the prompt, the layout boxes or the template's closing words)
            upsampled = generate_prompt(
                config,
                user_message,
                current="",
                negative=negative,
                instructions=system,
                images=image_uris or None,
                temperature=temperature,
                max_tokens=max_tokens,
                preserve_newlines=preserve_newlines,
                max_chars=None,
                post=post,
            )
    finally:
        for pil_img in pil_images:
            try:
                pil_img.close()
            except Exception:
                pass

    layout_rows: list[dict[str, Any]] | None = None
    vgl_json: dict[str, Any] | None = None

    if resolved_mode in ("t2i_bbox", "i2i_bbox"):
        if locked_rows:
            upsampled = flux3_layout.reconcile_boxes(upsampled, locked_rows)
        _, parsed_rows = flux3_layout.split_caption(upsampled)
        if parsed_rows is not None:
            layout_rows = parsed_rows
            for problem in flux3_layout.layout_problems(upsampled):
                warnings.append(Sentence("Layout warning: {{problem}}", problem=problem))
    elif resolved_mode == "vgl_json" or family == "fibo":
        try:
            from backend.nn.fibo import parse_vgl_prompt, validate_vgl_prompt

            parsed_vgl = parse_vgl_prompt(upsampled)
            if parsed_vgl is not None:
                vgl_json = parsed_vgl
                upsampled = json.dumps(parsed_vgl, ensure_ascii=False, indent=2)
                for issue in validate_vgl_prompt(parsed_vgl):
                    warnings.append(Sentence("VGL warning: {{issue}}", issue=issue))
            elif resolved_mode == "vgl_json":
                warnings.append(Sentence("VGL JSON mode expected a JSON object; returning raw text output."))
        except Exception:
            pass

    return UpsampleResult(
        prompt=upsampled,
        original_prompt=original,
        model_family=family,
        mode=resolved_mode,
        backend=backend,
        used_reference_count=len(image_uris),
        aspect_ratio=norm_aspect,
        layout_rows=layout_rows,
        vgl_json=vgl_json,
        warnings=tuple(warnings),
    )
