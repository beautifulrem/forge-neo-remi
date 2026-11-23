"""Bria Fibo 2.0 (8.1B Dev & 4-step Edit Turbo) DiT, VGL JSON protocol & Diffusers support.

License metadata:
- Bria Fibo 2.0 weights are distributed under **CC BY-NC 4.0** (non-commercial
  use unless licensed directly from Bria). License metadata is explicitly
  surfaced on the model, capability, and workflow advice.

Architecture & shared building blocks:
- Reuses Forge's existing ``Qwen3-VL-4B`` text encoder and ``FLUX.2`` 128-ch VAE
  stack (identical latent packing & ``EmbedND`` 3D RoPE);
- Single-stream AdaLN-Zero DiT (``FiboTransformer2DModel``) with QK-RMSNorm,
  SwiGLU FFN, optional guidance embedding (Dev vs 4-step Edit Turbo), multi-ref
  conditioning, lossless cross-step text/RoPE caching, per-block
  ``torch.compile`` (`transformer_blocks`), and Spectrum pre-head seam
  (`final_input` + `compute_final`);
- First-class **VGL 2.0 JSON** structured prompt parser, validator, builder,
  and deterministic encoder formatter.
"""

from __future__ import annotations

import json
import os
import re
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.args import dynamic_args
from backend.attention import attention_function
from backend.nn.anima_diffusers import (
    _DIFFUSERS_INDEX_FILES,
    _SHARD_INDEX_FILES,
    _load_component_sd,
    _read_json,
)
from backend.nn.flux import EmbedND, timestep_embedding
from backend.operations import main_stream_worker, weights_manual_cast
from backend.quant_ops import ck
from backend.utils import fp16_fix

FIBO_LICENSE = "CC-BY-NC-4.0"
FIBO_COMMERCIAL_USE = False
FIBO_LICENSE_NOTICE = (
    "Bria Fibo 2.0 weights are licensed under CC BY-NC 4.0 "
    "(non-commercial use only without a commercial license from Bria)."
)
FIBO_MAX_REFERENCES = 4

DIFFUSERS_CLASS = "FiboTransformer2DModel"
DIFFUSERS_PIPELINE_CLASSES = ("FiboPipeline", "Fibo2Pipeline", "FiboEditPipeline", "BriaFiboPipeline")

FIBO_SENSITIVE_PREFIXES: tuple[str, ...] = (
    "img_in.",
    "txt_in.",
    "time_in.",
    "guidance_in.",
    "final_layer.",
)

FIBO_INT8_LINEAR_SUFFIXES: tuple[str, ...] = (
    ".attn.to_q.weight",
    ".attn.to_k.weight",
    ".attn.to_v.weight",
    ".attn.to_out.0.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)

# region VGL 2.0 JSON structured prompt protocol

VGL_SCHEMA_VERSION = "2.0"

VGL_REQUIRED_FIELDS: tuple[str, ...] = (
    "short_description",
    "objects",
    "background_setting",
    "lighting",
    "aesthetics",
    "photographic_characteristics",
    "style_medium",
)

VGL_OPTIONAL_FIELDS: tuple[str, ...] = (
    "context",
    "artistic_style",
    "text_render",
    "edit_instruction",
    "reference_binding",
    "negative_elements",
)

VGL_OBJECT_FIELDS: tuple[str, ...] = (
    "description",
    "location",
    "relationship",
    "relative_size",
    "shape_and_color",
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_vgl_prompt(prompt: str) -> dict[str, Any] | None:
    """Parse a VGL 2.0 JSON structured prompt, or return ``None`` for plain text."""

    if not isinstance(prompt, str):
        return None
    stripped = _CODE_FENCE_RE.sub("", prompt.strip()).strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_vgl_prompt(vgl: Mapping[str, Any]) -> list[str]:
    """Validate a VGL 2.0 structured prompt dict; returns a list of issues (empty = valid)."""

    if not isinstance(vgl, Mapping):
        return ["VGL prompt must be a JSON object."]

    issues: list[str] = []
    for field in VGL_REQUIRED_FIELDS:
        if field not in vgl:
            issues.append(f"Missing required VGL field '{field}'.")

    if "short_description" in vgl and (not isinstance(vgl["short_description"], str) or not vgl["short_description"].strip()):
        issues.append("'short_description' must be a non-empty string.")

    if "style_medium" in vgl and (not isinstance(vgl["style_medium"], str) or not vgl["style_medium"].strip()):
        issues.append("'style_medium' must be a non-empty string.")

    if "objects" in vgl:
        objs = vgl["objects"]
        if not isinstance(objs, list):
            issues.append("'objects' must be a JSON array.")
        else:
            for idx, item in enumerate(objs):
                if not isinstance(item, Mapping):
                    issues.append(f"objects[{idx}] must be a JSON object.")
                    continue
                if not isinstance(item.get("description"), str) or not str(item.get("description")).strip():
                    issues.append(f"objects[{idx}] is missing a non-empty 'description'.")

    for sub_obj in ("lighting", "aesthetics", "photographic_characteristics"):
        if sub_obj in vgl and not isinstance(vgl[sub_obj], Mapping):
            issues.append(f"'{sub_obj}' must be a JSON object.")

    return issues


def build_vgl_prompt(
    short_description: str,
    *,
    objects: list[dict[str, Any]] | None = None,
    background_setting: str = "natural cohesive environment",
    lighting: dict[str, str] | None = None,
    aesthetics: dict[str, str] | None = None,
    photographic_characteristics: dict[str, str] | None = None,
    style_medium: str = "photograph",
    edit_instruction: str | None = None,
    reference_binding: dict[str, str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a canonical, valid VGL 2.0 JSON dict for Bria Fibo 2.0."""

    doc: dict[str, Any] = {
        "short_description": short_description.strip(),
        "objects": list(objects) if objects is not None else [],
        "background_setting": background_setting,
        "lighting": dict(
            lighting
            or {
                "conditions": "natural daylight",
                "direction": "soft front-side key",
                "shadows": "gentle grounded shadows",
            }
        ),
        "aesthetics": dict(
            aesthetics
            or {
                "composition": "balanced rule-of-thirds framing",
                "color_scheme": "natural rich tones",
                "mood_atmosphere": "clear and lifelike",
            }
        ),
        "photographic_characteristics": dict(
            photographic_characteristics
            or {
                "depth_of_field": "moderate",
                "focus": "sharp on primary subject",
                "camera_angle": "eye-level",
                "lens_focal_length": "50mm",
            }
        ),
        "style_medium": style_medium,
    }
    if edit_instruction:
        doc["edit_instruction"] = edit_instruction
    if reference_binding:
        doc["reference_binding"] = dict(reference_binding)
    for k, v in extra.items():
        if v is not None:
            doc[k] = v
    return doc


def format_vgl_for_encoder(prompt: str) -> str:
    """Canonicalize VGL JSON into deterministic compact JSON for Qwen3-VL-4B,
    or return plain prose prompts untouched."""

    parsed = parse_vgl_prompt(prompt)
    if parsed is None:
        return prompt
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# endregion


# region Fibo 2.0 DiT architecture

_TEXT_CACHE_MAX = 4
_ROPE_CACHE_MAX = 4


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else int(getattr(tensor, "_version", 0))


class FiboAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

    def forward(self, x: torch.Tensor, pe: torch.Tensor, transformer_options: dict = {}) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.num_heads, self.head_dim)
        k = self.to_k(x).view(b, n, self.num_heads, self.head_dim)
        v = self.to_v(x)
        q_scale, _, q_stream = weights_manual_cast(self.norm_q, q)
        k_scale, _, k_stream = weights_manual_cast(self.norm_k, k)
        with main_stream_worker(q_scale, None, q_stream), main_stream_worker(k_scale, None, k_stream):
            q, k = ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)
        out = attention_function(q.flatten(2), k.flatten(2), v, self.num_heads, transformer_options=transformer_options)
        return self.to_out[0](out)


class FiboFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FiboTransformerBlock(nn.Module):
    """Single-stream AdaLN-Zero transformer block for Bria Fibo 2.0."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = FiboAttention(dim, num_heads, eps=eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.mlp = FiboFeedForward(dim, int(dim * mlp_ratio))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x: torch.Tensor, temb: torch.Tensor, pe: torch.Tensor, transformer_options: dict = {}) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(temb).unsqueeze(1).chunk(6, dim=-1)
        x = x + gate1.tanh() * self.attn(self.norm1(x) * (1.0 + scale1) + shift1, pe, transformer_options=transformer_options)
        x = x + gate2.tanh() * self.mlp(self.norm2(x) * (1.0 + scale2) + shift2)
        return fp16_fix(x)


class FiboFinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.linear = nn.Linear(dim, out_channels, bias=True)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(temb).unsqueeze(1).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1.0 + scale) + shift)


class FiboTransformer2DModel(nn.Module):
    """Bria Fibo 2.0 (8.1B Dev & 4-step Edit Turbo) DiT backbone.

    Uses 128-ch FLUX.2 latents (patch_size=1 in packed space), Qwen3-VL-4B
    conditioning, 3D RoPE, multi-reference token concatenation, lossless
    cross-step text/RoPE caching, per-block ``torch.compile`` support, and a
    zero-param Spectrum pre-head capture seam (``final_input``).
    """

    license = FIBO_LICENSE
    commercial_use = FIBO_COMMERCIAL_USE

    def __init__(
        self,
        *,
        in_channels: int = 128,
        out_channels: int = 128,
        hidden_size: int = 3840,
        num_attention_heads: int = 30,
        num_layers: int = 32,
        text_in_dim: int = 2560,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        rope_theta: float = 10000.0,
        guidance_embeds: bool = True,
        is_edit_turbo: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.guidance_embeds = bool(guidance_embeds and not is_edit_turbo)
        self.is_edit_turbo = bool(is_edit_turbo)

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Sequential(
            nn.RMSNorm(text_in_dim, eps=eps),
            nn.Linear(text_in_dim, self.hidden_size, bias=True),
        )
        self.time_in = nn.Sequential(
            nn.Linear(256, self.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )
        self.guidance_in = (
            nn.Sequential(
                nn.Linear(256, self.hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            )
            if self.guidance_embeds
            else None
        )
        self.pe_embedder = EmbedND(dim=self.head_dim, theta=rope_theta, axes_dim=list(axes_dims_rope))
        self.transformer_blocks = nn.ModuleList(
            [FiboTransformerBlock(self.hidden_size, self.num_attention_heads, mlp_ratio=mlp_ratio, eps=eps) for _ in range(num_layers)]
        )
        self.fibo_final_input = nn.Identity()  # Spectrum pre-head capture seam (0 params)
        self.final_layer = FiboFinalLayer(self.hidden_size, self.out_channels, eps=eps)

        self._text_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._rope_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    def clear_caches(self) -> None:
        self._text_cache.clear()
        self._rope_cache.clear()

    def _apply(self, fn):
        self.clear_caches()
        return super()._apply(fn)

    def _embed_time(self, timestep: torch.Tensor, dtype: torch.dtype, guidance: torch.Tensor | None = None) -> torch.Tensor:
        t_vec = self.time_in(timestep_embedding(timestep.float(), 256).to(dtype))
        if self.guidance_in is not None and guidance is not None:
            t_vec = t_vec + self.guidance_in(timestep_embedding(guidance.float(), 256).to(dtype))
        return t_vec

    def _cached_rope(self, h: int, w: int, text_len: int, ref_shapes: tuple[tuple[int, int], ...], device: torch.device) -> torch.Tensor:
        key = (h, w, text_len, ref_shapes, device)
        cached = self._rope_cache.get(key)
        if cached is not None:
            self._rope_cache.move_to_end(key)
            return cached

        parts: list[torch.Tensor] = []
        txt_ids = torch.zeros((text_len, 3), device=device, dtype=torch.float32)
        txt_ids[:, 0] = torch.arange(text_len, device=device, dtype=torch.float32)
        parts.append(txt_ids)

        hh = torch.arange(h, device=device, dtype=torch.float32)
        ww = torch.arange(w, device=device, dtype=torch.float32)
        img_ids = torch.stack(
            [torch.full((h, w), float(text_len), device=device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)],
            dim=-1,
        ).flatten(0, 1)
        parts.append(img_ids)

        for idx, (rh, rw) in enumerate(ref_shapes):
            rhh = torch.arange(rh, device=device, dtype=torch.float32)
            rww = torch.arange(rw, device=device, dtype=torch.float32)
            ref_ids = torch.stack(
                [torch.full((rh, rw), float(text_len + idx + 1), device=device, dtype=torch.float32), rhh[:, None].expand(rh, rw), rww[None, :].expand(rh, rw)],
                dim=-1,
            ).flatten(0, 1)
            parts.append(ref_ids)

        pe = self.pe_embedder(torch.cat(parts, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        self._rope_cache[key] = pe
        while len(self._rope_cache) > _ROPE_CACHE_MAX:
            self._rope_cache.popitem(last=False)
        return pe

    def _embed_text(self, context: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() or any(getattr(m, "weight_function", None) for m in self.txt_in.modules()):
            return self.txt_in(context)
        params_sig = tuple((id(p), p.data_ptr(), _tensor_version(p), p.device, p.dtype) for p in self.txt_in.parameters())
        key = (id(context), context.data_ptr(), _tensor_version(context), tuple(context.shape), context.dtype, context.device, params_sig)
        entry = self._text_cache.get(key)
        if entry is not None:
            wref, cached = entry
            if wref() is context:
                self._text_cache.move_to_end(key)
                return cached
            self._text_cache.pop(key, None)
        out = self.txt_in(context)
        try:
            wref = weakref.ref(context)
        except TypeError:
            return out
        self._text_cache[key] = (wref, out)
        while len(self._text_cache) > _TEXT_CACHE_MAX:
            self._text_cache.popitem(last=False)
        return out

    def compute_fibo_final(
        self,
        feature: torch.Tensor,
        timestep: torch.Tensor,
        input_shape: tuple[int, ...],
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Spectrum pre-head fast-forward from captured target image tokens."""

        b = feature.shape[0]
        h, w = int(input_shape[-2]), int(input_shape[-1])
        t = timestep.reshape(-1)
        if t.shape[0] == 1 and b > 1:
            t = t.expand(b)
        temb = self._embed_time(t, feature.dtype, guidance=guidance)
        out = self.final_layer(feature, temb)
        return out.transpose(1, 2).reshape(b, self.out_channels, h, w)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        guidance: torch.Tensor | None = None,
        ref_latents: list[torch.Tensor] | None = None,
        transformer_options: dict = {},
        **kwargs,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        if ref_latents is None:
            ref_latents = list(dynamic_args.ref_latents or [])
        ref_latents = ref_latents[:FIBO_MAX_REFERENCES]

        txt = self._embed_text(context.to(device=device, dtype=dtype))
        if txt.shape[0] != b:
            txt = txt[:1].expand(b, -1, -1) if txt.shape[0] == 1 else txt.repeat(b // txt.shape[0], 1, 1)
        text_len = txt.shape[1]

        img = self.img_in(x.flatten(2).transpose(1, 2))
        parts = [txt, img]
        ref_shapes: list[tuple[int, int]] = []
        for ref in ref_latents:
            r = ref.to(device=device, dtype=dtype)
            if r.shape[0] != b:
                r = r[:1].expand(b, -1, -1, -1) if r.shape[0] == 1 else r.repeat(b // r.shape[0], 1, 1, 1)
            ref_shapes.append((int(r.shape[-2]), int(r.shape[-1])))
            parts.append(self.img_in(r.flatten(2).transpose(1, 2)))

        hidden = torch.cat(parts, dim=1)
        pe = self._cached_rope(h, w, text_len, tuple(ref_shapes), device)
        temb = self._embed_time(timesteps, dtype, guidance=guidance)

        opts = dict(transformer_options) if isinstance(transformer_options, dict) else {}
        blocks_replace = opts.get("patches_replace", {}).get("dit", {})
        opts["total_blocks"] = len(self.transformer_blocks)
        opts["block_type"] = "single"

        for i, block in enumerate(self.transformer_blocks):
            opts["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["vec"], args["pe"], args["transformer_options"])}

                hidden = blocks_replace[("single_block", i)]({"img": hidden, "vec": temb, "pe": pe, "transformer_options": opts}, {"original_block": block_wrap})["img"]
            else:
                hidden = block(hidden, temb, pe, opts)

        target_tokens = self.fibo_final_input(hidden[:, text_len : text_len + h * w])
        out = self.final_layer(target_tokens, temb)
        return out.transpose(1, 2).reshape(b, self.out_channels, h, w)


# endregion


# region Detection, Diffusers dir loading, LoRA, Spectrum & workflow advice


def _dir_primary(directory: str) -> str | None:
    from backend.nn.moss_diffusers import _dir_primary_safetensors

    return _dir_primary_safetensors(directory)


def _strip_prefix(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_fibo_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    keys = {_strip_prefix(k) for k in state_dict}
    return (
        "img_in.weight" in keys
        and "txt_in.1.weight" in keys
        and "time_in.0.weight" in keys
        and "transformer_blocks.0.adaln.1.weight" in keys
        and "final_layer.adaln.1.weight" in keys
    )


def fibo_diffusers_primary_file(directory: os.PathLike | str) -> str | None:
    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    if os.path.isdir(transformer_dir) and (primary := _dir_primary(transformer_dir)):
        return primary
    return _dir_primary(root)


def is_fibo_diffusers_dir(path: os.PathLike | str) -> bool:
    root = os.fspath(path)
    transformer_dir = os.path.join(root, "transformer")
    if not os.path.isdir(root) or not os.path.isdir(transformer_dir) or _dir_primary(transformer_dir) is None:
        return False
    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if not index:
            continue
        cls_name = str(index.get("_class_name", ""))
        entry = index.get("transformer")
        if any(p in cls_name for p in DIFFUSERS_PIPELINE_CLASSES) or "Fibo" in cls_name:
            return True
        if isinstance(entry, list) and DIFFUSERS_CLASS in entry:
            return True
    cfg = _read_json(os.path.join(transformer_dir, "config.json"))
    return bool(cfg and cfg.get("_class_name") == DIFFUSERS_CLASS)


def is_sharded_fibo_transformer_dir(path: os.PathLike | str) -> bool:
    root = os.fspath(path)
    if not os.path.isdir(root) or os.path.isdir(os.path.join(root, "transformer")) or _dir_primary(root) is None:
        return False
    cfg = _read_json(os.path.join(root, "config.json"))
    if cfg and cfg.get("_class_name") == DIFFUSERS_CLASS:
        return True
    for name in _SHARD_INDEX_FILES:
        index = _read_json(os.path.join(root, name))
        weight_map = index.get("weight_map") if index else None
        if isinstance(weight_map, dict) and "transformer_blocks.0.adaln.1.weight" in weight_map and "final_layer.adaln.1.weight" in weight_map:
            return True
    return False


def is_fibo_checkpoint_dir(path: os.PathLike | str) -> bool:
    return is_fibo_diffusers_dir(path) or is_sharded_fibo_transformer_dir(path)


def load_fibo_diffusers_dir(directory: os.PathLike | str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a Bria Fibo 2.0 Diffusers folder (reusing Qwen3-VL-4B + FLUX.2 VAE)
    with explicit CC BY-NC 4.0 license metadata."""

    from backend.nn.krea_diffusers import normalize_qwen3vl_key

    root = os.fspath(directory)
    transformer_dir = os.path.join(root, "transformer")
    is_pipeline = os.path.isdir(transformer_dir)
    if not is_pipeline:
        transformer_dir = root

    raw, metadata = _load_component_sd(transformer_dir)
    if not raw:
        raise FileNotFoundError(f"Bria Fibo 2.0 Diffusers directory has no transformer weights: {transformer_dir}")

    sd: dict[str, torch.Tensor] = {
        (k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"): v for k, v in raw.items()
    }
    del raw

    metadata = dict(metadata or {})
    metadata.setdefault("license", FIBO_LICENSE)
    metadata.setdefault("license_notice", FIBO_LICENSE_NOTICE)
    metadata.setdefault("commercial_use", "false")

    if not is_pipeline:
        return sd, metadata

    for index_name in _DIFFUSERS_INDEX_FILES:
        index = _read_json(os.path.join(root, index_name))
        if index and ("Turbo" in str(index.get("_class_name", "")) or index.get("is_edit_turbo")):
            metadata["fibo_variant"] = "edit_turbo"
            break

    te_dir = os.path.join(root, "text_encoder")
    if os.path.isdir(te_dir):
        te_sd, _ = _load_component_sd(te_dir)
        for key, value in te_sd.items():
            sd[f"text_encoders.qwen3vl_4b.transformer.{normalize_qwen3vl_key(key)}"] = value
        del te_sd

    vae_dir = os.path.join(root, "vae")
    if os.path.isdir(vae_dir):
        vae_sd, _ = _load_component_sd(vae_dir)
        if "decoder.up_blocks.0.resnets.0.norm1.weight" in vae_sd:
            from modules_forge.packages.huggingface_guess.diffusers_convert import convert_vae_state_dict

            vae_sd = convert_vae_state_dict(vae_sd)
        for key, value in vae_sd.items():
            if key.startswith("decoder.post_quant_conv."):
                key = "post_quant_conv." + key[len("decoder.post_quant_conv.") :]
            elif key.startswith("encoder.quant_conv."):
                key = "quant_conv." + key[len("encoder.quant_conv.") :]
            sd[f"vae.{key}"] = value
        del vae_sd

    return sd, metadata


def fibo_lora_key_map(sd: Mapping[str, torch.Tensor]) -> dict[str, object]:
    key_map: dict[str, object] = {}

    def register(bare: str, target: str) -> None:
        flat = bare.replace(".", "_")
        for prefix in ("", "diffusion_model.", "transformer.", "base_model.model.", "base_model.model.diffusion_model.", "base_model.model.transformer."):
            key_map[f"{prefix}{bare}"] = target
        key_map[f"lora_unet_{flat}"] = target
        key_map[f"lycoris_{flat}"] = target
        key_map[f"lora_transformer_{flat}"] = target

    for k in sd:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            register(k[len("diffusion_model.") : -len(".weight")], k)
    return key_map


def fast_forward_fibo(dit, predictor, sigma, input_x, feature):
    """Spectrum pre-head fast-forward for ``FiboTransformer2DModel``."""

    internal_timestep = predictor.timestep(sigma).float()
    model_output = dit.compute_fibo_final(feature, internal_timestep, tuple(input_x.shape))
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


_FIBO_TURBO_RE = re.compile(r"(?:^|[-_ /])(?:turbo|4[-_]?steps?|edit[-_ ]?turbo)(?:$|[-_ /.])", re.IGNORECASE)


def fibo_workflow_advice(p, *, opts=None) -> list[str]:
    """Non-blocking workflow advice for Bria Fibo 2.0 (4-step Edit Turbo vs Dev,
    VGL JSON validation, reference limits, and CC BY-NC 4.0 license notice)."""

    sd_model = getattr(p, "sd_model", None)
    if sd_model is None or not (getattr(sd_model, "is_fibo_engine", False) or type(sd_model).__name__ == "Fibo2"):
        return []

    from backend.nn.dit_registry import _prompt_and_checkpoint_names

    ckpt_name, lora_names = _prompt_and_checkpoint_names(p)
    combined = " ".join([ckpt_name, *lora_names]).strip()
    is_turbo = bool(getattr(sd_model, "is_fibo_turbo", False)) or bool(combined and _FIBO_TURBO_RE.search(combined))

    advice: list[str] = []
    steps = getattr(p, "steps", None)
    cfg = getattr(p, "cfg_scale", None)
    if is_turbo:
        if (isinstance(steps, (int, float)) and steps > 8) or (isinstance(cfg, (int, float)) and cfg > 1.5):
            advice.append(
                f"Bria Fibo 2.0 Edit Turbo detected with steps={steps}, cfg={cfg}: the 4-step Edit Turbo "
                "variant works best at 4 steps with CFG 1.0."
            )
    elif isinstance(steps, (int, float)) and steps <= 8 and isinstance(cfg, (int, float)) and cfg <= 1.2:
        advice.append(
            f"Bria Fibo 2.0 Dev detected with steps={steps}, cfg={cfg}: the undistilled 8.1B Dev checkpoint "
            "typically needs ~28-50 steps and CFG 3.5-5.0."
        )

    prompt = getattr(p, "prompt", None)
    if isinstance(prompt, str) and prompt.strip().startswith("{"):
        parsed = parse_vgl_prompt(prompt)
        if parsed is None:
            advice.append("Bria Fibo 2.0: prompt begins with '{' but is not valid JSON; fix syntax or use the VGL JSON Builder.")
        else:
            for issue in validate_vgl_prompt(parsed):
                advice.append(f"Bria Fibo 2.0 VGL: {issue}")

    refs = [*(getattr(sd_model, "ref_latents", None) or ())]
    if getattr(sd_model, "ini_latent", None) is not None:
        refs.append(sd_model.ini_latent)
    if len(refs) > FIBO_MAX_REFERENCES:
        advice.append(f"Bria Fibo 2.0 supports at most {FIBO_MAX_REFERENCES} reference images; {len(refs)} are attached.")

    return advice


# endregion
