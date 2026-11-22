"""Qwen3.5 4B, the second text encoder of Anima-3.8B (lylogummy/Anima-3.8B).

Supports:
- the community release ``qwen35_4b.safetensors`` (32 hybrid layers without
  layer 31's MLP, plus its unused 2560->1024 projection head);
- original HuggingFace ``Qwen/Qwen3.5-4B`` safetensors (single-file or sharded
  via ``model.safetensors.index.json``), stripping the ``model.language_model.``
  prefix and ignoring the unused vision tower, MTP layers, ``lm_head``, final
  RMSNorm, and layer 31's MLP;
- GGUF-quantized ``Qwen3.5-4B`` (remapped by ``backend.loader_gguf``).

Anima-3.8B's semantic connector reads the hidden states after layers 7, 15, 23
and 31. Built from the Qwen3.5 layers MOSS-Image already uses
(``backend.nn.llm.llama``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from backend.nn.llm.llama import MLP, GatedAttention, GatedDeltaNet, Qwen35_9BConfig, attention_function, precompute_freqs_cis

SEMANTIC_LAYERS = (7, 15, 23, 31)
QWEN35_4B_HIDDEN_SIZE = 2560
_STRIP_PREFIXES = ("model.language_model.", "language_model.", "model.")
_IGNORE_PREFIXES = ("visual.", "model.visual.", "mtp.", "nextn.", "lm_head.", "layers.31.mlp.", "layers.31.post_attention_layernorm.")


def _strip_qwen35_prefix(key: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def qwen35_4b_needed_key(key: str) -> bool:
    """Whether a raw safetensors key from ``Qwen/Qwen3.5-4B`` or ``qwen35_4b.safetensors``
    belongs to Anima-3.8B's text backbone (used to skip vision/MTP shards on load)."""

    if key.startswith(("model.visual.", "visual.", "mtp.", "nextn.", "lm_head.")):
        return False
    stripped = _strip_qwen35_prefix(key)
    if stripped.startswith(_IGNORE_PREFIXES) or stripped == "norm.weight":
        return False
    return stripped == "embed_tokens.weight" or stripped.startswith(("layers.", "norm."))


def is_qwen35_4b_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """True when ``state_dict`` holds Qwen3.5 4B (hidden_size 2560) in release,
    HuggingFace, or GGUF-remapped layout."""

    for prefix in ("", "model.language_model.", "language_model.", "model."):
        if f"{prefix}layers.0.linear_attn.A_log" in state_dict and (norm := state_dict.get(f"{prefix}layers.0.input_layernorm.weight")) is not None:
            if norm.shape[0] == QWEN35_4B_HIDDEN_SIZE:
                return True
    return False


def normalize_qwen35_4b_state_dict(state_dict: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    """Normalize release / HF / GGUF-remapped Qwen3.5 4B weights to ``AnimaQwen35`` keys.

    Returns ``(normalized_state_dict, has_release_head)``.
    """

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        if not qwen35_4b_needed_key(raw_key):
            continue
        key = _strip_qwen35_prefix(raw_key)
        if key.startswith(_IGNORE_PREFIXES) or key == "norm.weight":
            continue
        normalized[key] = value
    has_release_head = "norm.0.weight" in normalized
    return normalized, has_release_head


@dataclass
class Qwen35_4BConfig(Qwen35_9BConfig):
    hidden_size: int = QWEN35_4B_HIDDEN_SIZE
    intermediate_size: int = 9216
    final_norm: bool = False
    transformer_type: str = "qwen35_4b"
    no_mlp_layers: tuple = (31,)


HEAD_WIDTH = 1024  # the projection head the release ships in place of the final norm


class Qwen35Layer(nn.Module):
    def __init__(self, config: Qwen35_4BConfig, index: int):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        if index in config.no_mlp_layers:
            self.mlp = None
        else:
            self.mlp = MLP(config)
            self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)

    def forward(self, x, attention_mask=None, freqs_cis=None):
        if self.layer_type == "linear_attention":
            h, _ = self.linear_attn(self.input_layernorm(x))
        else:
            h, _ = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, freqs_cis=freqs_cis, optimized_attention=attention_function)
        x = x + h
        if self.mlp is not None:
            x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class AnimaQwen35(nn.Module):
    """Qwen3.5 4B backbone: ``forward`` returns the hidden states after the given layers."""

    def __init__(self, config_dict: dict | None = None, *, include_head: bool = True):
        super().__init__()
        config = Qwen35_4BConfig()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen35Layer(config, index) for index in range(config.num_hidden_layers)])
        if include_head:
            width = HEAD_WIDTH
            self.norm = nn.Sequential(
                nn.Linear(config.hidden_size, width, bias=True),
                nn.RMSNorm(width, eps=config.rms_norm_eps, add=True),
                nn.SiLU(),
                nn.Linear(width, width, bias=True),
            )
        else:
            self.norm = None

    def forward(self, input_ids: torch.Tensor, layers: Sequence[int] = SEMANTIC_LAYERS, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
        layers = tuple(int(layer) for layer in layers)
        if not layers or min(layers) < 0 or max(layers) >= self.num_layers:
            raise ValueError(f"Qwen3.5 layers must be in 0-{self.num_layers - 1}, got {layers}")
        x = self.embed_tokens(input_ids)
        if dtype is not None:
            x = x.to(dtype)
        length = x.shape[1]
        positions = torch.arange(length, device=x.device).unsqueeze(0)
        rotary = int(self.config.head_dim * self.config.partial_rotary_factor)
        freqs_cis = precompute_freqs_cis(rotary, positions, self.config.rope_theta, self.config.rope_scale, self.config.mrope_section, interleaved_mrope=True, device=x.device)
        mask = None
        if length > 1:
            mask = torch.empty(length, length, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)

        wanted = set(layers)
        states: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.layers[: max(layers) + 1]):
            x = layer(x, attention_mask=mask, freqs_cis=freqs_cis)
            if index in wanted:
                states[index] = x.clone()
        return [states[layer] for layer in layers]
