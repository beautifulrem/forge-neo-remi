"""Anima-3.8B's Semantic Connector v2: Qwen3.5 4B semantics in Anima's LLM adapter.

Anima-3.8B (lylogummy/Anima-3.8B v1.1) ships, beside its 52 DiT blocks, a connector
(``net.anima_v2_connector.*``) that wraps the 6 blocks of Anima's own LLM adapter (the one
that turns Qwen3 0.6B states and T5 ids into the DiT's 512-token context). After each
adapter block it adds two residuals:

- the quality anchor: cross-attention to a per-block softmax mix of Qwen3.5 4B layers
  7/15/23/31;
- the v2 attention: cross-attention to a 64-token semantic bank that a resampler distils
  from those layers, modulated by the diffusion timestep.

The bank depends on the timestep, so the DiT's context is recomputed at every step
(``Anima.forward``). With ``include_inserted_blocks=False`` it is the plain adapter.

Ported from https://github.com/GumGum10/forge-anima-3.8B (``anima3b/semantic_v2.py``,
``anima3b/adapter.py``), used under the MIT License:

Copyright (c) 2026 GumGum10 contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.nn.anima import Attention, apply_rotary_pos_emb
from backend.operations import scaled_dot_product_attention

CONNECTOR_PREFIX = "anima_v2_connector."
SEMANTIC_DIM = 2560
SEMANTIC_LAYERS = (7, 15, 23, 31)
_STATIC_CACHE_MAX_ENTRIES = 2
_STATIC_CACHE_MAX_BYTES = 128 * 1024 * 1024


def _tensor_signature(tensor: torch.Tensor | None) -> tuple | None:
    if tensor is None:
        return None
    version = None if tensor.is_inference() else tensor._version
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        version,
    )


def _tensor_owner_ref(tensor: torch.Tensor) -> weakref.ref[torch.Tensor]:
    owner = tensor._base if tensor._base is not None else tensor
    return weakref.ref(owner)


def _has_dynamic_hooks(module: nn.Module) -> bool:
    return bool(
        getattr(module, "weight_function", None)
        or getattr(module, "bias_function", None)
        or getattr(module, "_forward_hooks", None)
        or getattr(module, "_forward_pre_hooks", None)
    )


def _param_signature(param: torch.Tensor | None) -> tuple | None:
    if param is None:
        return None
    version = None if param.is_inference() else param._version
    return (param.data_ptr(), version, param.dtype, param.device)


@dataclass
class _StaticConnectorCache:
    owner_refs: tuple[weakref.ref[torch.Tensor], ...]
    adapter_ref: weakref.ref[nn.Module]
    embedded_in: torch.Tensor
    source_cast: torch.Tensor
    anchor_mask: torch.Tensor | None
    resampler_mask: torch.Tensor | None
    query_rope: tuple[torch.Tensor, torch.Tensor]
    source_rope: tuple[torch.Tensor, torch.Tensor]
    bank_rope: tuple[torch.Tensor, torch.Tensor]
    resampler_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    anchor_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def alive(self, adapter: nn.Module) -> bool:
        return self.adapter_ref() is adapter and all(ref() is not None for ref in self.owner_refs)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal embedding of flow timesteps in [0, 1] (scaled by 1000, as trained)."""

    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1))
    angles = timesteps.float().reshape(-1, 1) * 1_000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(embedding, (0, dim % 2)) if dim % 2 else embedding


def _attention_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(torch.bool)
    return mask[:, None, None, :] if mask.ndim == 2 else mask


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False)

    def project_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = context.shape
        k = self.k_proj(context).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward_with_kv(self, query: torch.Tensor, k_heads: torch.Tensor, v_heads: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, width = query.shape
        q_heads = self.q_proj(query).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, attn_mask=_attention_mask(mask))
        return self.o_proj(attended.transpose(1, 2).reshape(batch, tokens, width))

    def forward(self, query: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        k_heads, v_heads = self.project_kv(context)
        return self.forward_with_kv(query, k_heads, v_heads, mask)


class ModulatedNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, value: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self.norm(value) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ResamplerBlock(nn.Module):
    def __init__(self, dim: int, source_dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.cross_norm = ModulatedNorm(dim)
        self.self_norm = ModulatedNorm(dim)
        self.mlp_norm = ModulatedNorm(dim)
        self.source_norm = nn.LayerNorm(source_dim)
        self.cross_attention = MultiHeadAttention(dim, source_dim, num_heads)
        self.self_attention = MultiHeadAttention(dim, dim, num_heads)
        self.mlp_in = nn.Linear(dim, 2 * mlp_hidden_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_dim, dim, bias=False)
        self.time_modulation = nn.Linear(dim, 6 * dim, bias=True)

    def forward(
        self,
        queries: torch.Tensor,
        source: torch.Tensor | None,
        time: torch.Tensor,
        mask: torch.Tensor | None,
        source_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cross_scale, cross_shift, self_scale, self_shift, mlp_scale, mlp_shift = self.time_modulation(F.silu(time)).chunk(6, dim=-1)
        q_cross = self.cross_norm(queries, cross_scale, cross_shift)
        if source_kv is not None:
            queries = queries + self.cross_attention.forward_with_kv(q_cross, source_kv[0], source_kv[1], mask)
        else:
            queries = queries + self.cross_attention(q_cross, self.source_norm(source), mask)
        normalized = self.self_norm(queries, self_scale, self_shift)
        queries = queries + self.self_attention(normalized, normalized)
        gate, value = self.mlp_in(self.mlp_norm(queries, mlp_scale, mlp_shift)).chunk(2, dim=-1)
        return queries + self.mlp_out(F.silu(gate) * value)


class SemanticResampler(nn.Module):
    """Distils the Qwen3.5 layers into ``num_queries`` tokens, per timestep."""

    def __init__(self, source_dim: int, output_dim: int, num_layers: int, num_queries: int, num_blocks: int, dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, dim))
        self.layer_embeddings = nn.Parameter(torch.empty(num_layers, 1, source_dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(ResamplerBlock(dim, source_dim, num_heads, mlp_hidden_dim) for _ in range(num_blocks))
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, output_dim, bias=False)

    def project_static_kv(self, layers: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        embeddings = self.layer_embeddings.to(device=layers[0].device, dtype=layers[0].dtype)
        source = torch.cat([state + embeddings[index].unsqueeze(0) for index, state in enumerate(layers)], dim=1)
        return tuple(block.cross_attention.project_kv(block.source_norm(source)) for block in self.blocks)

    def forward_with_kv(
        self,
        source_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        timesteps: torch.Tensor,
        source_mask: torch.Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        time = self.time_mlp(timestep_embedding(timesteps, self.dim).to(device=device, dtype=dtype))
        queries = self.query_tokens.to(device=device, dtype=dtype).expand(batch, -1, -1)
        for block, kv in zip(self.blocks, source_kv, strict=True):
            queries = block(queries, None, time, source_mask, source_kv=kv)
        return self.output_projection(self.output_norm(queries))

    def forward(self, layers: list[torch.Tensor], timesteps: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        embeddings = self.layer_embeddings.to(device=layers[0].device, dtype=layers[0].dtype)
        source = torch.cat([state + embeddings[index].unsqueeze(0) for index, state in enumerate(layers)], dim=1)
        source_mask = None if mask is None else _attention_mask(torch.cat([mask] * self.num_layers, dim=1))
        time = self.time_mlp(timestep_embedding(timesteps, self.dim).to(source.dtype))
        queries = self.query_tokens.to(device=source.device, dtype=source.dtype).expand(source.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, source, time, source_mask)
        return self.output_projection(self.output_norm(queries))


class QualityAnchor(nn.Module):
    """The frozen v1 adapter: per adapter block, attention to a learned mix of the layers."""

    def __init__(self, blocks: int, layers: int, model_dim: int, source_dim: int, num_heads: int):
        super().__init__()
        self.layer_mix_logits = nn.Parameter(torch.zeros(blocks, layers))
        self.query_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in range(blocks))
        self.source_norms = nn.ModuleList(nn.RMSNorm(source_dim, eps=1e-6) for _ in range(blocks))
        self.semantic_attentions = nn.ModuleList(Attention(model_dim, source_dim, num_heads, model_dim // num_heads) for _ in range(blocks))


class SemanticConnector(nn.Module):
    """Semantic Connector v2 (``anima_qwen35_quality_anchored_semantic_connector_v2``).

    Holds only its own weights: the native adapter (the DiT's ``llm_adapter``) is passed to
    ``forward``, so the two keep their own state-dict prefixes.
    """

    def __init__(
        self,
        adapter_blocks: int = 6,
        model_dim: int = 1024,
        adapter_heads: int = 16,
        source_dim: int = SEMANTIC_DIM,
        layers: tuple[int, ...] = SEMANTIC_LAYERS,
        num_queries: int = 64,
        resampler_blocks: int = 6,
        resampler_dim: int = 2048,
        resampler_heads: int = 16,
        mlp_hidden_dim: int = 5632,
    ):
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        self.quality_anchor = QualityAnchor(adapter_blocks, len(self.layers), model_dim, source_dim, adapter_heads)
        self.semantic_resampler = SemanticResampler(source_dim, model_dim, len(self.layers), num_queries, resampler_blocks, resampler_dim, resampler_heads, mlp_hidden_dim)
        self.v2_query_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in range(adapter_blocks))
        self.v2_semantic_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in range(adapter_blocks))
        self.v2_attentions = nn.ModuleList(Attention(model_dim, model_dim, adapter_heads, model_dim // adapter_heads) for _ in range(adapter_blocks))
        self._static_cache: OrderedDict[tuple, _StaticConnectorCache] = OrderedDict()

    def clear_cache(self) -> None:
        self._static_cache.clear()

    def _apply(self, fn, recurse: bool = True):
        # Clear cached KV tensors whenever the connector moves device/dtype (e.g. VRAM offload).
        self._static_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _weight_signature(self, adapter: nn.Module) -> tuple | None:
        modules: list[nn.Module] = [adapter.embed, adapter.in_proj, adapter.rotary_emb]
        params: list[torch.Tensor | None] = [
            getattr(adapter.embed, "weight", None),
            getattr(adapter.in_proj, "weight", None),
            getattr(adapter.in_proj, "bias", None),
            getattr(adapter.rotary_emb, "inv_freq", None),
            self.semantic_resampler.layer_embeddings,
            self.quality_anchor.layer_mix_logits,
        ]
        for block in self.semantic_resampler.blocks:
            modules.extend((block.source_norm, block.cross_attention, block.cross_attention.k_proj, block.cross_attention.v_proj))
            params.extend((block.source_norm.weight, block.source_norm.bias, block.cross_attention.k_proj.weight, block.cross_attention.v_proj.weight))
        for norm, attn in zip(self.quality_anchor.source_norms, self.quality_anchor.semantic_attentions, strict=True):
            modules.extend((norm, attn, attn.k_proj, attn.k_norm, attn.v_proj))
            params.extend((norm.weight, attn.k_proj.weight, attn.k_norm.weight, attn.v_proj.weight))
        if any(_has_dynamic_hooks(module) for module in modules):
            return None
        signature: list[tuple | None] = []
        for param in params:
            if param is None:
                signature.append(None)
                continue
            item = _param_signature(param)
            if item is None:
                return None
            signature.append(item)
        return tuple(signature)

    def _estimated_cache_bytes(self, batch: int, sem_len: int, elem_size: int) -> int:
        resampler = self.semantic_resampler
        resampler_kv = len(resampler.blocks) * 2 * batch * (len(self.layers) * sem_len) * resampler.dim
        anchor = self.quality_anchor
        model_dim = anchor.semantic_attentions[0].n_heads * anchor.semantic_attentions[0].head_dim
        anchor_kv = len(anchor.semantic_attentions) * 2 * batch * sem_len * model_dim
        return (resampler_kv + anchor_kv) * elem_size

    @staticmethod
    def _anchor_attention_with_kv(
        attn: Attention,
        query: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor | None,
        query_rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        input_shape = query.shape[:-1]
        q_shape = (*input_shape, attn.n_heads, attn.head_dim)
        query_states = attn.q_norm(attn.q_proj(query).view(q_shape)).transpose(1, 2)
        query_states = apply_rotary_pos_emb(query_states, query_rope[0], query_rope[1])
        attn_output = scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output)

    def _build_static_cache(
        self,
        adapter: nn.Module,
        source: torch.Tensor,
        target_ids: torch.Tensor,
        semantic_layers: list[torch.Tensor],
        semantic_mask: torch.Tensor | None,
        dtype: torch.dtype | None,
        owner_tensors: list[torch.Tensor],
    ) -> _StaticConnectorCache:
        embedded = adapter.embed(target_ids)
        embedded_in = adapter.in_proj(embedded if dtype is None else embedded.to(dtype))
        compute_dtype = embedded_in.dtype
        source_cast = source.to(dtype=compute_dtype)
        semantic_cast = [state.to(dtype=compute_dtype) for state in semantic_layers]
        anchor_mask = _attention_mask(semantic_mask)
        resampler_mask = None if semantic_mask is None else _attention_mask(torch.cat([semantic_mask] * len(self.layers), dim=1))

        def rope(length: int):
            return adapter.rotary_emb(embedded_in, torch.arange(length, device=embedded_in.device).unsqueeze(0))

        query_rope = rope(embedded_in.shape[1])
        source_rope = rope(source_cast.shape[1])
        anchor_rope = rope(semantic_cast[0].shape[1])
        bank_rope = rope(self.semantic_resampler.query_tokens.shape[1])

        resampler_kv = self.semantic_resampler.project_static_kv(semantic_cast)

        anchor = self.quality_anchor
        mix = anchor.layer_mix_logits.float().softmax(dim=-1).to(device=embedded_in.device, dtype=compute_dtype)
        anchor_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index in range(len(adapter.blocks)):
            mixed = sum(state * mix[index, layer] for layer, state in enumerate(semantic_cast))
            ctx = anchor.source_norms[index](mixed)
            attn = anchor.semantic_attentions[index]
            kv_shape = (*ctx.shape[:-1], attn.n_heads, attn.head_dim)
            key_states = attn.k_norm(attn.k_proj(ctx).view(kv_shape)).transpose(1, 2)
            key_states = apply_rotary_pos_emb(key_states, anchor_rope[0], anchor_rope[1])
            value_states = attn.v_proj(ctx).view(kv_shape).transpose(1, 2)
            anchor_kv.append((key_states, value_states))

        return _StaticConnectorCache(
            owner_refs=tuple(_tensor_owner_ref(t) for t in owner_tensors),
            adapter_ref=weakref.ref(adapter),
            embedded_in=embedded_in,
            source_cast=source_cast,
            anchor_mask=anchor_mask,
            resampler_mask=resampler_mask,
            query_rope=query_rope,
            source_rope=source_rope,
            bank_rope=bank_rope,
            resampler_kv=resampler_kv,
            anchor_kv=tuple(anchor_kv),
        )

    def forward(
        self,
        adapter: nn.Module,
        source: torch.Tensor,
        target_ids: torch.Tensor,
        semantic: list[torch.Tensor] | torch.Tensor,
        timesteps: torch.Tensor,
        semantic_mask: torch.Tensor | None = None,
        include_inserted_blocks: bool = True,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The DiT context (before the T5 weights) for ``source`` (Qwen3 0.6B states),
        ``target_ids`` (T5 ids) and ``semantic`` (the Qwen3.5 layers, a list or stacked on dim 1).
        ``dtype``: the one to compute in (the DiT's); else the T5 embedding's."""

        semantic_raw = semantic
        if isinstance(semantic, torch.Tensor):
            semantic_layers = list(semantic.unbind(1))
        else:
            semantic_layers = list(semantic)
        if len(semantic_layers) != len(self.layers):
            raise ValueError(f"Semantic Connector v2 expects {len(self.layers)} Qwen3.5 layers, got {len(semantic_layers)}")
        if len(adapter.blocks) != len(self.v2_attentions):
            raise ValueError(f"Semantic Connector v2 wraps a {len(self.v2_attentions)}-block LLM adapter, got {len(adapter.blocks)} blocks")

        use_cache = (
            include_inserted_blocks
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and not self.training
            and not adapter.training
        )
        if use_cache:
            compute_dtype = dtype if dtype is not None else adapter.embed.weight.dtype
            est_bytes = self._estimated_cache_bytes(semantic_layers[0].shape[0], semantic_layers[0].shape[1], compute_dtype.itemsize)
            weight_sig = self._weight_signature(adapter) if est_bytes <= _STATIC_CACHE_MAX_BYTES else None
            if weight_sig is not None:
                if isinstance(semantic_raw, torch.Tensor):
                    sem_sig = (_tensor_signature(semantic_raw),)
                    owner_tensors = [source, target_ids, semantic_raw]
                else:
                    sem_sig = tuple(_tensor_signature(t) for t in semantic_layers)
                    owner_tensors = [source, target_ids, *semantic_layers]
                if semantic_mask is not None:
                    owner_tensors.append(semantic_mask)
                cache_key = (
                    id(adapter),
                    dtype,
                    _tensor_signature(source),
                    _tensor_signature(target_ids),
                    sem_sig,
                    _tensor_signature(semantic_mask),
                    weight_sig,
                )
                cached = self._static_cache.get(cache_key)
                if cached is not None and cached.alive(adapter):
                    self._static_cache.move_to_end(cache_key)
                else:
                    cached = self._build_static_cache(adapter, source, target_ids, semantic_layers, semantic_mask, dtype, owner_tensors)
                    self._static_cache[cache_key] = cached
                    self._static_cache.move_to_end(cache_key)
                    while len(self._static_cache) > _STATIC_CACHE_MAX_ENTRIES:
                        self._static_cache.popitem(last=False)

                # TransformerBlock.forward mutates x in-place (x.add_), so clone the cached seed.
                x = cached.embedded_in.clone()
                bank = self.semantic_resampler.forward_with_kv(
                    cached.resampler_kv,
                    timesteps,
                    cached.resampler_mask,
                    batch=x.shape[0],
                    device=x.device,
                    dtype=x.dtype,
                )
                anchor = self.quality_anchor
                for index, block in enumerate(adapter.blocks):
                    x = block(x, cached.source_cast, position_embeddings=cached.query_rope, position_embeddings_context=cached.source_rope)
                    key_states, value_states = cached.anchor_kv[index]
                    x = x + self._anchor_attention_with_kv(
                        anchor.semantic_attentions[index],
                        anchor.query_norms[index](x),
                        key_states,
                        value_states,
                        cached.anchor_mask,
                        cached.query_rope,
                    )
                    x = x + self.v2_attentions[index](
                        self.v2_query_norms[index](x),
                        context=self.v2_semantic_norms[index](bank),
                        position_embeddings=cached.query_rope,
                        position_embeddings_context=cached.bank_rope,
                    )
                return adapter.norm(adapter.out_proj(x))

        embedded = adapter.embed(target_ids)
        # The embedding can be stored wider than the rest (a float32 table in a bfloat16
        # checkpoint): without manual casting (CUDA) the Linears take only their own dtype
        x = adapter.in_proj(embedded if dtype is None else embedded.to(dtype))
        source = source.to(dtype=x.dtype)
        semantic_layers = [state.to(dtype=x.dtype) for state in semantic_layers]
        anchor_mask = _attention_mask(semantic_mask)
        bank = self.semantic_resampler(semantic_layers, timesteps, semantic_mask) if include_inserted_blocks else None

        def rope(length: int):
            return adapter.rotary_emb(x, torch.arange(length, device=x.device).unsqueeze(0))

        query_rope, source_rope, anchor_rope = rope(x.shape[1]), rope(source.shape[1]), rope(semantic_layers[0].shape[1])
        bank_rope = rope(bank.shape[1]) if bank is not None else None
        anchor = self.quality_anchor
        mix = anchor.layer_mix_logits.float().softmax(dim=-1).to(device=x.device, dtype=x.dtype)

        for index, block in enumerate(adapter.blocks):
            x = block(x, source, position_embeddings=query_rope, position_embeddings_context=source_rope)
            if not include_inserted_blocks:
                continue
            mixed = sum(state * mix[index, layer] for layer, state in enumerate(semantic_layers))
            x = x + anchor.semantic_attentions[index](anchor.query_norms[index](x), mask=anchor_mask, context=anchor.source_norms[index](mixed), position_embeddings=query_rope, position_embeddings_context=anchor_rope)
            x = x + self.v2_attentions[index](self.v2_query_norms[index](x), context=self.v2_semantic_norms[index](bank), position_embeddings=query_rope, position_embeddings_context=bank_rope)
        return adapter.norm(adapter.out_proj(x))


def connector_config(state_dict: dict, prefix: str = "", metadata: dict | None = None) -> dict | None:
    """``SemanticConnector`` arguments read from a checkpoint's weights (and metadata for
    what the shapes do not tell), or None when it has no connector."""

    query_tokens = state_dict.get(f"{prefix}{CONNECTOR_PREFIX}semantic_resampler.query_tokens")
    if query_tokens is None:
        return None
    metadata = metadata or {}

    def count(stem: str) -> int:
        marker = f"{prefix}{CONNECTOR_PREFIX}{stem}."
        return len({key[len(marker) :].split(".", 1)[0] for key in state_dict if key.startswith(marker)})

    layer_mix = state_dict[f"{prefix}{CONNECTOR_PREFIX}quality_anchor.layer_mix_logits"]
    mlp_out = state_dict[f"{prefix}{CONNECTOR_PREFIX}semantic_resampler.blocks.0.mlp_out.weight"]
    head_norm = state_dict[f"{prefix}{CONNECTOR_PREFIX}v2_attentions.0.q_norm.weight"]
    model_dim = state_dict[f"{prefix}{CONNECTOR_PREFIX}v2_attentions.0.q_proj.weight"].shape[0]
    layers = SEMANTIC_LAYERS
    if raw := metadata.get("anima_v2_adapter_layer_indices"):
        try:
            layers = tuple(int(value) for value in raw.strip("[] ").split(","))
        except ValueError:
            pass
    if len(layers) != layer_mix.shape[1]:
        raise ValueError(f"Semantic Connector v2 mixes {layer_mix.shape[1]} Qwen3.5 layers, metadata names {layers}")
    return {
        "adapter_blocks": int(layer_mix.shape[0]),
        "model_dim": int(model_dim),
        "adapter_heads": int(model_dim // head_norm.shape[0]),
        "source_dim": int(state_dict[f"{prefix}{CONNECTOR_PREFIX}semantic_resampler.layer_embeddings"].shape[-1]),
        "layers": layers,
        "num_queries": int(query_tokens.shape[1]),
        "resampler_blocks": count("semantic_resampler.blocks"),
        "resampler_dim": int(query_tokens.shape[2]),
        "resampler_heads": int(metadata.get("anima_v2_adapter_semantic_resampler_heads", 16)),
        "mlp_hidden_dim": int(mlp_out.shape[1]),
    }
