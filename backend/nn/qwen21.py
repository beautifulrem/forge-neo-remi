# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/qwen_image21/model.py

from collections import OrderedDict
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import memory_management
from backend.args import dynamic_args
from backend.attention import attention_function
from backend.nn.flux import EmbedND, timestep_embedding
from backend.nn.qwen import TimestepEmbedding
from backend.operations import main_stream_worker, weights_manual_cast
from backend.operations_mixed_precision import linear_input_act
from backend.quant_ops import ck
from backend.utils import fp16_fix

_PREFIX_KV_CACHE_MAX = 2
_ROPE_CACHE_MAX = 4
_CPU_KV_CACHE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else int(getattr(tensor, "_version", 0))


def _broadcast_ref_batch(ref: torch.Tensor, batch: int) -> torch.Tensor:
    if ref.shape[0] == batch:
        return ref
    if ref.shape[0] == 1:
        return ref.expand(batch, *ref.shape[1:])
    if batch % ref.shape[0] == 0:
        return ref.repeat(batch // ref.shape[0], *([1] * (ref.ndim - 1)))
    return ref[:1].expand(batch, *ref.shape[1:])


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x):
        w = memory_management.cast_to(self.weight, dtype=x.dtype, device=x.device) + 1.0
        return F.rms_norm(x.float(), w.shape, w, self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False)

    def forward(self, timestep, dtype):
        return self.timestep_embedder(timestep_embedding(timestep.float(), 256).to(dtype))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, fused=True):
        super().__init__()
        self.fused = fused
        if fused:
            self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        else:
            self.proj = nn.Linear(dim, hidden_dim, bias=False)
            self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        if self.fused:
            return linear_input_act(self.out, self.gate_up(x), "swiglu")
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False)])
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def project_kv(self, x, pe):
        """Project and RoPE-rotate only K and V (used for the last block's prefix KV)."""
        B, N, _ = x.shape
        k = self.to_k(x).view(B, N, self.heads, -1)
        v = self.to_v(x).view(B, N, self.heads, -1)
        k_scale, _, k_stream = weights_manual_cast(self.norm_k, k)
        with main_stream_worker(k_scale, None, k_stream):
            _, k = ck.rms_rope(k, k, pe, k_scale, k_scale, self.norm_k.eps)
        return k.flatten(2), v.flatten(2)

    def forward(self, x, pe, attn_fn, prefix_len, transformer_options={}, prefix_kv=None, kv_out=None):
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, -1)
        k = self.to_k(x).view(B, N, self.heads, -1)
        v = self.to_v(x).view(B, N, self.heads, -1)

        q_scale, _, q_stream = weights_manual_cast(self.norm_q, q)
        k_scale, _, k_stream = weights_manual_cast(self.norm_k, k)
        with main_stream_worker(q_scale, None, q_stream), main_stream_worker(k_scale, None, k_stream):
            q, k = ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)

        if kv_out is not None:
            kv_out[0] = k.flatten(2)
            kv_out[1] = v.flatten(2)

        if prefix_kv is not None:
            k_pref, v_pref = prefix_kv
            k_all = torch.cat((k_pref, k.flatten(2)), dim=1)
            v_all = torch.cat((v_pref, v.flatten(2)), dim=1)
            return self.to_out[0](attention_function(q.flatten(2), k_all, v_all, self.heads, transformer_options=transformer_options))

        if attn_fn is None:
            return self.to_out[0](attention_function(q.flatten(2), k.flatten(2), v.flatten(2), self.heads, transformer_options=transformer_options))

        return self.to_out[0](attn_fn(q, k, v, self.heads))


def _split_rows(p):
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def _modulated_norm(norm, x, scale, prefix_len, zero):
    s_prefix, s_target = scale
    out = ck.adaln(x, s_target, zero, norm.eps)
    if prefix_len:
        out[:, :prefix_len] = ck.adaln(x[:, :prefix_len], s_prefix, zero, norm.eps)
    return out


def _gated_residual(x, y, gate, prefix_len):
    g_prefix, g_target = gate
    x[:, prefix_len:].addcmul_(y[:, prefix_len:], g_target)
    if prefix_len:
        x[:, :prefix_len].addcmul_(y[:, :prefix_len], g_prefix)
    return x


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3, eps=1e-6, fused_mlp=True):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio, fused=fused_mlp)

    def forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}, prefix_kv=None, kv_out=None):
        scale1, gate1, scale2, gate2, zero = mod
        x = _gated_residual(
            x,
            self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len, zero), pe, attn_fn, prefix_len, transformer_options, prefix_kv=prefix_kv, kv_out=kv_out),
            gate1,
            prefix_len,
        )
        x = _gated_residual(x, self.img_mlp(_modulated_norm(self.img_norm2, x, scale2, prefix_len, zero)), gate2, prefix_len)

        return fp16_fix(x)


class LastLayer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        return ck.adaln(x, scale, torch.zeros_like(scale[:1]), self.norm.eps)


def block_causal_attention(segments, transformer_options={}):

    def attn(q, k, v, heads, preferred_attention=None):
        outs = [attention_function(q[:, start:end].flatten(2), k[:, :end].flatten(2), v[:, :end].flatten(2), heads, mask=mask, causal=mask is not None, transformer_options=transformer_options, preferred_attention=preferred_attention) for start, end, mask in segments]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    return attn


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        fused_mlp=True,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList([QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, fused_mlp=fused_mlp) for _ in range(num_layers)])

        self.final_input = nn.Identity()  # Spectrum pre-head capture seam (0 params)
        self.norm_out = LastLayer(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)

        # Lossless cross-step prefix KV & RoPE caches (cleared on sampling_cleanup / _apply).
        self._prefix_kv_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._target_rope_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    def clear_caches(self) -> None:
        """Free cross-step prefix KV and target RoPE caches."""

        self._prefix_kv_cache.clear()
        self._target_rope_cache.clear()

    def _apply(self, fn):
        self.clear_caches()
        return super()._apply(fn)

    def _prefix_modules(self):
        yield from self.txt_in.modules()
        yield self.img_in
        yield from self.time_text_embed.modules()
        yield from self.modulation.modules()
        for block in self.transformer_blocks:
            yield from getattr(block, "_orig_mod", block).modules()

    def _prefix_cacheable(self, transformer_options: dict) -> bool:
        if torch.is_grad_enabled():
            return False
        if isinstance(transformer_options, dict):
            if transformer_options.get("optimized_attention_override") is not None:
                return False
            patches = transformer_options.get("patches")
            if isinstance(patches, dict) and any(patches.values()):
                return False
            blocks_replace = transformer_options.get("patches_replace", {}).get("dit")
            if isinstance(blocks_replace, dict) and blocks_replace:
                return False
        return not any(getattr(m, "weight_function", None) or getattr(m, "bias_function", None) for m in self._prefix_modules())

    def _prefix_weight_probes(self) -> list[torch.Tensor]:
        probes = [
            self.txt_in.text_norm.weight,
            self.txt_in.in_layer.weight,
            self.txt_in.out_layer.weight,
            self.img_in.weight,
            self.time_text_embed.timestep_embedder.linear_1.weight,
            self.time_text_embed.timestep_embedder.linear_2.weight,
            self.modulation[1].weight,
        ]
        for block in self.transformer_blocks:
            raw = getattr(block, "_orig_mod", block)
            probes.extend(
                (
                    raw.attn.to_q.weight,
                    raw.attn.to_k.weight,
                    raw.attn.to_v.weight,
                    raw.attn.to_out[0].weight,
                    raw.img_mlp.out.weight,
                )
            )
            if raw.img_mlp.fused:
                probes.append(raw.img_mlp.gate_up.weight)
            else:
                probes.extend((raw.img_mlp.gate_layer.weight, raw.img_mlp.proj.weight))
        return probes

    def _prefix_weight_key(self, probes: list[torch.Tensor] | None = None) -> tuple:
        if probes is None:
            probes = self._prefix_weight_probes()
        return tuple((w.data_ptr(), _tensor_version(w), w.device, w.dtype) for w in probes)

    def _prefix_kv_fits_budget(self, batch: int, prefix_len: int, device: torch.device, dtype: torch.dtype) -> bool:
        kv_bytes = batch * len(self.transformer_blocks) * 2 * prefix_len * self.inner_dim * dtype.itemsize
        if device.type == "cpu":
            return kv_bytes <= _CPU_KV_CACHE_LIMIT_BYTES
        free = memory_management.get_free_memory(device)
        if free > 0 and kv_bytes > free * 0.45:
            # Evict stale entries once before giving up on caching this prompt.
            if self._prefix_kv_cache:
                self._prefix_kv_cache.clear()
                memory_management.soft_empty_cache()
                free = memory_management.get_free_memory(device)
            return free <= 0 or kv_bytes <= free * 0.45
        return True

    def _target_rope(self, h: int, w: int, pos: int, device: torch.device) -> torch.Tensor:
        key = (h, w, pos, device)
        cached = self._target_rope_cache.get(key)
        if cached is not None:
            self._target_rope_cache.move_to_end(key)
            return cached
        hh = torch.arange(h, device=device, dtype=torch.float32) - (h - h // 2)
        ww = torch.arange(w, device=device, dtype=torch.float32) - (w - w // 2)
        ids = torch.stack(
            [
                torch.full((h, w), pos, device=device, dtype=torch.float32),
                hh[:, None].expand(h, w),
                ww[None, :].expand(h, w),
            ],
            dim=-1,
        ).flatten(0, 1)
        pe = self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()
        self._target_rope_cache[key] = pe
        while len(self._target_rope_cache) > _ROPE_CACHE_MAX:
            self._target_rope_cache.popitem(last=False)
        return pe

    def _build_prefix_sequence(
        self,
        batch: int,
        target_parity: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        context: torch.Tensor,
        ref_latents: list[torch.Tensor],
        image_slots: list[int],
    ):
        txt = self.txt_in(context)
        if txt.shape[0] != batch:
            txt = _broadcast_ref_batch(txt, batch)
        slots = (image_slots + [txt.shape[1]] * len(ref_latents))[: len(ref_latents)]
        bounds = [0] + slots + [txt.shape[1]]

        parts, ids, segments = [], [], []
        pos, length = 0, 0
        th_mod, tw_mod = target_parity
        for idx, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
                segments.append((length, length + n, torch.ones((1, 1, n, length + n), dtype=torch.bool, device=device).tril(length)))
                pos += n
                length += n
            if idx < len(ref_latents):
                img = _broadcast_ref_batch(ref_latents[idx], batch)
                h, w = img.shape[-2:]
                parts.append(self.img_in(img.flatten(2).transpose(1, 2).to(device=device, dtype=dtype)))
                hh = torch.arange(h, device=device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - th_mod)
                ww = torch.arange(w, device=device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - tw_mod)
                ids.append(torch.stack([torch.full((h, w), pos, device=device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
                segments.append((length, length + h * w, None))
                pos += max(h, w)
                length += h * w

        if not parts:
            return None, None, [], pos
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments, pos

    def preprocess_prefix(
        self,
        batch: int,
        target_parity: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        context: torch.Tensor,
        ref_latents: list[torch.Tensor],
        image_slots: list[int],
        transformer_options: dict = {},
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        """Run the block-causal prefix (text + reference latents at timestep 0)
        once across all layers and capture each layer's post-RoPE ``(K, V)``."""

        hidden, pe_prefix, segments, prefix_pos = self._build_prefix_sequence(
            batch, target_parity, device, dtype, context, ref_latents, image_slots
        )
        if hidden is None:
            return [], prefix_pos

        temb_zero = self.time_text_embed(torch.zeros(1, device=device, dtype=dtype), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb_zero).chunk(4, dim=-1)
        s1 = scale1.unsqueeze(1)
        g1 = gate1.tanh().unsqueeze(1)
        s2 = scale2.unsqueeze(1)
        g2 = gate2.tanh().unsqueeze(1)
        zero = torch.zeros_like(s1)
        mod_prefix = ((s1, s1), (g1, g1), (s2, s2), (g2, g2), zero)

        attn_fn = block_causal_attention(segments, transformer_options)
        num_layers = len(self.transformer_blocks)
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.transformer_blocks):
            # One-shot prefix prefill runs once per prompt/ref set: keep eager so
            # per-block torch.compile only traces the repeated target-token step.
            eager_block = getattr(block, "_orig_mod", block)
            if i == num_layers - 1:
                # The last block's prefix output is discarded before norm_out;
                # only its rotated (K, V) are read by target tokens.
                normed = ck.adaln(hidden, s1, zero, eager_block.img_norm1.eps)
                prefix_kv.append(eager_block.attn.project_kv(normed, pe_prefix))
            else:
                kv_slot: list = [None, None]
                hidden = eager_block(hidden, mod_prefix, pe_prefix, attn_fn, 0, transformer_options, kv_out=kv_slot)
                prefix_kv.append((kv_slot[0], kv_slot[1]))
        return prefix_kv, prefix_pos

    @staticmethod
    def _refs_match(entry: dict, ref_latents: list[torch.Tensor]) -> bool:
        stored_refs = entry["ref_weakrefs"]
        stored_meta = entry["ref_meta"]
        if len(stored_refs) != len(ref_latents):
            return False
        for wref, (ptr, ver, snapshot), cur in zip(stored_refs, stored_meta, ref_latents):
            if wref() is cur and cur.data_ptr() == ptr and _tensor_version(cur) == ver:
                continue
            if snapshot.shape != cur.shape or snapshot.dtype != cur.dtype or snapshot.device != cur.device:
                return False
            if not torch.equal(snapshot, cur):
                return False
        return True

    def _get_prefix_kv(
        self,
        batch: int,
        target_parity: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        context: torch.Tensor,
        ref_latents: list[torch.Tensor],
        image_slots: list[int],
        transformer_options: dict,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int] | None:
        prefix_len = int(context.shape[1]) + sum(int(r.shape[-2] * r.shape[-1]) for r in ref_latents)
        if prefix_len == 0:
            return [], 0

        weight_probes = self._prefix_weight_probes()
        weight_key = self._prefix_weight_key(weight_probes)
        meta_key = (
            batch,
            tuple(context.shape),
            context.dtype,
            context.device,
            target_parity,
            tuple(int(s) for s in image_slots),
            tuple((tuple(r.shape), r.dtype, r.device) for r in ref_latents),
            device,
            dtype,
            weight_key,
        )

        # Look up matching entry (handles both pointer-stable inputs and fresh
        # torch.cat tensors produced each step by sampling_function.cond_cat).
        for cache_key, entry in list(self._prefix_kv_cache.items()):
            if entry["meta_key"] != meta_key:
                continue
            if not all(wr() is w for wr, w in zip(entry["weight_weakrefs"], weight_probes)):
                continue
            same_ctx = (
                entry["ctx_weakref"]() is context
                and entry["ctx_ptr"] == context.data_ptr()
                and entry["ctx_ver"] == _tensor_version(context)
            )
            if not same_ctx:
                same_ctx = torch.equal(entry["ctx_snapshot"], context)
            if same_ctx and self._refs_match(entry, ref_latents):
                self._prefix_kv_cache.move_to_end(cache_key)
                return entry["prefix_kv"], entry["prefix_pos"]

        if not self._prefix_kv_fits_budget(batch, prefix_len, device, dtype):
            return None

        prefix_kv, prefix_pos = self.preprocess_prefix(
            batch, target_parity, device, dtype, context, ref_latents, image_slots, transformer_options
        )
        ref_weakrefs = []
        ref_meta = []
        for r in ref_latents:
            try:
                wref = weakref.ref(r)
            except TypeError:
                wref = lambda: None
            ref_weakrefs.append(wref)
            ref_meta.append((r.data_ptr(), _tensor_version(r), r.detach().clone()))
        try:
            ctx_wref = weakref.ref(context)
        except TypeError:
            ctx_wref = lambda: None

        weight_weakrefs = tuple(weakref.ref(w) for w in weight_probes)
        entry_id = (meta_key, id(prefix_kv))
        self._prefix_kv_cache[entry_id] = {
            "meta_key": meta_key,
            "weight_weakrefs": weight_weakrefs,
            "ctx_weakref": ctx_wref,
            "ctx_ptr": context.data_ptr(),
            "ctx_ver": _tensor_version(context),
            "ctx_snapshot": context.detach().clone(),
            "ref_weakrefs": tuple(ref_weakrefs),
            "ref_meta": tuple(ref_meta),
            "prefix_kv": prefix_kv,
            "prefix_pos": prefix_pos,
        }
        while len(self._prefix_kv_cache) > _PREFIX_KV_CACHE_MAX:
            self._prefix_kv_cache.popitem(last=False)
        return prefix_kv, prefix_pos

    def compute_final(self, feature: torch.Tensor, timestep: torch.Tensor, input_shape: tuple[int, ...]) -> torch.Tensor:
        """Run timestep embedding + ``norm_out`` + ``proj_out`` on a cached or
        forecast pre-head target feature (Spectrum)."""

        B = feature.shape[0]
        H, W = int(input_shape[-2]), int(input_shape[-1])
        dtype = feature.dtype
        t = ((timestep.reshape(-1) * 1000).to(dtype) / 1000).to(dtype)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        temb = self.time_text_embed(t, dtype)
        out = self.proj_out(self.norm_out(feature, temb))
        return out.transpose(1, 2).reshape(B, self.out_channels, H, W)

    def build_sequence(self, x: torch.Tensor, context: torch.Tensor, ref_latents: list[torch.Tensor], image_slots: list[int]):
        txt = self.txt_in(context)
        if txt.shape[0] != x.shape[0]:
            txt = _broadcast_ref_batch(txt, x.shape[0])
        slots = (image_slots + [txt.shape[1]] * len(ref_latents))[: len(ref_latents)]
        bounds = [0] + slots + [txt.shape[1]]

        parts, ids, segments = [], [], []
        pos, length = 0, 0
        for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x]):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
                segments.append((length, length + n, torch.ones((1, 1, n, length + n), dtype=torch.bool, device=x.device).tril(length)))
                pos += n
                length += n
            h, w = img.shape[-2:]

            if img is not x:
                img = _broadcast_ref_batch(img, x.shape[0])

            parts.append(self.img_in(img.flatten(2).transpose(1, 2).to(x)))
            hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2)
            ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - x.shape[-1] % 2)
            ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
            segments.append((length, length + h * w, None))
            pos += max(h, w)
            length += h * w

        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments

    def forward(self, x, timesteps, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device
        if ref_latents is None:
            ref_latents = list(dynamic_args.ref_latents or [])
        if image_slots is None:
            image_slots = list(dynamic_args.image_slots or [])

        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)

        if self._prefix_cacheable(transformer_options):
            cached = self._get_prefix_kv(B, (H % 2, W % 2), device, dtype, context, ref_latents, image_slots, transformer_options)
            if cached is not None:
                prefix_kv, prefix_pos = cached
                temb_tgt = self.time_text_embed(t, dtype)
                scale1, gate1, scale2, gate2 = self.modulation(temb_tgt).chunk(4, dim=-1)
                s1 = scale1.unsqueeze(1)
                g1 = gate1.tanh().unsqueeze(1)
                s2 = scale2.unsqueeze(1)
                g2 = gate2.tanh().unsqueeze(1)
                zero = torch.zeros_like(s1[:1])
                mod_tgt = ((s1, s1), (g1, g1), (s2, s2), (g2, g2), zero)

                hidden_states = self.img_in(x.flatten(2).transpose(1, 2))
                pe_tgt = self._target_rope(H, W, prefix_pos, device)

                transformer_options["total_blocks"] = len(self.transformer_blocks)
                transformer_options["block_type"] = "single"
                for i, block in enumerate(self.transformer_blocks):
                    transformer_options["block_index"] = i
                    kv_i = prefix_kv[i] if prefix_kv else None
                    hidden_states = block(hidden_states, mod_tgt, pe_tgt, None, 0, transformer_options, prefix_kv=kv_i)

                feature = self.final_input(hidden_states)
                out = self.proj_out(self.norm_out(feature, temb_tgt))
                return out.transpose(1, 2).reshape(B, self.out_channels, H, W)

        hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, image_slots)
        prefix_len = hidden_states.shape[1] - H * W
        patches = transformer_options.get("patches", {})
        for p in patches.get("post_input", []):
            out = p({"img": hidden_states, "pe": pe, "transformer_options": transformer_options})
            hidden_states, pe = out["img"], out.get("pe", pe)

        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()), torch.zeros_like(scale1[:1, None]))

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "single"

        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            attn_fn = block_causal_attention(segments, transformer_options)
            if ("single_block", i) in blocks_replace:

                def block_wrap(args):
                    return {"img": block(args["img"], mod, args["pe"], attn_fn, prefix_len, args["transformer_options"])}

                args = {"img": hidden_states, "vec": temb, "pe": pe, "mod": mod, "attn_fn": attn_fn, "prefix_len": prefix_len, "transformer_options": transformer_options}
                hidden_states = blocks_replace[("single_block", i)](args, {"original_block": block_wrap})["img"]
            else:
                hidden_states = block(hidden_states, mod, pe, attn_fn, prefix_len, transformer_options)
            for p in patches.get("single_block", []):
                hidden_states = p({"img": hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})["img"]

        feature = self.final_input(hidden_states[:, prefix_len:])
        hidden_states = self.norm_out(feature, temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
