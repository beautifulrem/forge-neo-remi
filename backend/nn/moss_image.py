# https://github.com/Comfy-Org/ComfyUI/blob/v0.42.0/comfy/ldm/moss_image/model.py

from collections import OrderedDict
import weakref

import torch
import torch.nn as nn

from backend.args import dynamic_args
from backend.attention import attention_function
from backend.nn.flux import EmbedND, timestep_embedding
from backend.operations import main_stream_worker, weights_manual_cast
from backend.operations_mixed_precision import _fp16_linear_wanted, linear_input_act
from backend.quant_ops import QuantizedTensor, ck
from backend.utils import fp16_fix, resize_to_batch_size

_TEXT_EMBED_CACHE_MAX = 4
_ROPE_CACHE_MAX = 4


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else getattr(tensor, "_version", 0)


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _uses_fused_input_act(linear: nn.Module, x: torch.Tensor) -> bool:
    weight = getattr(linear, "weight", None)
    if getattr(linear, "_full_precision_mm", False):
        return False
    if isinstance(weight, QuantizedTensor):
        return weight._layout_cls == "TensorWiseINT8Layout" and not getattr(weight._params, "transposed", False)
    return _fp16_linear_wanted(x) and getattr(weight, "gguf_cls", None) is None


def modulated_rms_norm(x, norm, scale, shift):
    weight, _, stream = weights_manual_cast(norm, x)
    with main_stream_worker(weight, None, stream):
        return ck.rms_adaln(x, weight * (1 + scale) - 1, shift, norm.eps)


def gated_residual(x, y, gate):
    return x.addcmul_(y, gate)


def modulate(x, norm, mod, i, ref):
    if ref is None:
        return modulated_rms_norm(x, norm, mod[i + 1], mod[i])
    s, ref_mod = ref[0], ref[1]
    return torch.cat((modulated_rms_norm(x[:, :s], norm, mod[i + 1], mod[i]), modulated_rms_norm(x[:, s:], norm, ref_mod[i + 1], ref_mod[i])), dim=1)


def residual(x, y, mod, i, ref):
    if ref is None:
        return gated_residual(x, y, mod[i])
    s, ref_mod = ref[0], ref[1]
    gated_residual(x[:, :s], y[:, :s], mod[i])
    gated_residual(x[:, s:], y[:, s:], ref_mod[i])
    return x


def attention(q, k, v, q_norm, k_norm, pe, heads, transformer_options={}, ref_kv=None, kv_out=None):
    B, N, _ = q.shape
    q = q.view(B, N, heads, -1)
    k = k.view(B, N, heads, -1)
    q_scale, _, q_stream = weights_manual_cast(q_norm, q)
    k_scale, _, k_stream = weights_manual_cast(k_norm, k)
    with main_stream_worker(q_scale, None, q_stream), main_stream_worker(k_scale, None, k_stream):
        q, k = ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, q_norm.eps)
    q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
    if kv_out is not None:
        kv_out[:, 0].copy_(k)
        kv_out[:, 1].copy_(v)
    if ref_kv is not None:
        k = torch.cat((k, ref_kv[:, 0]), dim=1)
        v = torch.cat((v, ref_kv[:, 1]), dim=1)
    return attention_function(q, k, v, heads, transformer_options=transformer_options)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, pe, transformer_options={}):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, transformer_options))
        del q, k, v
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, num_heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads) for _ in range(2)])

    def forward(self, x, pe, transformer_options={}):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, pe, transformer_options)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, fused_qkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({name: nn.Linear(dim, 3 * dim if name == "qkv" else dim, bias=False) for name in names})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None):
        y = modulate(x, self.norm1, mod, 0, ref)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        del y
        out = attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, transformer_options, ref_kv, kv_out)
        del q, k, v
        x = residual(x, self.proj["o"](out), mod, 2, ref)
        del out
        y = modulate(x, self.norm2, mod, 3, ref)
        y = self.proj["ffn_up_gate"](y)
        ffn_down = self.proj["ffn_down"]
        if _uses_fused_input_act(ffn_down, y):
            ffn_out = linear_input_act(ffn_down, y, "situ")
        else:
            hidden, gate = y.chunk(2, dim=-1)
            ffn_out = ffn_down(situ(hidden, gate))
        z = residual(x, ffn_out, mod, 5, ref)
        del x, y, ffn_out
        return fp16_fix(z)


class MOSSImage(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, latent_channels=128, fused_qkv=False):
        super().__init__()
        self.latent_channels = latent_channels
        head_dim = dim // num_heads
        self.pe_embedder = EmbedND(dim=head_dim, theta=10000, axes_dim=[head_dim // 4] * 4)
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextEmbedder(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads, fused_qkv=fused_qkv) for _ in range(num_layers)])
        self.video_head_input = nn.Identity()  # hook point for Spectrum pre-head feature forecasting (0 params)
        self.final_norm_video = nn.RMSNorm(dim, eps=1e-6)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self._text_embed_cache: OrderedDict = OrderedDict()
        self._rope_cache: OrderedDict = OrderedDict()
        self._ref_kv_meta: tuple | None = None
        self._ref_kv_refs: tuple = ()

    def clear_caches(self) -> None:
        """Free step-level TextEmbedder / RoPE caches and invalidate cached ref KV."""

        self._text_embed_cache.clear()
        self._rope_cache.clear()
        self._ref_kv_meta = None
        self._ref_kv_refs = ()

    def _apply(self, fn):
        self.clear_caches()
        if dynamic_args.get("moss_ref_kv", None) is not None:
            dynamic_args.moss_ref_kv = None
        return super()._apply(fn)

    @staticmethod
    def _has_online_hooks(module: nn.Module) -> bool:
        for child in module.modules():
            if getattr(child, "weight_function", None) or getattr(child, "bias_function", None):
                return True
        return False

    @staticmethod
    def _module_state_token(module: nn.Module) -> tuple:
        params = tuple((id(p), p.data_ptr(), _tensor_version(p), p.device, p.dtype) for p in module.parameters())
        hooks = tuple(id(f) for child in module.modules() for f in (*getattr(child, "weight_function", ()), *getattr(child, "bias_function", ())))
        return params, hooks

    @staticmethod
    def _text_options_cacheable(transformer_options: dict) -> bool:
        if not transformer_options:
            return True
        if transformer_options.get("optimized_attention_override") is not None:
            return False
        patches = transformer_options.get("patches")
        return not (isinstance(patches, dict) and any(patches.values()))

    def _cached_rope(self, h: int, w: int, text_len: int, device, ref_shapes: tuple[tuple[int, int], ...], f_offset: int) -> torch.Tensor:
        key = (h, w, text_len, ref_shapes, f_offset, device)
        cached = self._rope_cache.get(key)
        if cached is not None:
            self._rope_cache.move_to_end(key)
            return cached
        pe = self.rope(h, w, text_len, device, ref_shapes, f_offset)
        self._rope_cache[key] = pe
        while len(self._rope_cache) > _ROPE_CACHE_MAX:
            self._rope_cache.popitem(last=False)
        return pe

    def _embed_text(self, context: torch.Tensor, text_pe: torch.Tensor, f_offset: int, transformer_options: dict) -> torch.Tensor:
        if self._has_online_hooks(self.text_embedder) or not self._text_options_cacheable(transformer_options):
            return self.text_embedder(context, text_pe, transformer_options)
        key = (
            id(context),
            context.data_ptr(),
            _tensor_version(context),
            tuple(context.shape),
            context.dtype,
            context.device,
            f_offset,
            self._module_state_token(self.text_embedder),
        )
        entry = self._text_embed_cache.get(key)
        if entry is not None:
            ref, txt = entry
            if ref() is context:
                self._text_embed_cache.move_to_end(key)
                return txt
            self._text_embed_cache.pop(key, None)
        txt = self.text_embedder(context, text_pe, transformer_options)
        try:
            ctx_ref = weakref.ref(context)
        except TypeError:
            return txt
        self._text_embed_cache[key] = (ctx_ref, txt)
        while len(self._text_embed_cache) > _TEXT_EMBED_CACHE_MAX:
            self._text_embed_cache.popitem(last=False)
        return txt

    def compute_video_head(self, video_tokens: torch.Tensor, timestep: torch.Tensor, out_shape: tuple[int, ...]) -> torch.Tensor:
        vec = self.time_embedding(timestep_embedding(timestep, 256).to(video_tokens.dtype))
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
        out = self.final_linear_video(modulated_rms_norm(video_tokens, self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(out_shape)

    def rope(self, h, w, text_len, device, ref_shapes=(), f_offset=0):
        n = h * w + text_len
        ids = torch.zeros(n + sum(rh * rw for rh, rw in ref_shapes), 4, device=device, dtype=torch.float32)
        ids[: h * w, 1] = text_len + f_offset
        ids[: h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
        ids[: h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
        ids[h * w : n, 1] = torch.arange(text_len, device=device, dtype=torch.float32) + f_offset
        for i, (rh, rw) in enumerate(ref_shapes):
            ref = ids[n : n + rh * rw]
            ref[:, 0] = i + 1
            ref[:, 2] = (torch.arange(rh, device=device, dtype=torch.float32) - (rh - 1) / 2).repeat_interleave(rw)
            ref[:, 3] = (torch.arange(rw, device=device, dtype=torch.float32) - (rw - 1) / 2).repeat(rh)
            n += rh * rw
        return self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

    def preprocess_refs(self, refs, transformer_options={}):
        batch = max(ref.shape[0] for ref in refs)
        x = torch.cat([self.video_embedder(resize_to_batch_size(ref, batch).flatten(2).transpose(1, 2)) for ref in refs], dim=1)
        pe = self.rope(0, 0, 0, x.device, [ref.shape[-2:] for ref in refs])
        vec = self.time_embedding(timestep_embedding(torch.zeros(1, device=x.device), 256).to(x.dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        kv = x.new_empty((x.shape[0], len(self.layers), 2) + x.shape[1:])
        for i, block in enumerate(self.layers):
            # One-shot KV fill runs once per generation: bypass any compiled wrapper so
            # torch.compile does not compile a one-off kv_out graph variant.
            eager_block = getattr(block, "_orig_mod", block)
            x = eager_block(x, mod, pe, transformer_options, None, None, kv[:, i])
        return kv

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()
        text_len = context.shape[1]

        method: str = dynamic_args.moss_reference_method
        ref_latents: list[torch.Tensor] = dynamic_args.ref_latents

        if ref_latents and method not in ("index", "index_timestep_zero", None):
            ref_meta = (
                tuple((id(r), r.data_ptr(), _tensor_version(r), tuple(r.shape)) for r in ref_latents),
                device,
                dtype,
                self._module_state_token(self.layers),
                self._module_state_token(self.video_embedder),
            )
            refs_alive = len(self._ref_kv_refs) == len(ref_latents) and all(w() is r for w, r in zip(self._ref_kv_refs, ref_latents))
            ref_kv = dynamic_args.get("moss_ref_kv", None)
            if (
                not isinstance(ref_kv, torch.Tensor)
                or not refs_alive
                or self._ref_kv_meta != ref_meta
                or ref_kv.device != device
                or ref_kv.dtype != dtype
            ):
                ref_kv = self.preprocess_refs([ref.to(device=device, dtype=dtype) for ref in ref_latents])
                dynamic_args.moss_ref_kv = ref_kv
                self._ref_kv_meta = ref_meta
                try:
                    self._ref_kv_refs = tuple(weakref.ref(r) for r in ref_latents)
                except TypeError:
                    self._ref_kv_refs = ()
            ref_kv = ref_kv.expand(B, -1, -1, -1, -1) if ref_kv.shape[0] == 1 and B > 1 else resize_to_batch_size(ref_kv, B)
            refs = []
        else:
            ref_kv = None
            refs = [resize_to_batch_size(ref.to(device=device, dtype=dtype), B) for ref in ref_latents]
        use_refs = bool(refs)

        f_offset = -text_len if ((use_refs or ref_kv is not None) and method != "index_timestep_zero_text_start") else 0
        ref_shapes = tuple(tuple(ref.shape[-2:]) for ref in refs)
        pe = self._cached_rope(H, W, text_len, device, ref_shapes, f_offset)

        vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

        ref = None
        if use_refs and method != "index":
            ref_vec = self.time_embedding(timestep_embedding(torch.zeros_like(timestep), 256).to(dtype))
            ref = (H * W + text_len, self.shared_block_modulation(ref_vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1))

        txt = self._embed_text(context, pe[:, H * W : H * W + text_len], f_offset, transformer_options)
        x = torch.cat([self.video_embedder(x.flatten(2).transpose(1, 2)), txt] + [self.video_embedder(r.flatten(2).transpose(1, 2)) for r in refs], dim=1)
        del txt, refs

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "single"
        for i, block in enumerate(self.layers):
            transformer_options["block_index"] = i
            ref_kv_i = None if ref_kv is None else ref_kv[:, i]
            if ("single_block", i) in blocks_replace:

                def block_wrap(args):
                    return {"img": block(args["img"], args["mod"], args["pe"], args["transformer_options"], args["ref"], args["ref_kv"])}

                x = blocks_replace[("single_block", i)]({"img": x, "vec": vec, "mod": mod, "pe": pe, "ref": ref, "ref_kv": ref_kv_i, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                x = block(x, mod, pe, transformer_options, ref, ref_kv_i)

        video_tokens = self.video_head_input(x[:, : H * W])
        out = self.final_linear_video(modulated_rms_norm(video_tokens, self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)
