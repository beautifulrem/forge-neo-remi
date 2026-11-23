# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/lumina/model.py
# Reference: https://github.com/Alpha-VLLM/Lumina-Image-2.0

import math
import weakref
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from backend.args import dynamic_args
from backend.attention import attention_function
from backend.nn.flux import EmbedND
from backend.operations import main_stream_worker, weights_manual_cast
from backend.quant_ops import ck
from backend.utils import fp16_fix as clamp_fp16
from backend.utils import pad_to_patch_size

_CONTEXT_CACHE_MAX = 4
_ROPE_CACHE_MAX = 4


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else int(getattr(tensor, "_version", 0))


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256, output_size=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size or hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t, dtype, **kwargs):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class JointAttention(nn.Module):

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool, out_bias: bool = False):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=out_bias)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor, transformer_options={}) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.qk_norm:
            q_scale, _, q_offload_stream = weights_manual_cast(self.q_norm, xq)
            k_scale, _, k_offload_stream = weights_manual_cast(self.k_norm, xk)
            epsilon = self.q_norm.eps if self.q_norm.eps is not None else torch.finfo(torch.float32).eps
            with main_stream_worker(q_scale, None, q_offload_stream), main_stream_worker(k_scale, None, k_offload_stream):
                if self.n_local_heads == self.n_local_kv_heads:
                    xq, xk = ck.rms_rope(xq, xk, freqs_cis, q_scale, k_scale, epsilon)
                else:
                    xq = ck.rms_rope1(xq, freqs_cis, q_scale, epsilon)
                    xk = ck.rms_rope1(xk, freqs_cis, k_scale, epsilon)
        else:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
            xq, xk = ck.apply_rope(xq, xk, freqs_cis)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = attention_function(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2), self.n_local_heads, x_mask, skip_reshape=True, transformer_options=transformer_options)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return clamp_fp16(F.silu(x1) * x3)

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int, multiple_of: int, ffn_dim_multiplier: float, norm_eps: float, qk_norm: bool, modulation=True, z_image_modulation=False, attn_out_bias=False):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm, out_bias=attn_out_bias)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.modulation = modulation
        if modulation:
            if z_image_modulation:
                self.adaLN_modulation = nn.Sequential(
                    nn.Linear(min(dim, 256), 4 * dim, bias=True),
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(min(dim, 1024), 4 * dim, bias=True),
                )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor, adaln_input: torch.Tensor = None, transformer_options={}):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                clamp_fp16(
                    self.attention(
                        modulate(self.attention_norm1(x), scale_msa),
                        x_mask,
                        freqs_cis,
                        transformer_options=transformer_options,
                    )
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                clamp_fp16(
                    self.feed_forward(
                        modulate(self.ffn_norm1(x), scale_mlp),
                    )
                )
            )
        else:
            assert adaln_input is None
            x = x + self.attention_norm2(
                clamp_fp16(
                    self.attention(
                        self.attention_norm1(x),
                        x_mask,
                        freqs_cis,
                        transformer_options=transformer_options,
                    )
                )
            )
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels, z_image_modulation=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        if z_image_modulation:
            min_mod = 256
        else:
            min_mod = 1024

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, min_mod), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class NextDiT(nn.Module):

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: int = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 4.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (1, 512, 512),
        rope_theta: float = 10000.0,
        z_image_modulation: bool = False,
        time_scale: float = 1.0,
        pad_tokens_multiple: int = None,
        clip_text_dim=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple

        self.x_embedder = nn.Linear(in_features=patch_size * patch_size * in_channels, out_features=dim, bias=True)

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_image_modulation=z_image_modulation,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, 1024), output_size=min(dim, 256) if z_image_modulation else None)
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=True),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.clip_text_pooled_proj = None

        if clip_text_dim is not None:
            self.clip_text_dim = clip_text_dim
            self.clip_text_pooled_proj = nn.Sequential(
                nn.RMSNorm(clip_text_dim, eps=norm_eps, elementwise_affine=True),
                nn.Linear(clip_text_dim, clip_text_dim, bias=True),
            )
            self.time_text_embed = nn.Sequential(
                nn.SiLU(),
                nn.Linear(min(dim, 1024) + clip_text_dim, min(dim, 1024), bias=True),
            )

        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    z_image_modulation=z_image_modulation,
                    attn_out_bias=False,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.zimage_final_input = nn.Identity()  # Spectrum pre-head capture seam (0 params)
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels, z_image_modulation=z_image_modulation)

        if self.pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=axes_dims)
        self.dim = dim
        self.n_heads = n_heads
        self.z_image_modulation = bool(z_image_modulation)
        self._context_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._rope_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._last_cap_len: int = 0
        self._last_img_size: list[tuple[int, int]] = []
        self._last_cap_size: list[int] = []

    def clear_caches(self) -> None:
        """Free cross-step refined context and RoPE caches."""

        self._context_cache.clear()
        self._rope_cache.clear()

    def _apply(self, fn):
        self.clear_caches()
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
    def _options_cacheable(transformer_options: dict) -> bool:
        if torch.is_grad_enabled():
            return False
        if not transformer_options:
            return True
        if transformer_options.get("optimized_attention_override") is not None:
            return False
        patches = transformer_options.get("patches")
        return not (isinstance(patches, dict) and any(patches.values()))

    def _cached_rope(
        self,
        bsz: int,
        cap_len: int,
        h_tokens: int,
        w_tokens: int,
        pad_extra: int,
        ref_grids: tuple[tuple[int, int], ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (bsz, cap_len, h_tokens, w_tokens, pad_extra, ref_grids, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is not None:
            self._rope_cache.move_to_end(key)
            return cached

        cap_pos_ids = torch.zeros(bsz, cap_len, 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_len, dtype=torch.float32, device=device) + 1.0

        x_pos_ids = torch.zeros((bsz, h_tokens * w_tokens, 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = cap_len + 1
        x_pos_ids[:, :, 1] = torch.arange(h_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, w_tokens).flatten()
        x_pos_ids[:, :, 2] = torch.arange(w_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(h_tokens, 1).flatten()
        if pad_extra > 0:
            x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))

        id_parts = [cap_pos_ids, x_pos_ids]
        for ref_idx, (rh, rw) in enumerate(ref_grids):
            r_ids = torch.zeros((bsz, rh * rw, 3), dtype=torch.float32, device=device)
            r_ids[:, :, 0] = cap_len + 2 + ref_idx
            r_ids[:, :, 1] = torch.arange(rh, dtype=torch.float32, device=device).view(-1, 1).repeat(1, rw).flatten()
            r_ids[:, :, 2] = torch.arange(rw, dtype=torch.float32, device=device).view(1, -1).repeat(rh, 1).flatten()
            id_parts.append(r_ids)

        freqs_cis = self.rope_embedder(torch.cat(id_parts, dim=1)).movedim(1, 2).to(dtype)
        self._rope_cache[key] = freqs_cis
        while len(self._rope_cache) > _ROPE_CACHE_MAX:
            self._rope_cache.popitem(last=False)
        return freqs_cis

    def _refine_context(
        self,
        raw_context: torch.Tensor,
        cap_mask: torch.Tensor | None,
        cap_freqs: torch.Tensor,
        transformer_options: dict,
    ) -> torch.Tensor:
        cacheable = (
            cap_mask is None
            and self._options_cacheable(transformer_options)
            and not self._has_online_hooks(self.cap_embedder)
            and not self._has_online_hooks(self.context_refiner)
        )
        if cacheable:
            key = (
                id(raw_context),
                raw_context.data_ptr(),
                _tensor_version(raw_context),
                tuple(raw_context.shape),
                raw_context.dtype,
                raw_context.device,
                tuple(cap_freqs.shape),
                self._module_state_token(self.cap_embedder),
                self._module_state_token(self.context_refiner),
            )
            entry = self._context_cache.get(key)
            if entry is not None:
                wref, cached = entry
                if wref() is raw_context:
                    self._context_cache.move_to_end(key)
                    return cached
                self._context_cache.pop(key, None)

        cap_feats = self.cap_embedder(raw_context)
        if self.pad_tokens_multiple is not None:
            pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
            if pad_extra:
                cap_feats = torch.cat(
                    (cap_feats, self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0).repeat(cap_feats.shape[0], pad_extra, 1)),
                    dim=1,
                )
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs, transformer_options=transformer_options)

        if cacheable:
            try:
                wref = weakref.ref(raw_context)
            except TypeError:
                return cap_feats
            self._context_cache[key] = (wref, cap_feats)
            while len(self._context_cache) > _CONTEXT_CACHE_MAX:
                self._context_cache.popitem(last=False)
        return cap_feats

    def compute_zimage_final(
        self,
        feature: torch.Tensor,
        timestep: torch.Tensor,
        input_shape: tuple[int, ...],
        *,
        img_size: list[tuple[int, int]] | None = None,
        cap_size: list[int] | None = None,
        clip_text_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run timestep embedding + ``final_layer`` + ``unpatchify`` + negation
        on a cached/forecast pre-head feature (Spectrum)."""

        bs = feature.shape[0]
        h, w = int(input_shape[-2]), int(input_shape[-1])
        p = self.patch_size
        h_pad = ((h + p - 1) // p) * p
        w_pad = ((w + p - 1) // p) * p

        t_in = 1.0 - timestep.reshape(-1)
        if t_in.shape[0] == 1 and bs > 1:
            t_in = t_in.expand(bs)
        t_emb = self.t_embedder(t_in * self.time_scale, dtype=feature.dtype)
        adaln_input = t_emb
        if self.clip_text_pooled_proj is not None:
            if clip_text_pooled is not None:
                pooled = self.clip_text_pooled_proj(clip_text_pooled.to(device=feature.device, dtype=feature.dtype))
            else:
                pooled = torch.zeros((1, self.clip_text_dim), device=feature.device, dtype=feature.dtype)
            adaln_input = self.time_text_embed(torch.cat((t_emb, pooled), dim=-1))

        out = self.final_layer(feature, adaln_input)
        sizes = img_size if img_size is not None else [(h_pad, w_pad)] * bs
        if cap_size is not None:
            caps = cap_size
        elif self._last_cap_size and len(self._last_cap_size) == bs:
            caps = self._last_cap_size
        else:
            caps = [self._last_cap_len] * bs
        unpatched = self.unpatchify(out, sizes, caps, return_tensor=True)[:, :, :h, :w]
        return -unpatched

    def unpatchify(self, x: torch.Tensor, img_size: list[tuple[int, int]], cap_size: list[int], return_tensor=False) -> list[torch.Tensor]:
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(x[i][begin:end].view(H // pH, W // pW, pH, pW, self.out_channels).permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2))

        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        return imgs

    def patchify_and_embed(
        self,
        x: list[torch.Tensor] | torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        t: torch.Tensor,
        num_tokens,
        transformer_options={},
        *,
        raw_context: torch.Tensor | None = None,
        ref_latents: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[int], torch.Tensor]:
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device
        dtype = x[0].dtype

        raw_cap_len = raw_context.shape[1] if raw_context is not None else cap_feats.shape[1]
        cap_pad_extra = (-raw_cap_len) % self.pad_tokens_multiple if self.pad_tokens_multiple is not None else 0
        cap_len = raw_cap_len + cap_pad_extra

        B, C, H, W = x.shape
        x_tok = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))
        H_tokens, W_tokens = H // pH, W // pW
        x_pad_extra = (-x_tok.shape[1]) % self.pad_tokens_multiple if self.pad_tokens_multiple is not None else 0
        if x_pad_extra:
            x_tok = torch.cat((x_tok, self.x_pad_token.to(device=x_tok.device, dtype=dtype, copy=True).unsqueeze(0).repeat(x_tok.shape[0], x_pad_extra, 1)), dim=1)

        # Optional Z-Image-Omni / Edit reference latents
        ref_tokens_list: list[torch.Tensor] = []
        ref_grids: list[tuple[int, int]] = []
        if ref_latents:
            for ref in ref_latents:
                r = pad_to_patch_size(ref.to(device=device, dtype=dtype), (pH, pW))
                if r.shape[0] != B:
                    r = r[:1].expand(B, -1, -1, -1) if r.shape[0] == 1 else r.repeat(B // r.shape[0], 1, 1, 1)
                _, rc, rh, rw = r.shape
                r_tok = self.x_embedder(r.view(B, rc, rh // pH, pH, rw // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))
                ref_tokens_list.append(r_tok)
                ref_grids.append((rh // pH, rw // pW))

        freqs_cis = self._cached_rope(bsz, cap_len, H_tokens, W_tokens, x_pad_extra, tuple(ref_grids), device, dtype)

        # Refine context (cached across denoising steps when raw_context is provided)
        if raw_context is not None:
            cap_feats = self._refine_context(raw_context, cap_mask, freqs_cis[:, :cap_len], transformer_options)
        else:
            if cap_pad_extra:
                cap_feats = torch.cat(
                    (cap_feats, self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0).repeat(cap_feats.shape[0], cap_pad_extra, 1)),
                    dim=1,
                )
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_len], transformer_options=transformer_options)

        x_end = cap_len + x_tok.shape[1]
        padded_img_mask = None
        for layer in self.noise_refiner:
            x_tok = layer(x_tok, padded_img_mask, freqs_cis[:, cap_len:x_end], t, transformer_options=transformer_options)

        parts = [cap_feats, x_tok]
        if ref_tokens_list:
            # Clean reference latents enter noise_refiner at t=0 (clean image embedding)
            t_zero = self.t_embedder(torch.zeros((B,), device=device, dtype=dtype), dtype=dtype)
            offset = x_end
            for r_tok in ref_tokens_list:
                r_end = offset + r_tok.shape[1]
                for layer in self.noise_refiner:
                    r_tok = layer(r_tok, None, freqs_cis[:, offset:r_end], t_zero, transformer_options=transformer_options)
                parts.append(r_tok)
                offset = r_end

        padded_full_embed = torch.cat(parts, dim=1)
        mask = None
        img_sizes = [(H, W)] * bsz
        l_effective_cap_len = [cap_feats.shape[1]] * bsz
        self._last_cap_len = int(cap_feats.shape[1])
        self._last_img_size = img_sizes
        self._last_cap_size = l_effective_cap_len
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    def forward(self, x, timesteps, context, num_tokens=None, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        t = 1.0 - timesteps
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = pad_to_patch_size(x, (self.patch_size, self.patch_size))

        t = self.t_embedder(t * self.time_scale, dtype=x.dtype)  # (N, D)
        adaln_input = t

        if self.clip_text_pooled_proj is not None:
            pooled = kwargs.get("clip_text_pooled", None)
            if pooled is not None:
                pooled = self.clip_text_pooled_proj(pooled)
            else:
                pooled = torch.zeros((1, self.clip_text_dim), device=x.device, dtype=x.dtype)

            adaln_input = self.time_text_embed(torch.cat((t, pooled), dim=-1))

        if ref_latents is None and self.z_image_modulation and getattr(dynamic_args, "zimage", False):
            ref_latents = list(dynamic_args.ref_latents or [])

        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
            x,
            context,
            cap_mask,
            t,
            num_tokens,
            transformer_options=transformer_options,
            raw_context=context,
            ref_latents=ref_latents or None,
        )
        freqs_cis = freqs_cis.to(x.device)

        opts = dict(transformer_options) if isinstance(transformer_options, dict) else {}
        blocks_replace = opts.get("patches_replace", {}).get("dit", {})
        opts["total_blocks"] = len(self.layers)
        opts["block_type"] = "single"
        for i, layer in enumerate(self.layers):
            opts["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": layer(args["img"], mask, args["pe"], args["vec"], transformer_options=args["transformer_options"])}

                x = blocks_replace[("single_block", i)]({"img": x, "vec": adaln_input, "pe": freqs_cis, "transformer_options": opts}, {"original_block": block_wrap})["img"]
            else:
                x = layer(x, mask, freqs_cis, adaln_input, transformer_options=opts)

        x = self.zimage_final_input(x)
        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]

        return -x
