# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/ldm/krea2/model.py
# https://github.com/lbouaraba/comfyui-krea2edit/blob/main/__init__.py

import weakref
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.args import dynamic_args
from backend.attention import attention_function
from backend.memory_management import cast_to
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import EmbedND, timestep_embedding
from backend.operations import repeat_kv_for_gqa
from backend.quant_ops import ck
from backend.utils import pad_to_patch_size


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else int(tensor._version)


def _broadcast_ref_batch(ref: torch.Tensor, bs: int) -> torch.Tensor:
    if ref.shape[0] == bs:
        return ref
    if ref.shape[0] == 1:
        return ref.expand(bs, -1, -1, -1)
    if bs % ref.shape[0] == 0:
        return ref.repeat(bs // ref.shape[0], 1, 1, 1)
    return ref[:1].expand(bs, -1, -1, -1)


def _imgids(bs: int, frame: int, h_: int, w_: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features))

    def fp32_weight(self, x):
        return cast_to(self.scale, dtype=torch.float32, device=x.device) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.fp32_weight(x).to(x.dtype), eps=self.eps)

    def modulated(self, x, scale, shift):
        if torch.is_grad_enabled():
            return torch.addcmul(shift, 1 + scale, self(x))
        # Fold the learned RMSNorm weight into the AdaLN scale before rounding.
        scale = (self.fp32_weight(x) * (1 + scale.float()) - 1).to(x.dtype)
        return ck.rms_adaln(x, scale, shift, self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()

        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = q.unflatten(-1, (self.heads, self.headdim))
        k = k.unflatten(-1, (self.kvheads, self.headdim))
        v = v.unflatten(-1, (self.kvheads, self.headdim))
        if freqs is not None and not torch.is_grad_enabled():
            qnorm, knorm = self.qknorm.qnorm, self.qknorm.knorm
            q, k = ck.rms_rope_(q, k, freqs.transpose(1, 2), qnorm.fp32_weight(q), knorm.fp32_weight(k), qnorm.eps)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        else:
            q, k = self.qknorm(q.transpose(1, 2), k.transpose(1, 2))
            if freqs is not None:
                q, k = ck.apply_rope(q, k, freqs)
        k, v = repeat_kv_for_gqa(k, v.transpose(1, 2), self.heads, 1)
        out = attention_function(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(2, dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(6 * dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None, transformer_options={}):
        x = x + self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = self.projector(x.unflatten(0, (b, l)).transpose(2, 3)).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x.addcmul_(pregate, self.attn(self.prenorm.modulated(x, prescale, preshift), freqs, mask, transformer_options=transformer_options))
        x.addcmul_(postgate, self.mlp(self.postnorm.modulated(x, postscale, postshift)))
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()

        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        return self.linear(self.norm.modulated(x, scale, shift))


class SingleStreamDiT(nn.Module):
    def __init__(
        self,
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        bias=False,
        theta=1e3,
        txtlayers=12,
        txtheads=20,
        txtkvheads=20,
        **kwargs,
    ):
        super().__init__()

        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last_input = nn.Identity()
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )
        # Step-invariant zero-loss caches (cleared on sampling_cleanup).
        # Bounded to 4 entries so CFG cond + uncond both stay warm across steps.
        self._txt_cache: dict[tuple, tuple[weakref.ReferenceType, tuple[weakref.ReferenceType, ...], torch.Tensor]] = {}
        self._rope_cache: dict[tuple, torch.Tensor] = {}
        self._ref_cache: dict[tuple, tuple[tuple[weakref.ReferenceType, ...], weakref.ReferenceType, list[torch.Tensor], list[tuple[int, int]]]] = {}

    def clear_caches(self) -> None:
        self._txt_cache.clear()
        self._rope_cache.clear()
        self._ref_cache.clear()

    def _encode_context_cached(self, context: torch.Tensor, target_dtype: torch.dtype, transformer_options: dict) -> torch.Tensor:
        patches = transformer_options.get("patches")
        has_dynamic_override = bool(transformer_options.get("optimized_attention_override")) or (isinstance(patches, dict) and any(patches.values()))
        if torch.is_grad_enabled() or has_dynamic_override:
            return self.txtmlp(self.txtfusion(context, mask=None, transformer_options=transformer_options))

        proj_w = self.txtfusion.projector.weight
        lw0_w = self.txtfusion.layerwise_blocks[0].attn.wq.weight
        mlp_w = self.txtmlp[1].weight
        cache_key = (
            context.data_ptr(),
            tuple(context.shape),
            context.dtype,
            context.device,
            _tensor_version(context),
            target_dtype,
            proj_w.data_ptr(),
            _tensor_version(proj_w),
            lw0_w.data_ptr(),
            _tensor_version(lw0_w),
            mlp_w.data_ptr(),
            _tensor_version(mlp_w),
        )
        entry = self._txt_cache.get(cache_key)
        if entry is not None and entry[0]() is context and entry[1][0]() is proj_w and entry[1][1]() is lw0_w and entry[1][2]() is mlp_w:
            return entry[2]

        encoded = self.txtmlp(self.txtfusion(context, mask=None, transformer_options=transformer_options))
        if len(self._txt_cache) >= 4:
            self._txt_cache.pop(next(iter(self._txt_cache)))
        self._txt_cache[cache_key] = (
            weakref.ref(context),
            (weakref.ref(proj_w), weakref.ref(lw0_w), weakref.ref(mlp_w)),
            encoded,
        )
        return encoded

    def _encode_refs_cached(
        self,
        ref_latents: list[torch.Tensor],
        bs: int,
        H: int,
        W: int,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        patch = self.patch
        first_w = self.first.weight
        cache_key = (
            tuple((r.data_ptr(), tuple(r.shape), r.dtype, r.device, _tensor_version(r)) for r in ref_latents),
            bs,
            H,
            W,
            target_dtype,
            target_device,
            first_w.data_ptr(),
            _tensor_version(first_w),
        )
        if not torch.is_grad_enabled():
            entry = self._ref_cache.get(cache_key)
            if entry is not None and entry[1]() is first_w and all(ref_w() is orig for ref_w, orig in zip(entry[0], ref_latents)):
                return entry[2], entry[3]

        refs: list[torch.Tensor] = []
        ref_grids: list[tuple[int, int]] = []
        for ref in ref_latents:
            ref_b = _broadcast_ref_batch(ref, bs)
            if ref_b.shape[-2:] != (H, W):
                ref_b = adaptive_resize(ref_b, W, H, "area", "center")
            ref_padded = pad_to_patch_size(ref_b.to(device=target_device, dtype=target_dtype), (patch, patch), padding_mode="replicate")
            ref_grids.append((ref_padded.shape[-2] // patch, ref_padded.shape[-1] // patch))
            ref_tokens = rearrange(ref_padded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
            refs.append(self.first(ref_tokens))

        if not torch.is_grad_enabled():
            if len(self._ref_cache) >= 4:
                self._ref_cache.pop(next(iter(self._ref_cache)))
            self._ref_cache[cache_key] = (tuple(weakref.ref(r) for r in ref_latents), weakref.ref(first_w), refs, ref_grids)
        return refs, ref_grids

    def _compute_freqs_cached(
        self,
        bs: int,
        txtlen: int,
        h_: int,
        w_: int,
        ref_grids: list[tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (bs, txtlen, h_, w_, tuple(ref_grids), device)
        if not torch.is_grad_enabled():
            cached = self._rope_cache.get(cache_key)
            if cached is not None:
                return cached

        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
        if ref_grids:
            refids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(ref_grids)]
            imgids = _imgids(bs, 0, h_, w_, device)
            pos = torch.cat((txtpos, *refids, imgids), dim=1)
        else:
            imgpos = _imgids(bs, 0, h_, w_, device)
            pos = torch.cat((txtpos, imgpos), dim=1)
        freqs = self.pe_embedder(pos)

        if not torch.is_grad_enabled():
            if len(self._rope_cache) >= 4:
                self._rope_cache.pop(next(iter(self._rope_cache)))
            self._rope_cache[cache_key] = freqs
        return freqs

    def compute_last(self, feature: torch.Tensor, timesteps: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
        """Run timestep embedding + final head + unpatchify on a cached/forecast pre-head feature (Spectrum)."""
        H_orig, W_orig = int(orig_shape[-2]), int(orig_shape[-1])
        patch = self.patch
        h_ = (H_orig + patch - 1) // patch
        w_ = (W_orig + patch - 1) // patch
        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(feature.dtype))
        out = self.last(feature, t)
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]
        return out.unsqueeze(2)

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        x = x.squeeze(2)
        bs, c, H_orig, W_orig = x.shape
        patch = self.patch

        x = pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        ref_latents: list[torch.Tensor] = dynamic_args.ref_latents
        _edit = bool(ref_latents)

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        img = self.first(img)

        if _edit:
            refs, ref_grids = self._encode_refs_cached(ref_latents, bs, H, W, x.dtype, x.device)
        else:
            refs, ref_grids = [], []

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self._encode_context_cached(context, img.dtype, transformer_options)

        txtlen, imglen = context.shape[1], img.shape[1]
        if _edit:
            reflen = sum(ref.shape[1] for ref in refs)
            combined = torch.cat((context, *refs, img), dim=1)
        else:
            reflen = 0
            combined = torch.cat((context, img), dim=1)

        freqs = self._compute_freqs_cached(bs, txtlen, h_, w_, ref_grids, combined.device)
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})

        for i, block in enumerate(self.blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], args["vec"], args["pe"], args["mask"], transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("single_block", i)](
                    {"img": combined, "vec": tvec, "pe": freqs, "mask": None, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                combined = out["img"]
            else:
                combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

        img_tokens = self.last_input(combined[:, txtlen + reflen : txtlen + reflen + imglen])
        out = self.last(img_tokens, t)
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]

        return out.unsqueeze(2)
