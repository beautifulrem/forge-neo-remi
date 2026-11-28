"""FLUX.3 Image Klein (Black Forest Labs, 2026) diffusion transformer.

Three token modes share one joint sequence: the noisy image tokens, the
optional VAE-encoded reference image tokens (``image_cond``) and the text
context (stacked Qwen3-VL-4B hidden states, 20480 channels). Each mode first
runs through its own per-mode blocks (self-attention within the mode only),
then image, references and text are concatenated for the joint single blocks;
only the image tokens reach the output layer. The image tokens are modulated
by the flow timestep, references and text by timestep 0, so their per-mode
blocks depend on the conditioning alone and are computed once per prompt
(cached across steps). A flow-map checkpoint (the step-distilled Klein)
additionally embeds the timestep each step jumps to (``ref_time_in``).

Module and parameter names follow the BFL / ComfyUI checkpoint layout
(``content_mode_blocks.image.0.attn_out.weight``, ...); Diffusers checkpoints
are converted by ``backend/nn/flux3_diffusers.py``.

References: https://github.com/black-forest-labs/flux-3-image-klein,
ComfyUI ``comfy/ldm/flux3/model.py``, Diffusers ``transformer_flux3.py``.
"""

from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F
from torch import nn

from backend.args import dynamic_args
from backend.attention import attention_function
from backend.nn.flux import EmbedND, MLPEmbedder, timestep_embedding
from backend.quant_ops import ck

CONTENT_MODES = ("image", "image_cond")
MODES = (*CONTENT_MODES, "txt")


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else int(tensor._version)


def _repeat_to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if batch % tensor.shape[0]:
        raise ValueError(f"FLUX.3: cannot broadcast a batch of {tensor.shape[0]} to {batch}")
    return tensor.repeat(batch // tensor.shape[0], *([1] * (tensor.ndim - 1)))


def flow_map_target(sampling_sigmas: torch.Tensor | None, timestep: torch.Tensor) -> torch.Tensor:
    """The flow time a step-distilled (flow-map) step jumps to: the schedule's
    next sigma after the current one, 0 after the last (and without a schedule)."""

    current = timestep.reshape(-1)[:1].float()
    if sampling_sigmas is None or not torch.is_tensor(sampling_sigmas) or sampling_sigmas.numel() == 0:
        return torch.zeros_like(current)
    sigmas = sampling_sigmas.reshape(-1).to(device=current.device, dtype=torch.float32)
    i = int((sigmas - current).abs().argmin())
    if i + 1 >= sigmas.numel():
        return torch.zeros_like(current)
    return sigmas[i + 1 : i + 2]


class Flux3Modulation(nn.Module):
    """One (shift, scale, gate) set from the SiLU of a (per-sample) time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        shift, scale, gate = self.lin(F.silu(vec)).chunk(3, dim=-1)
        return shift, scale, gate


class Flux3QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim, eps=1e-6)
        self.key_norm = nn.RMSNorm(dim, eps=1e-6)


class Flux3Block(nn.Module):
    """Parallel attention + SwiGLU MLP under one adaLN and one gate: the per-mode
    blocks and the joint single blocks share it. ``spans`` are ``(start, end,
    (shift, scale, gate))`` row ranges that partition the sequence."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 3.0, fused_qkv: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.fused_qkv = fused_qkv
        mlp_hidden = int(hidden_size * mlp_ratio)
        if fused_qkv:
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * mlp_hidden, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden_size, bias=False)
        self.norm = Flux3QKNorm(self.head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, spans, pe: torch.Tensor, transformer_options: dict | None = None) -> torch.Tensor:
        if transformer_options is None:
            transformer_options = {}
        B, N, _ = x.shape
        normed = self.pre_norm(x)
        if len(spans) == 1:
            _, _, (shift, scale, _) = spans[0]
            x_mod = torch.addcmul(shift, normed, 1 + scale)
        else:
            x_mod = torch.cat([torch.addcmul(shift, normed[:, s:e], 1 + scale) for s, e, (shift, scale, _) in spans], dim=1)
        del normed

        if self.fused_qkv:
            q, k, v = self.qkv(x_mod).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x_mod), self.k_proj(x_mod), self.v_proj(x_mod)
        mlp = self.mlp_in(x_mod)
        del x_mod

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # QK norm and RoPE in fp32 with one cast back (the BFL / Diffusers recipe)
        q = self.norm.query_norm(q.float())
        k = self.norm.key_norm(k.float())
        q, k = ck.apply_rope(q, k, pe)
        attn = attention_function(q.to(v.dtype), k.to(v.dtype), v, self.num_heads, skip_reshape=True, transformer_options=transformer_options)

        a, b = mlp.chunk(2, dim=-1)
        out = self.attn_out(attn) + self.mlp_out(F.silu(a) * b)
        del attn, mlp, a, b

        if len(spans) == 1:
            _, _, (_, _, gate) = spans[0]
            return x + out * gate
        return x + torch.cat([out[:, s:e] * gate for s, e, (_, _, gate) in spans], dim=1)


class Flux3LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(torch.addcmul(shift, self.norm_final(x), 1 + scale))


class Flux3(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads: int = 24,
        depth: int = 5,
        depth_single_blocks: int = 28,
        fused_qkv: bool = False,
        flow_map: bool = False,
        in_channels: int = 128,
        context_in_dim: int = 20480,
        mlp_ratio: float = 3.0,
        theta: int = 10000,
        axes_dim: tuple[int, ...] = (32, 32, 32, 32),
        image_rope_scale: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.flow_map = bool(flow_map)
        self.image_rope_scale = float(image_rope_scale)

        self.pe_embedder = EmbedND(dim=hidden_size // num_heads, theta=theta, axes_dim=list(axes_dim))
        self.emb_in = nn.ModuleDict({mode: nn.Linear(in_channels, hidden_size, bias=False) for mode in CONTENT_MODES})
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.time_in = MLPEmbedder(256, hidden_size, bias=False)
        if self.flow_map:
            self.ref_time_in = MLPEmbedder(256, hidden_size, bias=False)

        self.early_stream_modulations = nn.ModuleDict({mode: Flux3Modulation(hidden_size) for mode in MODES})
        self.single_stream_modulations = nn.ModuleDict({mode: Flux3Modulation(hidden_size) for mode in MODES})

        def make_block():
            return Flux3Block(hidden_size, num_heads, mlp_ratio, fused_qkv=fused_qkv)

        self.content_mode_blocks = nn.ModuleDict({mode: nn.ModuleList([make_block() for _ in range(depth)]) for mode in CONTENT_MODES})
        self.txt_mode_blocks = nn.ModuleList([make_block() for _ in range(depth)])
        self.single_blocks = nn.ModuleList([make_block() for _ in range(depth_single_blocks)])

        self.final_input = nn.Identity()  # Spectrum pre-head capture seam
        self.final_layer = nn.ModuleDict({"image": Flux3LastLayer(hidden_size, in_channels)})

        # BFL checkpoints name the QK-norm weights ``scale``
        self._register_load_state_dict_pre_hook(self._rename_norm_scale)

        self._cond_cache: dict[tuple, tuple[weakref.ReferenceType, tuple[weakref.ReferenceType, ...], torch.Tensor, torch.Tensor | None]] = {}
        self._last_sampling_sigmas: torch.Tensor | None = None

    @staticmethod
    def _rename_norm_scale(state_dict, prefix, *args, **kwargs):
        for key in [k for k in state_dict if k.startswith(prefix) and k.endswith(("query_norm.scale", "key_norm.scale"))]:
            state_dict[key[: -len("scale")] + "weight"] = state_dict.pop(key)

    def clear_caches(self) -> None:
        self._cond_cache.clear()
        self._last_sampling_sigmas = None

    # region position ids

    def image_ids(self, h: int, w: int, index: float, device) -> torch.Tensor:
        """(t, row, col, l) ids; references are told apart on the l axis."""

        ids = torch.zeros(h, w, 4, device=device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] * self.image_rope_scale
        ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] * self.image_rope_scale
        ids[..., 3] = index
        return ids.flatten(0, 1)

    def ref_ids(self, ref_shapes, device) -> torch.Tensor:
        return torch.cat([self.image_ids(h, w, -1 - i, device) for i, (h, w) in enumerate(ref_shapes)])

    @staticmethod
    def text_ids(length: int, device) -> torch.Tensor:
        ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
        ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
        return ids

    def rope(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pe_embedder(ids.unsqueeze(0))

    # endregion

    def _time_vec(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.time_in(timestep_embedding(timestep, 256).to(dtype))

    def _target_vec(self, timestep: torch.Tensor, transformer_options: dict, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.flow_map:
            return None
        sigmas = transformer_options.get("sampling_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is not None:
            self._last_sampling_sigmas = sigmas
        target = flow_map_target(sigmas, timestep).to(timestep.device)
        return self.ref_time_in(timestep_embedding(target, 256).to(dtype))

    def _cond_modules(self):
        yield self.txt_in
        yield self.emb_in["image_cond"]
        yield from self.time_in.modules()
        yield self.early_stream_modulations["txt"]
        yield self.early_stream_modulations["image_cond"]
        for block in (*self.txt_mode_blocks, *self.content_mode_blocks["image_cond"]):
            yield from block.modules()

    def _cond_cacheable(self, transformer_options: dict) -> bool:
        if self.flow_map or torch.is_grad_enabled():
            return False
        patches = transformer_options.get("patches") if isinstance(transformer_options, dict) else None
        if isinstance(patches, dict) and any(patches.values()):
            return False
        if isinstance(transformer_options, dict) and transformer_options.get("optimized_attention_override"):
            return False
        # online LoRA patches the weights at call time: nothing to key the cache on
        return not any(getattr(module, "weight_function", None) or getattr(module, "bias_function", None) for module in self._cond_modules())

    def _cond_weight_key(self) -> tuple:
        probes = (self.txt_in.weight, self.emb_in["image_cond"].weight, self.time_in.in_layer.weight)
        block = self.txt_mode_blocks[0] if len(self.txt_mode_blocks) else None
        if block is not None:
            probes += (block.mlp_in.weight, block.qkv.weight if block.fused_qkv else block.q_proj.weight)
        return tuple((w.data_ptr(), _tensor_version(w)) for w in probes)

    def preprocess_conds(self, context: torch.Tensor, ref_latents: list[torch.Tensor] | None, vec_target: torch.Tensor | None, transformer_options: dict | None = None):
        """The per-mode blocks of the text and the references (timestep 0): they
        depend on the conditioning alone (and, for a flow map, the target time)."""

        if transformer_options is None:
            transformer_options = {}

        device, dtype = context.device, context.dtype
        if ref_latents:
            batch = max([context.shape[0]] + [ref.shape[0] for ref in ref_latents])
            context = _repeat_to_batch(context, batch)
            ref_latents = [_repeat_to_batch(ref, batch) for ref in ref_latents]
        vec = self._time_vec(torch.zeros(1, device=device, dtype=torch.float32), dtype)
        if vec_target is not None:
            vec = vec + vec_target

        txt = self.txt_in(context)
        mod = self.early_stream_modulations["txt"](vec)
        pe = self.rope(self.text_ids(txt.shape[1], device))
        for block in self.txt_mode_blocks:
            txt = block(txt, [(0, txt.shape[1], mod)], pe, transformer_options)

        refs = None
        if ref_latents:
            refs = self.emb_in["image_cond"](torch.cat([ref.flatten(2).transpose(1, 2) for ref in ref_latents], dim=1))
            mod = self.early_stream_modulations["image_cond"](vec)
            pe = self.rope(self.ref_ids([tuple(ref.shape[-2:]) for ref in ref_latents], device))
            for block in self.content_mode_blocks["image_cond"]:
                refs = block(refs, [(0, refs.shape[1], mod)], pe, transformer_options)
        return txt, refs

    def _conds(self, context, ref_latents, vec_target, transformer_options):
        if not self._cond_cacheable(transformer_options):
            return self.preprocess_conds(context, ref_latents, vec_target, transformer_options)
        key = (
            context.data_ptr(),
            tuple(context.shape),
            context.dtype,
            context.device,
            _tensor_version(context),
            tuple((r.data_ptr(), tuple(r.shape), r.dtype, r.device, _tensor_version(r)) for r in ref_latents or ()),
            self._cond_weight_key(),
        )
        entry = self._cond_cache.get(key)
        if entry is not None and entry[0]() is context and all(ref() is orig for ref, orig in zip(entry[1], ref_latents or (), strict=True)):
            return entry[2], entry[3]
        txt, refs = self.preprocess_conds(context, ref_latents, vec_target, transformer_options)
        if len(self._cond_cache) >= 4:  # CFG cond + uncond stay warm across steps
            self._cond_cache.pop(next(iter(self._cond_cache)))
        self._cond_cache[key] = (weakref.ref(context), tuple(weakref.ref(r) for r in ref_latents or ()), txt, refs)
        return txt, refs

    def compute_final(self, feature: torch.Tensor, timestep: torch.Tensor, input_shape: tuple[int, ...]) -> torch.Tensor:
        """Time embedding + output layer on a cached / forecast pre-head image feature (Spectrum)."""

        H, W = int(input_shape[-2]), int(input_shape[-1])
        vec = self._time_vec(timestep.float(), feature.dtype)
        if self.flow_map:
            target = flow_map_target(self._last_sampling_sigmas, timestep).to(timestep.device)
            vec = vec + self.ref_time_in(timestep_embedding(target, 256).to(feature.dtype))
        out = self.final_layer["image"](feature, vec)
        return out.transpose(1, 2).reshape(feature.shape[0], self.in_channels, H, W)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, transformer_options: dict | None = None, **kwargs) -> torch.Tensor:
        if transformer_options is None:
            transformer_options = {}
        squeeze = x.ndim == 5
        if squeeze:
            x = x.squeeze(2)
        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        timestep = timestep.reshape(-1).float()

        ref_latents = [ref.to(device=device, dtype=dtype) for ref in dynamic_args.ref_latents] if dynamic_args.ref_latents else None
        if ref_latents:
            ref_latents = [ref.squeeze(2) if ref.ndim == 5 else ref for ref in ref_latents]

        vec_target = self._target_vec(timestep, transformer_options, dtype)
        txt, refs = self._conds(context.to(dtype), ref_latents, vec_target, transformer_options)
        txt, refs = _repeat_to_batch(txt, B), (None if refs is None else _repeat_to_batch(refs, B))

        vec = self._time_vec(torch.cat([timestep, timestep.new_zeros(1)]), dtype)
        if vec_target is not None:
            vec = vec + vec_target
        vec_img, vec_zero = vec[:-1], vec[-1:]
        if vec_img.shape[0] != B:
            vec_img = _repeat_to_batch(vec_img, B)

        # per sample the joint sequence is [image, image_cond, text]
        img = self.emb_in["image"](x.flatten(2).transpose(1, 2))
        n_img = img.shape[1]
        ids = [self.image_ids(H, W, 0, device)]
        if refs is not None:
            ids.append(self.ref_ids([tuple(r.shape[-2:]) for r in ref_latents], device))
        ids.append(self.text_ids(txt.shape[1], device))
        pe = self.rope(torch.cat(ids))

        early = self.early_stream_modulations["image"](vec_img)
        pe_img = pe[:, :, :n_img]
        for block in self.content_mode_blocks["image"]:
            img = block(img, [(0, n_img, early)], pe_img, transformer_options)

        parts = [(img, "image")] + ([(refs, "image_cond")] if refs is not None else []) + [(txt, "txt")]
        spans, start = [], 0
        for tokens, mode in parts:
            spans.append((start, start + tokens.shape[1], self.single_stream_modulations[mode](vec_img if mode == "image" else vec_zero)))
            start += tokens.shape[1]
        joint = torch.cat([tokens for tokens, _ in parts], dim=1)
        del img, parts

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {}) if isinstance(transformer_options, dict) else {}
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:

                def block_wrap(args, _block=block):
                    return {"img": _block(args["img"], args["spans"], args["pe"], args["transformer_options"])}

                joint = blocks_replace[("single_block", i)](
                    {"img": joint, "spans": spans, "vec": vec_img, "pe": pe, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )["img"]
            else:
                joint = block(joint, spans, pe, transformer_options)

        feature = self.final_input(joint[:, :n_img])
        out = self.final_layer["image"](feature, vec_img)
        out = out.transpose(1, 2).reshape(B, self.in_channels, H, W)
        return out.unsqueeze(2) if squeeze else out
