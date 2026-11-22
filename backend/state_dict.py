import json

import torch


def load_state_dict(model: torch.nn.Module, sd: dict[str, torch.Tensor], ignore_errors: list[str] = [], log_name: str = None, ignore_start: str | list[str] = None):
    gguf_targets = dict(model.named_parameters())
    gguf_targets.update(model.named_buffers())
    gguf_entries = [(name, tensor) for name, tensor in sd.items() if name in gguf_targets and getattr(tensor, "gguf_cls", None) is not None]

    if gguf_entries:
        from backend.loader_gguf import dequantize
        from backend.operations import ForgeOperationsGGUF

        gguf_layers = (ForgeOperationsGGUF.Linear, ForgeOperationsGGUF.Conv2d, ForgeOperationsGGUF.Embedding)
        for name, tensor in gguf_entries:
            if not isinstance(model.get_submodule(name.rpartition(".")[0]), gguf_layers):
                sd[name] = dequantize(tensor, dtype=gguf_targets[name].dtype)

    is_meta = any(p.is_meta for p in model.parameters())

    if is_meta:
        for name, param in [*model.named_parameters(), *model.named_buffers()]:
            if (entry := sd.get(name, None)) is not None and entry.dtype != param.dtype and getattr(entry, "gguf_cls", None) is None:
                sd[name] = entry.to(param.dtype)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=is_meta)

    if is_meta:
        for module in model.modules():
            for name, param in module._parameters.items():
                if param is not None and param.is_meta:
                    module._parameters[name] = torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype), requires_grad=False)
            for name, buffer in module._buffers.items():
                if buffer is not None and buffer.is_meta:
                    module._buffers[name] = torch.zeros(buffer.shape, dtype=buffer.dtype)

    missing = [x for x in missing if x not in ignore_errors]
    unexpected = [x for x in unexpected if x not in ignore_errors]

    if ignore_start is not None:
        missing = [x for x in missing if not x.startswith(tuple(ignore_start))]
        unexpected = [x for x in unexpected if not x.startswith(tuple(ignore_start))]

    unexpected = [x for x in unexpected if "comfy_attention" not in x]

    log_name = log_name or type(model).__name__
    if len(missing) > 0:
        print(f"{log_name} Missing: {missing}")
    if len(unexpected) > 0:
        print(f"{log_name} Unexpected: {unexpected}")


def state_dict_has(sd, prefix):
    return any(x.startswith(prefix) for x in sd.keys())


def filter_state_dict_with_prefix(sd, prefix, new_prefix=""):
    new_sd = {}

    for k, v in list(sd.items()):
        if k.startswith(prefix):
            new_sd[new_prefix + k[len(prefix) :]] = v
            del sd[k]

    return new_sd


def try_filter_state_dict(sd, prefix_list, new_prefix=""):
    for prefix in prefix_list:
        if state_dict_has(sd, prefix):
            return filter_state_dict_with_prefix(sd, prefix, new_prefix)
    return {}


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from * x : shape_from * (x + 1)]
    return sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def detect_quantization(state_dict: dict[str, torch.Tensor], *, is_unet: bool = False) -> dict | None:
    count: dict[str, int] = {}

    for key, val in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue

        fmt: str = json.loads(val.numpy().tobytes())["format"]
        count[fmt] = count.get(fmt, 0) + 1

    if count:
        from backend.quant_ops import QUANT_ALGOS

        layout: str = QUANT_ALGOS[max(count, key=count.get)]["comfy_tensor_layout"]
        return {"mixed_ops": True, "TE": not is_unet, "layout": layout}

    return None


def _detect_prefix(sd: list[str], meta: str) -> str:
    name = ""

    for key in sd:
        if key.endswith(f"{meta}.weight"):
            name = key
            break

    return name.replace(f"{meta}.weight", "")


def _strip_model_prefix(stem: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _convert_torchao_qdata(state_dict: dict[str, torch.Tensor], metadata: dict) -> dict[str, torch.Tensor]:
    """Convert TorchAO / Unsloth Studio ``_weight_qdata`` safetensors checkpoints
    (e.g. ``unsloth/MOSS-Image-SFT-FP8`` FP8 per-row, INT8 per-row, and
    INT8-ConvRot Hadamard group-256) into Forge's native ``comfy_quant`` layout
    without upcasting quantized weights to BF16."""

    qdata_keys = [k for k in state_dict if k.endswith("._weight_qdata")]
    if not qdata_keys:
        return state_dict

    unsloth_meta: dict = {}
    if isinstance(metadata.get("unsloth_metadata"), str):
        try:
            parsed = json.loads(metadata["unsloth_metadata"])
            if isinstance(parsed, dict):
                unsloth_meta = parsed
        except ValueError:
            pass

    act_rot = str(unsloth_meta.get("activation_rotation") or "").lower()
    use_convrot = act_rot.startswith("convrot")
    convrot_group = int(unsloth_meta.get("activation_rotation_group") or 256)
    raw_fqns = unsloth_meta.get("activation_rotation_fqns")
    convrot_fqns = set(raw_fqns) if isinstance(raw_fqns, list) and raw_fqns else None

    def _is_convrot_layer(stem: str) -> bool:
        if not use_convrot:
            return False
        if convrot_fqns is None:
            return True
        return stem in convrot_fqns or _strip_model_prefix(stem) in convrot_fqns

    for qkey in qdata_keys:
        stem = qkey[: -len("._weight_qdata")]
        qdata = state_dict.pop(qkey)
        scale = state_dict.pop(f"{stem}._weight_scale", None)
        zero_point = state_dict.pop(f"{stem}._weight_zero_point", None)

        if qdata.dtype == torch.int8 and scale is not None:
            if zero_point is not None and bool((zero_point != 0).any()):
                state_dict[f"{stem}.weight"] = ((qdata.float() - zero_point.float()) * scale.float()).to(torch.bfloat16)
                continue
            layer_conf: dict = {"format": "int8_tensorwise"}
            if _is_convrot_layer(stem):
                layer_conf["convrot"] = True
                layer_conf["convrot_groupsize"] = convrot_group
            state_dict[f"{stem}.weight"] = qdata
            state_dict[f"{stem}.weight_scale"] = scale.float()
            state_dict[f"{stem}.comfy_quant"] = torch.tensor(list(json.dumps(layer_conf).encode("utf-8")), dtype=torch.uint8)
        elif qdata.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and scale is not None:
            fmt_base = "float8_e4m3fn" if qdata.dtype == torch.float8_e4m3fn else "float8_e5m2"
            if scale.numel() == 1:
                state_dict[f"{stem}.weight"] = qdata
                state_dict[f"{stem}.weight_scale"] = scale.reshape(()).float()
                state_dict[f"{stem}.comfy_quant"] = torch.tensor(list(json.dumps({"format": fmt_base}).encode("utf-8")), dtype=torch.uint8)
            elif scale.dim() == 2 and scale.shape == (qdata.shape[0], 1):
                state_dict[f"{stem}.weight"] = qdata
                state_dict[f"{stem}.weight_scale"] = scale.float()
                state_dict[f"{stem}.comfy_quant"] = torch.tensor(list(json.dumps({"format": f"{fmt_base}_rowwise"}).encode("utf-8")), dtype=torch.uint8)
            else:
                state_dict[f"{stem}.weight"] = (qdata.float() * scale.float()).to(torch.bfloat16)
        elif scale is not None:
            state_dict[f"{stem}.weight"] = (qdata.float() * scale.float()).to(torch.bfloat16)
        else:
            state_dict[f"{stem}.weight"] = qdata

    return state_dict


def convert_quantization(state_dict: dict[str, torch.Tensor], metadata: dict) -> tuple[dict[str, torch.Tensor], dict]:
    # https://github.com/Comfy-Org/ComfyUI/blob/v0.19.0/comfy/utils.py#L1358
    if metadata is None:
        metadata = {}

    state_dict = _convert_torchao_qdata(state_dict, metadata)

    if "_quantization_metadata" in metadata:
        quant_metadata = json.loads(metadata["_quantization_metadata"]) or {}
    else:
        model_prefix = None

        for key in state_dict.keys():
            if key.endswith("scaled_fp8"):
                model_prefix = key.replace("scaled_fp8", "")
                break

        if model_prefix is None:
            return state_dict, metadata

        scaled_fp8_key = "{}scaled_fp8".format(model_prefix)
        scaled_fp8_weight = state_dict[scaled_fp8_key]
        scaled_fp8_dtype = scaled_fp8_weight.dtype
        if scaled_fp8_dtype is torch.float32:
            scaled_fp8_dtype = torch.float8_e4m3fn

        if scaled_fp8_weight.nelement() == 2:
            full_precision_matrix_mult = True
        else:
            full_precision_matrix_mult = False

        out_sd = {}
        layers = {}
        for k in list(state_dict.keys()):
            if k == scaled_fp8_key:
                continue
            if not k.startswith(model_prefix):
                out_sd[k] = state_dict[k]
                continue
            k_out = k
            w = state_dict.pop(k)
            layer = None
            if k_out.endswith(".scale_weight"):
                layer = k_out[: -len(".scale_weight")]
                k_out = "{}.weight_scale".format(layer)

            if layer is not None:
                layer_conf = {"format": "float8_e4m3fn"}
                if full_precision_matrix_mult:
                    layer_conf["full_precision_matrix_mult"] = full_precision_matrix_mult
                layers[layer] = layer_conf

            if k_out.endswith(".scale_input"):
                layer = k_out[: -len(".scale_input")]
                k_out = "{}.input_scale".format(layer)
                if w.item() == 1.0:
                    continue

            out_sd[k_out] = w

        state_dict = out_sd
        quant_metadata = {"layers": layers}

    if layers := quant_metadata.get("layers", None):
        prefix = _detect_prefix(state_dict.keys(), next(iter(layers.keys())))
        for k, v in layers.items():
            state_dict[f"{prefix}{k}.comfy_quant"] = torch.tensor(list(json.dumps(v).encode("utf-8")), dtype=torch.uint8)

    return state_dict, metadata
