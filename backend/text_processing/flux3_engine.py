"""Qwen3-VL-4B text conditioning for FLUX.3 Image Klein.

The prompt goes through the Qwen3-VL instruct chat template (no system turn)
and is right-padded with ``<|endoftext|>`` to a multiple of 80 tokens (capped
at 8192). The padding is masked inside Qwen3-VL but the DiT attends to it, so
the window size is part of the model recipe. Padding tokens sit at rope
position 1 (the transformers 4.57 behaviour the reference was trained with).
The hidden states entering layers 4, 8, ..., 32 are stacked per token into the
20480-channel context.

References: Diffusers ``modular_pipelines/flux3/encoders.py``
(``Flux3TextEncoderStep``), ComfyUI ``comfy/text_encoders/flux.py``
(``Flux3Tokenizer`` / ``Flux3Qwen3VL``).
"""

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import EMBEDDINGS, INF, SDClipModel, SDTokenizer

PAD_TOKEN = 151643  # <|endoftext|>
PAD_MULTIPLE = 80
MAX_LENGTH = 8192
HIDDEN_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def padded_length(real_length: int, pad_multiple: int = PAD_MULTIPLE, max_length: int = MAX_LENGTH) -> int:
    return min(-(-max(real_length, 1) // pad_multiple) * pad_multiple, max_length)


def padding_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Rope positions: 0, 1, ... over the real tokens, 1 on the padding."""

    return (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)


class Flux3ClipModel(SDClipModel):
    """``SDClipModel`` that passes the FLUX.3 padding position ids to Qwen3-VL."""

    def forward(self, tokens):
        device = memory_management.text_encoder_device()
        embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)
        outputs = self.transformer(
            None,
            attention_mask,
            embeds=embeds,
            num_tokens=num_tokens,
            intermediate_output=self.layer,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
            dtype=torch.float32,
            embeds_info=embeds_info,
            position_ids=padding_position_ids(attention_mask),
        )
        return outputs[1].float(), None, {"attention_mask": attention_mask}


class Flux3TextEngine:
    def __init__(self, text_encoder, tokenizer, *, pad_multiple: int = PAD_MULTIPLE, max_length: int = MAX_LENGTH):
        self.text_encoder = Flux3ClipModel(text_encoder, layer=list(HIDDEN_LAYERS), layer_idx=None, special_tokens={"pad": PAD_TOKEN}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD_TOKEN)
        self.pad_multiple = int(pad_multiple)
        self.max_length = int(max_length)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        if isinstance(texts, str):
            return self.tokenizer.tokenizer(TEMPLATE.format(texts))["input_ids"]
        return [self.tokenizer.tokenizer(TEMPLATE.format(t))["input_ids"] for t in texts]

    def tokenize_padded(self, text: str) -> list[list[tuple[int, float]]]:
        """One right-padded token window for ``text`` (truncated at ``max_length``)."""

        (batch,) = self.tokenizer.tokenize_with_weights(TEMPLATE.format(text), disable_weights=True)
        batch = batch[: self.max_length]
        batch.extend([(PAD_TOKEN, 1.0)] * (padded_length(len(batch), self.pad_multiple, self.max_length) - len(batch)))
        return [batch]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}
        for line in texts:
            if line not in cache:
                cond = self.text_encoder.encode_token_weights(self.tokenize_padded(line))[0]  # (1, layers, L, 2560)
                cache[line] = cond.movedim(1, 2).flatten(2)  # (1, L, layers * 2560)
            zs.extend(cache[line])
        return zs
