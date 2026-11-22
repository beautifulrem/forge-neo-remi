from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.nn.llm.llama import Qwen3_06B

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from backend.text_processing._comfy import INF, SDClipModel
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.qwen_tokens = []
        self.qwen_multipliers = []
        self.t5_tokens = []
        self.t5_multipliers = []


class Qwen06Engine:
    def __init__(self, text_encoder, qwen_tokenizer, t5_tokenizer, semantic_encoder=None, semantic_tokenizer=None, semantic_layers=None):
        self.text_encoder: "Qwen3_06B" = text_encoder
        self.clip_model = SDClipModel(
            text_encoder,
            max_length=INF,
            layer="last",
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            enable_attention_masks=True,
            return_attention_masks=True,
        )
        # Bundled NegPiP and FreeFuse use these raw tokenizers and PromptChunk
        # fields to align masks and concept positions with the T5 embeddings.
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer

        self.id_pad = 151643
        self.id_end = 1

        # Remi: Anima-3.8B's Qwen3.5 4B (backend.nn.anima_qwen35) and the layers its semantic
        # connector reads; None for the other Anima models
        self.semantic_encoder = semantic_encoder
        self.semantic_tokenizer = semantic_tokenizer
        self.semantic_layers = tuple(semantic_layers or ())

    @property
    def emphasis(self):
        return emphasis.get_current_option(opts.emphasis)()

    def tokenize(self, texts):
        return (
            self.qwen_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
            self.t5_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
        )

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        # BREAK is the parser's chunk marker (weight -1), not text: Anima has one chunk, so it is a space
        # (as modulation guidance reads the prompt); left as is it would encode "BREAK" at weight -1
        parsed = [(" ", 1.0) if text == "BREAK" and weight == -1 else (text, weight) for text, weight in parsed]
        qwen_tokenized, t5_tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            if not chunk.qwen_tokens:
                chunk.qwen_tokens.append(self.id_pad)
                chunk.qwen_multipliers.append(1.0)

            chunk.t5_tokens.append(self.id_end)
            chunk.t5_multipliers.append(1.0)

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens in qwen_tokenized:
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.qwen_tokens.append(token)
                chunk.qwen_multipliers.append(1.0)
                position += 1

        for tokens, (text, weight) in zip(t5_tokenized, parsed):
            if self.emphasis.name == "Ignore":
                weight = 1.0
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.t5_tokens.append(token)
                chunk.t5_multipliers.append(weight)
                position += 1

        if not chunks:
            next_chunk()

        return chunks

    def uses_semantic(self, negative: bool = False) -> bool:
        """Whether prompts are encoded for Anima-3.8B's semantic connector (the negative
        prompt only with the ``anima_semantic_negative`` setting)."""

        return self.semantic_encoder is not None and (not negative or bool(getattr(opts, "anima_semantic_negative", False)))

    def __call__(self, texts, semantic: bool = False):
        if any(emphasis.uses_emphasis(x) for x in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        zs = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            if line in cache:
                cond = cache[line]
            else:
                chunks: list[PromptChunk] = self.tokenize_line(line)
                assert len(chunks) == 1

                for chunk in chunks:
                    tokens = chunk.qwen_tokens
                    multipliers = chunk.qwen_multipliers

                    z: torch.Tensor = self.process_tokens([tokens], [multipliers])[0]
                    tok = torch.tensor(chunk.t5_tokens, dtype=torch.int)
                    mul = torch.tensor(chunk.t5_multipliers)

                cond = self.anima_preprocess(z, tok, mul)
                if semantic:
                    cond = {"crossattn": cond, **self.semantic_inputs(line, z, tok, mul)}
                cache[line] = cond

            zs.append(cond)

        del cache
        return zs

    def anima_preprocess(self, cross_attn: torch.Tensor, t5xxl_ids: torch.Tensor, t5xxl_weights: torch.Tensor) -> torch.Tensor:
        device = memory_management.text_encoder_device()

        cross_attn = cross_attn.unsqueeze(0).to(device=device)
        t5xxl_ids = t5xxl_ids.unsqueeze(0).to(device=device)

        cross_attn = self.text_encoder.preprocess_text_embeds(cross_attn, t5xxl_ids)
        if t5xxl_weights is not None:
            cross_attn *= t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(cross_attn)

        if cross_attn.shape[1] < 512:
            cross_attn = torch.nn.functional.pad(cross_attn, (0, 0, 0, 512 - cross_attn.shape[1]))

        return cross_attn

    def semantic_text(self, line: str) -> str:
        """The prompt as Qwen3.5 reads it: the text without emphasis syntax, BREAK as a space."""

        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        return "".join(" " if text == "BREAK" and weight == -1 else text for text, weight in parsed)

    def semantic_tokens(self, line: str) -> list[int]:
        # As the connector was trained: no special tokens, at most 1024 tokens, never empty
        ids = self.semantic_tokenizer.encode(self.semantic_text(line), add_special_tokens=False).ids[:1024]
        return ids or [self.id_pad]

    def semantic_inputs(self, line: str, source: torch.Tensor, t5_ids: torch.Tensor, t5_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """What ``Anima.semantic_context`` needs besides the timestep, per prompt: the Qwen3 0.6B
        states, the T5 ids and weights, and the Qwen3.5 layers (stacked) with their mask."""

        device = memory_management.text_encoder_device()
        ids = torch.tensor([self.semantic_tokens(line)], dtype=torch.long, device=device)
        dtype = getattr(self.semantic_encoder, "compute_dtype", None) or self.semantic_encoder.embed_tokens.weight.dtype
        layers = self.semantic_encoder(ids, self.semantic_layers, dtype=dtype)
        semantic = torch.stack(layers, dim=1)[0]
        return {
            "anima_source": source.to(device),
            "anima_ids": t5_ids.to(device=device, dtype=torch.long),
            "anima_weights": t5_weights.to(device=device, dtype=source.dtype),
            "anima_semantic": semantic.to(source.dtype),
            "anima_semantic_mask": torch.ones(semantic.shape[1], dtype=torch.long, device=device),
        }

    def process_embeds(self, batch_tokens):
        return self.clip_model.process_tokens(batch_tokens, memory_management.text_encoder_device())

    def process_tokens(self, batch_tokens, batch_multipliers):
        # Qwen weights are always one; emphasis applies after the T5 adapter.
        # Keep its output on the encoder device to avoid a CPU round trip.
        return self.clip_model(batch_tokens)[0]


# Retain imports used by extensions written against the original Remi engine.
AnimaTextProcessingEngine = Qwen06Engine
