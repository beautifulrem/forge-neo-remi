"""Anima-3.8B: the Semantic Connector v2 and its Qwen3.5 4B text encoder, without checkpoints.

The manifests in ``tests/fixtures`` are the tensor names and shapes of the release files
(``Anima-3.8B-v1.1.safetensors``, ``qwen35_4b.safetensors``): the modules must load them with
nothing missing or left over, since the loader fills missing weights with zeros and a zeroed
connector would silently do nothing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from backend.loader import process_anima
from backend.nn.anima import Anima, LLMAdapter
from backend.nn.anima_connector import SemanticConnector, connector_config
from backend.nn.anima_qwen35 import AnimaQwen35
from backend.operations import using_forge_operations
from backend.sampling.condition import compile_conditions
from test_anima_text_engine_compat import load_engine

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SMALL = dict(source_dim=96, num_queries=8, resampler_blocks=2, resampler_dim=256, resampler_heads=4, mlp_hidden_dim=512)


def manifest(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def meta_shapes(tensors: dict) -> dict:
    return {key: torch.empty(shape, device="meta") for key, shape in tensors.items()}


def test_connector_config_is_read_from_the_release_weights():
    release = manifest("anima_3_8b_connector_manifest.json")
    config = connector_config(meta_shapes(release["tensors"]), metadata=release["metadata"])

    assert config == {
        "adapter_blocks": 6,
        "model_dim": 1024,
        "adapter_heads": 16,
        "source_dim": 2560,
        "layers": (7, 15, 23, 31),
        "num_queries": 64,
        "resampler_blocks": 6,
        "resampler_dim": 2048,
        "resampler_heads": 16,
        "mlp_hidden_dim": 5632,
    }
    assert connector_config({"blocks.0.mlp.layer1.weight": torch.empty(1)}) is None


def test_the_3_8b_dit_has_exactly_the_release_adapter_and_connector_tensors():
    release = manifest("anima_3_8b_connector_manifest.json")
    config = connector_config(meta_shapes(release["tensors"]), metadata=release["metadata"])
    with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
        model = Anima(16, 16, 2, 1, model_channels=2048, crossattn_emb_channels=1024, adaln_lora_dim=256, num_blocks=2, num_heads=16, semantic_connector=config)

    ours = {key: list(value.shape) for key, value in model.state_dict().items() if key.startswith(("anima_v2_connector.", "llm_adapter."))}
    assert ours == release["tensors"]


def test_the_qwen35_encoder_has_exactly_the_release_tensors():
    release = manifest("anima_qwen35_4b_manifest.json")
    with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
        model = AnimaQwen35()

    assert {key: list(value.shape) for key, value in model.state_dict().items()} == release["tensors"]
    assert model.layers[31].mlp is None and model.layers[30].mlp is not None


@pytest.fixture
def connector():
    torch.manual_seed(0)
    adapter = LLMAdapter().eval()
    model = SemanticConnector(**SMALL).eval()
    for module in (adapter, model):
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
    return adapter, model


def inputs():
    generator = torch.Generator().manual_seed(1)
    source = torch.randn(1, 13, 1024, generator=generator)
    ids = torch.randint(0, 32128, (1, 9), generator=generator)
    semantic = torch.randn(1, 4, 11, 96, generator=generator)
    return source, ids, semantic


@torch.inference_mode()
def test_the_connector_depends_on_the_timestep_and_reduces_to_the_adapter(connector):
    adapter, model = connector
    source, ids, semantic = inputs()

    early = model(adapter, source, ids, semantic, torch.tensor([0.9]))
    late = model(adapter, source, ids, semantic, torch.tensor([0.1]))
    native = model(adapter, source, ids, semantic, torch.tensor([0.5]), include_inserted_blocks=False)

    assert early.shape == (1, 9, 1024)
    assert (early - late).abs().max() > 1e-3
    assert torch.allclose(native, adapter(source, ids), atol=1e-6)
    assert (early - native).abs().max() > 1e-3


@torch.inference_mode()
def test_semantic_context_replaces_the_encoded_context_with_weights_padding_and_negpip(connector):
    adapter, model = connector
    dit = SimpleNamespace(llm_adapter=adapter, anima_v2_connector=model)
    source, ids, semantic = inputs()
    context = torch.zeros(1, 512, 1024)
    weights = torch.linspace(0.5, 1.5, 9)[None]
    negpip = torch.ones(1, 512, 1)
    negpip[0, 3] = -1
    kwargs = {"anima_source": source, "anima_ids": ids, "anima_weights": weights, "anima_semantic": semantic, "anima_semantic_mask": torch.ones(1, 11, dtype=torch.long)}

    got = Anima.semantic_context(dit, context, torch.tensor([0.4]), {**kwargs, "c_negpip_mask": negpip})
    want = model(adapter, source, ids, semantic, torch.tensor([0.4])) * weights[..., None]

    assert got.shape == context.shape
    assert torch.allclose(got[:, :9] * negpip[:, :9], want, atol=1e-6)
    assert not got[:, 9:].any()
    assert Anima.semantic_context(dit, context, torch.tensor([0.4]), {}) is context
    with pytest.raises(RuntimeError, match="semantic connector"):
        Anima.semantic_context(SimpleNamespace(anima_v2_connector=None), context, torch.tensor([0.4]), kwargs)


@torch.inference_mode()
def test_semantic_context_computes_in_the_dit_dtype_with_a_wider_embedding_table(connector):
    """On CUDA the weights are not cast to the input: a float32 T5 embedding (as stored next to
    bfloat16 weights) must not leave the connector's Linears a float32 input."""
    adapter, model = connector
    adapter, model = adapter.to(torch.bfloat16), model.to(torch.bfloat16)
    adapter.embed.float()
    dit = SimpleNamespace(llm_adapter=adapter, anima_v2_connector=model)
    source, ids, semantic = inputs()
    kwargs = {"anima_source": source.bfloat16(), "anima_ids": ids, "anima_weights": torch.ones(1, 9, dtype=torch.bfloat16), "anima_semantic": semantic.bfloat16()}

    got = Anima.semantic_context(dit, torch.zeros(1, 512, 1024, dtype=torch.bfloat16), torch.tensor([0.4]), kwargs)

    assert got.dtype == torch.bfloat16 and got[:, :9].abs().sum() > 0


def test_conditions_without_a_pooled_vector_pass_their_extra_tensors_to_the_model():
    cond = {"crossattn": torch.zeros(2, 512, 4), "anima_ids": torch.zeros(2, 9, dtype=torch.long), "anima_semantic": torch.zeros(2, 4, 11, 8)}

    (compiled,) = compile_conditions(cond)

    assert set(compiled["model_conds"]) == {"c_crossattn", "anima_ids", "anima_semantic"}
    assert compiled["model_conds"]["anima_ids"].cond is cond["anima_ids"]


def test_the_3_8b_adapter_is_copied_not_moved_to_the_text_encoder():
    dit = {"llm_adapter.embed.weight": torch.ones(2), "blocks.0.x": torch.ones(1)}
    enc = {}

    process_anima(dit, enc, keep_in_dit=True)

    assert set(dit) == {"llm_adapter.embed.weight", "blocks.0.x"} and set(enc) == {"llm_adapter.embed.weight"}


def test_qwen35_reads_the_prompt_without_emphasis_syntax_and_negatives_only_when_enabled():
    namespace, opts = load_engine()
    engine = namespace["Qwen06Engine"](object(), None, None, semantic_encoder=object(), semantic_layers=(7, 15, 23, 31))

    assert engine.semantic_text("a (red hair:1.2), [blue] BREAK eyes") == "a red hair, blue eyes"
    assert engine.uses_semantic() and not engine.uses_semantic(negative=True)
    opts.anima_semantic_negative = True
    assert engine.uses_semantic(negative=True)
    assert not namespace["Qwen06Engine"](object(), None, None).uses_semantic()
