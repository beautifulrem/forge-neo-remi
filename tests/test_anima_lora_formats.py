import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from backend.nn import anima_layout
from modules.anima_lora_support import ANIMA_BLOCK_MAPPINGS
from modules_forge.packages.comfy.lora import model_lora_keys_unet

# networks.py can only be imported inside the running WebUI (circular module
# initialisation), so its Anima re-mapping is compiled from the source here.
_SOURCE = Path(__file__).parents[1] / "extensions-builtin" / "sd_forge_lora" / "networks.py"
_tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_opts = SimpleNamespace(anima_lora_inserted_blocks="copy")
_ns = {"torch": torch, "logger": logging.getLogger("test"), "Optional": Optional, "anima_layout": anima_layout, "shared": SimpleNamespace(opts=_opts)}
_wanted = {"process_anima", "anima_inserted_blocks_policy", "LoraNotApplied"}
exec(compile(ast.Module([n for n in _tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _wanted], []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
process_anima = _ns["process_anima"]
LoraNotApplied = _ns["LoraNotApplied"]


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.diffusion_model.blocks[0].mlp = torch.nn.Linear(2, 2, bias=False)
        self.diffusion_model.config = {}
        self.config = SimpleNamespace(huggingface_repo="circlestone-labs/Anima")


def test_key_maps_are_not_shared_between_calls():
    """Upstream's ``key_map={}`` default was one dict for every call: the keys of a model seen
    before (blocks 28-51 of a 3.8B) stayed in the map of the next (a 2B)."""

    deeper = _FakeModel()
    deeper.diffusion_model.blocks.append(torch.nn.Module())
    deeper.diffusion_model.blocks[1].mlp = torch.nn.Linear(2, 2, bias=False)

    assert "lora_unet_blocks_1_mlp" in model_lora_keys_unet(deeper)
    assert "lora_unet_blocks_1_mlp" not in model_lora_keys_unet(_FakeModel())


def test_trainer_prefixes_map_to_anima_weights():
    key_map = model_lora_keys_unet(_FakeModel(), {})
    target = "diffusion_model.blocks.0.mlp.weight"
    for alias in (
        "diffusion_model.blocks.0.mlp",
        "lora_unet_blocks_0_mlp",
        "transformer.blocks.0.mlp",
        "net.blocks.0.mlp",
        "lycoris_blocks_0_mlp",
        "lora_transformer_blocks_0_mlp",
        "blocks.0.mlp",
    ):
        assert key_map.get(alias) == target, alias


def test_block_remap_and_adapter_move_work_for_every_prefix():
    for prefix, sep in (("transformer.blocks.", "."), ("lycoris_blocks_", "_"), ("diffusion_model.blocks.", ".")):
        lora = {f"{prefix}{i}{sep}mlp.lora_down.weight": torch.full((1,), float(i)) for i in range(28)}
        lora["transformer.llm_adapter.x.lora_down.weight"] = torch.zeros(1)
        assert process_anima(lora, 40)
        assert f"{prefix}39{sep}mlp.lora_down.weight" in lora, prefix
        assert "text_encoders.qwen3_06b.llm_adapter.x.lora_down.weight" in lora


def test_bare_adapter_keys_move_to_the_text_encoder():
    lora = {f"blocks.{i}.mlp.lora_down.weight": torch.zeros(1) for i in range(28)}
    lora["llm_adapter.blocks.0.lora_down.weight"] = adapter = torch.zeros(1)
    assert process_anima(lora, 28) is None  # same depth: nothing re-mapped
    assert lora.get("text_encoders.qwen3_06b.llm_adapter.blocks.0.lora_down.weight") is adapter
    assert "llm_adapter.blocks.0.lora_down.weight" not in lora


def test_remapped_blocks_carry_their_source_block_without_copies():

    lora = {f"diffusion_model.blocks.{i}.mlp.lora_down.weight": torch.full((1,), float(i)) for i in range(28)}
    originals = {i: lora[f"diffusion_model.blocks.{i}.mlp.lora_down.weight"] for i in range(28)}
    assert process_anima(lora, 40)
    for target, source in enumerate(ANIMA_BLOCK_MAPPINGS[(28, 40)]):
        assert lora[f"diffusion_model.blocks.{target}.mlp.lora_down.weight"] is originals[source]


def _lora(blocks, prefix="diffusion_model.blocks.", sep="."):
    return {f"{prefix}{i}{sep}mlp.lora_down.weight": torch.full((1,), float(i)) for i in blocks}


def _blocks(lora, prefix="diffusion_model.blocks."):
    return {int(k[len(prefix):].split(".")[0]): float(v) for k, v in lora.items() if k.startswith(prefix)}


def test_skip_leaves_the_inserted_blocks_without_the_lora():
    lora = _lora(range(28))
    assert process_anima(lora, 40, inserted="skip") == "28→40"
    placed = _blocks(lora)
    assert len(placed) == 28
    # 2B block 2 is 2.9B block 3 (an inserted copy of block 1 sits at 2)
    assert placed[3] == 2.0 and 2 not in placed and placed[39] == 27.0


def test_copy_moves_every_source_key_off_its_old_index():
    # Without removing the source keys first, 2B block 2 would stay on 2.9B block 2 (an inserted copy of block 1)
    lora = _lora(range(28))
    process_anima(lora, 40, inserted="copy")
    assert _blocks(lora)[2] == 1.0 and _blocks(lora)[3] == 2.0


def test_a_deeper_lora_goes_down_onto_the_original_blocks():
    notes = {}
    lora = _lora(range(52))
    note = process_anima(lora, 28, notes=notes)
    assert note.startswith("52→28, dropped blocks")
    placed = _blocks(lora)
    assert sorted(placed) == list(range(28))
    # 3.8B blocks 2,3 are 2B block 1's copies; 2B block 2 is 3.8B block 4
    assert placed[2] == 4.0 and placed[27] == 51.0
    assert notes["depth"] == 52


def test_2_9b_lora_on_3_8b():
    lora = _lora(range(40))
    assert process_anima(lora, 52, inserted="skip") == "40→52"
    assert _blocks(lora)[4] == 3.0  # 2.9B block 3 sits after the copy inserted at 3


def test_declared_depth_wins_over_the_guess():
    # A 2.9B LoRA that only trains blocks 0-20 looks like a 2B one
    lora = _lora(range(21))
    assert process_anima(lora, 40, declared=40) is None
    assert sorted(_blocks(lora)) == list(range(21))
    guessed = _lora(range(21))
    assert process_anima(guessed, 40, inserted="skip") == "28→40"


def test_text_encoder_only_lora_fits_every_depth():
    lora = {"lora_te_layers_0_self_attn_q_proj.lora_down.weight": torch.zeros(1)}
    assert process_anima(lora, 52) is None


def test_too_deep_or_unknown_depth_is_reported():
    with pytest.raises(LoraNotApplied, match="beyond"):
        process_anima(_lora(range(60)), 28)
    with pytest.raises(LoraNotApplied, match="no Anima block layout"):
        process_anima(_lora(range(28)), 30)
    with pytest.raises(LoraNotApplied, match="Declared"):
        process_anima(_lora(range(40)), 28, declared=28)
