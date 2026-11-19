"""Block layouts of the depth-expanded Anima models, in one place.

Anima 2B has 28 DiT blocks. Anima-2.9B (Gazingstars123/Anima-2.9B) grew it to
40 by LLaMA-Pro block expansion, and Anima-3.8B (lylogummy/Anima-3.8B) grew 2.9B
to 52 the same way: each inserted block started as a copy of its neighbour with
zeroed output projections. Both releases publish where the blocks went
(``expand_manifest.json`` of 2.9B, the safetensors metadata of 3.8B), which
``INSERTIONS`` records.

Weights trained for one depth (LoRA, ControlNet-LLLite, control-LoRA) are moved
to another depth with ``block_map``. Upward, an original block keeps its
weights and an inserted block either gets a copy of its source block's weights
(``"copy"``) or none (``"skip"``); downward, the inserted blocks' weights are
dropped. Every loader goes through this module, so a LoRA, an LLLite and the
per-block settings written for the 2B layout always agree.

Pure Python: imported by ``modules`` code that must not pull in torch.
"""

from __future__ import annotations

from functools import cache
from typing import Final, Literal

# (old depth, new depth) -> {inserted position: source block}, from the published manifests
INSERTIONS: Final[dict[tuple[int, int], dict[int, int]]] = {
    (28, 40): {2: 1, 5: 3, 8: 5, 11: 7, 14: 9, 17: 11, 21: 14, 24: 16, 27: 18, 30: 20, 33: 22, 36: 24},
    (40, 52): {3: 2, 7: 5, 11: 8, 15: 11, 19: 14, 23: 17, 27: 20, 31: 23, 35: 26, 39: 29, 43: 32, 47: 35},
}

ANIMA_DEPTHS: Final[tuple[int, ...]] = (28, 40, 52)

Inserted = Literal["copy", "skip"]
INSERTED_POLICIES: Final[tuple[str, ...]] = ("copy", "skip")


def _expand(old: int, new: int) -> tuple[int, ...]:
    """Source block of each of the ``new`` blocks of one expansion step."""

    inserted = INSERTIONS[(old, new)]
    layout: list[int] = []
    source = 0
    for position in range(new):
        if position in inserted:
            layout.append(inserted[position])
        else:
            layout.append(source)
            source += 1
    if source != old:
        raise AssertionError(f"Anima expansion {old}->{new} does not keep all {old} blocks")
    return tuple(layout)


@cache
def expansion(src: int, dst: int) -> tuple[int, ...] | None:
    """For an expansion from ``src`` to ``dst`` blocks, the ``src`` block each
    ``dst`` block came from (inserted blocks name the block they copied), or
    None when ``dst`` is not an expansion of ``src``."""

    if src == dst:
        return tuple(range(src))
    if (src, dst) in INSERTIONS:
        return _expand(src, dst)
    for (old, new) in INSERTIONS:
        if old == src and new < dst and (rest := expansion(new, dst)) is not None:
            first = _expand(old, new)
            return tuple(first[block] for block in rest)
    return None


@cache
def original_blocks(src: int, dst: int) -> tuple[int, ...] | None:
    """The ``dst`` block that *is* each ``src`` block (not an inserted copy of
    it) when ``dst`` expands ``src``."""

    if (layout := expansion(src, dst)) is None:
        return None
    return tuple(layout.index(block) for block in range(src))


# Upstream's table (backend.nn.anima.BLOCK_MAPPINGS): target block -> source block
BLOCK_MAPPINGS: Final[dict[tuple[int, int], list[int]]] = {
    (src, dst): list(expansion(src, dst))  # type: ignore[arg-type]
    for src in ANIMA_DEPTHS
    for dst in ANIMA_DEPTHS
    if src < dst
}


def block_map(src: int, dst: int, inserted: Inserted = "copy") -> tuple[dict[int, tuple[int, ...]], tuple[int, ...]] | None:
    """Where the weights of each of ``src`` blocks go on a ``dst``-block model.

    Returns ``(targets, dropped)``: ``targets[block]`` are the model blocks
    that take block ``block``'s weights, ``dropped`` the source blocks that
    have no place on the model (the inserted blocks, when going down). None
    when the two depths are not related by the known expansions.
    """

    if src == dst:
        return {block: (block,) for block in range(src)}, ()
    if src < dst:
        if (layout := expansion(src, dst)) is None:
            return None
        if inserted == "skip":
            originals = original_blocks(src, dst)
            return {block: (originals[block],) for block in range(src)}, ()  # type: ignore[index]
        targets: dict[int, list[int]] = {}
        for target, source in enumerate(layout):
            targets.setdefault(source, []).append(target)
        return {block: tuple(blocks) for block, blocks in targets.items()}, ()
    if (originals := original_blocks(dst, src)) is None:
        return None
    back = {model_block: block for block, model_block in enumerate(originals)}
    targets_down = {block: (back[block],) for block in range(src) if block in back}
    dropped = tuple(block for block in range(src) if block not in back)
    return targets_down, dropped


class AnimaDepthError(ValueError):
    pass


def trained_depth(trained_blocks, *, declared: int | None = None) -> tuple[int, bool]:
    """``(depth, guessed)`` of weights trained for an Anima DiT, from the block
    indices they touch (``trained_blocks``) or a ``declared`` depth (the
    user's override or the file's metadata), which wins.

    A file that touches every block of a depth is unambiguous. One that
    touches fewer blocks is taken as the smallest depth it fits, and
    ``guessed`` is True: a 2.9B LoRA trained on blocks 0-20 only looks the
    same as a 2B one.
    """

    if declared is not None:
        if declared not in ANIMA_DEPTHS:
            raise AnimaDepthError(f"Unknown Anima depth {declared} (known: {', '.join(map(str, ANIMA_DEPTHS))})")
        if trained_blocks and max(trained_blocks) >= declared:
            raise AnimaDepthError(f"Declared Anima depth {declared} but block {max(trained_blocks)} is trained")
        return declared, False
    if not trained_blocks:
        raise AnimaDepthError("No Anima DiT blocks are trained")
    needed = max(trained_blocks) + 1
    depth = next((size for size in ANIMA_DEPTHS if needed <= size), None)
    if depth is None:
        raise AnimaDepthError(f"Block {needed - 1} is beyond the deepest known Anima model ({ANIMA_DEPTHS[-1]} blocks)")
    return depth, needed != depth


def declared_depth(metadata: dict | None) -> int | None:
    """An Anima depth written into a file's safetensors metadata, if any.

    Looks at the fields expansion tools and trainers write: ``new_block_count``
    (the 3.8B release), ``num_blocks`` / ``ss_num_blocks`` / ``anima_num_blocks``,
    and a ``2.9B`` / ``3.8B`` / ``40 blocks`` / ``52 blocks`` hint in the base
    model name fields.
    """

    if not metadata:
        return None
    for key in ("anima_num_blocks", "num_blocks", "ss_num_blocks", "new_block_count"):
        value = metadata.get(key)
        if value is not None:
            try:
                depth = int(str(value).strip())
            except ValueError:
                continue
            if depth in ANIMA_DEPTHS:
                return depth
    hints = " ".join(
        str(metadata.get(key, ""))
        for key in ("ss_base_model_version", "ss_sd_model_name", "modelspec.architecture", "modelspec.base_model", "ss_base_model")
    ).lower()
    if not hints.strip() or "anima" not in hints:
        return None
    for depth, names in ((52, ("3.8b", "3_8b", "3-8b", "52 block", "52block", "pro52")), (40, ("2.9b", "2_9b", "2-9b", "40 block", "40block"))):
        if any(name in hints for name in names):
            return depth
    return None
