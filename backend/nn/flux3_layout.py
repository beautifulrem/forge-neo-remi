"""FLUX.3 Image Klein caption and layout protocol (no model code).

FLUX.3 is trained on the captions its prompt upsampler writes (a Qwen3.5-9B
fine-tune, ``black-forest-labs/flux-3-image-klein-pup``): long English prose
for plain prompts, and for layouts one global caption citing every element as
``<id>``, one space, then one JSON array of ``{"id", "bbox", "desc"}`` rows with
``[y_min, x_min, y_max, x_max]`` boxes on a 0-1000 canvas (edits use
``{"id", "from", "src_bbox", "tgt_bbox", "desc"}`` rows).

Adapted from Diffusers ``modular_pipelines/flux3/prompt_upsampler.py``
(Apache-2.0), which follows BFL's ``src/flux3/bbox.py`` and ``pup.py``; the
system prompts are the ones shipped with the upsampler weights. Remi uses them
with any OpenAI-compatible endpoint (``remi.prompt_tools`` "Generate") and to
validate layout prompts built by the JSON Builder.
"""

from __future__ import annotations

import json
import math
import re

CANVAS = 1000
MAX_REFERENCES = 10
BOX_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

I2I_SCAFFOLD = {
    False: "Reference images follow in order (the first image, the second image, ...).\nUser request: {instruction}\nOutput the caption.",
    True: "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: {instruction}\nDecide the edit, imagine the target image, and output the annotation record.",
}

SYSTEM_PROMPTS = {
    "t2i": (
        "You are the prompt upsampler of a text-to-image model. Expand the user's short request into ONE "
        "detailed, self-contained caption: the exact text the image model receives. Output only the caption, "
        "with no greetings, explanations, refusals, questions, Markdown, headings, JSON, coordinates, ids or "
        "angle-bracket tokens, and no quotes around it. If the input is not clearly an image request, "
        'caption a visualization of its concept. The request may begin with a "TARGET CANVAS: W:H" line: '
        "compose for that canvas (wide spreads content sideways, tall stacks it) and never mention the ratio "
        "or the canvas.\n"
        "\n"
        "Fidelity and planning: keep every explicit subject, attribute, count, spatial constraint, style cue "
        "and piece of wording; never contradict, substitute, drop or narrow the request, and keep an "
        "abstract concept abstract. Decide everything left open as ONE concrete, consistent plan; never "
        "offer alternatives or hedge. Never re-express a stated constraint through another mechanism; add "
        "camera optics (depth of field, bokeh, blur) only when asked for or implied. Without a stated style, "
        'plan an actual photograph and open with the medium convention ("Full-color outdoor photograph, '
        '...", "Full-color indoor photograph, ...", "Studio photograph, ...", "Close-up photograph, ..."); '
        'keep "Photorealistic" for renders, and turn "photorealistic" or "realistic" requests for photos '
        "into the photograph convention. Give a photograph a few ordinary, non-ideal specifics drawn from "
        "the scene's own objects instead of the stock-photo ideal. Do not invent names, brands, logos or "
        "celebrities; keep user-named ones. A long request keeps its content but not its wording: drop "
        "appraisal and sales vocabulary, turn moods into observable physical facts, and add only new "
        "concrete content with positions. Time of day is lighting, not rendered text.\n"
        "\n"
        'Caption format: one continuous prose block in the generative voice, never "this image shows", "we '
        'can see" or "is visible". Lead with medium and style, then scene, lighting and camera, then the '
        "subjects with concrete colors, materials and state. Every person in focus gets an observable "
        'profile: ethnicity, gender, age range, skin tone ("fair skin", "brown skin"), hair and clothing. '
        "Make the layout reconstructable from words: frame-relative positions (frame-left, frame-right, "
        'upper, lower, foreground, midground, background), sentence-initial anchors ("On the left, ...", "In '
        'the background, ..."), relations (holds, faces, behind) and a consistent order; describe up to '
        "about ten instances one by one, beyond that give the exact count and arrangement. Out-of-focus or "
        "distant planes are shapes, colors and silhouettes, never identities or readable text, and the "
        "background takes about a sixth of the caption. Be definite: no hedging adverbs, no quality or "
        "emotion judgments, no closing sentence about mood; state negations as what is present. Rendered "
        "text: transcribe every wording verbatim inside double quotes in its original language and script, "
        "keep the line breaks of multi-line text, and give its typography, color and placement; author "
        "concrete, domain-correct wording when the request needs text it does not supply, and name canonical "
        "structures (calendar, keyboard, periodic table) exactly with their full layout. All prose is "
        "English; only quoted text keeps its language. Typical length is 100 to 350 words, more for complex "
        "scenes.\n"
        "\n"
        "Output only the caption."
    ),
    "t2i_bbox": (
        "You are the prompt upsampler of a text-to-image model. Plan the full image the user's short request "
        "implies and output ONE structured caption on a single line: the global caption paragraph, one "
        "space, then ONE JSON array, the element table that places every cited element with a bounding box "
        "in the image to be generated. Output nothing else (no Markdown fences, commentary or refusals); "
        'visualize the concept of a non-image input. The request may begin with a "TARGET CANVAS: W:H" line: '
        "compose for that canvas and never mention it. Boxes live on a 0-1000 grid stretched onto the "
        "canvas: an element of physical width-to-height ratio a gets an x-span of about y-span x a x H/W.\n"
        "\n"
        "Planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece of "
        "wording, and decide everything left open as ONE concrete composition that the caption and the table "
        "both describe. Without a stated style plan an actual photograph and open with the medium convention "
        '("Full-color outdoor photograph, ...", "Studio photograph, ..."); add camera optics only when asked '
        "for; do not invent names, brands, logos or celebrities; drop appraisal and sales vocabulary.\n"
        "\n"
        "The table: one row per distinct object, person, animal, building or salient region the caption "
        "mentions and one per coherent text block; no rows for style, lighting or atmosphere. Structured "
        "layouts (tables, grids, calendars, keyboards, charts, maps, posters, pages) decompose into their "
        "regions, each row carrying that region's complete verbatim content; more than about ten "
        "interchangeable items become one collection row giving the exact count and arrangement, while items "
        'whose positions matter keep a row each. Row keys, in order: "id", "bbox", "desc". id: the bare '
        "class noun plus an index numbered left to right, then top to bottom, from 1 (person_1, dog_2); a "
        "text block is its two-letter language code plus _Text_ plus an index (En_Text_1); one entity is one "
        "row and ids are unique. bbox: [y_min, x_min, y_max, x_max] as integers from 0 to 1000, top-left "
        "origin, clamped at the borders, with plausible relative sizes and a small margin around text. desc: "
        "one self-contained paragraph in the generative voice that opens with the element's identity (never "
        '"the X"): shape, material, colors, state and markings, but no position. People in focus get the '
        "richest desc: ethnicity, gender, age range, skin tone, build, hair, face, each garment, "
        "accessories, pose and expression. A text row carries its complete wording verbatim in double "
        "quotes, with line breaks as \\n, plus its typography.\n"
        "\n"
        "The global caption: one flowing paragraph in the generative voice (medium first, then scene, "
        "lighting, camera and frame-relative layout; no quality judgments) that names each table element at "
        'identity level at its first mention and cites it with its id in its own angle brackets: "a wicker '
        'table <table_1> holding a tiered stand <stand_1>". Cite every table id and no other; appearance '
        "detail and text content live in the descs; no coordinates. Text-dense documents are laid out region "
        "by region with every visible text transcribed verbatim at a readable size (cut content rather than "
        "shrink type), and a grid quotes every visible cell. All prose is English; only quoted text keeps "
        "its language.\n"
        "\n"
        "Output only the caption and the table."
    ),
}


def boxes_to_canvas(boxes: list[dict]) -> list[dict]:
    """User boxes (0-1 coordinates) as the rows of the upsampler's box table (0-1000 canvas)."""

    def to_grid(box):
        if not box:
            return None
        if len(box) != 4:
            raise ValueError(f"A box is [y_min, x_min, y_max, x_max], got {box}.")
        return [round(max(0.0, min(1.0, float(v))) * CANVAS) for v in box]

    rows = []
    for i, box in enumerate(boxes):
        row_id, desc = str(box.get("id") or f"obj_{i + 1}"), str(box.get("desc") or "")
        if box.get("bbox"):
            rows.append({"id": row_id, "bbox": to_grid(box["bbox"]), "desc": desc})
        elif box.get("src_bbox") or box.get("tgt_bbox"):
            rows.append({"id": row_id, "from": str(box.get("from") or ""), "src_bbox": to_grid(box.get("src_bbox")), "tgt_bbox": to_grid(box.get("tgt_bbox")), "desc": desc})
        else:
            raise ValueError(f"Box {i} has neither `bbox` nor `src_bbox` / `tgt_bbox`.")
    return rows


def split_caption(text: str) -> tuple[str, list | None]:
    """``(caption, rows)``: the rows are the JSON list that closes the text, if any."""

    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def valid_canvas_box(box) -> list | None:
    """The box as a list if it is a well-formed ``[y_min, x_min, y_max, x_max]`` on the 0-1000 canvas."""

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf") for v in box):
        return None
    y1, x1, y2, x2 = box
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > CANVAS or x2 > CANVAS or max(box) <= 1:
        return None
    return list(box)


def format_layout_request(prompt: str, rows: list[dict]) -> str:
    return (
        f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
        "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
        "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
        f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}"
    )


def box_iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def valid_added_row(row, caption: str) -> dict | None:
    """A row the upsampler added, kept only if the caption cites its id and it is well formed."""

    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if not isinstance(row_id, str) or not BOX_ID.fullmatch(row_id) or f"<{row_id}>" not in caption or not isinstance(desc, str) or not desc.strip():
        return None
    if (box := valid_canvas_box(row.get("bbox"))) is not None:
        return {"id": row_id, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = valid_canvas_box(src) if src is not None else None
    tgt_box = valid_canvas_box(tgt) if tgt is not None else None
    if (
        not isinstance(origin, str)
        or not re.fullmatch(r"ref_image_\d+|target", origin)
        or (src is not None and src_box is None)
        or (tgt is not None and tgt_box is None)
        or (src_box is None and tgt_box is None)
        or (origin == "target") == (src_box is not None)
    ):
        return None
    return {"id": row_id, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def reconcile_boxes(text: str, locked: list[dict]) -> str:
    """The upsampler's answer with the user's boxes re-imposed (BFL ``reconcile``)."""

    caption, rows = split_caption(text)
    # only the model's rows with a string id can be renamed or kept (another id may not even be hashable)
    rows = [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get("id"), str)]
    locked_ids = {row["id"] for row in locked}
    row_ids = {row.get("id") for row in rows if isinstance(row, dict)}
    renamed = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = valid_canvas_box(row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox"))
            if box is not None and box_iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")

    by_id = {row.get("id"): row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock.get("desc", "")})
    for row in rows:
        added = valid_added_row(row, caption)
        if added is not None and added["id"] not in {row["id"] for row in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


def layout_problems(prompt: str) -> list[str]:
    """What a FLUX.3 layout prompt gets wrong (empty for a plain caption or a well-formed layout):
    malformed boxes, duplicate ids, and table ids the caption does not cite as ``<id>``. Each is a
    :class:`remi.ui.Sentence` the UI translates."""

    from remi.ui import Sentence

    caption, rows = split_caption(prompt)
    if rows is None:
        return []
    problems = []
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            problems.append(Sentence("row {{row}} has no id", row=i + 1))
            continue
        row_id = row["id"]
        if row_id in seen:
            problems.append(Sentence("duplicate id {{id}}", id=row_id))
        seen.add(row_id)
        boxes = [row[key] for key in ("bbox", "src_bbox", "tgt_bbox") if row.get(key) is not None]
        if not boxes or any(valid_canvas_box(box) is None for box in boxes):
            problems.append(Sentence("{{id}}: boxes must be [y_min, x_min, y_max, x_max] integers on the 0-1000 canvas", id=row_id))
        if f"<{row_id}>" not in caption:
            problems.append(Sentence("{{id}} is not cited as <{{id}}> in the caption", id=row_id))
    return problems


def reference_size(width: int, height: int, num_references: int) -> tuple[int, int]:
    """The upsampler's reference size: sides a multiple of 32, under the pixel budget for the reference count."""

    budget = {1: 1024 * 1024, 2: 724 * 724}.get(num_references, 512 * 512)

    def floor(value):
        return max(32, int(math.floor(value / 32)) * 32)

    w = max(32, round(width / 32) * 32)
    h = max(32, round(height / 32) * 32)
    if w * h > budget:
        scale = math.sqrt(width * height / budget)
        w, h = floor(width / scale), floor(height / scale)
    if w * h > budget:
        if w == 32:
            h = floor(budget / w)
        else:
            w = floor(budget / h)
    return w, h
