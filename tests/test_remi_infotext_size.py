"""Infotext of any size neither freezes the server nor fills the index.

An image's own text can be as long as its writer liked (a folder scan, an
Extras source), and the compare form takes pasted text. The parameter
regexes were quadratic on some runs ("<lora:a" repeated, a long line without
a colon): 40,000 characters held the GIL for seconds, freezing every request.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image

from remi.files import FileStore
from remi.image_meta import MAX_INFOTEXT_CHARS
from remi.library import diff_parameters, parse_infotext_fields
from remi.server.testing import server_app
from remi.testing import webui_state

# Runs that made a regex try every start to the end of the text
RUNS = ["<lora:a", "<lyco:a:", "word ", "a", "k:{a", "k:[a", 'k:"\\"a', " "]


@pytest.mark.parametrize("unit", RUNS)
def test_parsing_a_long_run_takes_linear_time(unit):
    run = unit * (120_000 // len(unit))
    start = time.perf_counter()
    parse_infotext_fields("a cat\nSteps: 20, " + run)  # the parameter line
    parse_infotext_fields(run + "\nNegative prompt: " + run + "\nSteps: 20")  # the prompts (LoRA tags)
    diff_parameters(run + "\nSteps: 20, " + run, "a dog\nSteps: 30")
    assert time.perf_counter() - start < 1.0


def test_the_parse_is_unchanged_on_ordinary_infotext():
    text = '<lora:style_a:0.8> a cat, <LYCO:b:1>\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, CFG scale: 7, Size: 832x1216, Lora hashes: "style_a: abc, b: def", Hires upscale: 1.5, Template: {"a": 1}, List: [1, 2]'
    parsed = parse_infotext_fields(text)
    assert parsed["prompt"] == "<lora:style_a:0.8> a cat, <LYCO:b:1>" and parsed["negative_prompt"] == "blurry"
    assert parsed["params"] == {"Steps": "20", "Sampler": "Euler a", "CFG scale": "7", "Size": "832x1216", "Lora hashes": "style_a: abc, b: def", "Hires upscale": "1.5", "Template": '{"a": 1}', "List": "[1, 2]"}
    assert parsed["loras"]["style_a"] == 0.8 and parsed["loras"]["b"] == 1.0


def test_an_output_keeps_its_infotext_up_to_the_cap(tmp_path):
    files = FileStore(tmp_path / "files")
    try:
        record = files.save_output_image(Image.new("RGB", (8, 8)), "x" * (MAX_INFOTEXT_CHARS + 50_000))
        assert len(record.infotext) == MAX_INFOTEXT_CHARS
        assert len(files.get(record.id).infotext) == MAX_INFOTEXT_CHARS
    finally:
        files.close()


def test_compare_takes_pasted_text_up_to_the_cap(tmp_path):
    with webui_state(), server_app(tmp_path) as server:
        long = "a\nSteps: 20, " + "x" * MAX_INFOTEXT_CHARS
        assert server.client.post("/api/v2/gallery/compare", json={"infotext_a": long, "infotext_b": "b"}).status_code == 422
        assert server.client.post("/api/v2/gallery/compare", json={"infotext_a": long[:MAX_INFOTEXT_CHARS], "infotext_b": "b"}).status_code == 200
