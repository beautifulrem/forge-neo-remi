"""``pasted_value``: the API's infotext replay (``modules/api/api.py``) turns
infotext values into request field values. modules.api.api cannot be
imported outside the WebUI, so the function is loaded from its source."""

import ast
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_function = next(node for node in _TREE.body if getattr(node, "name", None) == "pasted_value")
_ns = {"Any": Any, "Mapping": Mapping, "math": math, "re": re}
exec(compile(ast.Module([_function], []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
for _name in ("_PNGINFO_SURROGATE_RE", "_sanitize_pnginfo_scalar", "_sanitize_pnginfo_items"):
    _node = next(
        n
        for n in _TREE.body
        if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == _name for t in n.targets)) or getattr(n, "name", None) == _name
    )
    exec(compile(ast.Module([_node], []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
pasted_value = _ns["pasted_value"]
_sanitize_pnginfo_items = _ns["_sanitize_pnginfo_items"]


def test_none_sets_nothing():
    assert pasted_value(None, float) is None


def test_infotext_text_converts():
    # Infotext values arrive as text: bool("False") would be True
    assert pasted_value("False", bool) is False
    assert pasted_value("True", bool) is True
    assert pasted_value("3.0", int) == 3 and pasted_value("7", int) == 7


def test_plain_values_are_converted_to_the_target_type():
    assert pasted_value("1.25", float) == 1.25
    assert pasted_value(True, bool) is True
    assert pasted_value({"a": 1}, dict) == {"a": 1}


def test_an_out_of_range_number_raises_what_apply_infotext_skips():
    """apply_infotext skips a value pasted_value cannot convert; "Steps: 1e999" used to escape as a
    500 because int(inf) raises OverflowError, which it did not catch."""
    caught = next(node for node in ast.walk(ast.parse(_SOURCE.read_text(encoding="utf-8"))) if isinstance(node, ast.FunctionDef) and node.name == "apply_infotext")
    handler = next(node for node in ast.walk(caught) if isinstance(node, ast.ExceptHandler))
    skipped = {element.id for element in handler.type.elts}
    for value in ("1e999", float("inf")):
        with pytest.raises(Exception) as error:
            pasted_value(value, int)
        assert type(error.value).__name__ in skipped


def test_sanitize_pnginfo_items_handles_non_utf8_binary_values():
    """pnginfoapi sanitizes binary/non-UTF-8 bytes (e.g., WebP XMP) and nested
    containers so Pydantic PNGInfoResponse serialization never raises UnicodeDecodeError."""
    import json

    raw = {
        "xmp": b"\xff\xfe\x00\x01binary_xmp",
        "text_bytes": b"hello\nworld",
        "null_bytes": b"a\x00b",
        "extension": (b"NETSCAPE2.0", 39),
        "view": memoryview(b"\x80\x81"),
        "surrogate": "bad\udc80text",
        "nonfinite": float("nan"),
        "ok": 42,
    }
    clean = _sanitize_pnginfo_items(raw)
    assert clean["xmp"] == "<14 bytes>"
    assert clean["text_bytes"] == "hello\nworld"
    assert clean["null_bytes"] == "<3 bytes>"
    assert clean["extension"] == ["NETSCAPE2.0", 39]
    assert clean["view"] == "<2 bytes>"
    assert clean["surrogate"] == "bad\ufffdtext"
    assert clean["nonfinite"] == "nan"
    assert clean["ok"] == 42
    json.dumps(clean).encode("utf-8")
