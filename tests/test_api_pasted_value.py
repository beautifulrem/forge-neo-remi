"""``pasted_value``: the API's infotext replay (``modules/api/api.py``) turns
infotext values into request field values. modules.api.api cannot be
imported outside the WebUI, so the function is loaded from its source."""

import ast
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"
_function = next(node for node in ast.parse(_SOURCE.read_text(encoding="utf-8")).body if getattr(node, "name", None) == "pasted_value")
_ns = {}
exec(compile(ast.Module([_function], []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
pasted_value = _ns["pasted_value"]


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
