"""The reference docs say what the server sends: a client or extension written
from them gets the fields it will meet (the 422 example in api.md had no
``template``, schema-format.md §4 left out the capability fields, and
architecture.md named another default extension index)."""

import json
import pathlib
import re

from remi.ui import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "remi-webui"


def _json_after(text: str, marker: str) -> object:
    start = text.index("```json", text.index(marker)) + len("```json")
    return json.loads(text[start : text.index("```", start)])


def test_the_api_guide_shows_a_rejected_value_as_the_schema_format_does():
    guide = _json_after((DOCS / "api.md").read_text(encoding="utf-8"), "answers **422**")
    reference = _json_after((DOCS / "schema-format.md").read_text(encoding="utf-8"), "A rejected request answers with")
    assert guide == reference
    assert {"template", "params"} <= set(reference["errors"][0])


def test_the_tab_schema_reference_lists_every_field():
    text = (DOCS / "schema-format.md").read_text(encoding="utf-8")
    section = text[text.index("## 4. Tab schema") :]
    block = section[section.index("```jsonc") : section.index("```", section.index("```jsonc") + 8)]
    keys = set(types.TabSchema.__annotations__) | set(types.ScriptEntry.__annotations__) | set(types.AdviceItemSchema.__annotations__)
    assert sorted(key for key in keys if f'"{key}"' not in block) == []


def test_the_route_table_names_the_default_extension_index():
    source = (ROOT / "remi" / "extensions_admin.py").read_text(encoding="utf-8")
    assert re.search(r"^DEFAULT_INDEX_URL = ", source, flags=re.M)
    table = (DOCS / "architecture.md").read_text(encoding="utf-8")
    row = next(line for line in table.splitlines() if line.startswith("| GET | `/extensions/index"))
    assert "`DEFAULT_INDEX_URL`" in row
