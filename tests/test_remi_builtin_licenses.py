"""Every built-in extension's license is accounted for: a directory under
``extensions-builtin/`` either carries a license file of its own or has a row
in the table of ``docs/remi-webui/licensing.md`` (origin, license, third-party
code, evidence). A new directory with neither fails here."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "remi-webui" / "licensing.md"
BUILTIN = ROOT / "extensions-builtin"
SECTION = "## Built-in extensions"
LICENSE_FILE = re.compile(r"^(LICENSE|LICENCE|COPYING)", re.IGNORECASE)


def table_rows() -> dict[str, list[str]]:
    """The table's rows by directory (its first cell, in backticks)."""

    text = DOC.read_text(encoding="utf-8")
    assert SECTION in text, f"{DOC.name} has no {SECTION!r} section"
    section = text.split(SECTION, 1)[1].split("\n## ", 1)[0]
    rows: dict[str, list[str]] = {}
    for line in section.splitlines():
        match = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if match:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            assert match.group(1) not in rows, f"{match.group(1)}: two rows"
            rows[match.group(1)] = cells
    return rows


def directories() -> list[str]:
    return sorted(path.name for path in BUILTIN.iterdir() if path.is_dir() and not path.name.startswith((".", "__")))


def has_license_file(name: str) -> bool:
    return any(LICENSE_FILE.match(path.name) and path.is_file() for path in (BUILTIN / name).iterdir())


def test_every_builtin_extension_has_a_license_file_or_a_row():
    rows = table_rows()
    missing = [name for name in directories() if name not in rows and not has_license_file(name)]
    assert missing == [], f"no license file and no row in {DOC.relative_to(ROOT)}: {missing}"


def test_every_row_names_a_directory_and_fills_each_column():
    rows = table_rows()
    assert rows, "the table is empty"
    assert sorted(set(rows) - set(directories())) == [], "rows for directories that are gone"
    for name, cells in rows.items():
        assert len(cells) == 5 and all(cells), f"{name}: directory, origin, license, third-party code, evidence"


def test_license_files_the_table_names_are_there():
    for name, cells in table_rows().items():
        for file in re.findall(r"`((?:LICENSE|LICENCE|COPYING|NOTICE)[\w.-]*)`", " ".join(cells[2:])):
            assert (BUILTIN / name / file).is_file(), f"{name}: the table names {file}"
