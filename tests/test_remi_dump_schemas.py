"""``webui/tools/dump_schemas.py --check`` (the ``contract:schemas`` step of tools/check.sh): a
fixture the SPA's contract tests still render but the schema dump no longer writes (a renamed or
removed tab) is reported, as an out-of-date one is."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "webui" / "tools" / "dump_schemas.py"


@pytest.fixture
def dump(monkeypatch):
    """The script as a module, with the live schemas replaced by fixed text; ``main()``'s changes to
    the working directory, ``sys.argv`` and ``sys.path`` are undone after the test."""

    spec = importlib.util.spec_from_file_location("dump_schemas_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "schema_files", lambda out: {os.path.join(out, f"{tab}.json"): f'{{"tab": "{tab}"}}\n' for tab in module.TABS})
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return module


def check(module, monkeypatch, out: Path) -> int:
    monkeypatch.setattr(sys, "argv", ["dump_schemas.py", "--check", "--out", str(out)])
    return module.main()


def test_check_reports_fixtures_the_dump_no_longer_writes(dump, monkeypatch, tmp_path, capsys):
    for tab in dump.TABS:
        (tmp_path / f"{tab}.json").write_text(f'{{"tab": "{tab}"}}\n', encoding="utf-8")
    # settings.json is tools/dump_settings.py's fixture in the same folder; other files are no fixtures
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    assert check(dump, monkeypatch, tmp_path) == 0
    capsys.readouterr()

    (tmp_path / "old.json").write_text("{}\n", encoding="utf-8")
    assert check(dump, monkeypatch, tmp_path) == 1
    out = capsys.readouterr().out
    assert "old.json" in out and "settings.json" not in out and "notes.txt" not in out

    (tmp_path / "old.json").unlink()
    (tmp_path / "img2img.json").write_text("{}\n", encoding="utf-8")
    assert check(dump, monkeypatch, tmp_path) == 1
    assert "img2img.json is out of date" in capsys.readouterr().out
