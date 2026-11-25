"""``Options.save`` writes ``config.json`` whole or not at all. It dumped straight into the
open file: a value ``json`` cannot write (nested past the recursion limit, an unpaired
surrogate) stopped it half way, and the next start found a cut-short file and lost every
setting. ``modules/options.py`` is upstream's (upstream-absorption.md §1.2)."""

import json
import os
import sys

import pytest


@pytest.fixture
def opts():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state.opts


def _deep(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("bad", [_deep(200_000), "x\udc80"], ids=["nested", "surrogate"])
def test_a_value_json_cannot_write_leaves_the_file_as_it_was(tmp_path, monkeypatch, opts, bad):
    config = tmp_path / "config.json"
    opts.save(str(config))
    before = config.read_bytes()
    monkeypatch.setitem(opts.data, "remi_test_unwritable", bad)
    with pytest.raises((RecursionError, UnicodeEncodeError)):
        opts.save(str(config))
    assert config.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.json"]  # no temporary file left


def test_a_save_replaces_the_file_whole_keeping_its_link_and_mode(tmp_path, monkeypatch, opts):
    real = tmp_path / "data" / "config.json"
    real.parent.mkdir()
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "config.json"
    if sys.platform != "win32":
        os.chmod(real, 0o640)
        link.symlink_to(real)
    else:
        link = real
    monkeypatch.setitem(opts.data, "remi_test_value", "é")
    opts.save(str(link))
    assert json.loads(real.read_text(encoding="utf-8"))["remi_test_value"] == "é"
    if sys.platform != "win32":
        assert link.is_symlink() and os.readlink(link) == str(real)
        assert real.stat().st_mode & 0o777 == 0o640
    assert sorted(path.name for path in real.parent.iterdir()) == ["config.json"]


def test_a_deep_value_json_can_write_is_saved_whole_and_reads_back(tmp_path, monkeypatch, opts):
    config = tmp_path / "config.json"
    monkeypatch.setitem(opts.data, "remi_test_deep", _deep(3000))
    opts.save(str(config))
    assert json.loads(config.read_text(encoding="utf-8"))["remi_test_deep"] == _deep(3000)
