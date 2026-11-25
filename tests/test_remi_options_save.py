"""``Options.save`` writes ``config.json`` whole or not at all. It dumped straight into the
open file: a value ``json`` cannot write (nested past the recursion limit, an unpaired
surrogate) stopped it half way, and the next start found a cut-short file and lost every
setting. Such a value is now left out of the file. ``modules/options.py`` is upstream's
(upstream-absorption.md §1.2)."""

import json
import os
import sys
import time

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


@pytest.mark.parametrize("bad", [_deep(200_000), "x\udc80", {"a set"}], ids=["nested", "surrogate", "set"])
def test_a_value_json_cannot_write_is_left_out_with_an_error(tmp_path, monkeypatch, capsys, opts, bad):
    """Such a value (put into ``opts.data`` by an extension, past the API's checks) refused every
    later save until a restart: the other settings are saved, and the error names the setting."""
    config = tmp_path / "config.json"
    opts.save(str(config))
    before = json.loads(config.read_text(encoding="utf-8"))
    monkeypatch.setitem(opts.data, "remi_test_unwritable", bad)
    monkeypatch.setitem(opts.data, "remi_test_value", "saved")
    capsys.readouterr()
    opts.save(str(config))
    assert json.loads(config.read_text(encoding="utf-8")) == {**before, "remi_test_value": "saved"}
    assert "remi_test_unwritable" in capsys.readouterr().err and opts.data["remi_test_unwritable"] is bad
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and hard links")
def test_a_read_only_file_still_refuses_the_save_and_a_hard_link_is_kept(tmp_path, monkeypatch, opts):
    """Swapping a new file in replaced a ``chmod 444`` settings file (the directory allowed it),
    where upstream's ``open(..., "w")`` refused: the file stays locked. A hard link would be cut
    from the new file: such a file is written in place, as before."""
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    os.chmod(config, 0o444)
    monkeypatch.setitem(opts.data, "remi_test_value", "new")
    with pytest.raises(PermissionError):
        opts.save(str(config))
    assert config.read_text(encoding="utf-8") == "{}"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.json"]

    os.chmod(config, 0o644)
    other = tmp_path / "other-name.json"
    os.link(config, other)
    opts.save(str(config))
    assert json.loads(other.read_text(encoding="utf-8"))["remi_test_value"] == "new"
    assert config.stat().st_ino == other.stat().st_ino


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owners and groups")
def test_a_save_keeps_the_files_owner_and_group(tmp_path, monkeypatch, opts):
    """A new file swapped in belongs to the process: a settings file of another group (one shared
    with another account) or of another user changed hands at every save, where upstream's
    ``open(..., "w")`` kept them. The new file is given them, or the file is written in place when
    it cannot be (another user's file)."""
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    groups = [gid for gid in os.getgroups() if gid != config.stat().st_gid]
    if not groups:
        pytest.skip("the user is in one group only")
    os.chown(config, -1, groups[0])
    monkeypatch.setitem(opts.data, "remi_test_value", "new")
    opts.save(str(config))
    assert config.stat().st_gid == groups[0] and json.loads(config.read_text(encoding="utf-8"))["remi_test_value"] == "new"

    def refused(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chown", refused)  # as for a file of another user
    inode = config.stat().st_ino
    monkeypatch.setitem(opts.data, "remi_test_value", "newer")
    opts.save(str(config))
    assert config.stat().st_ino == inode and config.stat().st_gid == groups[0]
    assert json.loads(config.read_text(encoding="utf-8"))["remi_test_value"] == "newer"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.json"]


def test_a_temporary_file_a_crash_left_is_removed_at_the_next_save(tmp_path, opts):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    stale = tmp_path / f"config.json.{'a' * 32}.tmp"
    under_way = tmp_path / f"config.json.{'b' * 32}.tmp"  # another process's save, still being written
    other = tmp_path / "config.json.backup.tmp"
    for path in (stale, under_way, other):
        path.write_text("{", encoding="utf-8")
    hour_ago = time.time() - 3600
    os.utime(stale, (hour_ago, hour_ago))
    os.utime(other, (hour_ago, hour_ago))
    opts.save(str(config))
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(["config.json", under_way.name, other.name])


def test_a_value_held_for_the_run_is_saved_as_it_was_stored(tmp_path, monkeypatch, opts):
    """A value the WebUI had to fill in for this run (a frozen setting) is not saved while it
    is still the live value; once something else is set, that is saved."""

    config = tmp_path / "config.json"
    monkeypatch.setitem(opts.data, "sd_model_checkpoint", "")
    opts.hold_for_run("sd_model_checkpoint", "anima.safetensors")
    assert opts.sd_model_checkpoint == "anima.safetensors"
    opts.save(str(config))
    assert json.loads(config.read_text(encoding="utf8"))["sd_model_checkpoint"] == ""
    opts.data["sd_model_checkpoint"] = "other.safetensors"
    opts.save(str(config))
    assert json.loads(config.read_text(encoding="utf8"))["sd_model_checkpoint"] == "other.safetensors"
    opts.data.pop("remi_test_held", None)
    opts.hold_for_run("remi_test_held", 1)  # a key the file did not have stays out of it
    opts.save(str(config))
    assert "remi_test_held" not in json.loads(config.read_text(encoding="utf8"))
