"""``Options.save`` writes ``config.json`` whole or not at all. It dumped straight into the
open file: a value ``json`` cannot write (nested past the recursion limit, an unpaired
surrogate) stopped it half way, and the next start found a cut-short file and lost every
setting. Such a value is now left out of the file. ``modules/options.py`` is upstream's
(upstream-absorption.md §1.2)."""

import errno
import json
import os
import shlex
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


class _FullDisk:
    """A file handle on a full disk: what it writes stops part way with ENOSPC."""

    def __init__(self, handle):
        self.handle = handle

    def write(self, data):
        self.handle.write(data[: len(data) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()


@pytest.mark.parametrize("linked", [False, True], ids=["file", "hard-linked"])
def test_a_full_disk_leaves_the_file_as_it_was(tmp_path, monkeypatch, opts, linked):
    """The new file next to it could not be written for lack of space, and the fallback wrote
    the file in place: it truncated config.json, then ran out of space too, leaving it cut
    short. A hard-linked file was written in place straight away. The save now fails before
    the file is touched."""
    from modules import options

    config = tmp_path / "config.json"
    opts.save(str(config))
    if linked:
        os.link(config, tmp_path / "other-name.json")
    before = config.read_bytes()
    real_open = open
    monkeypatch.setattr(options, "open", lambda path, mode="r", *args, **kwargs: _FullDisk(real_open(path, mode, *args, **kwargs)) if "w" in mode or "x" in mode else real_open(path, mode, *args, **kwargs), raising=False)
    monkeypatch.setitem(opts.data, "remi_test_value", "x" * 1000)
    with pytest.raises(OSError) as error:
        opts.save(str(config))
    assert error.value.errno == errno.ENOSPC
    assert config.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(["config.json", *(["other-name.json"] if linked else [])])


def test_a_refused_swap_still_writes_the_file_in_place(tmp_path, monkeypatch, opts):
    """A file bind-mounted on its own (or held open on Windows) refuses the swap: once the new
    file showed the bytes fit, the file is written in place."""
    config = tmp_path / "config.json"
    opts.save(str(config))

    def refused(source, destination):
        raise OSError(errno.EBUSY, "Device or resource busy")

    monkeypatch.setattr(os, "replace", refused)
    monkeypatch.setitem(opts.data, "remi_test_value", "in place")
    opts.save(str(config))
    assert json.loads(config.read_text(encoding="utf-8"))["remi_test_value"] == "in place"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.json"]


def test_without_a_new_file_the_free_space_decides_before_the_file_is_written(tmp_path, monkeypatch, opts):
    """A folder this process cannot create files in (the file itself writable): no new file can
    show that the bytes fit, so the disk's free space must hold what the file grows by."""
    import shutil
    from types import SimpleNamespace

    from modules import options

    config = tmp_path / "config.json"
    opts.save(str(config))
    before = config.read_bytes()
    real_open = open

    def no_new_files(path, mode="r", *args, **kwargs):
        if "x" in mode:
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(options, "open", no_new_files, raising=False)
    monkeypatch.setitem(opts.data, "remi_test_value", "x" * 1000)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(total=10**9, used=10**9, free=10))
    with pytest.raises(OSError) as error:
        opts.save(str(config))
    assert error.value.errno == errno.ENOSPC and config.read_bytes() == before
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(total=10**9, used=0, free=10**9))
    opts.save(str(config))
    assert json.loads(config.read_text(encoding="utf-8"))["remi_test_value"] == "x" * 1000


def test_under_pytest_the_repositorys_own_settings_file_is_refused(tmp_path, monkeypatch, opts):
    """A test once saved to the repository's ``config.json``, the owner's settings (ignored by
    git, so in no history), and rewrote them with the defaults. Under pytest that file is refused
    whatever path names it, so a test outside ``tests/`` (which ``tests/conftest.py`` never
    reaches) cannot do it again; the WebUI itself still saves there."""
    from modules import options

    owners = tmp_path / "repository" / "config.json"  # stands for the repository's settings file
    owners.parent.mkdir()
    owners.write_text('{"owner": 1}', encoding="utf-8")
    monkeypatch.setattr(options, "_REPOSITORY_SETTINGS", str(owners))
    os.link(owners, tmp_path / "hard.json")
    spellings = [owners, tmp_path / "repository" / ".." / "repository" / "config.json", tmp_path / "hard.json"]
    if sys.platform != "win32":
        (tmp_path / "link.json").symlink_to(owners)
        spellings.append(tmp_path / "link.json")
    for path in spellings:
        with pytest.raises(RuntimeError, match="repository's"):
            opts.save(str(path))
    assert owners.read_text(encoding="utf-8") == '{"owner": 1}'

    opts.save(str(tmp_path / "config.json"))  # a file of the test's own is saved
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))

    # between tests and while pytest collects them (module-level code) only PYTEST_VERSION is set
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    with pytest.raises(RuntimeError, match="repository's"):
        opts.save(str(owners))
    monkeypatch.delenv("PYTEST_VERSION", raising=False)  # outside pytest: the WebUI's own save
    opts.save(str(owners))
    assert "owner" not in json.loads(owners.read_text(encoding="utf-8"))


def test_the_guarded_file_is_the_repositorys_default_settings_file():
    # a WebUI a test starts with --data-dir <its folder> saves its settings there
    from modules import cmd_args, options, paths_internal

    guarded = options._REPOSITORY_SETTINGS
    assert guarded == os.path.join(paths_internal.script_path, "config.json")
    assert cmd_args.parser.get_default("ui_settings_file") == os.path.join(paths_internal.data_path, "config.json")
    assert paths_internal.parser.get_default("data_dir") == paths_internal.script_path


@pytest.mark.parametrize("form", ["--data-dir {}", "--data-dir={}", "--data {}", "--ui-settings-file {file}", "--ui-settings-file={file}", "--ui-settings {file}"])
def test_under_pytest_the_settings_file_commandline_args_named_is_refused(tmp_path, monkeypatch, opts, form):
    """An extension suite (its conftest does not drop COMMANDLINE_ARGS) run from a shell that exports
    it takes its --data-dir or --ui-settings-file, as argparse reads them (abbreviated too): the
    owner's settings outside the repository. The value that reached the WebUI's command line at
    import counts, not the environment at save time (tests/conftest.py drops it later)."""
    from modules import options, paths_internal

    owners = tmp_path / "owner data" / "config.json"
    owners.parent.mkdir()
    owners.write_text('{"owner": 1}', encoding="utf-8")
    monkeypatch.setattr(options, "_TEMPORARY", str(tmp_path / "elsewhere"))  # the owner's folder is not a temporary one
    monkeypatch.setattr(paths_internal, "commandline_args", "--cpu " + form.format(shlex.quote(str(owners.parent)), file=shlex.quote(str(owners))))
    monkeypatch.delenv("COMMANDLINE_ARGS", raising=False)
    with pytest.raises(RuntimeError, match="COMMANDLINE_ARGS named"):
        opts.save(str(owners))
    assert owners.read_text(encoding="utf-8") == '{"owner": 1}'
    opts.save(str(tmp_path / "config.json"))  # a file of the test's own is saved


def test_a_temporary_folder_commandline_args_names_is_the_tests_own(tmp_path, monkeypatch, opts):
    # a WebUI a test starts with COMMANDLINE_ARGS="--data-dir <its folder>" saves there
    from modules import paths_internal

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(paths_internal, "commandline_args", f"--data-dir {shlex.quote(str(data))}")
    opts.save(str(data / "config.json"))
    assert json.loads((data / "config.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("value", ["--data-dir 'unbalanced", "--data-dir", "--data-dir="])
def test_commandline_args_the_guard_cannot_read_leave_the_repositorys_file_guarded(tmp_path, monkeypatch, opts, value):
    from modules import options, paths_internal

    monkeypatch.setattr(paths_internal, "commandline_args", value)
    owners = tmp_path / "config.json"
    owners.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(options, "_REPOSITORY_SETTINGS", str(owners))
    with pytest.raises(RuntimeError, match="repository's"):
        opts.save(str(owners))
    opts.save(str(tmp_path / "own.json"))
