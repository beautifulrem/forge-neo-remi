"""The test suite cannot write the owner's settings: the WebUI's settings file is a temporary one.

A test once sent ``PATCH /api/v2/options`` without pointing ``shared.config_filename`` at its own
folder, and the repository's ``config.json`` (the owner's settings, ignored by git, so in no
history) was rewritten with the defaults. ``tests/conftest.py`` now hands the WebUI a settings file
in a temporary folder before anything parses the command line.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_settings_file_is_the_conftests_own():
    from modules import launch_utils, shared, shared_cmd_options

    for where, path in {
        "shared.config_filename": shared.config_filename,
        "cmd_opts.ui_settings_file": shared_cmd_options.cmd_opts.ui_settings_file,
        "launch_utils.args.ui_settings_file": launch_utils.args.ui_settings_file,
    }.items():
        assert not Path(path).resolve().is_relative_to(ROOT), f"{where} is {path}"
        assert Path(path).parent.name.startswith("remi-tests-settings-"), f"{where} is {path}"


def test_the_settings_file_does_not_depend_on_which_test_runs_first(tmp_path):
    # The WebUI reads sys.argv when it is first imported. Test files that reset sys.argv before
    # importing it (test_anima_reference_size.py, the *_harness.py helpers) took the conftest's
    # settings file away when they were collected first, and COMMANDLINE_ARGS, which
    # modules.paths_internal adds to sys.argv, could name another file.
    first = tmp_path / "test_resets_argv_first.py"
    first.write_text("import sys\n\nsys.argv = sys.argv[:1]\nimport modules.shared  # noqa: E402,F401\n\n\ndef test_imported():\n    pass\n", encoding="utf-8")
    env = {**os.environ, "COMMANDLINE_ARGS": f"--ui-settings-file {tmp_path / 'elsewhere.json'}"}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", str(ROOT / "pyproject.toml"), "--rootdir", str(ROOT)]
    command += [str(first), "tests/test_settings_file_isolation.py::test_the_settings_file_is_the_conftests_own"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and "2 passed" in result.stdout, result.stdout[-4000:] + result.stderr[-2000:]
    assert not (tmp_path / "elsewhere.json").exists()


def test_webui_state_saves_settings_in_its_own_folder(tmp_path, monkeypatch):
    # The test that rewrote config.json was a scratch file outside tests/, which this conftest never
    # reaches: it built the WebUI with remi.testing.webui_state and sent PATCH /api/v2/options, which
    # saves to shared.config_filename, there still the command line's file. The helper keeps that
    # file in its own folder, as it does the styles.
    from modules import shared

    from remi.testing import webui_state

    data_folder = tmp_path / "config.json"  # stands for the data folder's settings file
    monkeypatch.setattr(shared.cmd_opts, "ui_settings_file", str(data_folder))
    monkeypatch.setattr(shared, "config_filename", str(data_folder))
    with webui_state() as webui:
        assert Path(shared.config_filename).parent != tmp_path
        webui.opts.save(shared.config_filename)
        assert Path(shared.config_filename).is_file()
    assert not data_folder.exists()
    assert shared.config_filename == str(data_folder)


def test_webui_state_keeps_a_settings_file_the_test_chose(tmp_path, monkeypatch):
    from modules import shared

    from remi.testing import webui_state

    chosen = tmp_path / "chosen.json"
    monkeypatch.setattr(shared, "config_filename", str(chosen))
    with webui_state():
        assert shared.config_filename == str(chosen)


def test_the_test_app_saves_settings_in_its_own_folder(tmp_path):
    # A script that set the WebUI up some other way and built the test app: PATCH /api/v2/options
    # saved to the command line's settings file. The app saves to its own folder instead.
    import json

    from modules import shared

    from remi.server.testing import server_app
    from remi.testing import webui_state

    command_line = tmp_path / "config.json"  # stands for the command line's settings file
    root = tmp_path / "app"
    root.mkdir()
    with webui_state() as webui:
        saved = shared.cmd_opts.ui_settings_file, shared.config_filename
        shared.cmd_opts.ui_settings_file = shared.config_filename = str(command_line)
        try:
            webui.opts.data["jpeg_quality"] = 80
            with server_app(root) as server:
                response = server.client.patch("/api/v2/options", json={"jpeg_quality": 90})
                assert response.status_code == 200, response.text
            assert not command_line.exists()
            assert json.loads((root / "config.json").read_text(encoding="utf-8"))["jpeg_quality"] == 90
            assert shared.config_filename == str(command_line)
        finally:
            shared.cmd_opts.ui_settings_file, shared.config_filename = saved


def test_a_settings_file_whose_folder_is_gone_is_not_taken_for_the_tests_own(tmp_path, monkeypatch):
    # A test that set shared.config_filename inside a helper whose exit ran before monkeypatch's undo
    # left the helper's deleted temporary path behind; the next helper took it for one the test chose
    from modules import shared

    from remi.server.testing import server_app
    from remi.testing import webui_state

    gone = tmp_path / "gone" / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(gone))
    with webui_state() as webui:
        assert Path(shared.config_filename).parent.is_dir()
        webui.opts.save(shared.config_filename)
    assert shared.config_filename == str(gone)

    root = tmp_path / "app"
    root.mkdir()
    with webui_state():
        shared.config_filename = str(gone)
        with server_app(root):
            assert shared.config_filename == str(root / "config.json")
        assert shared.config_filename == str(gone)
