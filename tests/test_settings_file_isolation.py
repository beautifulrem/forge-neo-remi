"""The test suite cannot write the owner's settings: the WebUI's settings file is a temporary one.

A test once sent ``PATCH /api/v2/options`` without pointing ``shared.config_filename`` at its own
folder, and the repository's ``config.json`` (the owner's settings, ignored by git, so in no
history) was rewritten with the defaults. ``tests/conftest.py`` now hands the WebUI a settings file
in a temporary folder before anything parses the command line.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_settings_file_is_outside_the_repository():
    from modules import launch_utils, shared, shared_cmd_options

    for where, path in {
        "shared.config_filename": shared.config_filename,
        "cmd_opts.ui_settings_file": shared_cmd_options.cmd_opts.ui_settings_file,
        "launch_utils.args.ui_settings_file": launch_utils.args.ui_settings_file,
    }.items():
        assert not Path(path).resolve().is_relative_to(ROOT), f"{where} is {path}"


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
