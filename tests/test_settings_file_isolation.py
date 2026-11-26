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
