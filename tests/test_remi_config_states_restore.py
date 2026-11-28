"""Pinning tests for config state extension enablement restore and
preserving disabled built-in extensions across restore / initialize_rest."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from remi.testing import webui_state


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return result.stdout.strip()


def _make_ext(extensions, name: str, path: Path, *, enabled: bool = True, is_builtin: bool = False):
    meta = extensions.ExtensionMetadata(str(path), name.lower())
    return extensions.Extension(name, str(path), enabled=enabled, is_builtin=is_builtin, metadata=meta)


def test_restore_extension_config_sets_enabled_and_preserves_disabled_builtins(tmp_path, monkeypatch):
    from modules import config_states, extensions, shared

    with webui_state():
        cfg_file = tmp_path / "config.json"
        monkeypatch.setattr(shared, "config_filename", str(cfg_file))

        ext_builtin_off = _make_ext(extensions, "builtin-off", tmp_path / "builtin" / "builtin-off", enabled=True, is_builtin=True)
        ext_builtin_on = _make_ext(extensions, "builtin-on", tmp_path / "builtin" / "builtin-on", enabled=True, is_builtin=True)
        ext_builtin_unlisted = _make_ext(extensions, "builtin-unlisted", tmp_path / "builtin" / "builtin-unlisted", enabled=False, is_builtin=True)

        user_a_dir = tmp_path / "extensions" / "user-a"
        user_a_dir.mkdir(parents=True)
        (user_a_dir / "README.md").write_text("v1", encoding="utf-8")
        _git("init", "-q", "-b", "main", cwd=user_a_dir)
        _git("add", ".", cwd=user_a_dir)
        _git("commit", "-q", "-m", "v1", cwd=user_a_dir)
        commit_v1 = _git("rev-parse", "HEAD", cwd=user_a_dir)

        ext_user_a = _make_ext(extensions, "user-a", user_a_dir, enabled=False, is_builtin=False)
        ext_user_b = _make_ext(extensions, "user-b", tmp_path / "extensions" / "user-b", enabled=True, is_builtin=False)

        monkeypatch.setattr(
            extensions,
            "extensions",
            [ext_builtin_off, ext_builtin_on, ext_builtin_unlisted, ext_user_a, ext_user_b],
        )
        monkeypatch.setattr(shared.opts, "disabled_extensions", ["builtin-unlisted"], raising=False)

        saved_state = {
            "created_at": 1_700_000_000.0,
            "webui": {},
            "extensions": {
                "builtin-off": {"name": "builtin-off", "enabled": False, "is_builtin": True},
                "builtin-on": {"name": "builtin-on", "enabled": True, "is_builtin": True},
                "user-a": {"name": "user-a", "enabled": True, "is_builtin": False, "commit_hash": commit_v1},
            },
        }
        monkeypatch.setattr(ext_user_a, "fetch_and_reset_hard", lambda commit=None: None)
        monkeypatch.setattr(ext_user_b, "read_info_from_repo", lambda: None)

        config_states.restore_extension_config(saved_state)

        assert ext_builtin_off.enabled is False
        assert ext_builtin_on.enabled is True
        assert ext_builtin_unlisted.enabled is False
        assert ext_user_a.enabled is True
        assert ext_user_b.enabled is False
        assert not hasattr(ext_user_a, "disabled")
        assert not hasattr(ext_user_b, "disabled")
        assert set(shared.opts.disabled_extensions) == {"builtin-off", "builtin-unlisted", "user-b"}


def test_initialize_rest_applies_restored_extension_enablement(tmp_path, monkeypatch):
    from modules import extensions, initialize, sd_samplers, shared

    with webui_state():
        builtin_dir = tmp_path / "extensions-builtin"
        ext_dir = tmp_path / "extensions"
        (builtin_dir / "builtin-core").mkdir(parents=True)
        user_dir = ext_dir / "user-ext"
        user_dir.mkdir(parents=True)
        (user_dir / "a.txt").write_text("v1", encoding="utf-8")
        _git("init", "-q", "-b", "main", cwd=user_dir)
        _git("add", ".", cwd=user_dir)
        _git("commit", "-q", "-m", "v1", cwd=user_dir)
        v1 = _git("rev-parse", "HEAD", cwd=user_dir)

        monkeypatch.setattr(extensions, "extensions_builtin_dir", str(builtin_dir))
        monkeypatch.setattr(extensions, "extensions_dir", str(ext_dir))
        monkeypatch.setattr(shared, "config_filename", str(tmp_path / "config.json"))
        monkeypatch.setattr(shared.opts, "disabled_extensions", ["builtin-core", "user-ext"], raising=False)

        backup_path = tmp_path / "state.json"
        backup_path.write_text(
            json.dumps(
                {
                    "created_at": 1_700_000_000.0,
                    "webui": {},
                    "extensions": {
                        "builtin-core": {"name": "builtin-core", "enabled": False, "is_builtin": True},
                        "user-ext": {"name": "user-ext", "enabled": True, "is_builtin": False, "commit_hash": v1},
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(shared.opts, "restore_config_state_file", str(backup_path), raising=False)
        monkeypatch.setattr(extensions.Extension, "fetch_and_reset_hard", lambda self, commit=None: None)
        monkeypatch.setattr(sd_samplers, "set_samplers", lambda: None)
        monkeypatch.setattr(shared.cmd_opts, "ui_debug_mode", True, raising=False)

        from modules import scripts

        monkeypatch.setattr(scripts, "load_scripts", lambda: None)

        initialize.initialize_rest()

        by_name = {ext.name: ext for ext in extensions.extensions}
        assert by_name["builtin-core"].enabled is False
        assert by_name["user-ext"].enabled is True
        assert shared.opts.disabled_extensions == ["builtin-core"]
        assert shared.opts.restore_config_state_file == ""


def test_list_config_states_skips_non_finite_and_out_of_range_timestamps(tmp_path, monkeypatch):
    from modules import config_states

    states_dir = tmp_path / "config_states"
    states_dir.mkdir()
    (states_dir / "bad_nan.json").write_text('{"created_at": NaN, "name": "NaNState"}', encoding="utf-8")
    (states_dir / "bad_huge.json").write_text('{"created_at": 1e30, "name": "HugeState"}', encoding="utf-8")
    (states_dir / "bad_bool.json").write_text('{"created_at": true, "name": "BoolState"}', encoding="utf-8")
    (states_dir / "bad_list.json").write_text("[]", encoding="utf-8")
    (states_dir / "valid.json").write_text(json.dumps({"created_at": 1_700_000_000.0, "name": "GoodState"}), encoding="utf-8")

    monkeypatch.setattr(config_states, "config_states_dir", str(states_dir))
    listed = config_states.list_config_states()
    assert len(listed) == 1
    assert next(iter(listed)).startswith("GoodState:")
