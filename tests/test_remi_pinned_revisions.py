"""The revisions the legacy-UI tools pin are commits of this history.

The recorder (``tools/record_legacy_ui.py``), its fixture list
(``tools/legacy_ui_recordings.py``), the extension settings record
(``tools/record_legacy_settings.py`` and its fixture) and the recorder's
environment test name revisions by hash. A history rewrite (2026-11) left them
naming commits only the old branch had: git keeps those only while a reflog
entry does, then prunes them, and the legacy-ui gates fail and these tests
skip from then on. Every quoted hash in those files must be an ancestor of
HEAD.
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tool_git_history_pinned", REPO / "tools" / "git_history.py")
history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(history)
PINNED_IN = ("tools/record_legacy_ui.py", "tools/legacy_ui_recordings.py", "tools/record_legacy_settings.py", "tests/test_remi_legacy_ui_env.py")
QUOTED_HASH = re.compile(r"""["']([0-9a-f]{7,40})["']""")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def pinned() -> dict[str, str]:
    revisions = {f"{path}: {rev}": rev for path in PINNED_IN for rev in QUOTED_HASH.findall((REPO / path).read_text(encoding="utf-8"))}
    fixture = json.loads((REPO / "tests" / "fixtures" / "legacy_ui" / "extension_settings.json").read_text(encoding="utf-8"))
    revisions[f"extension_settings.json: {fixture['revision']}"] = fixture["revision"]
    return revisions


def test_the_pinned_revisions_are_found():
    names = pinned()
    assert len(names) >= 8, names  # the recorder's three, the fixture list's three, the settings record and its fixture


def test_the_pinned_revisions_are_commits_of_this_history():
    # a shallow clone (CI) fetches the whole history first (tests/test_remi_git_history.py)
    if (reason := history.ensure_history(REPO)) is not None:
        pytest.skip(reason)
    gone = sorted(name for name, rev in pinned().items() if _git("merge-base", "--is-ancestor", rev, "HEAD").returncode != 0)
    assert gone == []
