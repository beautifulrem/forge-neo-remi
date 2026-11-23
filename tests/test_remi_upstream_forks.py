"""``tools/upstream_forks.py``: a Forge Neo merge keeps Remi's forked files.

A forked file that Remi left as it was at the merge base merges without a
conflict, so upstream's version would replace Remi's silently (Remi's
``lib_spectrum/__init__.py`` against upstream's Spectrum rewrite). ``restore``
puts every forked path the merge touched back to HEAD, conflicted or not.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "remi-webui" / "upstream-absorption.md"

spec = importlib.util.spec_from_file_location("tool_upstream_forks", REPO / "tools" / "upstream_forks.py")
forks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forks)


def git(repo: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false", *args]
    return subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False).stdout


def write(repo: Path, files: dict[str, str | None]) -> None:
    for name, text in files.items():
        path = repo / name
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def merging_repo(tmp_path: Path) -> Path:
    """A repository in the middle of ``git merge --no-commit theirs``."""

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, {"fork/init.py": "logger only\n", "fork/core.py": "base\n", "fork/old_ui.py": "ui\n", "runtime.py": "base\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "switch", "-q", "-c", "theirs")
    write(repo, {"fork/init.py": "from .core import Spectrum\n", "fork/core.py": "upstream rewrite\n", "fork/old_ui.py": "ui v2\n", "fork/added.py": "new upstream file\n", "runtime.py": "upstream fix\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "upstream")
    git(repo, "switch", "-q", "main")
    write(repo, {"fork/core.py": "remi fork\n", "fork/old_ui.py": None, "fork/remi_only.py": "remi\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "remi")
    git(repo, "merge", "--no-ff", "--no-commit", "theirs")
    return repo


def test_restore_keeps_remis_version_of_every_forked_file_the_merge_touched(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = merging_repo(tmp_path)
    assert forks.merge_head(repo) is not None
    # what the merge did: a clean merge of init.py, a conflict in core.py, modify/delete of old_ui.py, a new file
    assert (repo / "fork" / "init.py").read_text() == "from .core import Spectrum\n"
    assert set(forks.changed_paths(("fork",), str(repo))) == {"fork/init.py", "fork/core.py", "fork/old_ui.py", "fork/added.py"}

    planned = forks.restore(("fork",), str(repo), dry_run=True)
    assert (repo / "fork" / "init.py").read_text() == "from .core import Spectrum\n"  # a dry run changes nothing
    assert dict(planned) == {
        "fork/init.py": "kept Remi's version",
        "fork/core.py": "kept Remi's version",
        "fork/old_ui.py": "removed (not in Remi)",
        "fork/added.py": "removed (not in Remi)",
    }

    assert forks.restore(("fork",), str(repo)) == planned
    assert (repo / "fork" / "init.py").read_text() == "logger only\n"
    assert (repo / "fork" / "core.py").read_text() == "remi fork\n"
    assert (repo / "fork" / "remi_only.py").read_text() == "remi\n"
    assert not (repo / "fork" / "old_ui.py").exists() and not (repo / "fork" / "added.py").exists()
    assert forks.changed_paths(("fork",), str(repo)) == []
    assert git(repo, "ls-files", "--unmerged") == ""
    # files outside the forked paths keep the merge's result
    assert (repo / "runtime.py").read_text() == "upstream fix\n"
    assert forks.restore(("fork",), str(repo)) == []


def test_no_merge_no_merge_head(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, {"a.py": "a\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a")
    assert forks.merge_head(str(repo)) is None


def documented_forks() -> set[str]:
    """The ``extensions-builtin/…`` paths of the first column of upstream-absorption.md §1.1."""

    section = DOC.read_text(encoding="utf-8").split("### 1.1 ", 1)[1].split("\n### 1.2 ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    found = set()
    for row in rows:
        first = row.split(" | ", 1)[0]
        found |= {re.sub(r"/(\*\*?)?$", "", path) for path in re.findall(r"`(extensions-builtin/[^`]+)`", first)}
    return found


def test_the_tool_and_the_doc_list_the_same_forked_paths():
    assert documented_forks() == set(forks.FORKED)
    for path in forks.FORKED:
        assert (REPO / path).exists() or path.endswith("controlnet_ui"), path  # controlnet_ui: deleted by Remi, kept deleted
    text = DOC.read_text(encoding="utf-8")
    assert "tools/upstream_forks.py restore" in text and "tools/upstream_forks.py review" in text


def documented_divergences() -> dict[str, str]:
    """Upstream-absorption.md §1.2: each row's file and the test it names."""

    section = DOC.read_text(encoding="utf-8").split("### 1.2 ", 1)[1].split("\n## 2.", 1)[0]
    found = {}
    for row in section.splitlines():
        if row.startswith("| `"):
            cells = [cell.strip() for cell in row.strip("|").split(" | ")]
            found[cells[0].strip("`")] = cells[-1].strip("`")
    return found


def test_the_tool_and_the_doc_list_the_same_diverged_files():
    """Every file with a fix of an upstream bug is in ``DIVERGED`` and in §1.2 with the test that pins
    the fix; ``restore`` (the forks) and the divergences do not overlap."""

    rows = documented_divergences()
    assert set(rows) == set(forks.DIVERGED)
    assert len(forks.DIVERGED) == len(set(forks.DIVERGED))
    assert not set(forks.DIVERGED) & set(forks.FORKED)
    for path, test in rows.items():
        assert (REPO / path).is_file(), path
        assert (REPO / test).is_file() and Path(test).name.startswith("test_"), (path, test)
    assert "deterministic upstream bugs are fixed in Remi" in DOC.read_text(encoding="utf-8")


def _synthetic_doc(forked_paths: list[str], diverged_rows: list[tuple[str, str, str]]) -> str:
    fork_lines = "\n".join(f"| `{p}` | remi fork | none |" for p in forked_paths)
    div_lines = "\n".join(f"| `{p}` | {summary} | `{test}` |" for p, summary, test in diverged_rows)
    return (
        "# Absorbing Forge Neo updates\n\n"
        "### 1.1 Built-ins whose processing Remi forked\n\n"
        "| upstream file(s) | Remi's version | seams |\n|---|---|---|\n"
        f"{fork_lines}\n\n"
        "### 1.2 Upstream bugs Remi fixes (divergences)\n\n"
        "| upstream file | Remi's fix | pinning test |\n|---|---|---|\n"
        f"{div_lines}\n\n"
        "## 2. Translating each kind of change\n"
    )


def _setup_synthetic_repo(
    tmp_path: Path,
    *,
    forked: tuple[str, ...] = ("extensions-builtin/ext_a/script.py",),
    diverged_rows: list[tuple[str, str, str]] | None = None,
    create_sources: bool = True,
    create_tests: bool = True,
) -> tuple[Path, Path]:
    repo = tmp_path / "synthetic"
    repo.mkdir()
    if diverged_rows is None:
        diverged_rows = [
            ("backend/fix_a.py", "deterministic fix a", "tests/test_fix_a.py"),
            ("backend/hook_b.py", "remi architecture hook b", "tests/test_hook_b.py"),
        ]
    for path in forked:
        write(repo, {path: "# forked\n"})
    for path, _, test in diverged_rows:
        if create_sources:
            write(repo, {path: "# source\n"})
        if create_tests and test:
            write(repo, {test: "def test_ok(): pass\n"})
    doc = repo / "docs" / "remi-webui" / "upstream-absorption.md"
    write(repo, {"docs/remi-webui/upstream-absorption.md": _synthetic_doc(list(forked), diverged_rows)})
    return repo, doc


def test_budget_passes_on_repository_and_classifies_all_divergences(capsys):
    report = forks.audit_budget()
    assert report.ok, report.errors
    assert report.forked_count == len(forks.FORKED) <= forks.DEFAULT_MAX_FORKED
    assert report.diverged_count == len(forks.DIVERGED)
    assert report.pinning_tests_verified == len(forks.DIVERGED)
    assert report.missing_pinning_tests == ()
    assert set(forks.DIVERGED_CLASSIFICATION) == set(forks.DIVERGED)
    assert sum(report.counts_by_category.values()) == len(forks.DIVERGED)
    # Enough upstream PR candidates exist to achieve the P1.4 reduction target (32 -> <= 20)
    assert report.diverged_count - report.counts_by_category[forks.UPSTREAM_PR_CANDIDATE] <= forks.DEFAULT_TARGET_DIVERGED

    code = forks.main(["budget"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Upstream divergence budget:" in out
    assert f"FORKED:        {len(forks.FORKED)} / max 6 [OK]" in out
    assert f"Pinning tests: {len(forks.DIVERGED)} / {len(forks.DIVERGED)} verified on disk" in out
    assert "Status: PASS" in out

    code_verbose = forks.main(["budget", "--verbose"])
    verbose_out = capsys.readouterr().out
    assert code_verbose == 0
    assert "backend/attention.py" in verbose_out
    assert forks.UPSTREAM_PR_CANDIDATE in verbose_out
    assert forks.REMI_ARCHITECTURE_HOOK in verbose_out


def test_budget_json_output_and_target_diverged_advisory(capsys):
    code = forks.main(["budget", "--max-forked", "6", "--target-diverged", "20", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["forked"]["count"] == len(forks.FORKED)
    assert payload["forked"]["max"] == 6
    assert payload["forked"]["ok"] is True
    assert payload["diverged"]["count"] == len(forks.DIVERGED)
    assert payload["diverged"]["target"] == 20
    assert payload["diverged"]["within_target"] is (len(forks.DIVERGED) <= 20)
    assert payload["diverged"]["over_target"] == max(0, len(forks.DIVERGED) - 20)
    assert payload["pinning_tests"]["verified"] == len(forks.DIVERGED)
    assert payload["pinning_tests"]["missing"] == []
    assert payload["errors"] == []
    assert len(payload["entries"]) == len(forks.DIVERGED)

    code_high = forks.main(["budget", "--target-diverged", str(len(forks.DIVERGED) + 5), "--json"])
    payload_high = json.loads(capsys.readouterr().out)
    assert code_high == 0
    assert payload_high["diverged"]["within_target"] is True
    assert payload_high["diverged"]["over_target"] == 0
    assert payload_high["warnings"] == []


def test_budget_enforces_max_forked_cap(capsys):
    too_low = len(forks.FORKED) - 1
    code = forks.main(["budget", "--max-forked", str(too_low)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FORKED guardrail exceeded" in out
    assert "Status: FAIL" in out

    code_json = forks.main(["budget", "--max-forked", str(too_low), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code_json == 1
    assert payload["ok"] is False
    assert payload["forked"]["ok"] is False
    assert any("FORKED guardrail exceeded" in err for err in payload["errors"])


def test_budget_detects_missing_pinning_test_and_bad_test_name(tmp_path):
    rows = [
        ("backend/fix_a.py", "deterministic fix", "tests/test_missing.py"),
        ("backend/hook_b.py", "remi hook", "tests/helper_not_a_test.py"),
    ]
    repo, doc = _setup_synthetic_repo(tmp_path, diverged_rows=rows, create_tests=False)
    write(repo, {"tests/helper_not_a_test.py": "# not a test_ file\n"})

    report = forks.audit_budget(
        root=str(repo),
        doc_path=str(doc),
        forked=("extensions-builtin/ext_a/script.py",),
        diverged=("backend/fix_a.py", "backend/hook_b.py"),
        classifications={
            "backend/fix_a.py": forks.UPSTREAM_PR_CANDIDATE,
            "backend/hook_b.py": forks.REMI_ARCHITECTURE_HOOK,
        },
        max_forked=2,
        target_diverged=5,
    )
    assert not report.ok
    assert report.pinning_tests_verified == 0
    assert len(report.missing_pinning_tests) == 2
    assert any("pinning test missing on disk for backend/fix_a.py" in e for e in report.errors)
    assert any("pinning test must start with test_ for backend/hook_b.py" in e for e in report.errors)


def test_budget_detects_missing_source_file_and_doc_code_drift(tmp_path):
    rows = [
        ("backend/fix_a.py", "deterministic fix", "tests/test_fix_a.py"),
        ("backend/doc_only.py", "only in doc", "tests/test_doc_only.py"),
        ("backend/fix_a.py", "duplicate row", "tests/test_fix_a.py"),
    ]
    repo, doc = _setup_synthetic_repo(tmp_path, diverged_rows=rows, create_sources=False)
    write(repo, {"backend/doc_only.py": "# exists\n"})

    report = forks.audit_budget(
        root=str(repo),
        doc_path=str(doc),
        forked=("extensions-builtin/ext_a/script.py", "extensions-builtin/ext_b/undoc.py"),
        diverged=("backend/fix_a.py", "backend/code_only.py"),
        classifications={
            "backend/fix_a.py": forks.UPSTREAM_PR_CANDIDATE,
            "backend/code_only.py": forks.REMI_ARCHITECTURE_HOOK,
        },
        max_forked=2,
        target_diverged=5,
    )
    assert not report.ok
    assert any("duplicate §1.2 row" in e for e in report.errors)
    assert any("FORKED path missing from §1.1: extensions-builtin/ext_b/undoc.py" in e for e in report.errors)
    assert any("DIVERGED path missing from §1.2: backend/code_only.py" in e for e in report.errors)
    assert any("§1.2 path missing from DIVERGED: backend/doc_only.py" in e for e in report.errors)
    assert any("DIVERGED file missing on disk: backend/fix_a.py" in e for e in report.errors)


def test_budget_detects_overlap_unclassified_invalid_and_stale_classifications(tmp_path):
    rows = [
        ("backend/fix_a.py", "deterministic fix", "tests/test_fix_a.py"),
        ("backend/hook_b.py", "remi hook", "tests/test_hook_b.py"),
    ]
    repo, doc = _setup_synthetic_repo(
        tmp_path,
        forked=("backend/fix_a.py",),
        diverged_rows=rows,
    )
    report = forks.audit_budget(
        root=str(repo),
        doc_path=str(doc),
        forked=("backend/fix_a.py",),
        diverged=("backend/fix_a.py", "backend/hook_b.py"),
        classifications={
            "backend/fix_a.py": "unknown-category",
            "backend/stale.py": forks.UPSTREAM_PR_CANDIDATE,
        },
        max_forked=2,
        target_diverged=5,
    )
    assert not report.ok
    assert any("both FORKED and DIVERGED: backend/fix_a.py" in e for e in report.errors)
    assert any("invalid classification for backend/fix_a.py" in e for e in report.errors)
    assert any("DIVERGED path lacks classification: backend/hook_b.py" in e for e in report.errors)
    assert any("classification entry not in DIVERGED: backend/stale.py" in e for e in report.errors)


def test_budget_rejects_negative_cli_limits():
    with pytest.raises(SystemExit) as exc1:
        forks.main(["budget", "--max-forked", "-1"])
    assert exc1.value.code == 2

    with pytest.raises(SystemExit) as exc2:
        forks.main(["budget", "--target-diverged", "-1"])
    assert exc2.value.code == 2
