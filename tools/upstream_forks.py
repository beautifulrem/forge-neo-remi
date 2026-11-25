"""Keep Remi's version of the upstream files it forked while merging Forge Neo.

Most files Remi shares with Forge Neo are merged as usual. A few built-ins had
their processing rewritten by Remi (docs/remi-webui/upstream-absorption.md
§1.1): for them the rule is *keep Remi's version and port upstream's fixes
selectively*. A conflict there is easy to see, but Git merges a forked file
without a conflict whenever Remi left it as it was at the merge base (Remi's
``lib_spectrum/__init__.py`` only holds a logger, so upstream's 2026 rewrite of
it would come in silently and import a ``Spectrum`` class Remi's
``forecaster.py`` does not have). So after ``git merge --no-commit`` every
forked path the merge touched, conflicted or not, goes back to Remi's version
here, and the tool prints the upstream diff to read for the selective port.

The files where Remi fixed an upstream bug but otherwise keeps upstream's code
(§1.2, ``DIVERGED``) are not restored: they merge as usual, ``review`` lists the
upstream commits on them, and the test each §1.2 row names fails if a merge
dropped the fix.

    $PYTHON tools/upstream_forks.py review               # before the merge: upstream commits and
                                                         # diffstat on the forked paths
    $PYTHON tools/upstream_forks.py restore --dry-run    # during the merge: what restore would do
    $PYTHON tools/upstream_forks.py restore              # during the merge: forked paths back to HEAD
    $PYTHON tools/upstream_forks.py budget               # governance check: FORKED cap, §1.2 pinning
                                                         # tests, and DIVERGED classification

``restore`` refuses to run outside a merge (no ``MERGE_HEAD``), and should run
right after ``git merge`` (it also undoes edits to these paths made since).
``$PYTHON``: any Python 3.10+ (standard library only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = "upstream/neo"
DOC_REL_PATH = os.path.join("docs", "remi-webui", "upstream-absorption.md")
DEFAULT_MAX_FORKED = 6
DEFAULT_TARGET_DIVERGED = 20

UPSTREAM_PR_CANDIDATE = "upstream-pr-candidate"
REMI_ARCHITECTURE_HOOK = "remi-architecture-hook"
DIVERGED_CATEGORIES = frozenset({UPSTREAM_PR_CANDIDATE, REMI_ARCHITECTURE_HOOK})

FORKED = (
    "extensions-builtin/sd_forge_spectrum/lib_spectrum",
    "extensions-builtin/sd_forge_spectrum/scripts/spectrum.py",
    "extensions-builtin/sd_forge_compile/scripts/compile.py",
    "extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui",
    "extensions-builtin/sd_forge_controlnet/lib_controlnet/infotext.py",
    "extensions-builtin/sd_forge_controlnet/lib_controlnet/external_code.py",
)
"""The paths of upstream-absorption.md §1.1 (files and folders), one per forked
file or folder; ``tests/test_remi_upstream_forks.py`` keeps the two in step."""

DIVERGED = (
    "backend/attention.py",
    "backend/memory_management.py",
    "backend/nn/krea.py",
    "backend/nn/qwen21.py",
    "backend/sampling/sampling_function.py",
    "backend/text_processing/anima_engine.py",
    "extensions-builtin/forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py",
    "extensions-builtin/forge_preprocessor_reference/scripts/forge_reference.py",
    "extensions-builtin/forge_preprocessor_tile/scripts/preprocessor_tile.py",
    "extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite.py",
    "extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite_anima.py",
    "extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py",
    "extensions-builtin/sd_forge_controlnet/lib_controlnet/global_state.py",
    "extensions-builtin/sd_forge_controlnet/scripts/controlnet.py",
    "extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py",
    "extensions-builtin/sd_forge_lora/networks.py",
    "extensions-builtin/sd_forge_lora_schedule/scripts/control.py",
    "extensions-builtin/sd_forge_radial/scripts/forge_radial.py",
    "extensions-builtin/soft-inpainting/scripts/soft_inpainting.py",
    "modules/api/api.py",
    "modules/extras.py",
    "modules/images.py",
    "modules/img2img.py",
    "modules/infotext_utils.py",
    "modules/ngrok.py",
    "modules/processing.py",
    "modules/processing_scripts/refiner.py",
    "modules/sd_samplers.py",
    "modules/sd_samplers_common.py",
    "modules/sd_schedulers.py",
    "modules_forge/packages/comfy/lora.py",
    "modules_forge/packages/k_diffusion/external.py",
    "modules_forge/supported_controlnet.py",
)
"""Upstream files that keep upstream's code but carry Remi's fixes of upstream bugs
(upstream-absorption.md §1.2, one row per file naming the test that pins the fix).
They merge normally, and ``restore`` never touches them: a merge that conflicts
there keeps the fix, and the pinning tests fail if a clean merge drops one."""

DIVERGED_CLASSIFICATION: dict[str, str] = {
    "backend/attention.py": UPSTREAM_PR_CANDIDATE,
    "backend/memory_management.py": UPSTREAM_PR_CANDIDATE,
    "backend/nn/krea.py": REMI_ARCHITECTURE_HOOK,
    "backend/nn/qwen21.py": REMI_ARCHITECTURE_HOOK,
    "backend/sampling/sampling_function.py": REMI_ARCHITECTURE_HOOK,
    "backend/text_processing/anima_engine.py": REMI_ARCHITECTURE_HOOK,
    "extensions-builtin/forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/forge_preprocessor_reference/scripts/forge_reference.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/forge_preprocessor_tile/scripts/preprocessor_tile.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite_anima.py": REMI_ARCHITECTURE_HOOK,
    "extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py": REMI_ARCHITECTURE_HOOK,
    "extensions-builtin/sd_forge_controlnet/lib_controlnet/global_state.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/sd_forge_controlnet/scripts/controlnet.py": REMI_ARCHITECTURE_HOOK,
    "extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/sd_forge_lora/networks.py": REMI_ARCHITECTURE_HOOK,
    "extensions-builtin/sd_forge_lora_schedule/scripts/control.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/sd_forge_radial/scripts/forge_radial.py": UPSTREAM_PR_CANDIDATE,
    "extensions-builtin/soft-inpainting/scripts/soft_inpainting.py": UPSTREAM_PR_CANDIDATE,
    "modules/api/api.py": REMI_ARCHITECTURE_HOOK,
    "modules/extras.py": UPSTREAM_PR_CANDIDATE,
    "modules/images.py": UPSTREAM_PR_CANDIDATE,
    "modules/img2img.py": UPSTREAM_PR_CANDIDATE,
    "modules/infotext_utils.py": UPSTREAM_PR_CANDIDATE,
    "modules/ngrok.py": UPSTREAM_PR_CANDIDATE,
    "modules/processing.py": REMI_ARCHITECTURE_HOOK,
    "modules/processing_scripts/refiner.py": UPSTREAM_PR_CANDIDATE,
    "modules/sd_samplers.py": UPSTREAM_PR_CANDIDATE,
    "modules/sd_samplers_common.py": UPSTREAM_PR_CANDIDATE,
    "modules/sd_schedulers.py": REMI_ARCHITECTURE_HOOK,
    "modules_forge/packages/comfy/lora.py": REMI_ARCHITECTURE_HOOK,
    "modules_forge/packages/k_diffusion/external.py": UPSTREAM_PR_CANDIDATE,
    "modules_forge/supported_controlnet.py": UPSTREAM_PR_CANDIDATE,
}
"""Governance classification of every ``DIVERGED`` path: ``upstream-pr-candidate``
(pure deterministic Forge Neo bug fix eligible to drop out of ``DIVERGED`` once
upstreamed) vs ``remi-architecture-hook`` (carries Remi-specific runtime hooks or
DiT architecture extensions; see ``docs/remi-webui/upstream-pr-candidates.md``)."""


class DocumentedDivergence(NamedTuple):
    path: str
    summary: str
    pinning_test: str


class DivergedAuditRow(NamedTuple):
    path: str
    category: str | None
    summary: str
    pinning_test: str | None
    file_exists: bool
    pinning_test_exists: bool
    in_code: bool
    in_doc: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category,
            "pinning_test": self.pinning_test,
            "file_exists": self.file_exists,
            "pinning_test_exists": self.pinning_test_exists,
            "in_code": self.in_code,
            "in_doc": self.in_doc,
            "summary": self.summary,
        }


class BudgetReport(NamedTuple):
    forked_count: int
    max_forked: int
    forked_ok: bool
    forked_paths: tuple[str, ...]
    diverged_count: int
    target_diverged: int
    over_target: int
    within_target: bool
    counts_by_category: dict[str, int]
    pinning_tests_total: int
    pinning_tests_verified: int
    missing_pinning_tests: tuple[str, ...]
    rows: tuple[DivergedAuditRow, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "forked": {
                "count": self.forked_count,
                "max": self.max_forked,
                "ok": self.forked_ok,
                "paths": list(self.forked_paths),
            },
            "diverged": {
                "count": self.diverged_count,
                "target": self.target_diverged,
                "over_target": self.over_target,
                "within_target": self.within_target,
                "counts_by_category": dict(self.counts_by_category),
            },
            "pinning_tests": {
                "total": self.pinning_tests_total,
                "verified": self.pinning_tests_verified,
                "missing": list(self.missing_pinning_tests),
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "entries": [row.to_dict() for row in self.rows],
        }


def parse_documented_forks(doc_text: str) -> set[str]:
    """Extract the forked paths from ``upstream-absorption.md`` §1.1."""

    if "### 1.1 " not in doc_text or "\n### 1.2 " not in doc_text:
        raise ValueError("upstream-absorption.md is missing ### 1.1 / ### 1.2 sections")
    section = doc_text.split("### 1.1 ", 1)[1].split("\n### 1.2 ", 1)[0]
    found: set[str] = set()
    for row in section.splitlines():
        if not row.startswith("| `"):
            continue
        first = row.split(" | ", 1)[0]
        found |= {re.sub(r"/(\*\*?)?$", "", path) for path in re.findall(r"`([^`\s]+/[^`\s]*)`", first)}
    return found


def parse_documented_divergences(doc_text: str) -> tuple[dict[str, DocumentedDivergence], list[str]]:
    """Extract every §1.2 row (file, fix summary, pinning test) and any duplicate paths."""

    if "### 1.2 " not in doc_text or "\n## 2." not in doc_text:
        raise ValueError("upstream-absorption.md is missing ### 1.2 / ## 2. sections")
    section = doc_text.split("### 1.2 ", 1)[1].split("\n## 2.", 1)[0]
    rows: dict[str, DocumentedDivergence] = {}
    duplicates: list[str] = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split(" | ")]
        if len(cells) < 3:
            continue
        path = cells[0].strip("`")
        summary = cells[1].strip()
        pinning_test = cells[-1].strip("`")
        if path in rows:
            duplicates.append(path)
        rows[path] = DocumentedDivergence(path=path, summary=summary, pinning_test=pinning_test)
    return rows, duplicates


def _duplicates(items: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    dups: list[str] = []
    for item in items:
        if item in seen and item not in dups:
            dups.append(item)
        seen.add(item)
    return dups


def audit_budget(
    *,
    root: str = ROOT,
    doc_path: str | None = None,
    forked: tuple[str, ...] = FORKED,
    diverged: tuple[str, ...] = DIVERGED,
    classifications: dict[str, str] | None = None,
    max_forked: int = DEFAULT_MAX_FORKED,
    target_diverged: int = DEFAULT_TARGET_DIVERGED,
) -> BudgetReport:
    """Enforce the FORKED guardrail and audit DIVERGED rows against §1.2 and disk."""

    root_path = Path(root)
    doc = Path(doc_path) if doc_path is not None else root_path / DOC_REL_PATH
    class_map = DIVERGED_CLASSIFICATION if classifications is None else classifications

    errors: list[str] = []
    warnings: list[str] = []

    forked_ok = len(forked) <= max_forked
    if not forked_ok:
        errors.append(f"FORKED guardrail exceeded: {len(forked)} > max {max_forked}")

    for dup in _duplicates(forked):
        errors.append(f"duplicate FORKED entry: {dup}")
    for dup in _duplicates(diverged):
        errors.append(f"duplicate DIVERGED entry: {dup}")

    overlap = sorted(set(forked) & set(diverged))
    for path in overlap:
        errors.append(f"path listed in both FORKED and DIVERGED: {path}")

    doc_forks: set[str] | None = None
    doc_diverged: dict[str, DocumentedDivergence] = {}
    if not doc.is_file():
        errors.append(f"missing governance doc: {doc}")
    else:
        doc_text = doc.read_text(encoding="utf-8")
        try:
            doc_forks = parse_documented_forks(doc_text)
            doc_diverged, doc_dups = parse_documented_divergences(doc_text)
        except ValueError as exc:
            errors.append(str(exc))
            doc_dups = []
        for dup in doc_dups:
            errors.append(f"duplicate §1.2 row in upstream-absorption.md: {dup}")

    if doc_forks is not None:
        for path in sorted(set(forked) - doc_forks):
            errors.append(f"FORKED path missing from §1.1: {path}")
        for path in sorted(doc_forks - set(forked)):
            errors.append(f"§1.1 path missing from FORKED: {path}")

    code_set = set(diverged)
    doc_set = set(doc_diverged)
    if doc.is_file() and doc_forks is not None:
        for path in sorted(code_set - doc_set):
            errors.append(f"DIVERGED path missing from §1.2: {path}")
        for path in sorted(doc_set - code_set):
            errors.append(f"§1.2 path missing from DIVERGED: {path}")

    for extra in sorted(set(class_map) - code_set):
        errors.append(f"classification entry not in DIVERGED: {extra}")

    counts_by_category: dict[str, int] = {
        UPSTREAM_PR_CANDIDATE: 0,
        REMI_ARCHITECTURE_HOOK: 0,
    }
    rows: list[DivergedAuditRow] = []
    missing_pinning: list[str] = []
    verified_pinning = 0

    for path in sorted(code_set | doc_set):
        in_code = path in code_set
        in_doc = path in doc_set
        doc_row = doc_diverged.get(path)
        category = class_map.get(path) if in_code else None
        if in_code:
            if category is None:
                errors.append(f"DIVERGED path lacks classification: {path}")
            elif category not in DIVERGED_CATEGORIES:
                errors.append(f"invalid classification for {path}: {category!r}")
            else:
                counts_by_category[category] += 1

        file_exists = (root_path / path).is_file()
        if in_code and not file_exists:
            errors.append(f"DIVERGED file missing on disk: {path}")

        pinning_test = doc_row.pinning_test if doc_row is not None else None
        pinning_exists = False
        if pinning_test:
            pinning_path = root_path / pinning_test
            pinning_exists = pinning_path.is_file() and pinning_path.name.startswith("test_")
            if not pinning_path.is_file():
                errors.append(f"pinning test missing on disk for {path}: {pinning_test}")
                missing_pinning.append(f"{path} -> {pinning_test}")
            elif not pinning_path.name.startswith("test_"):
                errors.append(f"pinning test must start with test_ for {path}: {pinning_test}")
                missing_pinning.append(f"{path} -> {pinning_test}")
            elif in_code:
                verified_pinning += 1
        elif in_doc:
            errors.append(f"§1.2 row missing pinning test for {path}")
            missing_pinning.append(f"{path} -> <missing>")

        rows.append(
            DivergedAuditRow(
                path=path,
                category=category,
                summary=doc_row.summary if doc_row is not None else "",
                pinning_test=pinning_test,
                file_exists=file_exists,
                pinning_test_exists=pinning_exists,
                in_code=in_code,
                in_doc=in_doc,
            )
        )

    over_target = max(0, len(diverged) - target_diverged)
    within_target = len(diverged) <= target_diverged
    if not within_target:
        warnings.append(
            f"DIVERGED count {len(diverged)} exceeds target <= {target_diverged} "
            f"({over_target} over target; {counts_by_category[UPSTREAM_PR_CANDIDATE]} {UPSTREAM_PR_CANDIDATE} eligible)"
        )

    return BudgetReport(
        forked_count=len(forked),
        max_forked=max_forked,
        forked_ok=forked_ok,
        forked_paths=tuple(forked),
        diverged_count=len(diverged),
        target_diverged=target_diverged,
        over_target=over_target,
        within_target=within_target,
        counts_by_category=counts_by_category,
        pinning_tests_total=len(diverged),
        pinning_tests_verified=verified_pinning,
        missing_pinning_tests=tuple(missing_pinning),
        rows=tuple(rows),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def format_budget_report(report: BudgetReport, *, verbose: bool = False) -> str:
    """Format a concise governance summary suitable for ``tools/check.sh``."""

    forked_state = "OK" if report.forked_ok else f"EXCEEDED by {report.forked_count - report.max_forked}"
    diverged_state = "ON TARGET" if report.within_target else f"{report.over_target} over target"
    out = [
        "Upstream divergence budget:",
        f"  FORKED:        {report.forked_count} / max {report.max_forked} [{forked_state}]",
        f"  DIVERGED:      {report.diverged_count} / target <= {report.target_diverged} [{diverged_state}]",
        f"  Classes:       {report.counts_by_category[UPSTREAM_PR_CANDIDATE]} {UPSTREAM_PR_CANDIDATE}, "
        f"{report.counts_by_category[REMI_ARCHITECTURE_HOOK]} {REMI_ARCHITECTURE_HOOK}",
        f"  Pinning tests: {report.pinning_tests_verified} / {report.pinning_tests_total} verified on disk",
    ]
    if verbose:
        out.append("  Entries:")
        for row in report.rows:
            cat = row.category or "unclassified"
            test = row.pinning_test or "<missing>"
            marker = "ok" if (row.file_exists and row.pinning_test_exists and row.in_code and row.in_doc) else "FAIL"
            out.append(f"    [{marker:4s}] {cat:22s} {row.path} ({test})")
    if report.errors:
        out.append("Errors:")
        out.extend(f"  - {err}" for err in report.errors)
        out.append(f"Status: FAIL ({len(report.errors)} governance error(s))")
    elif report.warnings:
        out.append(f"Status: PASS (guardrails met; {report.over_target} diverged file(s) above target)")
    else:
        out.append("Status: PASS (all guardrails and divergence target met)")
    return "\n".join(out)


def git(*args: str, root: str = ROOT, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line]


def merge_head(root: str = ROOT) -> str | None:
    found = git("rev-parse", "-q", "--verify", "MERGE_HEAD", root=root, check=False).strip()
    return found or None


def changed_paths(paths: tuple[str, ...] = FORKED, root: str = ROOT) -> list[str]:
    """Files under ``paths`` whose index or working tree differs from HEAD (merged, conflicted, added or deleted)."""

    found = set(lines(git("diff", "--no-renames", "--name-only", "HEAD", "--", *paths, root=root)))
    found |= set(lines(git("diff", "--no-renames", "--name-only", "--cached", "HEAD", "--", *paths, root=root)))
    found |= {line.split("\t", 1)[1] for line in lines(git("ls-files", "--unmerged", "--", *paths, root=root))}
    return sorted(found)


def in_head(path: str, root: str = ROOT) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"HEAD:{path}"], cwd=root, capture_output=True, check=False).returncode == 0


def restore(paths: tuple[str, ...] = FORKED, root: str = ROOT, dry_run: bool = False) -> list[tuple[str, str]]:
    """Put every changed file under ``paths`` back to HEAD's state; returns ``(path, action)``."""

    done = []
    for path in changed_paths(paths, root):
        if in_head(path, root):
            action = "kept Remi's version"
            if not dry_run:
                git("checkout", "HEAD", "--", path, root=root)
        else:
            action = "removed (not in Remi)"
            if not dry_run:
                git("rm", "-q", "-f", "--ignore-unmatch", "--", path, root=root)
                full = os.path.join(root, path)
                if os.path.lexists(full):
                    os.remove(full)
        done.append((path, action))
    return done


def review(base: str | None, upstream: str, root: str = ROOT) -> str:
    base = base or git("merge-base", "HEAD", upstream, root=root).strip()
    found = []
    for paths, what in (
        (FORKED, "the files Remi forked (keep Remi's, port selectively)"),
        (DIVERGED, "the files carrying Remi's fixes of upstream bugs (merge, keep each fix; §1.2 names its test)"),
    ):
        log = git("log", "--oneline", "--no-merges", f"{base}..{upstream}", "--", *paths, root=root)
        if not log.strip():
            found.append(f"No upstream commit after {base[:10]} touches {what.split(' (')[0]}.")
            continue
        stat = git("diff", "--stat=200", base, upstream, "--", *paths, root=root)
        found.append(f"Upstream commits after {base[:10]} on {what}:\n{log}\n{stat}")
    return "\n".join(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    shown = sub.add_parser("review", help="upstream commits and diffstat on the forked and diverged paths (before merging)")
    shown.add_argument("--base", help="merge base (default: git merge-base HEAD <upstream>)")
    shown.add_argument("--upstream", default=UPSTREAM, help="upstream revision (default: %(default)s)")
    back = sub.add_parser("restore", help="during a merge: every forked path the merge touched back to HEAD")
    back.add_argument("--dry-run", action="store_true", help="only list what would be restored")
    cap = sub.add_parser("budget", help="enforce the FORKED cap, audit §1.2 pinning tests, and classify DIVERGED files")
    cap.add_argument("--max-forked", type=int, default=DEFAULT_MAX_FORKED, help="maximum allowed FORKED paths (default: %(default)s)")
    cap.add_argument("--target-diverged", type=int, default=DEFAULT_TARGET_DIVERGED, help="target ceiling for DIVERGED files (default: %(default)s)")
    cap.add_argument("--json", action="store_true", dest="as_json", help="emit machine-readable JSON governance report")
    cap.add_argument("-v", "--verbose", action="store_true", help="list every classified DIVERGED file and pinning test")
    options = parser.parse_args(argv)

    if options.command == "review":
        print(review(options.base, options.upstream))
        return 0

    if options.command == "budget":
        if options.max_forked < 0:
            parser.error("--max-forked must be >= 0")
        if options.target_diverged < 0:
            parser.error("--target-diverged must be >= 0")
        report = audit_budget(max_forked=options.max_forked, target_diverged=options.target_diverged)
        if options.as_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_budget_report(report, verbose=options.verbose))
        return 0 if report.ok else 1

    theirs = merge_head()
    if theirs is None:
        print("No merge in progress (MERGE_HEAD): run this right after `git merge --no-ff --no-commit upstream/neo`.", file=sys.stderr)
        return 2
    base = git("merge-base", "HEAD", theirs).strip()
    done = restore(dry_run=options.dry_run)
    if not done:
        print("The merge left every forked path as Remi has it.")
        return 0
    verb = "would restore" if options.dry_run else "restored"
    print(f"{verb} {len(done)} forked path(s); port upstream's fixes selectively (with a test) and list each upstream commit in the sync note's \"Adapted for Remi\" table:")
    for path, action in done:
        print(f"  {path}: {action}\n      git diff {base[:10]} {theirs[:10]} -- {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
