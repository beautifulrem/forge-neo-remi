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

``restore`` refuses to run outside a merge (no ``MERGE_HEAD``), and should run
right after ``git merge`` (it also undoes edits to these paths made since).
``$PYTHON``: any Python 3.10+ (standard library only).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = "upstream/neo"

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
    "backend/memory_management.py",
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
    options = parser.parse_args(argv)

    if options.command == "review":
        print(review(options.base, options.upstream))
        return 0

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
