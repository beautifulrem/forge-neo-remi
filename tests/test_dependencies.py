"""Every third-party module the WebUI imports at module level comes from
requirements.txt (or what those packages depend on): the WebUI starts in an
environment with nothing else installed. Imports inside functions and
``try`` blocks are optional features and are not checked."""

from __future__ import annotations

import ast
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ("webui.py", "launch.py", "modules", "modules_forge", "remi", "scripts", "extensions-builtin")
SKIPPED = ("modules_forge/packages", "extensions-builtin/forge_legacy_preprocessors", "/tests/", "/install.py", "/preload.py")
# Installed by the launcher itself (modules/launch_utils.py), not by requirements.txt
LAUNCHER = {"torch", "torchvision"}
# Modules imported only on a path that needs them, each with the reason
OPTIONAL = {
    "ngrok": "modules/ngrok.py is imported only with --ngrok, whose launcher step installs it",
    "ldm_patched": "lib_negpip's fallback for the classic Forge backend (IS_NEO is false only there)",
}


def _name(requirement: str) -> str:
    return re.split(r"[\s;<>=!~\[]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _closure(names: set[str]) -> set[str]:
    """``names`` and every distribution they require, as installed here."""

    seen: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        pending += [_name(item) for item in requires if "extra ==" not in item]
    return seen


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    found: set[str] = set()

    def visit(body):
        for node in body:
            if isinstance(node, ast.Import):
                found.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.partition(".")[0])
            elif isinstance(node, ast.If) and not (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"):
                visit(node.body)
                visit(node.orelse)

    visit(tree.body)
    return found


def _local_names() -> set[str]:
    names = {path.stem for path in ROOT.iterdir()}
    names |= {path.name for path in (ROOT / "modules_forge" / "packages").iterdir()}
    for extension in (ROOT / "extensions-builtin").iterdir():
        for folder in [extension, *extension.glob("*")]:
            if folder.is_dir():
                names |= {path.stem for path in folder.iterdir()}
    return names


def test_module_level_imports_are_declared_dependencies():
    requirements = {_name(line) for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    allowed = _closure(requirements | LAUNCHER)
    distributions = metadata.packages_distributions()
    local = _local_names()
    undeclared: dict[str, list[str]] = {}
    for entry in CODE:
        paths = [ROOT / entry] if entry.endswith(".py") else sorted((ROOT / entry).rglob("*.py"))
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if any(part in f"/{relative}" for part in SKIPPED):
                continue
            for module in _top_level_imports(path):
                if module in sys.stdlib_module_names or module in local or module in OPTIONAL or module == "__future__":
                    continue
                owners = {_name(owner) for owner in distributions.get(module, ())}
                if not owners & allowed:
                    undeclared.setdefault(module, []).append(relative)
    assert undeclared == {}, undeclared


# Remi's own code may lean on these without declaring them, each with the reason
NOT_DIRECT = {
    "llama_cpp": "remi/prompt_tools/pup.py imports it only for a local GGUF model and says how to install it",
}


def _all_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="replace"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.partition(".")[0])
    return found


def test_remi_declares_every_library_it_uses():
    """A library Remi's own code calls (inside functions too) is in requirements.txt itself, not
    only pulled in by another package that could drop it (requests for the translation provider)."""

    requirements = {_name(line) for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    distributions = metadata.packages_distributions()
    local = _local_names()
    undeclared: dict[str, list[str]] = {}
    for path in sorted((ROOT / "remi").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if "/tests/" in f"/{relative}":
            continue
        for module in _all_imports(path):
            if module in sys.stdlib_module_names or module in local or module in NOT_DIRECT or module == "__future__":
                continue
            owners = {_name(owner) for owner in distributions.get(module, ())}
            if not owners & requirements:
                undeclared.setdefault(module, []).append(relative)
    assert undeclared == {}, undeclared


# Releases of the web server every request goes through that fix its advisories (docs/remi-webui/security.md,
# "Dependency advisories"): starlette 1.3.1 (GHSA-82w8-qh3p-5jfq, the last of five fixed in 1.0.1-1.3.1);
# fastapi 0.133.0 is the first that admits a Starlette 1.x
WEB_SERVER_FLOORS = {"starlette": "1.3.1", "fastapi": "0.133.0"}


def test_the_web_server_is_pinned_past_its_advisories():
    """Starlette is pinned itself: fastapi only names a floor that an installed older
    Starlette still meets, and the launcher reinstalls requirements.txt by its == pins."""

    from packaging.version import Version

    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name, sep, version = line.partition("==")
        if sep:
            pins[_name(name)] = version.split(";")[0].strip()
    assert {name: Version(pins.get(name, "0")) >= Version(floor) for name, floor in WEB_SERVER_FLOORS.items()} == dict.fromkeys(WEB_SERVER_FLOORS, True), pins
