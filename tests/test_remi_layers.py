"""The layers of architecture.md hold: extension and script code, and the
layers they build on (``remi.ui``, ``remi.core``, ``remi.pages``), never reach
into the server (``remi.server``, ``request.app.state.services``). Actions get
files through ``ctx.files`` (``remi.ui.ActionFiles``) instead."""

import ast
import re
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Code that must not know the server
CLIENT_CODE = ("extensions-builtin", "scripts", "modules/processing_scripts", "remi/ui", "remi/core", "remi/pages")

# WebUI glue in modules/ that is part of the server, each with its reason
SERVER_GLUE = {
    "modules/extension_route_guard.py": "the A1111 extension route guard the server installs (bind address, local-client and login checks)",
    "modules/util.py": "A1111's util.open_folder, delegating to the server's checked open-folder",
}

# `app.state.services`, and `getattr(app.state, "services")` with or without a default
SERVICES_RE = re.compile(r"\.state\.services\b|[\"']services[\"']\s*[,)]")

# Extension code that is handed the services, each with its reason (Remi's own prompt tools: their
# history and tag data live with the server's file store)
SERVICES_GLUE = {
    "extensions-builtin/sd-webui-prompt-all-in-one-neo/scripts/prompt_tools.py": "hands the app's services to remi.prompt_tools.attach in on_app_started",
    "extensions-builtin/sd-webui-tagcomplete-neo/scripts/tag_autocomplete.py": "hands the app's services to remi.prompt_tools.attach in on_app_started",
}


def _python_files(*roots: str):
    for root in roots:
        base = ROOT / root
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if "/tests/" in f"/{relative}" or "/node_modules/" in relative:
                continue
            yield relative, path


def _is_server(module: str) -> bool:
    return module == "remi.server" or module.startswith("remi.server.")


def _server_imports(source: str) -> list[int]:
    lines = []
    with warnings.catch_warnings():
        # Vendored third-party code (mmcv, OneFormer in forge_legacy_preprocessors) has "\s" in
        # docstrings: that is upstream's to fix, and this test reads imports only.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        if any(_is_server(name) for name in names):
            lines.append(node.lineno)
    return lines


def test_extension_and_layer_code_does_not_import_the_server():
    found = []
    for relative, path in _python_files(*CLIENT_CODE):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found += [f"{relative}:{line}" for line in _server_imports(source)]
    assert found == [], f"import remi.server outside the server (use ctx.files / remi.ui): {found}"


def _absolute_imports(source: str, module: str, is_package: bool) -> list[tuple[int, str]]:
    """``(line, module)`` of every import in ``source`` (the code of ``module``), relative ones
    resolved, with ``from X import name`` also read as ``X.name``."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    package = module if is_package else module.rpartition(".")[0]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parent = ".".join(parts[: len(parts) - node.level + 1])
                base = f"{parent}.{base}" if base else parent
            found += [(node.lineno, base)] + [(node.lineno, f"{base}.{alias.name}") for alias in node.names]
    return found


def test_remi_below_the_server_does_not_import_it():
    """architecture.md "Layers": remi/server sits on top of remi/jobs, remi/worker, remi.ui, …
    Nothing else under remi/ imports it, not even inside a function."""

    found = []
    for relative, path in _python_files("remi"):
        if relative.startswith("remi/server/"):
            continue
        module = relative.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        imports = _absolute_imports(path.read_text(encoding="utf-8"), module, path.name == "__init__.py")
        found += sorted({f"{relative}:{line}" for line, name in imports if _is_server(name)})
    assert found == [], f"remi.server imported below the server: {found}"


def test_the_layer_check_resolves_relative_imports():
    assert (1, "remi.server.services") in _absolute_imports("from ..server import services", "remi.jobs.importers", False)
    assert (1, "remi.server") in _absolute_imports("from .server import x", "remi", True)
    assert (1, "remi.server.runtime.not_ported") in _absolute_imports("from remi.server.runtime import not_ported", "remi.x", False)


def test_extension_and_layer_code_does_not_reach_the_server_services():
    found = []
    for relative, path in _python_files(*CLIENT_CODE):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found += [f"{relative}:{number}" for number, line in enumerate(source.splitlines(), 1) if SERVICES_RE.search(line)]
    reached = {item.rsplit(":", 1)[0] for item in found}
    assert reached == set(SERVICES_GLUE), f"request.app.state.services outside the server (use ctx.files): {[item for item in found if item.rsplit(':', 1)[0] not in SERVICES_GLUE]}; gone: {sorted(set(SERVICES_GLUE) - reached)}"


def test_the_services_pattern_sees_every_spelling():
    for line in ('services = request.app.state.services', 'getattr(app.state, "services")', 'getattr(app.state, "services", None)', "getattr(app.state, 'services' , None)"):
        assert SERVICES_RE.search(line), line


def test_only_the_listed_webui_glue_imports_the_server():
    found = set()
    for relative, path in _python_files("modules", "modules_forge"):
        if relative.startswith("modules/processing_scripts/"):
            continue
        if _server_imports(path.read_text(encoding="utf-8")):
            found.add(relative)
    assert found == set(SERVER_GLUE), f"new server imports in modules/: {sorted(found - set(SERVER_GLUE))}; gone: {sorted(set(SERVER_GLUE) - found)}"


def test_action_files_reads_uploads_and_outputs_only(tmp_path):
    from remi.testing import webui_state

    with webui_state():  # images are read through modules.images (the WebUI's import order)
        _check_action_files(tmp_path)


def _check_action_files(tmp_path):
    from PIL import Image

    from remi import ui
    from remi.files import FileStore

    store = FileStore(tmp_path / "files")
    try:
        files = ui.ActionFiles(store)
        image_id = files.store_image(Image.new("RGB", (6, 4), (255, 0, 0)), "red.png")
        assert files.image_size(image_id) == (6, 4)
        assert files.image(image_id).size == (6, 4) and files.image(image_id).filename == "red.png"
        assert files.path(image_id) and files.get(image_id).is_image
        text_id = files.store_bytes(b"a,b\n", "table.csv")
        assert files.get(text_id) is not None and files.image(text_id) is None and files.image_size(text_id) is None
        preview = tmp_path / "net.png"
        Image.new("RGB", (2, 2)).save(preview)
        network = store.register_path(preview)
        assert network is not None and files.get(network.id) is None  # network previews are not action inputs
        for bad in (None, "", "unknown", 3):
            assert files.get(bad) is None and files.image(bad) is None and files.path(bad) is None
    finally:
        store.close()
