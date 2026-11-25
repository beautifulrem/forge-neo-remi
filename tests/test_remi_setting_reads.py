"""Remi reads settings through ``remi.setting_values.setting()``.

A setting can hold a value of the wrong kind or out of range: a hand-edited
config.json, an extension's assignment or, before it checked values, an
``/sdapi/v1/options`` write. Read as is, one such value took down a page, a route or
the server (B13-1: the extensions page with ``disabled_extensions = 5``, the
LoRA list with a multiplier of ``10**400``, the gallery's Save, the Detail
Daemon form). ``setting()`` reads the value as the kind of its default, within
its slider's range and among its fixed choices.

The scope is Remi's own code: the package ``remi/`` (the routes, services and
forms of the WebUI) and the extensions Remi wrote (their tests aside). Upstream
code and the third-party extensions bundled with Remi keep their reads; the
server's start (``repair()``) leaves them values of their kind.

The guard flags every read of ``shared.opts`` (a name bound to it,
``shared.opts`` under any name the file gives the module, ``modules.shared.opts``,
or a call of a function named for it, ``_opts()`` or ``_resolve_opts(...)``)
that is not a method of
``Options``: attribute reads, ``getattr(opts, ...)`` and the raw ``opts.data``
mapping. Assignments, ``del`` and the ``opts.data`` methods called only to
write (``update``, ``clear``, a ``pop`` whose result is unused) pass. A place
that wants the stored value as it is says why in ``RAW``. A parameter that is
passed the settings under another name is out of its sight.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPES = (
    ROOT / "remi",
    # extensions Remi wrote: the Hires guard rounds to the grid processing's sRound reads with setting()
    ROOT / "extensions-builtin" / "sd_forge_anima_hires_guard",
    ROOT / "extensions-builtin" / "sd-postprocess-suite",
)
IMPLEMENTATION = {"remi/setting_values.py"}

OPTIONS_MEMBERS = {
    # methods and the bookkeeping of modules.options.Options: not settings
    "data_labels",
    "restricted_opts",
    "set",
    "get_default",
    "save",
    "same_type",
    "load",
    "onchange",
    "add_option",
    "reorder",
    "cast_value",
    "dumpjson",
}

RAW = {
    # (path, function): why the stored value is read as it is
    ("remi/server/options.py", "coerce_values"): "compares the posted value with the stored one (an unchanged value outside stale choices is kept)",
    ("remi/ui/settings.py", "check_api_values"): "skips a posted value equal to the stored one, as upstream's set_config does",
    ("remi/server/options.py", "apply_values"): "tells a setting that refused the value (its stored value unchanged) from one that took it",
    ("remi/prompt_tools/options.py", "register_tag_options"): "moves values stored under the tools' earlier names to the current ones",
    ("remi/prompt_tools/options.py", "take_legacy_translation_key"): "removes the API key an earlier version stored in the settings",
}


# opts.data methods that only write (their result is None, or unused)
DATA_WRITES = {"update", "clear", "pop", "setdefault"}


@dataclass
class _Names:
    """What a file calls the settings (``opts``, ``options = shared.opts``) and
    the module that holds them (``shared``, ``import modules.shared as sh``)."""

    opts: set[str] = field(default_factory=lambda: {"opts"})
    shared: set[str] = field(default_factory=lambda: {"shared"})


def _is_shared(node: ast.AST, names: _Names) -> bool:
    """``shared`` (or a name bound to it) or ``modules.shared``."""

    if isinstance(node, ast.Name):
        return node.id in names.shared
    return isinstance(node, ast.Attribute) and node.attr == "shared" and isinstance(node.value, ast.Name) and node.value.id == "modules"


def _is_opts(node: ast.AST, names: _Names) -> bool:
    """``shared.opts`` (``modules.shared.opts``), a name bound to it, or a call
    of a function named for it: ``opts()``, ``_opts()``, ``self.opts()``,
    ``_resolve_opts(opts)``."""

    if isinstance(node, ast.Name):
        return node.id in names.opts
    if isinstance(node, ast.Attribute):
        return node.attr == "opts" and _is_shared(node.value, names)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        return (name == "opts" or name.endswith("_opts")) and not name.endswith("cmd_opts")  # cmd_opts: the command line
    return False


def _is_data(node: ast.AST, names: _Names) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "data" and _is_opts(node.value, names)


def _aliases(tree: ast.AST) -> _Names:
    """The names a file binds to the settings (``from modules.shared import
    opts as o``, ``options = shared.opts``) and to their module (``from modules
    import shared as s``, ``import modules.shared as sh``)."""

    names = _Names()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "modules.shared":
            names.opts |= {alias.asname or alias.name for alias in node.names if alias.name == "opts"}
        elif isinstance(node, ast.ImportFrom) and node.module == "modules":
            names.shared |= {alias.asname or alias.name for alias in node.names if alias.name == "shared"}
        elif isinstance(node, ast.Import):
            names.shared |= {alias.asname for alias in node.names if alias.name == "modules.shared" and alias.asname}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _is_opts(node.value, names):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in names.opts:
                        names.opts.add(target.id)
                        changed = True
    return names


class _Reads(ast.NodeVisitor):
    def __init__(self, names: _Names) -> None:
        self.found: list[tuple[int, str, str]] = []
        self.function = ""
        self.names = names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        outer, self.function = self.function, node.name
        self.generic_visit(node)
        self.function = outer

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # opts.data[key] = value (or del): a write
        if not isinstance(node.ctx, ast.Load) and _is_data(node.value, self.names):
            self.visit(node.slice)
            return
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # opts.data.update(...), opts.data.pop(key, None) as a statement: a write
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in DATA_WRITES and _is_data(call.func.value, self.names):
            for argument in [*call.args, *call.keywords]:
                self.visit(argument)
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and _is_opts(node.value, self.names) and node.attr not in OPTIONS_MEMBERS:
            self.found.append((node.lineno, self.function, f"opts.{node.attr}"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args and _is_opts(node.args[0], self.names):
            name = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "<name>"
            if name not in OPTIONS_MEMBERS:
                self.found.append((node.lineno, self.function, f"getattr(opts, {name!r})"))
        self.generic_visit(node)


def reads_in(source: str, filename: str) -> list[tuple[int, str, str]]:
    tree = ast.parse(source, filename)
    visitor = _Reads(_aliases(tree))
    visitor.visit(tree)
    return visitor.found


def reads(path: Path) -> list[tuple[int, str, str]]:
    return reads_in(path.read_text(encoding="utf-8"), str(path))


def direct_reads() -> list[str]:
    found = []
    for path in sorted(path for scope in SCOPES for path in scope.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in IMPLEMENTATION or "tests" in path.relative_to(ROOT).parts[1:]:
            continue
        for line, function, what in reads(path):
            if (relative, function) not in RAW:
                found.append(f"{relative}:{line} {function or '<module>'}: {what}")
    return found


def test_remi_reads_settings_through_setting():
    assert direct_reads() == []


def test_the_guard_sees_reads_through_a_resolver_and_the_full_module_path():
    source = """
import modules.shared
from remi.core.capabilities import _resolve_opts

def f(opts=None):
    active = _resolve_opts(opts)
    a = getattr(active, "forge_preset", None)
    b = modules.shared.opts.samples_format
    opts.data.update({"x": 1})
    opts.data.pop("legacy", None)
    opts.data.clear()
    c = opts.data.pop("taken", None)
    d = _cmd_opts().lora_dir

def g():
    import modules.shared as sh
    from modules import shared as s
    return sh.opts.outdir_save, s.opts.live_preview_fps, sh.cmd_opts.ckpt
"""
    assert [what for _line, _function, what in reads_in(source, "probe.py")] == [
        "getattr(opts, 'forge_preset')",
        "opts.samples_format",
        "opts.data",
        "opts.outdir_save",
        "opts.live_preview_fps",
    ]


def test_the_guard_sees_each_form_of_read():
    source = """
from modules import shared
from modules.shared import opts

def f(name):
    a = shared.opts.disabled_extensions
    b = opts.data.get("detail_daemon_count", 6)
    c = getattr(shared.opts, "outdir_save", "")
    d = getattr(opts, name)
    e = _opts().extra_networks_default_multiplier
    shared.opts.disabled_extensions = []
    shared.opts.data["sd_model_checkpoint"] = name
    del opts.data["legacy"]
    opts.save("config.json")
    shared.opts.data_labels.get("x")
    shared.cmd_opts.ckpt
    options = shared.opts
    g = options.data.get("x")
    h = getattr(self.opts(), "live_preview_fps", None)

def k():
    from modules.shared import opts as settings
    return settings.samples_format
"""
    assert [what for _line, _function, what in reads_in(source, "probe.py")] == [
        "opts.disabled_extensions",
        "opts.data",
        "getattr(opts, 'outdir_save')",
        "getattr(opts, '<name>')",
        "opts.extra_networks_default_multiplier",
        "opts.data",
        "getattr(opts, 'live_preview_fps')",
        "opts.samples_format",
    ]
