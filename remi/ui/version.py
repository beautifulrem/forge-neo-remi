"""Public ``remi.ui`` API version, the Remi version, and SemVer compatibility checks.

Extension authors declare ``"api_version"`` (e.g. ``"1.0"``, ``"^1.0"``,
``">=1.0,<2.0"``) and optionally ``"min_remi_version"`` in ``remi-extension.json``
so Remi can validate compatibility explicitly at load time and surface
actionable diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from remi.ui.sentence import Sentence

REMI_VERSION = "1.0.0"
"""The running Remi WebUI's own version, which a manifest's ``min_remi_version``
is compared with (extension-guide.md). It is Remi's, not Forge Neo's (bootstrap
``version``, ``modules_forge.forge_version``, follows upstream) and not the
extension API's: :data:`REMI_API_VERSION` stays 1.0 while the API grows, so an
extension that needs something a later build added (``ui.Sentence``, a widget)
says so with ``min_remi_version``. Raise the minor version with a change an
extension may come to rely on, the patch version for fixes."""

REMI_API_VERSION = "1.0"
"""Current public ``remi.ui`` extension API version. It stays 1.0 while the
API grows compatibly: a bare ``"1.0"`` in a manifest matches 1.0.x only, so a
minor bump would refuse every extension that declares it (extension-guide.md)."""

REMI_MIN_SUPPORTED_API_VERSION = "1.0"
"""Oldest extension API version still supported by this runtime."""

_SEMVER_RE = re.compile(r"^v?(?P<major>0|[1-9]\d*)(?:\.(?P<minor>0|[1-9]\d*))?(?:\.(?P<patch>0|[1-9]\d*))?(?:-(?P<pre>[0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")
_COMPARATOR_RE = re.compile(r"^(?P<op>==|!=|>=|<=|>|<|\^|~|=)?\s*(?P<ver>.+)$")
_OPERATOR_SPACE_RE = re.compile(r"(==|!=|>=|<=|[<>=^~])\s+")


@dataclass(frozen=True, order=True)
class SemVer:
    major: int
    minor: int = 0
    patch: int = 0
    prerelease: str = ""

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base

    def core_tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)


def parse_semver(text: str) -> SemVer:
    """Parse ``"1.0"``, ``"1.2.3"``, or ``"1.2.3-beta.1"`` into :class:`SemVer`."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("version must be a non-empty string")
    match = _SEMVER_RE.match(text.strip())
    if not match:
        raise ValueError(f"invalid version string: {text!r}")
    return SemVer(
        major=int(match.group("major")),
        minor=int(match.group("minor") or 0),
        patch=int(match.group("patch") or 0),
        prerelease=match.group("pre") or "",
    )


def _prerelease_key(prerelease: str) -> tuple[tuple[int, int | str], ...]:
    """SemVer precedence of a prerelease: identifier by identifier, numbers as numbers and
    below words (``rc.9`` < ``rc.10``, ``9`` < ``alpha``); a shorter run of equal identifiers
    comes first (``alpha`` < ``alpha.1``)."""

    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in prerelease.split("."))


def _compare(left: SemVer, right: SemVer) -> int:
    """SemVer precedence (build metadata is not part of a parsed version)."""

    if left.core_tuple() != right.core_tuple():
        return -1 if left.core_tuple() < right.core_tuple() else 1
    if left.prerelease == right.prerelease:
        return 0
    if not left.prerelease:
        return 1  # release > prerelease
    if not right.prerelease:
        return -1
    return -1 if _prerelease_key(left.prerelease) < _prerelease_key(right.prerelease) else 1


def compare_versions(left: str, right: str) -> int:
    """``-1``, ``0`` or ``1`` as version ``left`` comes before, equals or follows ``right``
    by SemVer precedence (``"1.10"`` follows ``"1.9"``; ``"1.0"`` is ``"1.0.0"``). Raises
    ``ValueError`` for a string that is no version."""

    return _compare(parse_semver(left), parse_semver(right))


def _matches_clause(current: SemVer, clause: str) -> bool:
    clause = clause.strip()
    if not clause or clause in ("*", "x", "X"):
        return True
    if ".*" in clause or ".x" in clause.lower():
        parts = clause.lstrip("v").split(".")
        if parts[0].isdigit():
            if current.major != int(parts[0]):
                return False
            if len(parts) > 1 and parts[1].isdigit():
                return current.minor == int(parts[1])
            return True
        raise ValueError(f"invalid wildcard range: {clause!r}")
    match = _COMPARATOR_RE.match(clause)
    if not match:
        raise ValueError(f"invalid comparator: {clause!r}")
    op = match.group("op") or "=="
    raw_ver = match.group("ver").strip()
    target = parse_semver(raw_ver)
    cmp = _compare(current, target)
    if op in ("==", "="):
        # "1.0" matches "1.0.x" when patch omitted; explicit patch requires exact match
        parts = raw_ver.lstrip("v").split("-", 1)[0].split(".")
        if len(parts) == 1:
            return current.major == target.major
        if len(parts) == 2:
            return current.major == target.major and current.minor == target.minor
        return cmp == 0
    if op == "!=":
        return cmp != 0
    if op == ">=":
        return cmp >= 0
    if op == "<=":
        return cmp <= 0
    if op == ">":
        return cmp > 0
    if op == "<":
        return cmp < 0
    if op == "^":
        if cmp < 0:
            return False
        if target.major > 0:
            return current.major == target.major
        if target.minor > 0:
            return current.major == 0 and current.minor == target.minor
        return current.major == 0 and current.minor == 0 and current.patch == target.patch
    if op == "~":
        if cmp < 0:
            return False
        return current.major == target.major and current.minor == target.minor
    raise ValueError(f"unsupported operator {op!r}")


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    current: str
    required: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "compatible": self.compatible,
            "current": self.current,
            "required": self.required,
            "reason": self.reason,
        }


def check_api_compatibility(required_range: str | None, current: str = REMI_API_VERSION) -> CompatibilityResult:
    """Check whether ``current`` satisfies ``required_range``.

    Comma- or space-separated clauses are AND-ed (e.g. ``">=1.0,<2.0"``).
    ``||`` separates OR alternatives.
    """

    try:
        current_ver = parse_semver(current)
    except ValueError as error:
        return CompatibilityResult(False, current, required_range, str(error))
    if required_range is None or not str(required_range).strip():
        return CompatibilityResult(True, current, None, None)
    spec = str(required_range).strip()
    alternatives = [alt.strip() for alt in spec.split("||") if alt.strip()]
    if not alternatives:
        return CompatibilityResult(True, current, spec, None)
    for alt in alternatives:
        # glue each operator to its version first: ">= 1.0" is one clause, not ">=" and "1.0"
        clauses = [part for part in re.split(r"[,\s]+", _OPERATOR_SPACE_RE.sub(r"\1", alt)) if part]
        try:
            if all(_matches_clause(current_ver, clause) for clause in clauses):
                return CompatibilityResult(True, current, spec, None)
        except ValueError as error:
            return CompatibilityResult(False, current, spec, str(error))
    return CompatibilityResult(
        False,
        current,
        spec,
        Sentence("requires remi.ui API {{required}}, but runtime provides {{current}}", required=spec, current=current),
    )


def check_remi_version(minimum: str | None, current: str = REMI_VERSION) -> CompatibilityResult:
    """Whether the running Remi (``current``) is ``minimum`` or newer, compared as versions.
    The reason a newer Remi is needed is a :class:`Sentence` naming both (the UI translates it)."""

    if minimum is None or not str(minimum).strip():
        return CompatibilityResult(True, current, None, None)
    required = str(minimum).strip()
    try:
        if compare_versions(current, required) >= 0:
            return CompatibilityResult(True, current, required, None)
    except ValueError as error:
        return CompatibilityResult(False, current, required, str(error))
    return CompatibilityResult(False, current, required, Sentence("requires Remi {{required}} or newer; this is Remi {{running}}", required=required, running=current))


__all__ = [
    "REMI_API_VERSION",
    "REMI_MIN_SUPPORTED_API_VERSION",
    "REMI_VERSION",
    "CompatibilityResult",
    "SemVer",
    "check_api_compatibility",
    "check_remi_version",
    "compare_versions",
    "parse_semver",
]
