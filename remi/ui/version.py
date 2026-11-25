"""Public ``remi.ui`` API version and SemVer compatibility checker.

Extension authors declare ``"api_version"`` (e.g. ``"1.0"``, ``"^1.0"``,
``">=1.0,<2.0"``) in ``remi-extension.json`` so Remi can validate compatibility
explicitly at load time and surface actionable diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REMI_API_VERSION = "1.1"
"""Current public ``remi.ui`` extension API version (1.1 added ``ui.Sentence``
and ``ui.extension_of_owner``)."""

REMI_MIN_SUPPORTED_API_VERSION = "1.0"
"""Oldest extension API version still supported by this runtime."""

_SEMVER_RE = re.compile(r"^v?(?P<major>0|[1-9]\d*)(?:\.(?P<minor>0|[1-9]\d*))?(?:\.(?P<patch>0|[1-9]\d*))?(?:-(?P<pre>[0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")
_COMPARATOR_RE = re.compile(r"^(?P<op>==|!=|>=|<=|>|<|\^|~|=)?\s*(?P<ver>.+)$")


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


def _compare(left: SemVer, right: SemVer) -> int:
    if left.core_tuple() != right.core_tuple():
        return -1 if left.core_tuple() < right.core_tuple() else 1
    if left.prerelease == right.prerelease:
        return 0
    if not left.prerelease:
        return 1  # release > prerelease
    if not right.prerelease:
        return -1
    return -1 if left.prerelease < right.prerelease else 1


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
        clauses = [part for part in re.split(r"[,\s]+", alt) if part]
        try:
            if all(_matches_clause(current_ver, clause) for clause in clauses):
                return CompatibilityResult(True, current, spec, None)
        except ValueError as error:
            return CompatibilityResult(False, current, spec, str(error))
    return CompatibilityResult(
        False,
        current,
        spec,
        f"requires remi.ui API {spec}, but runtime provides {current}",
    )


__all__ = [
    "REMI_API_VERSION",
    "REMI_MIN_SUPPORTED_API_VERSION",
    "CompatibilityResult",
    "SemVer",
    "check_api_compatibility",
    "parse_semver",
]
