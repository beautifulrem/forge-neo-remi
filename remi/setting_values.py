"""Settings read as the kind of value their default is.

A setting can hold a value of the wrong kind: a hand-edited config.json (the
WebUI only warns about it) or an ``/sdapi/v1/options`` write, which checks no
types. Read as is, one such value took every request down (the upload limit
is read by the body-limit middleware) or the app's bootstrap and the settings
page, the only places to set it right. So:

- the server repairs the stored values when it starts (:func:`repair`);
- Remi's own code reads settings through :func:`setting`, which gives the
  default for a value of the wrong kind written while the server runs, and
  keeps a number within the range its slider offers: ``1e303`` MB is a number
  but no upload limit (multiplied into bytes it overflowed every request).
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)

_REGISTERED = object()


def _finite(value: Any) -> bool:
    """A number that is one: not inf or NaN, and not an int too large for a
    float (``10**400`` in a JSON body; ``math.isfinite`` raises on it)."""

    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def bounds(info: Any) -> tuple[float, float] | None:
    """The ``(minimum, maximum)`` a setting's slider offers, if it has both."""

    try:
        args = info.component_args() if callable(info.component_args) else info.component_args
        low, high = args["minimum"], args["maximum"]
    except Exception:  # no slider, a callable that fails, no range
        return None
    return (low, high) if _finite(low) and _finite(high) and low <= high else None


def fit(default: Any, value: Any, limits: tuple[float, float] | None = None) -> Any:
    """``value`` as a setting whose default is ``default``: the default when
    ``value`` is None or of another kind (a non-finite number, a list where a
    number belongs, a number where a list of names belongs...); a number
    within ``limits`` when given. A setting without a default (None) takes any
    value."""

    if value is None or default is None:
        return default if value is None else value
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        # 0 / 1 for a checkbox (an API client): the same answer
        return bool(value) if type(value) in (int, float) and value in (0, 1) else default
    if isinstance(default, (int, float)):
        if not _finite(value):
            return default
        return min(max(value, limits[0]), limits[1]) if limits is not None else value
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    if isinstance(default, list):
        kinds = {float if type(item) is int else type(item) for item in default} or {str}
        ok = isinstance(value, list) and all((float if type(item) is int else type(item)) in kinds for item in value)
        return value if ok else default
    return value


def setting(name: str, default: Any = _REGISTERED) -> Any:
    """The setting ``name`` read as the kind of ``default`` (its registered
    default when not given) and within its slider's range; unset,
    unregistered or of the wrong kind: the default."""

    from modules import shared

    opts = shared.opts
    info = getattr(opts, "data_labels", {}).get(name)
    if default is _REGISTERED:
        default = info.default if info is not None else None
    try:
        value = getattr(opts, name, default)
    except Exception:  # an Options stand-in without the setting
        return default
    return fit(default, value, bounds(info) if info is not None else None)


def repair(opts: Any) -> list[str]:
    """Replace the stored settings whose value is of the wrong kind with
    their default (in memory: the next save writes them), each one logged.
    Returns their names. A value of the right kind outside its slider's range
    is kept (upstream code may mean it); :func:`setting` reads it within."""

    repaired = []
    data = getattr(opts, "data", None)
    if not isinstance(data, dict):
        return repaired
    for key, info in getattr(opts, "data_labels", {}).items():
        value = data.get(key)
        if value is None:
            continue
        try:
            fixed = fit(info.default, value)
        except Exception:  # one setting must not stop the start: read as its default
            fixed = info.default
        if fixed is value:
            continue
        data[key] = fixed
        repaired.append(key)
        log.warning("setting %s held %r (%s), not a value of its kind: reset to %r", key, _short(value), type(value).__name__, fixed)
    return repaired


def _short(value: Any) -> Any:
    text = repr(value)
    return value if len(text) <= 80 else text[:77] + "..."
