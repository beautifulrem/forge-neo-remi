"""Settings read as the kind of value their default is.

A setting can hold a value of the wrong kind: a hand-edited config.json (the
WebUI only warns about it) or an ``/sdapi/v1/options`` write, which checks no
types. Read as is, one such value took every request down (the upload limit
is read by the body-limit middleware) or the app's bootstrap and the settings
page, the only places to set it right. So:

- the server repairs the stored values when it starts (:func:`repair`);
- Remi's own code reads settings through :func:`setting`, which gives the
  default for a value of the wrong kind written while the server runs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)

_REGISTERED = object()


def fit(default: Any, value: Any) -> Any:
    """``value`` as a setting whose default is ``default``: the default when
    ``value`` is None or of another kind (a non-finite number, a list where a
    number belongs, a number where a list of names belongs...). A setting
    without a default (None) takes any value."""

    if value is None or default is None:
        return default if value is None else value
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        # 0 / 1 for a checkbox (an API client): the same answer
        return bool(value) if type(value) in (int, float) and value in (0, 1) else default
    if isinstance(default, (int, float)):
        return value if type(value) in (int, float) and math.isfinite(value) else default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    if isinstance(default, list):
        kinds = {float if type(item) is int else type(item) for item in default} or {str}
        ok = isinstance(value, list) and all((float if type(item) is int else type(item)) in kinds for item in value)
        return value if ok else default
    return value


def setting(name: str, default: Any = _REGISTERED) -> Any:
    """The setting ``name`` read as the kind of ``default`` (its registered
    default when not given); unset, unregistered or of the wrong kind: the
    default."""

    from modules import shared

    opts = shared.opts
    info = getattr(opts, "data_labels", {}).get(name)
    if default is _REGISTERED:
        default = info.default if info is not None else None
    try:
        value = getattr(opts, name, default)
    except Exception:  # an Options stand-in without the setting
        return default
    return fit(default, value)


def repair(opts: Any) -> list[str]:
    """Replace the stored settings whose value is of the wrong kind with
    their default (in memory: the next save writes them), each one logged.
    Returns their names."""

    repaired = []
    data = getattr(opts, "data", None)
    if not isinstance(data, dict):
        return repaired
    for key, info in getattr(opts, "data_labels", {}).items():
        value = data.get(key)
        if value is None:
            continue
        fixed = fit(info.default, value)
        if fixed is value:
            continue
        data[key] = fixed
        repaired.append(key)
        log.warning("setting %s held %r (%s), not a value of its kind: reset to %r", key, _short(value), type(value).__name__, fixed)
    return repaired


def _short(value: Any) -> Any:
    text = repr(value)
    return value if len(text) <= 80 else text[:77] + "..."
