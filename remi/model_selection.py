"""The model selection (checkpoint, VAE / text encoder modules, low-bits
dtype) as the settings change it, kept away from running generations.

A generation reads the selection more than once: each batch of ``n_iter``
calls ``forge_model_reload()``, and the Hires checkpoint, the Refiner and
per-job overrides switch it and switch back by re-deriving the loading
parameters from ``shared.opts``. A pick written straight into ``shared.opts``
while a job runs therefore changed the model between its batches (with the
first batch's infotext written for the rest), or was undone by the switch
back at the end of the hires pass.

So a pick made while GPU work holds the generation lock is *held*: the
settings show it at once (:func:`held`), and it is applied under the lock
before the next job (the queue calls :func:`apply_held`), or by a waiter as
soon as the lock is free when nothing else is queued. A pick made while the
lock is free applies at once, as before. The setting request never waits for
the generation.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, cast

log = logging.getLogger(__name__)


def unet_dtypes() -> list[str]:
    """The low-bits dtypes ``main_entry`` offers (the stored one, or
    "Automatic", before it is imported)."""

    import sys

    from remi.setting_values import setting

    main_entry = sys.modules.get("modules_forge.main_entry")
    if main_entry is not None:
        return list(main_entry.forge_unet_storage_dtype_options)
    dtype = setting("forge_unet_storage_dtype", "")
    return [dtype] if dtype else ["Automatic"]


def check_value(key: str, value: Any, dtypes: Callable[[], list[str]] | None = None) -> Any:
    """A model pick (one of :data:`MODEL_OPTIONS`) as the settings store it,
    or :class:`remi.ui.ParamError`: a checkpoint the WebUI knows, a list of
    known VAE / text encoder names (their file names), an offered dtype.
    Both ``PATCH /api/v2/options`` and ``/sdapi/v1/options`` check with it
    (the latter a checkpoint by A1111's exact names instead).
    ``dtypes``: the offered dtypes (:func:`unet_dtypes`)."""

    from remi.ui import ParamError, Sentence

    if key == "sd_model_checkpoint":
        from modules import sd_models

        if not isinstance(value, str) or sd_models.get_closet_checkpoint_match(value) is None:
            raise ParamError(Sentence("unknown checkpoint {{value}}", value=repr(value)))
        return value
    if key == "forge_additional_modules":
        from remi.networks import module_files

        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ParamError("expected a list of module names")
        known = {name for kind in ("vae", "text_encoders") for name, _path, _root in module_files(kind)}
        names = [item.replace("\\", "/").rsplit("/", 1)[-1] for item in value]
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ParamError(Sentence("unknown module(s): {{names}}", names=", ".join(unknown)))
        return names
    if key == "forge_unet_storage_dtype":
        offered = (dtypes or unet_dtypes)()
        if not isinstance(value, str) or value not in offered:
            raise ParamError(Sentence("unknown dtype {{value}}", value=repr(value)))
        return value
    raise KeyError(key)


MODEL_OPTIONS = ("sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype")
"""The options that select what ``forge_model_reload()`` loads."""

PRESET_COPIES = {"sd_model_checkpoint": "forge_checkpoint_", "forge_additional_modules": "forge_additional_modules_", "forge_unet_storage_dtype": "forge_unet_storage_dtype_"}
"""The prefix of the UI preset's own copy of each pick, which ``main_entry`` writes with it."""

_held: dict[str, Any] = {}
_guard = threading.Lock()
_waiter: threading.Thread | None = None


def _generation_lock() -> Any:
    from modules.call_queue import queue_lock

    return queue_lock


def _main_entry() -> Any:
    from modules_forge import main_entry

    return main_entry


def _preset() -> str | None:
    """The UI preset whose own copy of a pick ``main_entry`` also writes, or None."""

    from modules import shared
    from remi.setting_values import setting

    # main_entry also stores the pick as the preset's own (forge_checkpoint_<preset>, ...), which only an
    # offered preset has: for another (a stored name --disable-preset removed, a value of another kind)
    # it set the global setting and then raised KeyError, half applied. None: the global setting alone
    preset = setting("forge_preset", "")
    return preset if f"forge_unet_storage_dtype_{preset}" in shared.opts.data_labels else None


def _apply_now(key: str, value: Any) -> bool:
    main_entry = _main_entry()
    preset = cast(str, _preset())  # modules_change and the others check it for None

    if key == "sd_model_checkpoint":
        return main_entry.checkpoint_change(value, preset)
    if key == "forge_additional_modules":
        if not main_entry.module_list:
            main_entry.refresh_models()
        return main_entry.modules_change(value, preset)
    if key == "forge_unet_storage_dtype":
        return main_entry.dtype_change(value, preset)
    raise KeyError(key)


def _as_stored(key: str, value: Any) -> Any:
    """``value`` as ``shared.opts`` stores it once applied (module names become
    their sorted paths, as ``modules_change`` stores them)."""

    if key != "forge_additional_modules" or not isinstance(value, list):
        return value
    import os

    main_entry = _main_entry()
    if not main_entry.module_list:
        main_entry.refresh_models()
    return sorted(str(main_entry.module_list[name]) for name in (os.path.basename(v) for v in value) if name in main_entry.module_list)


def select(key: str, value: Any, apply: Callable[[str, Any], bool] | None = None) -> bool:
    """Select ``value`` for the model option ``key``; ``True`` when the
    selection the settings show changed (applied now, or held). A frozen
    setting (``--freeze-settings*``: the option, or the UI preset's copy of
    it the model manager writes, see :func:`frozen_pick`) raises ``AssertionError``, as
    ``Options.set`` does, also when the pick would be held: the waiter's
    later apply failed there, only logged, after the request had answered
    as if it worked. The stored pick sent again (a client posting back what
    ``GET /sdapi/v1/options`` answered) is no change and passes; held, so is
    the pick the settings show already (the stored one, or the one held).
    Such a pick does nothing: the model manager would store the dtype again,
    save, reload the loading parameters and write the UI preset's copy of it,
    which a freeze may keep. Back to the stored pick, the held one is
    dropped and nothing is applied."""

    as_stored = _as_stored(key, value)
    if any(_same_pick(pick, _shown(key)) for pick in (value, as_stored)):
        return False  # the pick the settings show already (the stored one, or the one held)
    _check_not_frozen(key, value)
    if any(_same_pick(pick, _stored(key)) for pick in (value, as_stored)):
        with _guard:
            _held.pop(key, None)  # the stored pick, which nothing has to apply
        return True
    lock = _generation_lock()
    if lock.acquire(blocking=False):
        try:
            with _guard:
                _held.pop(key, None)  # a pick applied now replaces one still held
            return (apply or _apply_now)(key, value)
        finally:
            lock.release()
    value = as_stored
    with _guard:
        if _same_pick(value, _held[key] if key in _held else _stored(key)):
            return False  # held meanwhile by another request
        _held[key] = value
        _start_waiter()
    log.info("Model selection %s=%r held until the running generation ends", key, value)
    return True


def _stored(key: str) -> Any:
    """The pick as stored (a value equal to it is no change: Options.set and the model manager
    compare it so), not read as a setting."""

    from modules import shared

    return shared.opts.data.get(key)


def _shown(key: str) -> Any:
    """The pick the settings show: the one held, else the stored one."""

    with _guard:
        return _held[key] if key in _held else _stored(key)


def _same_pick(value: Any, shown: Any) -> bool:
    if isinstance(value, list) and isinstance(shown, list) and all(isinstance(item, str) for item in value + shown):
        return sorted(value) == sorted(shown)  # modules in another order (modules_change sorts them)
    return value == shown


def frozen_pick(key: str, value: Any) -> tuple[str, str] | None:
    """The frozen setting (``--freeze-settings*``) that keeps the pick
    ``value`` for ``key`` from being made, and why, or None: ``key`` itself,
    or the UI preset's own copy of it, which ``main_entry`` writes with it
    (``forge_unet_storage_dtype_<preset>``, ...) when it would change. The
    stored value sent again is no change. Both settings routes refuse such a
    pick with the request's other refused values (422) before anything is
    set."""
    from modules import shared
    from remi.setting_values import frozen_reason

    if _same_pick(value, _stored(key)):  # unchanged: Options.set and the model manager leave it as it is
        return None
    found = None
    if (reason := frozen_reason(shared.opts, key, shared.cmd_opts)) is not None:
        found = key, reason
    elif (preset := _preset()) is not None:
        copy = PRESET_COPIES[key] + preset
        if (reason := frozen_reason(shared.opts, copy, shared.cmd_opts)) is not None and not _same_pick(_as_stored(key, value), _stored(copy)):
            found = copy, reason
    if found is None or _same_pick(_as_stored(key, value), _stored(key)):  # modules named, not by path: the stored ones
        return None
    return found


def _check_not_frozen(key: str, value: Any) -> None:
    if (found := frozen_pick(key, value)) is not None:
        raise AssertionError(f"not possible to set '{found[0]}': {found[1]}")


def held() -> dict[str, Any]:
    """The picks waiting for the running generation to end (shown by the settings)."""

    with _guard:
        return dict(_held)


def apply_held(apply: Callable[[str, Any], bool] | None = None) -> list[str]:
    """Apply the held picks; the caller holds the generation lock. Returns
    the options applied."""

    with _guard:
        picks = dict(_held)
        _held.clear()
    done = []
    for key, value in picks.items():
        try:
            if (apply or _apply_now)(key, value):
                done.append(key)
        except Exception:  # a file removed meanwhile: keep the loaded selection
            log.exception("Could not apply the held model selection %s=%r", key, value)
    return done


def _start_waiter() -> None:
    """Under ``_guard``: a thread that applies the held picks once the lock is free."""

    global _waiter
    if _waiter is not None and _waiter.is_alive():
        return

    def wait() -> None:
        global _waiter
        while True:
            with _generation_lock():
                apply_held()
            with _guard:
                if not _held:  # a pick held while this one applied: go round again
                    _waiter = None
                    return

    _waiter = threading.Thread(target=wait, name="remi-model-selection", daemon=True)
    _waiter.start()
