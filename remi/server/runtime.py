"""Server-side startup before each app is created (also on "Reload UI"):
extension settings, the script runners and their layouts, the built-in tab
names and the model selection.

Scripts and Extras operations without ``params()`` (A1111's Gradio ``ui()``)
and script files that fail to import are left out; bootstrap lists them in
``not_ported``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from remi.setting_values import setting
from remi.ui.manifest import not_ported as not_ported

log = logging.getLogger(__name__)

BUILTIN_TABS = (("txt2img", "txt2img"), ("img2img", "img2img"), ("extras", "Extras"), ("png-info", "PNG Info"), ("gallery", "Gallery"), ("networks", "Networks"))
"""``(route id, title)`` of the built-in tabs; titles are what the
``ui_tab_order`` / ``hidden_tabs`` options store."""


def prepare(*, models: bool = True) -> None:
    """Settings, runners and model selection for a new app."""

    from modules import script_callbacks, shared, shared_items
    from modules.options import options_section
    from remi.server.settings import register_options
    from remi.setting_values import repair

    script_callbacks.ui_settings_callback()
    shared.opts.data_labels.update(options_section(("callbacks", "Callbacks", "system"), {**shared_items.callbacks_order_settings()}))
    register_options()
    # every setting is registered now: one of the wrong kind (config.json, /sdapi/v1/options) reads as its default
    repair(shared.opts)
    apply_warning_filters(setting("show_warnings", False))
    shared.tab_names[:] = [title for _id, title in BUILTIN_TABS]
    initialize_runners()
    shared.tab_names.extend(_page_titles())  # extension pages, as their tabs were
    if models:
        initialize_model_selection()


QUIET_WARNINGS = (UserWarning, FutureWarning)
_warning_filters: list[tuple] = []


def apply_warning_filters(show: bool) -> None:
    """The setting "Show warnings in console" (``show_warnings``): when off,
    Python's ``UserWarning`` / ``FutureWarning`` (library deprecation noise)
    are not printed. Applied at every app build ("Reload UI")."""

    for entry in _warning_filters:
        if entry in warnings.filters:
            warnings.filters.remove(entry)
    _warning_filters.clear()
    if not show:
        for category in QUIET_WARNINGS:
            warnings.filterwarnings("ignore", category=category)
            _warning_filters.append(warnings.filters[0])
    # Direct edits of ``warnings.filters`` must invalidate the per-module "already warned" caches
    # (CPython's private hook; without it only the warnings already shown once stay cached)
    mutated = getattr(warnings, "_filters_mutated", None)
    if mutated is not None:
        mutated()


def _page_titles() -> list[str]:
    """Registered extension pages' titles (their scripts are loaded by now)."""

    try:
        from remi import pages
    except ImportError:
        return []
    builtin = {title for _id, title in BUILTIN_TABS}
    return [title for title in dict.fromkeys(pages.tab_titles()) if title not in builtin]


def initialize_runners() -> None:
    """Build the txt2img / img2img runners and assign every script's and
    Extras operation's argument slots."""

    from modules import scripts
    from remi.core.extras import assign_args

    for runner, is_img2img in ((scripts.scripts_txt2img, False), (scripts.scripts_img2img, True)):
        runner.initialize_scripts(is_img2img=is_img2img)
        runner.ensure_arg_slots()

    postproc = scripts.scripts_postproc
    if postproc is not None:
        assign_args(postproc)


# region Model selection


def initialize_model_selection() -> None:
    """Pick a checkpoint when none is set, scan the module folders and set
    the loading parameters."""

    from modules import sd_models, shared

    if setting("sd_model_checkpoint", "") in ("", "None", "none"):
        if not sd_models.checkpoints_list:
            sd_models.list_models()
        if sd_models.checkpoints_list:
            first = next(iter(sd_models.checkpoints_list.values())).name
            try:
                shared.opts.set("sd_model_checkpoint", first)
            except AssertionError:
                # frozen settings (--freeze-settings and its narrower forms) refuse the setter:
                # this run still starts on the first checkpoint
                shared.opts.data["sd_model_checkpoint"] = first
    main_entry = _main_entry()
    main_entry.refresh_models()
    main_entry.refresh_model_loading_parameters()


def _main_entry() -> Any:
    from modules_forge import main_entry

    return main_entry


def load_selected_model(*, force: bool = False) -> None:
    """Load the selected checkpoint, modules and dtype now (call on the
    worker). ``force``: load again even when they are already loaded."""

    from modules import sd_models
    from remi import model_selection

    model_selection.apply_held()  # the caller holds the generation lock
    _main_entry().refresh_model_loading_parameters()
    if force:
        sd_models.model_data.forge_hash = ""
    sd_models.forge_model_reload()


def unet_dtypes() -> list[str]:
    from remi.model_selection import unet_dtypes as dtypes

    return dtypes()


def select_checkpoint(name: str) -> bool:
    from remi import model_selection

    return model_selection.select("sd_model_checkpoint", name)


def select_modules(names: list[str]) -> bool:
    from remi import model_selection

    return model_selection.select("forge_additional_modules", names)


def select_unet_dtype(dtype: str) -> bool:
    from remi import model_selection

    return model_selection.select("forge_unet_storage_dtype", dtype)


def refresh_models() -> None:
    """Re-scan checkpoints, VAE and text encoders (once each), and the
    upscalers."""

    _main_entry().refresh_models()
    refresh_upscalers()


def refresh_upscalers() -> None:
    """Re-scan the upscaler folders (``shared.sd_upscalers``): the Hires. fix,
    Extras, SD upscale and settings choices are built from that list on every
    schema request and validation, so a model added while the server runs
    can be picked without a restart."""

    from modules import modelloader

    modelloader.load_upscalers()


# endregion
