import contextlib
import errno
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass

from modules import errors
from modules.shared_cmd_options import cmd_opts
from remi import ui


class OptionInfo:
    """One setting. ``component`` is a ``remi.ui`` widget class (``ui.Slider``,
    ``ui.Choice``, ...; see remi/ui/settings.py) with its ``component_args``;
    ``None`` picks one from the default's type."""

    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, comment_before='', comment_after='', infotext=None, restrict_api=False, category_id=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.category_id = category_id
        self.refresh = refresh
        self.do_not_save = False

        # Help, structured for the settings schema (remi/ui/settings.py). A1111's
        # comment_before / comment_after arguments (HTML) are kept as html help.
        self.notes: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.html_blocks: list[str] = [text for text in (comment_before, comment_after) if text]
        self.choice_hints: dict[str, str] = {}
        """Description of each choice, by choice label (schema ``choice_hints``)"""
        self.default_order: list[str] = []
        """For an ordering setting: the order used while it is empty (schema ``default_order``)"""
        self.requires_restart = False
        self.requires_reload_ui = False

        self.infotext = infotext

        self.restrict_api = restrict_api
        """If True, the setting will not be accessible via API"""

    def link(self, label, url):
        self.links.append((label, url))
        return self

    def js(self, label, js_func):
        """A1111 link running a client-side function of the old settings page; ignored."""

        return self

    def info(self, *infos):
        """Help notes, one per argument (plain text; A1111's inline markup is reduced to text)."""

        for info in infos:
            self.notes.append(info)
        return self

    def choice_info(self, hints):
        """Describe the choices: ``{choice label: plain-text description}``."""

        self.choice_hints.update(hints)
        return self

    def html(self, html):
        """A1111 help markup; Remi's own options use notes, links and choice_info instead."""

        self.html_blocks.append(html)
        return self

    def needs_restart(self):
        self.requires_restart = True
        return self

    def needs_reload_ui(self):
        self.requires_reload_ui = True
        return self


class OptionHTML(OptionInfo):
    def __init__(self, text):
        super().__init__(str(text).strip(), label='', component=ui.Html)

        self.do_not_save = True


class OptionMarkdown(OptionInfo):
    """A help block between settings, written in Markdown: the settings schema
    sends it as a ``markdown`` node, which the SPA renders as text, links and
    emphasis (never as HTML). Remi's own options use it; ``OptionHTML`` stays
    for A1111 extensions."""

    def __init__(self, text):
        super().__init__(str(text).strip(), label='', component=ui.Markdown)

        self.do_not_save = True


class OptionDiv(OptionInfo):
    def __init__(self):
        super().__init__("", label="", component=ui.Separator)

        self.do_not_save = True


class OptionRow(OptionInfo):
    """Opens a row of settings; the next OptionRow of the section closes it."""

    def __init__(self):
        super().__init__("", label="", component=ui.Row)

        self.do_not_save = True


def options_section(section_identifier, options_dict):
    for v in options_dict.values():
        if len(section_identifier) == 2:
            v.section = section_identifier
        elif len(section_identifier) == 3:
            v.section = section_identifier[0:2]
            v.category_id = section_identifier[2]

    return options_dict


options_builtin_fields = {"data_labels", "data", "restricted_opts", "typemap"}
_NOT_STORED = object()


def write_whole(filename, data: bytes):
    """Remi: ``data`` written to ``filename`` whole: into a new file next to it (next to the link's
    target for a symlink, with the file's mode), then swapped in, so a failed write or a crash leaves
    the old file. Where the swap is refused (a file bind-mounted on its own, one another process
    holds open on Windows) the bytes, already encoded, are written in place. So is a hard-linked
    file (a new one would leave the other name behind), and a read-only one refuses the save as
    ``open(..., "w")`` did (the directory would have let a new file in). The new file gets the old
    one's owner and group; where it cannot (another user's file) the file is written in place, as
    upstream did, rather than change hands. A file is written in place only once the new file next
    to it showed the bytes fit (or, where none can be made, the disk's free space does): a full
    disk fails the save before the file is truncated. A new file a crash left behind is removed at
    the next save."""
    target = os.path.realpath(filename)
    _remove_stale_temporaries(target)
    try:
        status = os.stat(target)
    except OSError:
        status = None
    if status is not None and not os.access(target, os.W_OK):
        with open(target, "wb") as file:  # refused, before anything is written
            file.write(data)
        return
    mode = status.st_mode & 0o777 if status is not None else None
    temporary = f"{target}.{uuid.uuid4().hex}.tmp"
    try:
        handle = open(temporary, "xb")
    except OSError as error:
        if error.errno in _NO_SPACE:
            raise
        _write_in_place(target, data, status)  # no new file can go next to it (a folder it cannot write)
        return
    try:
        with handle as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:  # out of space (in place it would run out too, after the truncation)
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise
    try:
        if status is not None and status.st_nlink > 1:
            raise OSError(errno.EMLINK, "a new file would leave the other names behind")
        if status is not None and hasattr(os, "chown"):
            created = os.stat(temporary)
            if (created.st_uid, created.st_gid) != (status.st_uid, status.st_gid):
                os.chown(temporary, status.st_uid, status.st_gid)  # refused for another user's file
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, target)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        with open(target, "wb") as file:  # the new file held the bytes: they fit
            file.write(data)


_NO_SPACE = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}


def _write_in_place(target: str, data: bytes, status) -> None:
    """``data`` written over ``target`` when no new file could show that it fits: refused
    without touching the file when the free space is short of what the file grows by."""

    grows = len(data) - (status.st_size if status is not None else 0)
    if grows > 0 and shutil.disk_usage(os.path.dirname(target) or ".").free < grows:
        raise OSError(errno.ENOSPC, "No space left on device", target)
    with open(target, "wb") as file:
        file.write(data)


_STALE_AFTER = 60.0
"""Seconds after which a new file of ``write_whole`` was left by a crash, not a save under way."""


def _remove_stale_temporaries(target: str) -> None:
    folder, name = os.path.split(target)
    pattern = re.compile(re.escape(name) + r"\.[0-9a-f]{32}\.tmp")
    with contextlib.suppress(OSError), os.scandir(folder or ".") as entries:
        for entry in entries:
            if pattern.fullmatch(entry.name):
                with contextlib.suppress(OSError):
                    if entry.is_file(follow_symlinks=False) and time.time() - entry.stat(follow_symlinks=False).st_mtime > _STALE_AFTER:
                        os.remove(entry.path)


_UNWRITABLE = (RecursionError, UnicodeEncodeError, TypeError, ValueError)


def _encoded(data) -> bytes:
    encoded = json.dumps(data, indent=4, ensure_ascii=False).encode("utf8")
    json.loads(encoded)  # written only when the next start can read it back
    return encoded


def _writable(key, value) -> bool:
    try:
        _encoded({key: value})
    except _UNWRITABLE:
        return False
    return True


class Options:
    typemap = {int: float}

    def __init__(self, data_labels: dict[str, OptionInfo], restricted_opts):
        self.data_labels = data_labels
        self.data = {k: v.default for k, v in self.data_labels.items() if not v.do_not_save}
        self.restricted_opts = restricted_opts
        self._held_for_run = {}  # Remi: see hold_for_run

    def __setattr__(self, key, value):
        if key in options_builtin_fields:
            return super(Options, self).__setattr__(key, value)

        if self.data is not None:
            if key in self.data or key in self.data_labels:

                # Check that settings aren't globally frozen
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                # Get the info related to the setting being changed
                info = self.data_labels.get(key, None)
                if info.do_not_save:
                    return

                # Restrict component arguments
                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set '{key}' because it is restricted")

                # Check that this section isn't frozen
                if cmd_opts.freeze_settings_in_sections is not None:
                    frozen_sections = list(map(str.strip, cmd_opts.freeze_settings_in_sections.split(','))) # Trim whitespace from section names
                    section_key = info.section[0]
                    section_name = info.section[1]
                    assert section_key not in frozen_sections, f"not possible to set '{key}' because settings in section '{section_name}' ({section_key}) are frozen with --freeze-settings-in-sections"

                # Check that this section of the settings isn't frozen
                if cmd_opts.freeze_specific_settings is not None:
                    frozen_keys = list(map(str.strip, cmd_opts.freeze_specific_settings.split(','))) # Trim whitespace from setting keys
                    assert key not in frozen_keys, f"not possible to set '{key}' because this setting is frozen with --freeze-specific-settings"

                # Check shorthand option which disables editing options in "saving-paths"
                if cmd_opts.hide_ui_dir_config and key in self.restricted_opts:
                    raise RuntimeError(f"not possible to set '{key}' because it is restricted with --hide_ui_dir_config")

                self.data[key] = value
                return

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if item in options_builtin_fields:
            return super(Options, self).__getattribute__(item)

        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def set(self, key, value, is_api=False, run_callbacks=True):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""

        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        option = self.data_labels[key]
        if option.do_not_save:
            return False

        if is_api and option.restrict_api:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if run_callbacks and option.onchange is not None:
            try:
                option.onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def get_default(self, key):
        """returns the default value for the key"""

        data_label = self.data_labels.get(key)
        if data_label is None:
            return None

        return data_label.default

    def hold_for_run(self, key, value):
        """Remi: sets ``key`` to ``value`` for this run only, past the frozen-settings checks (a
        frozen setting the WebUI has to fill in to start). Saves keep the stored value while
        ``value`` is still the live one."""
        self._held_for_run[key] = (value, self.data.get(key, _NOT_STORED))
        self.data[key] = value

    def save(self, filename):
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        data = dict(self.data)
        for key, (value, stored) in self._held_for_run.items():  # Remi: see hold_for_run
            if key in data and data[key] == value:
                if stored is _NOT_STORED:
                    del data[key]
                else:
                    data[key] = stored

        # Remi: encoded first, then swapped in whole. Dumped straight into the open file, a value json
        # cannot write (nested past the recursion limit, an unpaired surrogate) stopped it half way and
        # left config.json cut short: every setting lost at the next start. Such a value (an extension's,
        # past the API's checks) is left out with an error, or every later save would fail with it
        try:
            data = _encoded(data)
        except _UNWRITABLE:
            unwritable = {key for key, value in data.items() if not _writable(key, value)}
            errors.report(f"Settings not saved, as JSON cannot write their values: {', '.join(sorted(map(str, unwritable)))}")
            data = _encoded({key: value for key, value in data.items() if key not in unwritable})
        write_whole(filename, data)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        try:
            with open(filename, "r", encoding="utf8") as file:
                self.data = json.load(file)
        except FileNotFoundError:
            self.data = {}

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func

        if call:
            func()

    def add_option(self, key, info):
        self.data_labels[key] = info
        if key not in self.data and not info.do_not_save:
            self.data[key] = info.default

    def reorder(self):
        """Reorder settings so that:
            - all items related to section always go together
            - all sections belonging to a category go together
            - sections inside a category are ordered alphabetically
            - categories are ordered by creation order

        Category is a superset of sections: for category "postprocessing" there could be multiple sections: "face restoration", "upscaling".

        This function also changes items' category_id so that all items belonging to a section have the same category_id.
        """

        category_ids = {}
        section_categories = {}

        settings_items = self.data_labels.items()
        for _, item in settings_items:
            if item.section not in section_categories:
                section_categories[item.section] = item.category_id

        for _, item in settings_items:
            item.category_id = section_categories.get(item.section)

        for category_id in categories.mapping:
            if category_id not in category_ids:
                category_ids[category_id] = len(category_ids)

        def sort_key(x):
            item: OptionInfo = x[1]
            category_order = category_ids.get(item.category_id, len(category_ids))
            section_order = item.section[1]

            return category_order, section_order

        self.data_labels = dict(sorted(settings_items, key=sort_key))

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """

        if value is None:
            return None

        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None

        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        else:
            value = expected_type(value)

        return value


@dataclass
class OptionsCategory:
    id: str
    label: str

class OptionsCategories:
    def __init__(self):
        self.mapping = {}

    def register_category(self, category_id, label):
        if category_id in self.mapping:
            return category_id

        self.mapping[category_id] = OptionsCategory(category_id, label)


categories = OptionsCategories()
