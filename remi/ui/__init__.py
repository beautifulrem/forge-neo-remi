"""``remi.ui``: the declarative parameter and layout model.

Scripts declare their parameters once, as a tree returned by ``params()``::

    from remi import ui

    def params(self, is_img2img):
        return ui.Section("DAVE", enable=ui.Toggle("enable", False, preset="dave.enabled"), children=[
            ui.Slider("strength", 0.0, 1.0, 0.3, 0.05, infotext="DAVE strength", preset="dave.strength"),
        ])

That tree drives the SPA (schema export), the HTTP API (named values to
positional script arguments), infotext paste, presets and conflict locks.
See docs/remi-webui/architecture.md and schema-format.md.
"""

from remi.ui import conditions
from remi.ui.actions import ActionContext, ActionSpec, ActionTimeoutError, action, actions_of, call_action, call_action_isolated, check_patch, merge_patches, message, toast, update
from remi.ui.conditions import all_of, any_of, evaluate, not_, param
from remi.ui.context import Context
from remi.ui.errors import ParamError, SchemaError, ValidationError
from remi.ui.files import ActionFiles
from remi.ui.layout import ScriptLayout
from remi.ui.manifest import MANIFEST_FILENAME, DiagnosticsCollector, ExtensionManifest, ManifestValidationResult, diagnostics, load_manifest, validate_manifest_dict
from remi.ui.nodes import (
    Button,
    Choice,
    Color,
    Column,
    Custom,
    File,
    Gallery,
    Group,
    Html,
    Image,
    ImageList,
    Input,
    Json,
    Markdown,
    MultiChoice,
    Node,
    Number,
    Output,
    RangeSlider,
    Row,
    Section,
    Seed,
    Separator,
    Slider,
    Slot,
    State,
    Tab,
    Tabs,
    Text,
    Toggle,
    without_model_hash,
)
from remi.ui.schema import SCHEMA_VERSION, jsonable, to_schema
from remi.ui.version import REMI_API_VERSION, REMI_MIN_SUPPORTED_API_VERSION, CompatibilityResult, SemVer, check_api_compatibility, parse_semver

__all__ = [
    "MANIFEST_FILENAME",
    "REMI_API_VERSION",
    "REMI_MIN_SUPPORTED_API_VERSION",
    "SCHEMA_VERSION",
    "ActionContext",
    "ActionFiles",
    "ActionSpec",
    "ActionTimeoutError",
    "Button",
    "Choice",
    "Color",
    "Column",
    "CompatibilityResult",
    "Context",
    "Custom",
    "DiagnosticsCollector",
    "ExtensionManifest",
    "File",
    "Gallery",
    "Group",
    "Html",
    "Image",
    "ImageList",
    "Input",
    "Json",
    "ManifestValidationResult",
    "Markdown",
    "MultiChoice",
    "Node",
    "Number",
    "Output",
    "ParamError",
    "RangeSlider",
    "Row",
    "SchemaError",
    "ScriptLayout",
    "Section",
    "Seed",
    "SemVer",
    "Separator",
    "Slider",
    "Slot",
    "State",
    "Tab",
    "Tabs",
    "Text",
    "Toggle",
    "ValidationError",
    "action",
    "actions_of",
    "all_of",
    "any_of",
    "call_action",
    "call_action_isolated",
    "check_api_compatibility",
    "check_patch",
    "conditions",
    "diagnostics",
    "evaluate",
    "jsonable",
    "load_manifest",
    "merge_patches",
    "message",
    "not_",
    "param",
    "parse_semver",
    "to_schema",
    "toast",
    "update",
    "validate_manifest_dict",
    "without_model_hash",
]
