"""Typed shapes of the exported JSON (docs/remi-webui/schema-format.md).

Plain ``TypedDict``s: the exporters build ordinary dicts, these only give
them names and keys for type checkers.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

Condition: TypeAlias = bool | dict[str, Any]
ChoiceValue: TypeAlias = str | int | float


class ChoiceEntry(TypedDict):
    value: ChoiceValue
    label: str


class NodeSchema(TypedDict, total=False):
    """One exported node (§1). ``type`` is always present; the other keys
    depend on the kind."""

    type: str
    visible_if: Condition
    enabled_if: Condition
    # inputs
    name: str
    label: str
    info: str
    hint: str
    default: Any
    on_change: str
    on_input: str
    width: str | int
    hidden: bool
    infotext: str
    preset: str
    exclusive: dict[str, str]
    inactive_if: Condition
    minimum: float
    maximum: float
    step: float
    integer: bool
    precision: int
    lines: int
    max_lines: int
    placeholder: str
    prompt: bool
    choices: list[ChoiceEntry]
    choice_hints: dict[str, str]
    style: str
    allow_custom: bool
    kind: str
    source: str
    mode: str
    accept: str
    widget: str
    props: dict[str, Any]
    persist: bool
    submit: bool
    locks: str
    # layout / display
    title: str
    open: bool
    enable: NodeSchema
    children: list[NodeSchema]
    bind: str
    value: Any
    text: str
    action: str
    variant: str
    section: str


class PatchEntry(TypedDict, total=False):
    """One param's part of an action patch (§5)."""

    value: Any
    choices: list[ChoiceEntry]
    visible: bool
    enabled: bool
    label: str
    info: str
    hint: str
    minimum: float
    maximum: float
    step: float


Patch: TypeAlias = dict[str, Any]
"""``{param | "owner.param": PatchEntry}`` plus the reserved ``__toast__``,
``__outputs__``, ``__settings__``, ``__refresh__``, ``__advice__``, and
``__capabilities__`` entries."""


class AdviceItemSchema(TypedDict):
    """One pre-submit workflow advice chip (``TabSchema.advice`` / ``Patch.__advice__``)."""

    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    patch: dict[str, Any] | None
    action_label: NotRequired[str]
    # a sentence with values in it: its catalog key (``{{name}}`` placeholders) and the values
    template: NotRequired[str]
    params: NotRequired[dict[str, str | list[str]]]


class ModelCapabilitySchema(TypedDict, total=False):
    """Active model capability metadata exposed on ``TabSchema.capabilities``."""

    preset: str
    dit_name: str | None
    engine_label: str
    capabilities: list[str]
    max_references: int
    reference_slot_format: str | None
    reference_slot_start: int
    supports_rgba: bool
    supports_control_blocks: bool
    supports_cfg_guidance: bool
    supports_negative_prompt: bool
    attention_backends: list[str]
    quant_formats: list[str]
    compile_preset: str | None
    is_video: bool
    frames: int
    use_distill: bool
    use_shift: bool
    defaults: dict[str, Any]
    badges: list[str]


class ScriptEntry(TypedDict):
    """One script of a tab schema (§4)."""

    id: str
    title: str
    kind: Literal["alwayson", "selectable"]
    section: str | None
    order: int | float
    visible_if: NotRequired[Condition]
    layout: NodeSchema | None
    defaults: dict[str, Any]
    args: list[str]
    requires_capabilities: NotRequired[list[str]]
    applicable: NotRequired[bool]
    inapplicable_reason: NotRequired[str]
    # the reason's catalog key and values (the reason is English)
    inapplicable_template: NotRequired[str]
    inapplicable_params: NotRequired[dict[str, str | list[str]]]


class RequiredInput(TypedDict):
    param: str
    message: str
    when: Condition


class OwnerSchema(TypedDict):
    owner: str
    layout: NodeSchema | None
    defaults: dict[str, Any]
    required: NotRequired[list[RequiredInput]]


class TabSchema(TypedDict):
    version: int
    tab: str
    core: OwnerSchema
    scripts: list[ScriptEntry]
    conflict_params: NotRequired[list[str]]
    capabilities: NotRequired[ModelCapabilitySchema]
    advice: NotRequired[list[AdviceItemSchema]]
