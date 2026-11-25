/**
 * TypeScript mirror of docs/remi-webui/schema-format.md (v1). Keep in sync with
 * remi/ui/schema.py. Optional keys may be absent; the renderer never relies
 * on them being present.
 */

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }

// ---------------------------------------------------------------- conditions

export type LeafCondition = { param: string } & (
  | { eq?: never; ne?: never; in?: never; gt?: never; gte?: never; lt?: never; lte?: never }
  | { eq: Json }
  | { ne: Json }
  | { in: Json[] }
  // Ordered comparisons follow Python: numbers (and booleans) with numbers, strings with strings.
  | { gt: number | string }
  | { gte: number | string }
  | { lt: number | string }
  | { lte: number | string }
)

export type Condition =
  | boolean
  | LeafCondition
  | { all: Condition[] }
  | { any: Condition[] }
  | { not: Condition }

// ---------------------------------------------------------------- common keys

type NodeBase = {
  visible_if?: Condition
  enabled_if?: Condition
}

/** 12-column span, or "auto"/"fill". */
export type Width = "auto" | "fill" | number

export type InputBase = NodeBase & {
  name: string
  /** "" renders the control without a label. */
  label: string
  info?: string | null
  /** Tooltip on the label (also the control's accessible description). */
  hint?: string | null
  default?: Json
  on_change?: string | null
  /** Runs only after a user edit of this control (not pastes, presets or action patches). */
  on_input?: string | null
  width?: Width
  hidden?: boolean
  infotext?: string | null
  preset?: string | null
  /** `false`: kept out of drafts; each page load starts from the current default. */
  persist?: boolean
  /** `false`: held for actions only, never sent with a job (the server uses the default). */
  submit?: boolean
  /** While true the control is dimmed but stays editable: its value has no effect with the current values. */
  inactive_if?: Condition
}

export type ChoiceOption = { value: string | number; label: string }

// ---------------------------------------------------------------- inputs

export type ToggleNode = InputBase & {
  type: "toggle"
  exclusive?: { group: string; member: string } | null
  /** `"owner.param"` (or a param of this owner) protected from programmatic writes while on. */
  locks?: string | null
}
export type SliderNode = InputBase & { type: "slider"; minimum: number; maximum: number; step: number; integer?: boolean }
export type NumberNode = InputBase & {
  type: "number"
  minimum?: number | null
  maximum?: number | null
  step?: number | null
  precision?: number | null
  integer?: boolean
}
export type TextNode = InputBase & {
  type: "text"
  lines?: number
  max_lines?: number | null
  placeholder?: string | null
  prompt?: boolean
}
export type ChoiceNode = InputBase & {
  type: "choice"
  choices: ChoiceOption[]
  style?: "dropdown" | "radio"
  allow_custom?: boolean
  /** Tooltips by (untranslated) choice label. */
  choice_hints?: Record<string, string>
}
export type MultiChoiceNode = InputBase & {
  type: "multichoice"
  choices: ChoiceOption[]
  style?: "dropdown" | "checkboxes"
  allow_custom?: boolean
  choice_hints?: Record<string, string>
}
export type SeedNode = InputBase & { type: "seed" }
export type ColorNode = InputBase & { type: "color" }
export type ImageNode = InputBase & {
  type: "image"
  kind?: "image" | "mask" | "sketch"
  source?: string | null
  mode?: "RGB" | "RGBA" | "L" | null
}
/** `widget` edits the list with a registered widget (value: upload ids) instead of the stock control. */
export type ImageListNode = InputBase & { type: "image_list"; widget?: string | null; props?: Record<string, Json> }
export type FileNode = InputBase & { type: "file"; accept?: string | null }
export type JsonNode = InputBase & { type: "json" }
export type CustomNode = InputBase & { type: "custom"; widget: string; props?: Record<string, Json> }
/** A node a registered widget renders: `custom`, or an input kind with a `widget` hint. */
export type WidgetNode = CustomNode | (ImageListNode & { widget: string })
export type StateNode = InputBase & { type: "state" }

export type InputNode =
  | ToggleNode
  | SliderNode
  | NumberNode
  | TextNode
  | ChoiceNode
  | MultiChoiceNode
  | SeedNode
  | ColorNode
  | ImageNode
  | ImageListNode
  | FileNode
  | JsonNode
  | CustomNode
  | StateNode

export type InputType = InputNode["type"]

// ---------------------------------------------------------------- layout / display

export type SectionNode = NodeBase & {
  type: "section"
  title: string
  open?: boolean
  enable?: ToggleNode | null
  children: Node[]
}
export type GroupNode = NodeBase & { type: "group"; title?: string | null; open?: boolean; children: Node[] }
export type RowNode = NodeBase & { type: "row"; children: Node[] }
export type ColumnNode = NodeBase & { type: "column"; width?: Width; children: Node[] }
export type TabNode = NodeBase & { type: "tab"; title: string; value?: string | number | null; children: Node[] }
export type TabsNode = NodeBase & { type: "tabs"; bind?: string | null; children: TabNode[] }
export type MarkdownNode = NodeBase & { type: "markdown"; text: string }
export type HtmlNode = NodeBase & { type: "html"; text: string }
export type SeparatorNode = NodeBase & { type: "separator" }
export type ButtonNode = NodeBase & {
  type: "button"
  label: string
  action: string
  variant?: "primary" | "secondary" | "ghost" | "tool" | "danger"
}
export type GalleryNode = NodeBase & { type: "gallery"; name: string; label?: string | null }
export type OutputNode = NodeBase & {
  type: "output"
  name: string
  kind?: "text" | "html" | "image" | "json"
  label?: string | null
}
export type SlotNode = NodeBase & { type: "slot"; section: string }

export type LayoutNode =
  | SectionNode
  | GroupNode
  | RowNode
  | ColumnNode
  | TabsNode
  | TabNode
  | MarkdownNode
  | HtmlNode
  | SeparatorNode
  | ButtonNode
  | GalleryNode
  | OutputNode
  | SlotNode

export type Node = InputNode | LayoutNode
export type NodeType = Node["type"]

const INPUT_TYPES: ReadonlySet<string> = new Set<InputType>([
  "toggle",
  "slider",
  "number",
  "text",
  "choice",
  "multichoice",
  "seed",
  "color",
  "image",
  "image_list",
  "file",
  "json",
  "custom",
  "state",
])

export function isInput(node: Node): node is InputNode {
  return INPUT_TYPES.has(node.type)
}

/** Child nodes of a layout node (empty for leaves). */
export function childrenOf(node: Node): readonly Node[] {
  return "children" in node && Array.isArray(node.children) ? node.children : []
}

// ---------------------------------------------------------------- documents

export type OwnerDefaults = Record<string, Json>

/** Declarative active-model capability summary (`backend/nn/model_capabilities.py`). */
export type ModelCapabilitySummary = {
  preset: string
  dit_name: string | null
  engine_label: string | null
  capabilities: string[]
  max_references: number
  reference_slot_format: string | null
  /** 1 for `<image_1>`, 0 for `ref_image_0` / `Picture 0`. */
  reference_slot_start?: number
  supports_rgba: boolean
  supports_control_blocks: boolean
  supports_cfg_guidance?: boolean
  supports_negative_prompt?: boolean
  attention_backends?: string[]
  quant_formats?: string[]
  compile_preset?: string | null
  is_video?: boolean
  frames?: number
  use_distill?: boolean
  use_shift?: boolean
  defaults?: Record<string, Json>
  badges?: string[]
}

export type AdviceSeverity = "info" | "warning" | "error"

/**
 * The catalog key of a sentence the server filled in (`remi.ui.Sentence`): `template` with
 * `{{name}}` placeholders and the values that fill them; a list value holds catalog names.
 */
export type SentenceParams = Record<string, string | readonly string[]>

/** Pre-submit workflow advice chip with an optional one-click parameter/option fix. */
export type AdviceItem = {
  code: string
  severity: AdviceSeverity
  /** English; translated by `template` when it has values in it. */
  message: string
  template?: string
  params?: SentenceParams
  /** Core-relative ParamPatch entries and/or `__options__` server option updates. */
  patch?: (Record<string, unknown> & { __options__?: Record<string, unknown> }) | null
  action_label?: string | null
}

export type ScriptSchema = {
  id: string
  title: string
  kind: "alwayson" | "selectable"
  section?: string | null
  order?: number
  visible_if?: Condition | null
  layout: Node | null
  defaults: OwnerDefaults
  args?: string[]
  /** Declarative capability tags this script requires (e.g. `["anima_hooks"]`). */
  requires_capabilities?: string[]
  /** False when the active model lacks one or more `requires_capabilities`. */
  applicable?: boolean
  /** Human-readable explanation shown when `applicable === false` (English; see `inapplicable_template`). */
  inapplicable_reason?: string | null
  /** The reason's catalog key and values (it names the active model). */
  inapplicable_template?: string
  inapplicable_params?: SentenceParams
}

/** A core input that must not be empty while `when` holds (schema-format.md §3). */
export type RequiredInput = { param: string; message: string; when: Condition }

export type TabSchema = {
  version: number
  tab: "txt2img" | "img2img" | "extras"
  /** `required`: checked before submitting; the server applies the same rules. */
  core: { owner: "core"; layout: Node; defaults: OwnerDefaults; required?: RequiredInput[] }
  scripts: ScriptSchema[]
  /**
   * `"owner.param"` references (core params as `"core.<name>"`) whose changes
   * run the core `__conflicts__` action with `selected` set to the reference.
   */
  conflict_params?: string[]
  /** Active model capability metadata driving panel visibility, badges and slot formatting. */
  capabilities?: ModelCapabilitySummary | null
  /** Initial pre-submit workflow advice items for the current form defaults/options. */
  advice?: AdviceItem[]
}

/**
 * Schema-rendered extension page (`GET /api/v2/pages/{id}`). Not specified in
 * schema-format.md v1; assumed to mirror a single owner of a tab schema.
 */
export type PageSchema = {
  version: number
  id: string
  title: string
  /** Action owner, `page:<id>`. */
  owner: string
  layout: Node | null
  defaults: OwnerDefaults
}

/** A settings input (§6): an input node plus the setting's help and flags. */
export type SettingNode = InputNode & {
  notes?: string[]
  links?: { label: string; url: string }[]
  /** An ordering setting's order while it is empty (plain names, not translated). */
  default_order?: string[]
  /** Help markup of a third-party option (A1111 `.html()`; sanitised like `html` nodes). */
  html?: string | null
  needs_restart?: boolean
  needs_reload_ui?: boolean
  refresh?: boolean
  onchange?: boolean
  /** Not changeable through the A1111 API (/sdapi/v1); the WebUI can change it. */
  restricted?: boolean
  /** Why the server refuses changes (frozen, read-only); such a setting is `enabled_if: false`. */
  locked?: string
}

export type SettingsSchema = {
  version: number
  categories: {
    id: string
    label: string
    sections: { id: string; label: string; items: (SettingNode | MarkdownNode | HtmlNode | SeparatorNode | RowNode)[] }[]
  }[]
  quicksettings: string[]
  values: Record<string, Json>
  /** Each stored setting's built-in default (absent from older servers). */
  defaults?: Record<string, Json>
}

/** One rejected value of a request (§3); `owner` is `"core"`, a script id or `"settings"`. */
/** A rejected value's message: English, with its catalog key when it names the value (schema-format.md §3). */
export type FieldMessage = { message: string; template?: string; params?: SentenceParams }
export type ValidationError = FieldMessage & { owner: string; param: string }
/** A value a job refused when it ran, as the server stores it: with its field when the entry names one. */
export type RefusedValue = FieldMessage & { owner?: string; param?: string }

/** Body of a rejected request (§3). */
export type ValidationErrors = { errors: ValidationError[] }
