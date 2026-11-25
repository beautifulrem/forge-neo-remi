/**
 * Rebuilds an editable Layout draft from a compiled MOSS-Image prompt, or
 * checks a saved draft (upstream `sd_forge_json_builder/javascript/decompile.js`).
 * A compiled prompt has no canvas size: its boxes are relative to the
 * centre, so the caller gives the width and height it was made for.
 */
import { CANVAS_TYPES, check, isObject, LABELS } from "./compile"

export type Label = (typeof LABELS)[number]
export type CanvasType = (typeof CANVAS_TYPES)[number]
export type Box = [x1: number, y1: number, x2: number, y2: number]

/** What the builder edits: boxes in pixels from the top-left corner. */
export type LayoutDraft = {
  width: number
  height: number
  summary: { caption: string; meta: string[]; canvas: CanvasType }
  items: { label: Label; caption: string; bbox: Box; text?: string }[]
}

function normalizeDraft(value: unknown): LayoutDraft {
  check(isObject(value), "Draft must be an object")
  const { width, height } = value
  for (const [field, dim] of [
    ["width", width],
    ["height", height],
  ] as const) {
    check(typeof dim === "number" && Number.isFinite(dim) && dim > 0, "{{field}} must be a positive number", { field })
  }

  const summary = value.summary
  check(isObject(summary), "{{location}} must be an object", { location: "summary" })
  check(typeof summary.caption === "string" && summary.caption.trim().length > 0, "{{location}} must be nonempty text", { location: "summary.caption" })
  const canvas = summary.canvas === undefined ? "single" : summary.canvas
  check((CANVAS_TYPES as readonly unknown[]).includes(canvas), "summary.canvas must be single, multi_panel or collage")
  let meta: string[] = []
  if (summary.meta !== undefined) {
    check(Array.isArray(summary.meta), "{{location}} must be an array", { location: "summary.meta" })
    meta = summary.meta.map((entry: unknown, index) => {
      check(typeof entry === "string" && entry.trim().length > 0, "{{location}} must be nonempty text", { location: `summary.meta[${index}]` })
      return entry
    })
  }

  const items = value.items
  check(Array.isArray(items) && items.length > 0, "items must be a nonempty array")
  const cleanItems = items.map((item: unknown, index) => {
    const location = `items[${index}]`
    check(isObject(item), "{{location}} must be an object", { location })
    check((LABELS as readonly unknown[]).includes(item.label), "{{location}} must be object, character or text", { location: `${location}.label` })
    check(typeof item.caption === "string" && item.caption.trim().length > 0, "{{location}} must be nonempty text", { location: `${location}.caption` })
    const box = item.bbox
    check(Array.isArray(box) && box.length === 4, "{{location}} must be [x1,y1,x2,y2]", { location: `${location}.bbox` })
    check(
      box.every((v) => typeof v === "number" && Number.isFinite(v)),
      "{{location}} must contain four finite numbers",
      { location: `${location}.bbox` },
    )
    const bbox = box.slice() as Box
    check(bbox[0] < bbox[2] && bbox[1] < bbox[3], "{{location}} must have positive width and height", { location: `${location}.bbox` })
    const clean: LayoutDraft["items"][number] = { label: item.label as Label, caption: item.caption, bbox }
    if (item.label === "text") {
      check(typeof item.text === "string" && item.text.trim().length > 0, "{{location}} must be nonempty text", { location: `${location}.text` })
      clean.text = item.text
    }
    return clean
  })

  return { width: width as number, height: height as number, summary: { caption: summary.caption, meta, canvas: canvas as CanvasType }, items: cleanItems }
}

/** `\(` / `\)` back to JSON string escapes (`\\(`): an escaped parenthesis is not valid JSON. */
export function restoreJsonParens(prompt: string): string {
  return prompt.replace(/(?<!\\)((?:\\\\)*)\\([()])/g, (_match, runs: string, paren: string) => `${runs}\\\\${paren}`)
}

/**
 * A compiled prompt and the text the builder appended after it ("Extra", e.g.
 * a LoRA tag): `{…} <lora:x:1>` → `{ json: "{…}", extra: "<lora:x:1>" }`.
 * Text that does not start with a JSON object is returned whole.
 */
export function splitPrompt(prompt: string): { json: string; extra: string } {
  const text = prompt.trim()
  if (!text.startsWith("{")) return { json: text, extra: "" }
  let depth = 0
  let inString = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (inString) {
      if (c === "\\") i++
      else if (c === '"') inString = false
    } else if (c === '"') inString = true
    else if (c === "{") depth++
    else if (c === "}" && --depth === 0) return { json: text.slice(0, i + 1), extra: text.slice(i + 1).trim() }
  }
  return { json: text, extra: "" }
}

/**
 * A draft from a compiled prompt (boxes for a `width` × `height` canvas) or
 * from a saved draft (its own size; `width` and `height` are not used).
 * Throws a {@link DraftError} or a `SyntaxError` (not JSON).
 */
export function decompilePrompt(input: unknown, width: number, height: number): LayoutDraft {
  const draft = typeof input === "string" ? (JSON.parse(restoreJsonParens(input)) as unknown) : input
  check(isObject(draft), "Prompt must be a JSON object")
  if (typeof draft.width === "number" && typeof draft.height === "number") return normalizeDraft(draft)

  check(Number.isFinite(width) && width > 0, "A valid width is required to decompile a prompt")
  check(Number.isFinite(height) && height > 0, "A valid height is required to decompile a prompt")
  check(draft.bbox_coord_space === "abs_centralized", "Prompt bbox_coord_space must be abs_centralized")
  check(draft.bbox_format === "yxyx", "Prompt bbox_format must be yxyx")
  check(Array.isArray(draft.items), "Prompt items must be an array")

  const halfWidth = Math.floor(width / 2)
  const halfHeight = Math.floor(height / 2)
  const items = draft.items.map((item: unknown, index) => {
    check(isObject(item), "{{location}} must be an object", { location: `items[${index}]` })
    const box = item.bbox
    check(Array.isArray(box) && box.length === 4, "{{location}} must be [y1,x1,y2,x2]", { location: `items[${index}].bbox` })
    check(
      box.every((v) => typeof v === "number" && Number.isFinite(v)),
      "{{location}} must contain four finite numbers",
      { location: `items[${index}].bbox` },
    )
    const [y1, x1, y2, x2] = box as [number, number, number, number]
    return { ...item, bbox: [x1 + halfWidth, y1 + halfHeight, x2 + halfWidth, y2 + halfHeight] }
  })

  return normalizeDraft({ width, height, bbox_coord_space: "abs", bbox_format: "xyxy", summary: draft.summary, items })
}
