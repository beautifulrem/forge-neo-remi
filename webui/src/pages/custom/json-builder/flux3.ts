/**
 * FLUX.3 Image Klein layout prompts from the JSON Builder draft.
 *
 * FLUX.3 is trained on the structured captions its prompt upsampler writes
 * (black-forest-labs/flux-3-image-klein `bbox.py`, Diffusers
 * `modular_pipelines/flux3/prompt_upsampler.py`): one global caption that cites
 * every element by its id in angle brackets, one space, then one JSON array —
 * the element table — with each element's box as `[y_min, x_min, y_max, x_max]`
 * integers on a 0–1000 canvas (top-left origin, independent of the output
 * size). Ids are a class noun plus an index numbered left to right, then top to
 * bottom; text blocks are `En_Text_1`, ….
 *
 * The prompt goes to the text encoder as written (no parenthesis escaping: the
 * FLUX.3 engine reads the text literally).
 */
import { check, DraftError, isObject } from "./compile"
import type { Box, Label, LayoutDraft } from "./decompile"

export const PROMPT_FORMATS = ["moss", "flux3", "vgl"] as const
export type PromptFormat = (typeof PROMPT_FORMATS)[number]
export const PROMPT_FORMAT_NAMES: Record<PromptFormat, string> = { moss: "MOSS-Image", flux3: "FLUX.3", vgl: "Fibo VGL" }

/** The 0–1000 grid FLUX.3 boxes live on. */
export const CANVAS = 1000

export type Flux3Row = { id: string; bbox: [number, number, number, number]; desc: string }

type DraftLike = {
  width: number
  height: number
  caption: string
  meta: string
  extra: string
  items: { label: Label; caption: string; text: string; bbox: Box }[]
}

const clampGrid = (v: number) => Math.min(CANVAS, Math.max(0, v))

/** A pixel box `[x1, y1, x2, y2]` as `[y_min, x_min, y_max, x_max]` on the 0–1000 canvas (outward-rounded). */
export function canvasBox(bbox: Box, width: number, height: number): [number, number, number, number] {
  check(width > 0 && height > 0, "The canvas needs a positive size")
  const [x1, y1, x2, y2] = bbox
  check(x1 < x2 && y1 < y2, "A box must have a positive width and height")
  const box: [number, number, number, number] = [
    clampGrid(Math.floor((y1 / height) * CANVAS)),
    clampGrid(Math.floor((x1 / width) * CANVAS)),
    clampGrid(Math.ceil((y2 / height) * CANVAS)),
    clampGrid(Math.ceil((x2 / width) * CANVAS)),
  ]
  check(box[0] < box[2] && box[1] < box[3], "A box is too small for the 0-1000 canvas")
  return box
}

const ID_NOUNS: Record<Label, string> = { object: "object", character: "person", text: "En_Text" }

const flat = (text: string) => text.replace(/\s+/g, " ").trim()

/** The element table: one row per item, ids numbered left to right, then top to bottom. */
export function flux3Rows(state: DraftLike): Flux3Row[] {
  const order = state.items.map((item, index) => ({ item, index })).sort((a, b) => a.item.bbox[0] - b.item.bbox[0] || a.item.bbox[1] - b.item.bbox[1] || a.index - b.index)
  const counts: Partial<Record<Label, number>> = {}
  const rows = new Map<number, Flux3Row>()
  for (const { item, index } of order) {
    const caption = flat(item.caption)
    check(caption.length > 0, "Item {{n}} needs a caption", { n: index + 1 })
    const n = (counts[item.label] = (counts[item.label] ?? 0) + 1)
    let desc = caption
    if (item.label === "text") {
      const wording = item.text.replace(/\r?\n/g, "\\n").trim()
      check(wording.length > 0, "Item {{n}} needs the text to render", { n: index + 1 })
      desc = `Text reading "${wording}", ${caption}`
    }
    rows.set(index, { id: `${ID_NOUNS[item.label]}_${n}`, bbox: canvasBox(item.bbox, state.width, state.height), desc })
  }
  return state.items.map((_, index) => rows.get(index) as Flux3Row)
}

/** The identity phrase an element is cited with: its description's first clause, at most eight words. */
function identity(row: Flux3Row): string {
  const clause = row.desc.split(/[,.;:]/)[0] ?? row.desc
  const words = clause.trim().split(/\s+/).slice(0, 8).join(" ")
  return words.charAt(0).toLowerCase() + words.slice(1)
}

/** The global caption with every row cited as `<id>` (rows the caption does not cite yet are listed after it). */
export function citedCaption(caption: string, meta: string, rows: Flux3Row[]): string {
  let text = flat(caption)
  check(text.length > 0, "The summary caption must be nonempty text")
  const tags = meta
    .split(",")
    .map((tag) => flat(tag))
    .filter(Boolean)
  if (tags.length) text = `${text.replace(/[.\s]*$/, ".")} Style: ${tags.join(", ")}.`
  const missing = rows.filter((row) => !text.includes(`<${row.id}>`))
  if (missing.length) text = `${text.replace(/[.\s]*$/, ".")} Elements: ${missing.map((row) => `${identity(row)} <${row.id}>`).join(", ")}.`
  return text
}

/** The FLUX.3 layout prompt for the draft, then the extra text on its own line. Throws a `DraftError`. */
export function finalFlux3Prompt(state: DraftLike): string {
  check(state.items.length > 0, "Add at least one item")
  const rows = flux3Rows(state)
  const prompt = `${citedCaption(state.caption, state.meta, rows)} ${JSON.stringify(rows)}`
  const extra = state.extra.replace(/\s+/g, " ").trim()
  return extra ? `${prompt}\n${extra}` : prompt
}

/** `(caption, rows)`: the JSON array that closes the text, if any (BFL `split`). */
export function splitFlux3Prompt(text: string): { caption: string; rows: unknown[] | null; extra: string } {
  const lines = text.trim().split(/\r?\n/)
  // "Send to" writes the extra text (a LoRA tag) on its own line after the prompt
  for (let cut = lines.length; cut > 0; cut--) {
    const body = lines.slice(0, cut).join(" ").trimEnd()
    const extra = lines.slice(cut).join(" ").trim()
    for (let i = body.length - 1; i >= 0; i--) {
      if (body[i] !== "[") continue
      try {
        const rows: unknown = JSON.parse(body.slice(i))
        if (Array.isArray(rows)) return { caption: body.slice(0, i).trimEnd(), rows, extra }
      } catch {
        // not the opening bracket of the table
      }
    }
  }
  return { caption: text.trim(), rows: null, extra: "" }
}

const isBoxRow = (row: unknown): row is { id: string; bbox: number[]; desc: string } =>
  isObject(row) && typeof row.id === "string" && typeof row.desc === "string" && Array.isArray(row.bbox) && row.bbox.length === 4 && row.bbox.every((v) => typeof v === "number" && Number.isFinite(v))

/** True for a FLUX.3 layout prompt: a caption closed by a nonempty table of `{id, bbox, desc}` rows. */
export function isFlux3Prompt(text: string): boolean {
  const { rows } = splitFlux3Prompt(text)
  return rows !== null && rows.length > 0 && rows.every(isBoxRow)
}

/** A FLUX.3 layout prompt back into an editable draft for a `width` × `height` canvas. */
export function decompileFlux3Prompt(text: string, width: number, height: number): { draft: LayoutDraft; extra: string } {
  check(Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0, "width and height must be positive numbers")
  const { caption, rows, extra } = splitFlux3Prompt(text)
  if (rows === null || rows.length === 0) throw new DraftError("Not a FLUX.3 layout prompt: no element table at the end")
  const items = rows.map((row, index) => {
    check(isBoxRow(row), "Row {{n}} must have an id, a bbox and a desc", { n: index + 1 })
    const [y1, x1, y2, x2] = row.bbox.map((v) => clampGrid(v)) as [number, number, number, number]
    check(y1 < y2 && x1 < x2, "Row {{n}}: the box must have a positive size", { n: index + 1 })
    const bbox: Box = [Math.round((x1 / CANVAS) * width), Math.round((y1 / CANVAS) * height), Math.round((x2 / CANVAS) * width), Math.round((y2 / CANVAS) * height)]
    const quoted = /^Text reading "((?:[^"\\]|\\.)*)",\s*/.exec(row.desc)
    if (/(^|_)Text_\d+$/.test(row.id) || quoted) {
      const wording = quoted ? (quoted[1] ?? "").replace(/\\n/g, "\n") : (/"((?:[^"\\]|\\.)*)"/.exec(row.desc)?.[1] ?? "")
      return { label: "text" as const, caption: quoted ? row.desc.slice(quoted[0].length) || row.desc : row.desc, text: wording, bbox }
    }
    return { label: (/^(person|man|woman|girl|boy|character|child)_\d+$/i.test(row.id) ? "character" : "object") as Label, caption: row.desc, bbox }
  })
  let summary = caption
    .replace(/\s*Elements: [^]*$/, "")
    .replace(/\s*<[A-Za-z][A-Za-z0-9_]*>/g, "")
    .trim()
  let meta: string[] = []
  const style = /\s*Style: ([^.]*)\.\s*$/.exec(summary)
  if (style) {
    meta = (style[1] ?? "")
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean)
    summary = summary.slice(0, style.index).trim()
  }
  check(summary.length > 0, "The caption before the element table is empty")
  return { draft: { width, height, summary: { caption: summary, meta, canvas: "single" }, items }, extra }
}
