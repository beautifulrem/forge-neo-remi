/**
 * The Source card's frame, shared by the panel (`SourcePanel`, its own chunk) and the stand-in shown
 * while that chunk loads (`SourceFrame.tsx`): the mode grid's tracks and tiles, the hint under them,
 * the image lines and the source blocks the card shows. The stand-in is the same markup without the
 * controls, so it is as tall as the panel at every width and in every mode (the grid's rows and a
 * label's or a hint's wrapping go by the width, the lines and options by the mode); a fixed block
 * cannot be both 149 px (one row of modes) and 204 px (two rows of two-line labels).
 */
import type { CSSProperties } from "react"
import { cn } from "@/lib/utils"
import { evaluate } from "@/ui/conditions"
import { useFormState } from "@/ui/hooks"
import type { Node, TabNode, TabsNode } from "@/ui/schema"
import type { EditorBinding } from "../core/modes"
import { pruneSource } from "./source-layout"

const MODE_PARAM = "mode"

/**
 * The core form's modes (`remi.core.forms.IMG2IMG_MODES`): the chrome label (the tiles are app
 * chrome, translated without the server catalog) and a short description shown under the picker.
 * Unknown modes use the tab title.
 */
export const MODE_TEXT: Record<string, { label: string; description: string }> = {
  img2img: { label: "img2img", description: "Redraws the whole image." },
  Sketch: { label: "Sketch", description: "Paint colours over the image; the result is redrawn." },
  Inpaint: { label: "Inpaint", description: "Paint a mask; only the masked area is redrawn." },
  "Inpaint sketch": { label: "Inpaint sketch", description: "Paint colours; the painted area is redrawn with them." },
  "Inpaint upload": { label: "Inpaint upload", description: "Upload a mask image; its white (or opaque) areas are redrawn." },
  Batch: { label: "Batch", description: "Processes many images with the same settings." },
}

/** The mode picker: the grid over its hint. */
export const MODE_PICKER = "@container flex flex-col gap-2"
/** The line under the picker that says what the mode does. */
export const MODE_HINT = "px-0.5 text-xs text-muted-foreground"

/**
 * The mode grid: one row when there is room, otherwise two balanced rows (3 + 3; 3 + 2 with the
 * short row centred, never an empty cell). Two tracks per mode, so an odd last row can start
 * half a mode in. One row from 22rem for up to five modes (the parameter column is wider than
 * that), from 30rem for more.
 */
export function modeGrid(count: number): { style: CSSProperties; className: string } {
  const perRow = count > 4 ? Math.ceil(count / 2) : count
  return {
    style: { "--cols": perRow * 2, "--cols-wide": count } as CSSProperties,
    className: cn(
      "grid grid-cols-[repeat(var(--cols),minmax(0,1fr))] gap-1 rounded-md bg-muted p-1",
      count <= 5 ? "@[22rem]:grid-cols-[repeat(var(--cols-wide),minmax(0,1fr))]" : "@[30rem]:grid-cols-[repeat(var(--cols-wide),minmax(0,1fr))]",
    ),
  }
}

/** The mode that opens a short second row (odd counts above 4), which starts one track in to centre the row. */
const centredIndex = (count: number) => (count > 4 && count % 2 === 1 ? Math.ceil(count / 2) : -1)

/** A mode tile among `count`, the one at `index` (checked: `data-checked`): `rounded-sm` inside the track's `rounded-md` and `p-1`. */
export function modeTile(count: number, index: number): string {
  return cn(
    "group/mode col-span-2 flex min-w-0 flex-col items-center justify-start gap-1 rounded-sm px-0.5 py-1.5 text-muted-foreground transition-colors outline-none select-none",
    count <= 5 ? "@[22rem]:col-span-1" : "@[30rem]:col-span-1",
    index === centredIndex(count) && (count <= 5 ? "col-start-2 @[22rem]:col-start-auto" : "col-start-2 @[30rem]:col-start-auto"),
    "hover:bg-background/60 hover:text-foreground focus-visible:ring-3 focus-visible:ring-focus-ring",
    "data-checked:bg-card data-checked:text-foreground data-checked:inset-ring data-checked:inset-ring-muted-foreground dark:data-checked:bg-accent",
  )
}

/** A tile's label: the chrome label of a known mode, else the tab's title. */
export const modeLabel = (tab: TabNode) => MODE_TEXT[String(tab.value)]?.label ?? tab.title

/** The modes on offer: the mode tabs whose `visible_if` holds and that set a value. */
export function useModeOptions(tabs: TabsNode): TabNode[] {
  const visible = useFormState((s) => tabs.children.map((tab) => (evaluate(tab.visible_if, "core", (o, n) => s.values[o]?.[n]) ? "1" : "0")).join(""))
  return tabs.children.filter((_, i) => visible[i] === "1" && tabs.children[i]?.value !== undefined)
}

/** The lines of the source images. */
export const IMAGE_LINES = "flex flex-col gap-1"
/** One image line: a 48 px thumbnail, its label over a line or two of text, the actions. */
export const IMAGE_LINE = "flex items-center gap-2.5 rounded-lg border border-transparent p-1 transition-colors"
export const IMAGE_THUMB = "grid size-12 shrink-0 place-items-center overflow-hidden rounded-md border"
export const IMAGE_LABEL = "truncate text-sm font-medium"
export const IMAGE_TEXT = "line-clamp-2 text-xs break-words tabular-nums"
/** The line that says the mode's layer is painted on the stage. */
export const PAINT_HINT = "flex items-center gap-1.5 px-1 text-xs"

/** The source blocks as the card shows them: the mode tabs taken out, the selected mode's options and the rest. */
export function useSourceParts(nodes: readonly Node[], binding: EditorBinding | null) {
  const handled = new Set(binding ? [binding.background, ...binding.layers.map((layer) => layer.param)] : [])
  const found: { tabs: TabsNode | null } = { tabs: null }
  const rest = pruneSource(nodes, handled, MODE_PARAM, found)
  const modeNode = useFormState((s) => s.index.inputs.get("core")?.get(MODE_PARAM) ?? null)
  const mode = useFormState((s) => s.values.core?.[MODE_PARAM])
  const selected = found.tabs?.children.find((tab) => tab.value === mode) ?? null
  return { tabs: found.tabs, modeNode, mode, selected, rest }
}

