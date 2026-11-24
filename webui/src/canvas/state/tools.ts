/**
 * Tool settings shared by every editor (brush size, colours, …), persisted
 * per browser like an image editor's preferences (`remi-canvas-tools`).
 */
import { create } from "zustand"
import { createJSONStorage, persist } from "zustand/middleware"
import { isRecord, mergeTyped } from "@/lib/persist"

export const TOOLS = ["brush", "eraser", "fill", "rect", "lasso", "eyedropper", "hand"] as const
export type Tool = (typeof TOOLS)[number]

export type BrushSettings = {
  /** Diameter in px (image px, or screen px with `forge_canvas_consistent_brush`). */
  size: number
  /** 0–1. */
  opacity: number
  /** 0 = hard edge, 1 = all falloff. */
  softness: number
}

export type ToolState = {
  tool: Tool
  brush: BrushSettings
  eraser: BrushSettings
  /** Sketch colour, `#rrggbb`. */
  color: string
  /** Recently used sketch colours, newest first. */
  recent: string[]
  fillTolerance: number
  /** Bucket fill looks at the layer alone or at the picture as shown. */
  fillSample: "layer" | "image"
  /** Mask display colour and the mask layer's opacity on screen. */
  maskColor: string
  maskOpacity: number
  /** Sketch layer opacity on screen (the export keeps its real alpha). */
  sketchOpacity: number
  layerVisible: boolean
  imageVisible: boolean
  /** Pen pressure scales the brush. */
  pressure: boolean
  /** What the mouse wheel does (pinch and Ctrl+wheel always zoom). */
  wheel: "zoom" | "pan"
  /** Keep the mask blur and the "Only masked" area previews on (they otherwise show while their settings change). */
  previewBlur: boolean
  previewArea: boolean
}

export type ToolActions = {
  setTool: (tool: Tool) => void
  setBrush: (patch: Partial<BrushSettings>) => void
  setEraser: (patch: Partial<BrushSettings>) => void
  setColor: (color: string) => void
  /** Remembers a colour in the recent list (after a stroke used it). */
  rememberColor: (color: string) => void
  set: (patch: Partial<Omit<ToolState, "brush" | "eraser">>) => void
}

export const MIN_SIZE = 1
export const MAX_SIZE = 1024
export const MASK_COLORS = ["#ff3b6b", "#ffffff", "#000000", "#2f80ff", "#22c55e", "#facc15"] as const
/** Accessible names of the mask colours (English chrome keys). */
export const MASK_COLOR_NAMES: Record<(typeof MASK_COLORS)[number], string> = {
  "#ff3b6b": "Pink",
  "#ffffff": "White",
  "#000000": "Black",
  "#2f80ff": "Blue",
  "#22c55e": "Green",
  "#facc15": "Yellow",
}
export const SWATCHES = ["#000000", "#ffffff", "#7f7f7f", "#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#00c7be", "#007aff", "#5856d6", "#af52de", "#ff2d55"] as const

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/** Brush size ↔ slider position 0–100 (log scale: fine control of small brushes). */
export const sizeToSlider = (size: number) => (Math.log(clamp(size, MIN_SIZE, MAX_SIZE)) / Math.log(MAX_SIZE)) * 100
export const sliderToSize = (value: number) => Math.round(clamp(MAX_SIZE ** (value / 100), MIN_SIZE, MAX_SIZE))

/** Size steps of `[` / `]` (and W + wheel): about 12 % per step, at least 1 px. */
export function stepSize(size: number, direction: 1 | -1): number {
  const next = direction > 0 ? Math.max(size + 1, size * 1.12) : Math.min(size - 1, size / 1.12)
  return Math.round(clamp(next, MIN_SIZE, MAX_SIZE))
}

const DEFAULTS: ToolState = {
  tool: "brush",
  brush: { size: 48, opacity: 1, softness: 0 },
  eraser: { size: 48, opacity: 1, softness: 0 },
  color: "#ff3b30",
  recent: [],
  fillTolerance: 32,
  fillSample: "layer",
  maskColor: MASK_COLORS[0],
  maskOpacity: 0.55,
  sketchOpacity: 1,
  layerVisible: true,
  imageVisible: true,
  pressure: true,
  wheel: "zoom",
  previewBlur: false,
  previewArea: false,
}

const fixBrush = (brush: BrushSettings): BrushSettings => ({
  size: clamp(Math.round(brush.size), MIN_SIZE, MAX_SIZE),
  opacity: clamp(brush.opacity, 0.01, 1),
  softness: clamp(brush.softness, 0, 1),
})

/** A colour as the colour inputs give it, `#rrggbb`. */
const isHex = (value: unknown): value is string => typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value)
const finite = (value: unknown, fallback: number) => (typeof value === "number" && Number.isFinite(value) ? value : fallback)

/** A stored brush: each setting a finite number in its range (a string made the size NaN). */
function storedBrush(value: unknown, fallback: BrushSettings): BrushSettings {
  const brush = isRecord(value) ? value : {}
  return fixBrush({ size: finite(brush.size, fallback.size), opacity: finite(brush.opacity, fallback.opacity), softness: finite(brush.softness, fallback.softness) })
}

/** The values the stored choices take: an unknown tool left the canvas dead. */
const RULES = { tool: TOOLS, fillSample: ["layer", "image"], wheel: ["zoom", "pan"], maskColor: MASK_COLORS, color: isHex } as const

export const useTools = create<ToolState & ToolActions>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      setTool: (tool) => set({ tool }),
      setBrush: (patch) => set((s) => ({ brush: fixBrush({ ...s.brush, ...patch }) })),
      setEraser: (patch) => set((s) => ({ eraser: fixBrush({ ...s.eraser, ...patch }) })),
      setColor: (color) => set({ color: color.toLowerCase() }),
      rememberColor: (color) => set((s) => ({ recent: [color.toLowerCase(), ...s.recent.filter((c) => c !== color.toLowerCase())].slice(0, 8) })),
      set: (patch) => set(patch),
    }),
    {
      name: "remi-canvas-tools",
      version: 1,
      storage: createJSONStorage(() => localStorage),
      // Older or hand-edited values fall back to the defaults key by key.
      merge: (persisted, current) => {
        const p = mergeTyped<ToolState & ToolActions>(persisted, current, RULES)
        return {
          ...p,
          brush: storedBrush(p.brush, DEFAULTS.brush),
          eraser: storedBrush(p.eraser, DEFAULTS.eraser),
          recent: p.recent.filter(isHex).slice(0, 8),
          fillTolerance: clamp(p.fillTolerance, 0, 255),
          maskOpacity: clamp(p.maskOpacity, 0.05, 1),
          sketchOpacity: clamp(p.sketchOpacity, 0.05, 1),
        }
      },
    },
  ),
)

/** Tools that need a paintable layer. */
export const PAINT_TOOLS: ReadonlySet<Tool> = new Set(["brush", "eraser", "fill", "rect", "lasso"])
