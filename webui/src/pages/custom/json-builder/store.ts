/**
 * The JSON Builder's draft, kept across visits and reloads (upstream's lived
 * in the tab's DOM for the session). One draft per browser.
 */
import { create } from "zustand"
import { createJSONStorage, persist } from "zustand/middleware"
import { CANVAS_TYPES, LABELS } from "./compile"
import type { Box, LayoutDraft } from "./decompile"
import { ASPECT_RATIOS, type AspectRatio, type BuilderDraft, fromDraft, type Item, newItem, rescaleBox } from "./draft"
import { PROMPT_FORMATS, type PromptFormat } from "./flux3"
import { DEFAULT_VGL, type VglFields } from "./vgl"

export type BuilderState = Omit<BuilderDraft, "format" | "vgl"> & {
  /** The model the prompt is compiled for. */
  format: PromptFormat
  /** Bria Fibo 2.0 VGL 2.0 structured prompt settings. */
  vgl: VglFields
  /** The size the boxes are laid out for (the last committed size; `width`/`height` can be a preview being typed). */
  committedWidth: number
  committedHeight: number
  /** Bumped by `load` so a field editing for the old draft drops its work. */
  draftRevision: number
  /** The resolution tool's megapixels and aspect ratio. */
  mp: number
  ar: AspectRatio
  /** The last item id handed out. */
  lastId: number
  set: (patch: Partial<Pick<BuilderState, "caption" | "meta" | "extra" | "canvas" | "mp" | "ar" | "format">>) => void
  setVgl: (patch: Partial<VglFields>) => void
  /** A new canvas size; when `rescale`, the boxes are scaled from the last committed size into it. */
  setSize: (width: number, height: number, rescale?: boolean) => void
  addItem: () => void
  updateItem: (id: number, patch: Partial<Omit<Item, "id">>) => void
  setBox: (id: number, bbox: Box) => void
  removeItem: (id: number) => void
  /** One empty item left. */
  clearItems: () => void
  load: (draft: LayoutDraft) => void
}

const INITIAL = { width: 1024, height: 1024, committedWidth: 1024, committedHeight: 1024, draftRevision: 0, caption: "", meta: "", extra: "", canvas: "single", mp: 1, ar: "1:1", format: "moss" } as const

export const useBuilder = create<BuilderState>()(
  persist(
    (set) => ({
      ...INITIAL,
      vgl: { ...DEFAULT_VGL },
      items: [newItem(1, INITIAL.width, INITIAL.height)],
      lastId: 1,
      set: (patch) => set(patch),
      setVgl: (patch) => set((s) => ({ vgl: { ...s.vgl, ...patch } })),
      setSize: (width, height, rescale = false) =>
        set((s) =>
          rescale
            ? { width, height, committedWidth: width, committedHeight: height, items: s.items.map((item) => ({ ...item, bbox: rescaleBox(item.bbox, s.committedWidth, s.committedHeight, width, height) })) }
            : { width, height },
        ),
      addItem: () => set((s) => ({ items: [...s.items, newItem(s.lastId + 1, s.width, s.height, s.items.length)], lastId: s.lastId + 1 })),
      updateItem: (id, patch) => set((s) => ({ items: s.items.map((item) => (item.id === id ? { ...item, ...patch } : item)) })),
      setBox: (id, bbox) => set((s) => ({ items: s.items.map((item) => (item.id === id ? { ...item, bbox } : item)) })),
      removeItem: (id) =>
        set((s) => {
          const items = s.items.filter((item) => item.id !== id)
          return items.length > 0 ? { items } : { items: [newItem(s.lastId + 1, s.width, s.height)], lastId: s.lastId + 1 }
        }),
      clearItems: () => set((s) => ({ items: [newItem(s.lastId + 1, s.width, s.height)], lastId: s.lastId + 1 })),
      load: (draft) => set((s) => ({ ...fromDraft(draft, s.lastId), committedWidth: draft.width, committedHeight: draft.height, draftRevision: s.draftRevision + 1 })),
    }),
    {
      name: "remi-json-builder",
      version: 1,
      storage: createJSONStorage(() => localStorage),
      // width/height persist the committed size: a preview being typed would otherwise reload boxes scaled for a size they don't have.
      partialize: ({ committedWidth, committedHeight, caption, meta, extra, canvas, items, mp, ar, lastId, format, vgl }) => ({ width: committedWidth, height: committedHeight, caption, meta, extra, canvas, items, mp, ar, lastId, format, vgl }),
      // A stored draft from another build or edited by hand: keep only what is well formed.
      merge: (persisted, current) => ({ ...current, ...restore(persisted) }),
    },
  ),
)

// Function declarations, not consts: `persist` rehydrates inside `create()` above, before a const below it exists.
function positive(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v > 0
}

function str(v: unknown): v is string {
  return typeof v === "string"
}

/** An item id that still counts up: past 2^53 (1e300 is an "integer") `id + 1` is `id` again. */
function itemId(v: unknown): v is number {
  return Number.isSafeInteger(v) && (v as number) >= 0 && (v as number) <= 1e9
}

function restoreItem(value: unknown): Item | null {
  if (!value || typeof value !== "object") return null
  const item = value as Record<string, unknown>
  const box = item.bbox
  if (!itemId(item.id) || !Array.isArray(box) || box.length !== 4 || !box.every((v) => typeof v === "number" && Number.isFinite(v))) return null
  if (!(LABELS as readonly unknown[]).includes(item.label) || !str(item.caption) || !str(item.text)) return null
  return { id: item.id as number, label: item.label as Item["label"], caption: item.caption, text: item.text, bbox: box.slice() as Box }
}

/** The well-formed parts of a stored draft. */
export function restore(persisted: unknown): Partial<BuilderState> {
  if (!persisted || typeof persisted !== "object") return {}
  const p = persisted as Record<string, unknown>
  const out: Partial<BuilderState> = {}
  if (positive(p.width) && positive(p.height)) {
    out.width = p.width
    out.height = p.height
    out.committedWidth = p.width
    out.committedHeight = p.height
  }
  if (str(p.caption)) out.caption = p.caption
  if (str(p.meta)) out.meta = p.meta
  if (str(p.extra)) out.extra = p.extra
  if ((CANVAS_TYPES as readonly unknown[]).includes(p.canvas)) out.canvas = p.canvas as BuilderState["canvas"]
  if ((PROMPT_FORMATS as readonly unknown[]).includes(p.format)) out.format = p.format as PromptFormat
  if (p.vgl && typeof p.vgl === "object") {
    const rawVgl = p.vgl as Record<string, unknown>
    const restored: VglFields = { ...DEFAULT_VGL }
    for (const key of Object.keys(DEFAULT_VGL) as (keyof VglFields)[]) {
      if (str(rawVgl[key])) restored[key] = rawVgl[key]
    }
    out.vgl = restored
  }
  if (positive(p.mp)) out.mp = p.mp
  if ((ASPECT_RATIOS as readonly unknown[]).includes(p.ar)) out.ar = p.ar as AspectRatio
  const items = Array.isArray(p.items) ? p.items.map(restoreItem).filter((item): item is Item => item !== null) : []
  if (items.length > 0 && new Set(items.map((item) => item.id)).size === items.length) {
    out.items = items
    out.lastId = Math.max(...items.map((item) => item.id), itemId(p.lastId) ? p.lastId : 0)
  }
  return out
}
