/**
 * Network browser view state, persisted (`remi-networks`): one set per
 * surface (the Networks page and the drawer in Generate keep their own kind,
 * search, view and filters), filters per kind, the drawer's open state and
 * width, and the folder tree's width.
 */
import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { GenerateTab } from "@/features/progress/run-controller"
import { SORT_KEYS, SORT_ORDERS, type SortKey, type SortOrder } from "./queries"
import { type FieldRules, keepEntries, mergeTyped } from "@/lib/persist"

export type Surface = "page" | "drawer"
export const VIEW_MODES = ["grid", "list"] as const
export type ViewMode = (typeof VIEW_MODES)[number]
export const CARD_SIZES = ["s", "m", "l"] as const
export type CardSize = (typeof CARD_SIZES)[number]
export type PromptTarget = "positive" | "negative"

export type SurfaceState = {
  kind: string
  view: ViewMode
  size: CardSize
  /** Chosen sort (null: the `extra_networks_card_order_field` / `_order` settings). */
  sort: SortKey | null
  order: SortOrder | null
  /** The folder tree pane is shown (wide layouts). */
  tree: boolean
  query: string
  /** Per kind: the selected folder ("" = all). */
  folder: Record<string, string>
  /** Per kind: selected architectures. */
  archs: Record<string, string[]>
  /** Per kind: only items for the current preset (unset: the `lora_preset_filter` setting for LoRA). */
  compatible: Record<string, boolean>
}

type NetworksUiState = {
  surfaces: Record<Surface, SurfaceState>
  drawerOpen: boolean
  /** Drawer width in px (desktop). */
  drawerWidth: number
  /** Folder tree pane width in px (the Networks page). */
  treeWidth: number
  /** The form the Networks page inserts into. */
  pageForm: GenerateTab
  /** Prompt chosen explicitly in the browser (else the last focused prompt editor decides). */
  target: PromptTarget | null
  update: (surface: Surface, patch: Partial<SurfaceState>) => void
  setKindFilter: <K extends "folder" | "archs" | "compatible">(surface: Surface, key: K, kind: string, value: SurfaceState[K][string] | undefined) => void
  setDrawer: (open: boolean) => void
  setDrawerWidth: (width: number) => void
  setTreeWidth: (width: number) => void
  setPageForm: (tab: GenerateTab) => void
  setTarget: (target: PromptTarget | null) => void
}

type PersistedState = Pick<NetworksUiState, "surfaces" | "drawerOpen" | "drawerWidth" | "treeWidth" | "pageForm">

export const DEFAULT_SURFACE: SurfaceState = {
  kind: "lora",
  view: "grid",
  size: "m",
  sort: null,
  order: null,
  tree: true,
  query: "",
  folder: {},
  archs: {},
  compatible: {},
}

export const DRAWER_WIDTH = { min: 360, default: 640 }
export const TREE_WIDTH = { min: 160, max: 480, default: 240 }

/** The values a stored surface's fields take (a card size "xl" crashed the page on every load). */
const SURFACE_RULES: FieldRules<SurfaceState> = { view: VIEW_MODES, size: CARD_SIZES, sort: [...SORT_KEYS, null], order: [...SORT_ORDERS, null] }

const drawerWidth = (width: number) => Math.max(DRAWER_WIDTH.min, Math.round(width))
const treeWidth = (width: number) => Math.min(TREE_WIDTH.max, Math.max(TREE_WIDTH.min, Math.round(width)))

export const useNetworksUi = create<NetworksUiState>()(
  persist(
    (set) => ({
      surfaces: { page: DEFAULT_SURFACE, drawer: { ...DEFAULT_SURFACE, size: "s", tree: false } },
      drawerOpen: false,
      drawerWidth: DRAWER_WIDTH.default,
      treeWidth: TREE_WIDTH.default,
      pageForm: "txt2img",
      target: null,
      update: (surface, patch) => set((s) => ({ surfaces: { ...s.surfaces, [surface]: { ...s.surfaces[surface], ...patch } } })),
      setKindFilter: (surface, key, kind, value) =>
        set((s) => {
          const current = s.surfaces[surface]
          const map = { ...(current[key] as Record<string, unknown>) }
          if (value === undefined) delete map[kind]
          else map[kind] = value
          return { surfaces: { ...s.surfaces, [surface]: { ...current, [key]: map } } }
        }),
      setDrawer: (drawerOpen) => set({ drawerOpen }),
      setDrawerWidth: (width) => set({ drawerWidth: drawerWidth(width) }),
      setTreeWidth: (width) => {
        const next = treeWidth(width)
        set((s) => (s.treeWidth === next ? s : { treeWidth: next }))
      },
      setPageForm: (pageForm) => set({ pageForm }),
      setTarget: (target) => set({ target }),
    }),
    {
      name: "remi-networks",
      version: 2,
      partialize: (s) => ({ surfaces: s.surfaces, drawerOpen: s.drawerOpen, drawerWidth: s.drawerWidth, treeWidth: s.treeWidth, pageForm: s.pageForm }),
      // v1 saved the old default sort ("name", ascending) as if chosen: the settings decide again.
      migrate: (persisted, version) => {
        const saved = (persisted ?? {}) as Partial<NetworksUiState>
        if (version < 2 && saved.surfaces) {
          for (const surface of Object.values(saved.surfaces) as Partial<SurfaceState>[]) {
            if (surface.sort === "name" && surface.order === "asc") Object.assign(surface, { sort: null, order: null })
          }
        }
        // the rest is filled in by `merge`
        return saved as PersistedState
      },
      merge: (persisted, current) => {
        const saved = mergeTyped(persisted, current)
        // a surface's fields keep their kinds too (a stored `query: null` threw on every render)
        const surface = (name: Surface): SurfaceState => {
          const merged = mergeTyped(saved.surfaces[name], current.surfaces[name], SURFACE_RULES)
          return {
            ...merged,
            folder: keepEntries(merged.folder, (value) => typeof value === "string"),
            archs: keepEntries(merged.archs, (value): value is string[] => Array.isArray(value) && value.every((arch) => typeof arch === "string")),
            compatible: keepEntries(merged.compatible, (value) => typeof value === "boolean"),
          }
        }
        return {
          ...current,
          ...saved,
          surfaces: { page: surface("page"), drawer: surface("drawer") },
          drawerWidth: drawerWidth(saved.drawerWidth),
          treeWidth: treeWidth(saved.treeWidth),
          pageForm: saved.pageForm === "img2img" ? "img2img" : "txt2img",
        }
      },
    },
  ),
)

/** One surface's state. */
export const useSurface = (surface: Surface) => useNetworksUi((s) => s.surfaces[surface])
