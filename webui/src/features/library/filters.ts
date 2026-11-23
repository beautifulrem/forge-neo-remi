/**
 * The gallery's view state: filters, sort, layout and tile size, persisted
 * per browser (`remi-gallery-view`), and the `GET /gallery` query built from it.
 */
import { create } from "zustand"
import { createJSONStorage, persist } from "zustand/middleware"
import type { GalleryCategory, GalleryQuery, GallerySort } from "@/api/types"
import type { TileSize } from "./layout"
import { mergeTyped } from "@/lib/persist"

export type GalleryLayout = "justified" | "masonry"

export type GalleryFilters = {
  q: string
  category: GalleryCategory | null
  model: string | null
  boardId?: string | null
  /** Local dates `YYYY-MM-DD` (both inclusive). */
  from: string | null
  to: string | null
  starred: boolean
  sort: GallerySort
}

export type ExtendedGalleryQuery = GalleryQuery & { board_id?: string | null }

export type GalleryView = Omit<GalleryFilters, "boardId"> & {
  boardId: string | null
  layout: GalleryLayout
  size: TileSize
  set: (patch: Partial<Omit<GalleryView, "set" | "reset">>) => void
  /** Clears the filters (layout, size and sort stay). */
  reset: () => void
}

export const NO_FILTERS: Omit<GalleryFilters, "sort"> & { boardId: string | null } = {
  q: "",
  category: null,
  model: null,
  boardId: null,
  from: null,
  to: null,
  starred: false,
}

export const useGalleryView = create<GalleryView>()(
  persist(
    (set) => ({
      ...NO_FILTERS,
      sort: "newest",
      layout: "justified",
      size: "m",
      set: (patch) => set(patch),
      reset: () => set(NO_FILTERS),
    }),
    {
      name: "remi-gallery-view",
      version: 1,
      storage: createJSONStorage(() => localStorage),
      // The search text is per visit; everything else is kept.
      partialize: ({ category, model, boardId, from, to, starred, sort, layout, size }) => ({ category, model, boardId, from, to, starred, sort, layout, size }),
      merge: mergeTyped,
    },
  ),
)

const DATE = /^(\d{4})-(\d{2})-(\d{2})$/

/** Local midnight of `YYYY-MM-DD` (plus `days`) as an ISO instant; null for anything else. */
export function localDayStart(date: string | null, days = 0): string | null {
  const match = date ? DATE.exec(date) : null
  if (!match) return null
  const day = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]) + days)
  return Number.isNaN(day.getTime()) ? null : day.toISOString()
}

/** The `GET /gallery` query for the filters (empty ones left out; `to` is inclusive). */
export function galleryQuery(filters: GalleryFilters): ExtendedGalleryQuery {
  const query: ExtendedGalleryQuery = { sort: filters.sort }
  const q = filters.q.trim()
  if (q) query.q = q
  if (filters.category) query.category = filters.category
  if (filters.model) query.model = filters.model
  if (filters.boardId) query.board_id = filters.boardId
  const since = localDayStart(filters.from)
  const until = localDayStart(filters.to, 1)
  if (since) query.since = since
  if (until) query.until = until
  if (filters.starred) query.starred = true
  return query
}

/** How many filters narrow the view (for the "clear filters" affordance). */
export function activeFilters(filters: GalleryFilters): number {
  return [filters.q.trim(), filters.category, filters.model, filters.boardId, filters.from || filters.to, filters.starred].filter(Boolean).length
}
