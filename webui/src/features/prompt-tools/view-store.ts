/** Per-prompt view choice (text editor or tag chips), remembered across reloads. */
import { create } from "zustand"
import { persist } from "zustand/middleware"
import { keepEntries, mergeTyped } from "@/lib/persist"

export type PromptView = "text" | "chips"

const isView = (value: unknown): value is PromptView => value === "text" || value === "chips"

type ViewState = {
  views: Record<string, PromptView>
  setView: (key: string, view: PromptView) => void
}

export const usePromptViews = create<ViewState>()(
  persist(
    (set) => ({
      views: {},
      setView: (key, view) => set((state) => ({ views: { ...state.views, [key]: view } })),
    }),
    {
      name: "remi-prompt-tools",
      version: 1,
      partialize: ({ views }) => ({ views }),
      merge: (persisted, current) => {
        const merged = mergeTyped(persisted, current)
        return { ...merged, views: keepEntries(merged.views, isView) }
      },
    },
  ),
)

/** The key of a prompt field: its kind and label (stable across reloads, unlike element ids). */
export function viewKey(kind: string, label: string): string {
  return `${kind}:${label}`
}
