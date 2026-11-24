/**
 * Every persisted store loads from what a browser may hold: corrupt JSON, an
 * unknown (newer) version, and fields of the wrong type. None of them may
 * throw, and each store ends with fields of its own types.
 */
import { afterEach, describe, expect, it } from "vitest"
import { useSession } from "@/features/gallery/session-store"
import { useExtrasRuns } from "@/features/extras/run"
import { useGalleryView } from "@/features/library/filters"
import { useNetworksUi } from "@/features/networks/store"
import { usePromptViews } from "@/features/prompt-tools/view-store"
import { useLocale } from "@/i18n"
import { useTools } from "@/canvas/state/tools"
import { useBuilder } from "@/pages/custom/json-builder/store"
import { APPEARANCE_KEY, useAppearance } from "./theme"

type Hydratable = { persist: { rehydrate: () => Promise<void> | void; getOptions: () => { name?: string } }; getState: () => object }

const stores: [string, Hydratable, Storage][] = [
  ["gallery view", useGalleryView, localStorage],
  ["session results", useSession, sessionStorage],
  ["extras runs", useExtrasRuns, sessionStorage],
  ["prompt views", usePromptViews, localStorage],
  ["networks", useNetworksUi, localStorage],
  ["appearance", useAppearance, localStorage],
  ["locale", useLocale, localStorage],
  ["canvas tools", useTools, localStorage],
  ["json builder", useBuilder, localStorage],
]

/** Each field's kind: a damaged snapshot must leave the same kinds. */
const kinds = (state: object) =>
  Object.fromEntries(
    Object.entries(state)
      .filter(([, value]) => typeof value !== "function")
      .map(([key, value]) => [key, value === null ? "null-or-string" : Array.isArray(value) ? "array" : typeof value]),
  )

const sameKinds = (before: Record<string, string>, after: Record<string, string>) => {
  for (const [key, kind] of Object.entries(before)) {
    const now = after[key]
    if (kind === "null-or-string") expect(["null-or-string", "string"], key).toContain(now)
    else if (now !== "null-or-string") expect(now, key).toBe(kind)
  }
}

describe("persisted stores", () => {
  afterEach(() => {
    localStorage.clear()
    sessionStorage.clear()
  })

  it.each(stores)("%s survives a damaged snapshot", async (_name, store, storage) => {
    const key = store.persist.getOptions().name!
    expect(key === APPEARANCE_KEY || key.startsWith("remi-")).toBe(true)
    const before = kinds(store.getState())
    const damaged = Object.fromEntries(Object.keys(before).map((field) => [field, field.length % 2 ? 7 : null]))
    for (const raw of ["{not json", JSON.stringify({ state: damaged, version: 1 }), JSON.stringify({ state: null, version: 99 }), JSON.stringify([1, 2])]) {
      storage.setItem(key, raw)
      await expect(Promise.resolve(store.persist.rehydrate())).resolves.not.toThrow()
      sameKinds(before, kinds(store.getState()))
    }
  })

  // The kinds were checked one level deep: a damaged surface, result or run crashed its page on
  // every load (`query.trim()`, `key.replace`, `outputs.find`), and "Try again" crashed again.
  it("keeps only the stored entries of their own shape", async () => {
    const item = { key: "job:0", fileId: "f", jobId: "job", tab: "txt2img", index: 0, infotext: null, createdAt: 1 }
    sessionStorage.setItem("remi-session-results", JSON.stringify({ state: { items: [{}, 5, "x", { key: "a:0" }, item], selected: null }, version: 1 }))
    await useSession.persist.rehydrate()
    expect(useSession.getState().items).toEqual([item])

    const run = { jobId: "j", outputs: [{ id: "o", kind: "image" }], source: null, info: null, at: 1 }
    const runs = { results: [{}, { jobId: "j", outputs: null }, 5, { ...run, outputs: [null] }, run], jobs: [5, "k"], sources: { k: 5, j: "s" } }
    sessionStorage.setItem("remi-extras-runs", JSON.stringify({ state: runs, version: 1 }))
    await useExtrasRuns.persist.rehydrate()
    expect(useExtrasRuns.getState()).toMatchObject({ results: [run], jobs: ["k"], sources: { j: "s" } })

    const page = { folder: null, archs: null, compatible: null, query: null, kind: 5 }
    const drawer = { query: "lora", folder: { lora: 5, lyco: "sub" }, archs: { lora: "sdxl", lyco: ["sdxl"] }, compatible: { lora: "yes", lyco: true } }
    localStorage.setItem("remi-networks", JSON.stringify({ state: { surfaces: { page, drawer }, drawerOpen: true }, version: 2 }))
    await useNetworksUi.persist.rehydrate()
    const { surfaces } = useNetworksUi.getState()
    expect(surfaces.page).toMatchObject({ kind: "lora", query: "", folder: {}, archs: {}, compatible: {} })
    expect(surfaces.drawer).toMatchObject({ query: "lora", folder: { lyco: "sub" }, archs: { lyco: ["sdxl"] }, compatible: { lyco: true } })
  })
})
