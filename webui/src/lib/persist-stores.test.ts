/**
 * Every persisted store loads from what a browser may hold: corrupt JSON, an
 * unknown (newer) version, and fields of the wrong type. None of them may
 * throw, and each store ends with fields of its own types.
 */
import { afterEach, describe, expect, it } from "vitest"
import { useSession } from "@/features/gallery/session-store"
import { useExtrasRuns } from "@/features/extras/run"
import { useQueueEdit } from "@/features/queue/edit-session"
import { useGalleryView } from "@/features/library/filters"
import { DRAWER_WIDTH, TREE_WIDTH, useNetworksUi } from "@/features/networks/store"
import { usePromptViews } from "@/features/prompt-tools/view-store"
import { useLocale } from "@/i18n"
import { useTools } from "@/canvas/state/tools"
import { useBuilder } from "@/pages/custom/json-builder/store"
import { APPEARANCE_KEY, useAppearance } from "./theme"

type Hydratable = { persist: { rehydrate: () => Promise<void> | void; getOptions: () => { name?: string } }; getState: () => object }

/** Each persisted store: its name, the store, where it keeps its state and its module (under src/). */
const stores: [string, Hydratable, Storage, string][] = [
  ["gallery view", useGalleryView, localStorage, "features/library/filters.ts"],
  ["session results", useSession, sessionStorage, "features/gallery/session-store.ts"],
  ["extras runs", useExtrasRuns, sessionStorage, "features/extras/run.ts"],
  ["queue edits", useQueueEdit, sessionStorage, "features/queue/edit-session.ts"],
  ["prompt views", usePromptViews, localStorage, "features/prompt-tools/view-store.ts"],
  ["networks", useNetworksUi, localStorage, "features/networks/store.ts"],
  ["appearance", useAppearance, localStorage, "lib/theme.ts"],
  ["locale", useLocale, localStorage, "i18n/index.ts"],
  ["canvas tools", useTools, localStorage, "canvas/state/tools.ts"],
  ["json builder", useBuilder, localStorage, "pages/custom/json-builder/store.ts"],
]

/** Modules that persist a store tested elsewhere, and where. */
const TESTED_ELSEWHERE: Record<string, string> = {
  // one store per form, created on demand (no module-level store to rehydrate here)
  "ui/form-store.ts": "form drafts: form-store.test.ts loads the defaults from damaged drafts",
}

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

  // the table was kept by hand: a store added later loaded damaged snapshots untested
  it("are all in the table, or tested elsewhere", async () => {
    const { readFileSync, readdirSync } = await import("node:fs")
    const { join, sep } = await import("node:path")
    const files = (readdirSync("src", { recursive: true }) as string[]).filter((file) => /\.tsx?$/.test(file) && !/\.test\.tsx?$/.test(file))
    // keyed on the import: a module may have a function of its own named `persist`
    const persisting = files
      .filter((file) => /import\s*\{[^}]*\bpersist\b[^}]*\}\s*from\s*"zustand\/middleware"/.test(readFileSync(join("src", file), "utf8")))
      .map((file) => file.split(sep).join("/"))
      .sort()
    expect(persisting.length).toBeGreaterThan(5) // the scan works
    const listed = [...stores.map((row) => row[3]), ...Object.keys(TESTED_ELSEWHERE)].sort()
    expect(persisting).toEqual(listed)
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

  // Values of the right type that a field does not take (an older build's choice, a hand edit, a
  // number out of range): a card size "xl" crashed the Networks page on every load, and Generate
  // with it while the drawer was kept open; an unknown tool left the canvas dead.
  const outOfRange: { name: string; store: Hydratable; stored: string; expected: Record<string, unknown> }[] = [
    {
      name: "networks",
      store: useNetworksUi,
      stored: '{"state":{"surfaces":{"page":{"size":"xl","view":"tiles","sort":"rating","order":"up"},"drawer":{"size":"xl","sort":"name","order":"desc"}},"drawerWidth":1e999,"treeWidth":5},"version":2}',
      expected: {
        surfaces: { page: expect.objectContaining({ size: "m", view: "grid", sort: null, order: null }), drawer: expect.objectContaining({ size: "s", sort: "name", order: "desc" }) },
        drawerWidth: DRAWER_WIDTH.default,
        treeWidth: TREE_WIDTH.min,
      },
    },
    {
      name: "gallery view",
      store: useGalleryView,
      stored: '{"state":{"category":"videos","sort":"random","layout":"grid","size":"xl","starred":true},"version":1}',
      expected: { category: null, sort: "newest", layout: "justified", size: "m", starred: true },
    },
    {
      name: "appearance",
      store: useAppearance,
      stored: '{"state":{"theme":"light","themeChosen":true,"density":"cozy","accent":"purple"},"version":3}',
      expected: { theme: "light", density: "comfortable", accent: "blue" },
    },
    {
      name: "canvas tools",
      store: useTools,
      stored:
        '{"state":{"tool":"spray","fillSample":"all","wheel":"scroll","maskColor":"red","color":"blue","recent":["#00ff00","nope"],"fillTolerance":900,"maskOpacity":7,"sketchOpacity":-1,"brush":{"size":"big","opacity":null,"softness":3},"eraser":{"size":1e999}},"version":1}',
      expected: {
        tool: "brush",
        fillSample: "layer",
        wheel: "zoom",
        maskColor: "#ff3b6b",
        color: "#ff3b30",
        recent: ["#00ff00"],
        fillTolerance: 255,
        maskOpacity: 1,
        sketchOpacity: 0.05,
        brush: { size: 48, opacity: 1, softness: 1 },
        eraser: { size: 48, opacity: 1, softness: 0 },
      },
    },
    {
      name: "prompt views",
      store: usePromptViews,
      stored: '{"state":{"views":{"positive:Prompt":"tags","negative:Negative prompt":"chips","x":5}},"version":1}',
      expected: { views: { "negative:Negative prompt": "chips" } },
    },
    {
      name: "json builder",
      store: useBuilder,
      stored: '{"state":{"canvas":"triple","format":"xml","ar":"7:3","mp":-1,"width":1e999,"height":512},"version":1}',
      expected: { canvas: "single", format: "moss", ar: "1:1", mp: 1, width: 1024, height: 1024 },
    },
  ]

  it.each(outOfRange)("$name keeps only the values its fields take", async ({ store, stored, expected }) => {
    const full = store as Hydratable & { getInitialState: () => object; setState: (state: object, replace: true) => void }
    full.setState(full.getInitialState(), true)
    localStorage.setItem(store.persist.getOptions().name!, stored)
    await store.persist.rehydrate()
    const state = store.getState() as Record<string, unknown>
    for (const [key, value] of Object.entries(expected)) expect(state[key], key).toEqual(value)
  })
})
