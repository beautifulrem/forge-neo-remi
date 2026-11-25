import { toast } from "sonner"
import { describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { createFormStore } from "@/ui/form-store"
import type { TabSchema } from "@/ui/schema"
import { loadSchema } from "./use-form-loader"

/** The img2img mock schema with other core defaults (as after a UI preset switch). */
function withDefaults(schema: TabSchema, defaults: Record<string, unknown>): TabSchema {
  return { ...schema, core: { ...schema.core, defaults: { ...schema.core.defaults, ...defaults } as TabSchema["core"]["defaults"] } }
}

describe("loadSchema", () => {
  it("keeps uploads and lets untouched params follow new defaults on a reload", () => {
    const store = createFormStore("loader-test", { persistDrafts: false })
    const schema = tabSchema("img2img")
    loadSchema(store, schema)
    store.getState().setValues("core", { init_img: "upload-1", cfg_scale: 9 })

    loadSchema(store, withDefaults(tabSchema("img2img"), { width: 512, height: 768, cfg_scale: 7 }))
    const core = store.getState().values.core
    expect(core?.init_img).toBe("upload-1") // `load` alone drops upload ids
    expect(core?.width).toBe(512) // untouched: follows the new default
    expect(core?.height).toBe(768)
    expect(core?.cfg_scale).toBe(9) // edited by the user: kept
  })

  // a new default nobody asked for: a lock keeps the value without a word
  it("carries defaults over silently, a locked param included", async () => {
    const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
    const keep = { id: "keep", title: "Keep size", kind: "alwayson" as const, layout: { type: "toggle" as const, name: "size", label: "Keep size", locks: "core.width" }, defaults: { size: true } }
    const store = createFormStore("loader-locked", { persistDrafts: false })
    const schema = tabSchema("img2img")
    loadSchema(store, { ...schema, scripts: [...schema.scripts, keep] })
    const width = store.getState().values.core?.width
    loadSchema(store, withDefaults({ ...schema, scripts: [...schema.scripts, keep] }, { width: 512, height: 768 }))
    await Promise.resolve()
    expect(store.getState().values.core).toMatchObject({ width, height: 768 })
    expect(info).not.toHaveBeenCalled()
    vi.restoreAllMocks()
  })

  it("does not carry over values on the first load", () => {
    const store = createFormStore("loader-first", { persistDrafts: false })
    loadSchema(store, tabSchema("img2img"))
    expect(store.getState().values.core?.init_img).toBeNull()
  })
})
