import { beforeEach, describe, expect, it, vi } from "vitest"
import { decompilePrompt } from "./decompile"
import { restore, useBuilder } from "./store"
import { COMPILED } from "./testdata"

const initial = useBuilder.getState()

beforeEach(() => useBuilder.setState(initial, true))

describe("builder store", () => {
  it("starts with one empty object item on a 1024 × 1024 canvas", () => {
    const s = useBuilder.getState()
    expect([s.width, s.height, s.canvas, s.committedWidth, s.committedHeight]).toEqual([1024, 1024, "single", 1024, 1024])
    expect(s.items).toEqual([{ id: 1, label: "object", caption: "", text: "", bbox: [102, 102, 409, 409] }])
  })

  it("steps a new item down-right from the last one", () => {
    useBuilder.getState().addItem()
    expect(useBuilder.getState().items.map((item) => item.bbox)).toEqual([
      [102, 102, 409, 409],
      [153, 153, 460, 460],
    ])
  })

  it("adds, edits and removes items; removing the last one leaves an empty item", () => {
    const { addItem, updateItem, removeItem } = useBuilder.getState()
    addItem()
    updateItem(2, { label: "text", text: "hi" })
    expect(useBuilder.getState().items.map((item) => [item.id, item.label])).toEqual([
      [1, "object"],
      [2, "text"],
    ])
    removeItem(1)
    removeItem(2)
    expect(useBuilder.getState().items.map((item) => item.id)).toEqual([3])
    useBuilder.getState().clearItems()
    expect(useBuilder.getState().items.map((item) => item.id)).toEqual([4])
  })

  it("rescales the boxes from the last committed size only when asked", () => {
    useBuilder.getState().setBox(1, [800, 800, 1000, 1000])
    // a typed value previews the size without touching the boxes
    useBuilder.getState().setSize(512, 512)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([800, 800, 1000, 1000])
    expect(useBuilder.getState().committedWidth).toBe(1024)
    // committing rescales from 1024 once — not twice from the preview
    useBuilder.getState().setSize(512, 512, true)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([400, 400, 500, 500])
    expect([useBuilder.getState().committedWidth, useBuilder.getState().committedHeight]).toEqual([512, 512])
    // and committing the old size restores the box
    useBuilder.getState().setSize(1024, 1024, true)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([800, 800, 1000, 1000])
  })

  it("loads a draft with ids after the last one and rescales from its size", () => {
    useBuilder.getState().set({ extra: "<lora:x:1>" })
    useBuilder.getState().load(decompilePrompt(COMPILED, 1024, 768))
    const s = useBuilder.getState()
    expect([s.width, s.height, s.extra, s.lastId]).toEqual([1024, 768, "", 3])
    expect([s.committedWidth, s.committedHeight]).toEqual([1024, 768])
    expect(s.items.map((item) => item.id)).toEqual([2, 3])
    // the rescale baseline is the loaded 1024 × 768, not a stale 1024 × 1024
    s.setSize(512, 384, true)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([50, 100, 251, 350])
  })

  it("persists the committed size, not a preview being typed", () => {
    useBuilder.getState().setBox(1, [800, 800, 1000, 1000])
    useBuilder.getState().setSize(512, 512)
    let stored = JSON.parse(localStorage.getItem("remi-json-builder")!).state
    expect(stored.width).toBe(1024)
    expect(stored.items[0].bbox).toEqual([800, 800, 1000, 1000])
    useBuilder.getState().setSize(512, 512, true)
    stored = JSON.parse(localStorage.getItem("remi-json-builder")!).state
    expect(stored.width).toBe(512)
    expect(stored.items[0].bbox).toEqual([400, 400, 500, 500])
  })

  it("is kept in the browser", () => {
    useBuilder.getState().set({ caption: "kept" })
    expect(JSON.parse(localStorage.getItem("remi-json-builder")!).state).toMatchObject({ caption: "kept", width: 1024 })
  })
})

describe("restore", () => {
  it("keeps only the well-formed parts of a stored draft", () => {
    expect(restore(null)).toEqual({})
    expect(restore({ width: -1, height: 512, caption: 3, canvas: "grid", ar: "7:5", mp: 0, items: "x" })).toEqual({})
    const item = { id: 4, label: "object", caption: "c", text: "", bbox: [0, 0, 16, 16] }
    expect(restore({ width: 512, height: 768, canvas: "collage", ar: "16:9", mp: 2, items: [item, { id: 5, label: "x" }], lastId: 2 })).toEqual({
      width: 512,
      height: 768,
      committedWidth: 512,
      committedHeight: 768,
      canvas: "collage",
      ar: "16:9",
      mp: 2,
      items: [item],
      lastId: 4,
    })
    // duplicate ids would break the list keys: the stored items are dropped
    expect(restore({ items: [item, item] })).toEqual({})
  })

  it("rescales from the restored size, not a stale 1024 one", () => {
    useBuilder.setState(restore({ width: 512, height: 768, items: [{ id: 1, label: "object", caption: "c", text: "", bbox: [10, 20, 110, 60] }] }))
    useBuilder.getState().setSize(1024, 1536, true)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([20, 40, 220, 120])
  })

  it("restores the stored draft when the page loads", async () => {
    // rehydration runs inside create(), before the rest of the module is evaluated
    const stored = { width: 768, height: 512, caption: "kept", items: [{ id: 3, label: "object", caption: "c", text: "", bbox: [10, 20, 110, 60] }], lastId: 3 }
    localStorage.setItem("remi-json-builder", JSON.stringify({ state: stored, version: 1 }))
    try {
      vi.resetModules()
      const fresh = (await import("./store")).useBuilder.getState()
      expect([fresh.width, fresh.height, fresh.caption, fresh.lastId]).toEqual([768, 512, "kept", 3])
      expect(fresh.items.map((item) => item.bbox)).toEqual([[10, 20, 110, 60]])
    } finally {
      localStorage.removeItem("remi-json-builder")
    }
  })
})
