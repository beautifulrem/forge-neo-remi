import { describe, expect, it } from "vitest"
import { mergeTyped } from "./persist"

const current = { items: [] as { id: string }[], selected: null as string | null, open: false, size: "m", views: {} as Record<string, string>, set: () => {} }

describe("mergeTyped", () => {
  it("keeps stored fields of the store's types", () => {
    const merged = mergeTyped({ items: [{ id: "a" }], selected: "a", open: true, size: "l", views: { x: "chips" } }, current)
    expect(merged).toMatchObject({ items: [{ id: "a" }], selected: "a", open: true, size: "l", views: { x: "chips" } })
    expect(merged.set).toBe(current.set)
  })

  it("drops fields of another type, unknown fields and empty list entries", () => {
    const merged = mergeTyped({ items: [null, { id: "b" }], selected: 5, open: "yes", size: null, views: [], extra: 1, set: 3 }, current)
    expect(merged).toEqual({ ...current, items: [{ id: "b" }] })
  })

  it("keeps the defaults for a snapshot that is no object", () => {
    for (const stored of [null, undefined, "x", 3, []]) expect(mergeTyped(stored, current)).toBe(current)
  })
})
