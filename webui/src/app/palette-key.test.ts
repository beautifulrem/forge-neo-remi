import { describe, expect, it } from "vitest"
import { isPaletteKey } from "./commands"

const key = (init: Partial<KeyboardEvent>) => ({ key: "k", metaKey: false, ctrlKey: false, altKey: false, defaultPrevented: false, ...init })

describe("the command palette shortcut", () => {
  it("is ⌘K on macOS and leaves Ctrl+K to text fields", () => {
    expect(isPaletteKey(key({ metaKey: true }), true)).toBe(true)
    expect(isPaletteKey(key({ ctrlKey: true }), true)).toBe(false)
  })

  it("is Ctrl+K elsewhere", () => {
    expect(isPaletteKey(key({ ctrlKey: true, key: "K" }), false)).toBe(true)
    expect(isPaletteKey(key({ metaKey: true }), false)).toBe(false)
  })

  it("leaves a key a field already handled, and other keys, alone", () => {
    expect(isPaletteKey(key({ metaKey: true, defaultPrevented: true }), true)).toBe(false)
    expect(isPaletteKey(key({ metaKey: true, altKey: true }), true)).toBe(false)
    expect(isPaletteKey(key({ metaKey: true, key: "j" }), true)).toBe(false)
  })
})
