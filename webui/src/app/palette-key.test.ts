import { afterEach, describe, expect, it, vi } from "vitest"
import { isMac } from "@/lib/platform"
import { formatShortcut, isPaletteKey } from "./commands"

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

// The notifications region was labelled "Alt+T" on a Mac too (App.tsx), and the canvas and the app
// each had their own Mac test.
describe("shortcut labels", () => {
  afterEach(() => vi.restoreAllMocks())
  const platform = (value: string) => vi.spyOn(navigator, "platform", "get").mockReturnValue(value)

  it("spell the keys of the platform they are shown on, read when shown", () => {
    platform("MacIntel")
    expect(isMac()).toBe(true)
    expect(formatShortcut("Alt+T")).toBe("⌥T")
    expect(formatShortcut("Mod+Shift+Enter")).toBe("⌘⇧Enter")
    platform("iPad")
    expect(isMac()).toBe(true)
    platform("Win32")
    expect(isMac()).toBe(false)
    expect(formatShortcut("Alt+T")).toBe("Alt+T")
    expect(formatShortcut("Mod+K")).toBe("Ctrl+K")
  })
})
