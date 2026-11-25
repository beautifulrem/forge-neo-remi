/**
 * A button acting as a toggle (`aria-pressed`) wears the selection tint from one place,
 * `PRESSED_TINT_CLASS` (toggle.tsx; docs/remi-webui/frontend.md). The image editor's tool buttons
 * wrote it by hand without its hover rules: a hovered pressed tool took the ghost button's
 * hover fill instead. (The image viewer's chrome tints white on its black backdrop, not the
 * selection tint.)
 */
import { describe, expect, it } from "vitest"
import { PRESSED_TINT_CLASS } from "./toggle"

const sources = import.meta.glob<string>(["../../**/*.{ts,tsx}", "!../../**/*.test.*"], { query: "?raw", import: "default", eager: true })

describe("the pressed tint", () => {
  it("is written once, in toggle.tsx", () => {
    expect(PRESSED_TINT_CLASS).toMatch(/aria-pressed:hover:bg-primary/)
    const scanned = Object.keys(sources)
    expect(scanned.length).toBeGreaterThan(100) // the scan works
    const byHand = Object.entries(sources)
      .filter(([path, source]) => path !== "./toggle.tsx" && /aria-pressed:bg-primary/.test(source))
      .map(([path]) => path)
    expect(byHand).toEqual([])
  })
})
