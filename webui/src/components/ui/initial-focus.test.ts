/**
 * A field in a dialog, popover or sheet takes the focus through the popup's
 * `initialFocus`, never `autoFocus`: focused as it mounted, the field became
 * the place the popup gave the focus back to on close, and the focus was left
 * on the page (JSON Builder's Load prompt, Capture recipe).
 */
import { describe, expect, it } from "vitest"

const sources = import.meta.glob<string>(["../../**/*.tsx", "!../../**/*.test.*"], { query: "?raw", import: "default", eager: true })

describe("popup fields", () => {
  it("take the focus through initialFocus", () => {
    const popups = Object.entries(sources).filter(([, source]) => /<(?:AlertDialog|Dialog|Popover|Sheet)Content\b/.test(source))
    expect(popups.length).toBeGreaterThan(10) // the scan works
    const autofocused = popups.filter(([, source]) => /(?<![`\w])autoFocus(?=[\s=/>])/.test(source)).map(([path]) => path)
    expect(autofocused).toEqual([])
  })
})
