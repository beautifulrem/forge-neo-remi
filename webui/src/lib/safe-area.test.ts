/**
 * Every popup Base UI positions keeps the safe area (docs/remi-webui/frontend.md, "Safe area"): its
 * `collisionPadding` is Base UI's 5 px from the edges of the safe area, not from the screen's. The
 * rule was missed once by a menu that uses Base UI directly instead of the shadcn wrappers.
 */
import { describe, expect, it } from "vitest"

const sources = import.meta.glob<string>(["../**/*.tsx", "!../**/*.test.*"], { query: "?raw", import: "default", eager: true })

/** The opening `<X.Positioner ...>` tags of `source` (multi-line tags included). */
function positioners(source: string): string[] {
  return [...source.matchAll(/<[A-Za-z]+\.Positioner\b[^>]*>/g)].map((match) => match[0])
}

describe("positioned popups", () => {
  it("keep their collision padding inside the safe area", () => {
    const found = Object.entries(sources).flatMap(([file, source]) => positioners(source).map((tag) => ({ file: file.replace(/^\.\.\//, ""), tag })))
    expect(found.length).toBeGreaterThanOrEqual(9)
    expect(found.filter(({ tag }) => !/collisionPadding=\{collisionPadding\}/.test(tag)).map(({ file }) => file)).toEqual([])
  })
})
