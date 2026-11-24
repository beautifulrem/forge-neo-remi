import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

// The tab showed a purple-gradient "R" from before the app had its mark: the favicon is the rail's
// logo, drawn from the same shapes.
const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8")

describe("the favicon", () => {
  const favicon = read("../../public/favicon.svg")
  const rail = read("./NavRail.tsx")

  it("is the app's mark", () => {
    const holes = [...rail.matchAll(/<(?:rect|circle) ([^>]*?) fill="black" \/>/g)].map((match) => match[1]!)
    const stroke = rail.match(/<path d="([^"]+)"/)![1]!
    const sheet = rail.match(/<rect (x="3" y="2"[^>]*?) fill="currentColor"/)![1]!
    expect(holes).toHaveLength(3)
    for (const shape of [...holes, `d="${stroke}"`, sheet]) expect(favicon).toContain(shape)
  })

  // a tab strip is light or dark whatever the page's theme: the mark brings its own tile
  it("sits on a tile of its own, without a gradient", () => {
    expect(favicon).toMatch(/<rect width="32" height="32" rx="[\d.]+" fill="#[0-9a-f]{6}"/)
    expect(favicon).not.toMatch(/Gradient/)
  })
})
