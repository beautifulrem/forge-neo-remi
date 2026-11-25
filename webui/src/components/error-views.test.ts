/**
 * Error views follow three patterns (docs/remi-webui/frontend.md, "Error views"): a list or panel
 * that could not load is `LoadFailed` (what failed, then Retry), a page that could not load is
 * `PagePlaceholder`, and an action's error is a plain destructive line by the action. Hand-written
 * copies had drifted: a smaller Retry in one list, a muted one in another, tinted boxes for some
 * action errors and plain lines for others.
 */
import { describe, expect, it } from "vitest"

const sources = import.meta.glob<string>(["../**/*.tsx", "!../**/*.test.*"], { query: "?raw", import: "default", eager: true })

type Alert = { file: string; tag: string; body: string }

/** Every element announced as an alert: its opening tag and what it holds. */
function alerts(file: string, source: string): Alert[] {
  const found: Alert[] = []
  for (const match of source.matchAll(/role="alert"/g)) {
    const start = source.lastIndexOf("<", match.index)
    const end = source.indexOf(">", match.index) + 1
    const name = /^<([\w.]+)/.exec(source.slice(start))?.[1]
    if (!name) continue
    let depth = 1
    let at = end
    const tags = new RegExp(`<(/?)${name.replace(".", "\\.")}\\b[^>]*?(/?)>`, "g")
    tags.lastIndex = end
    for (let tag = tags.exec(source); tag && depth > 0; tag = tags.exec(source)) {
      if (tag[1]) depth--
      else if (!tag[2]) depth++
      at = tag.index
    }
    found.push({ file: file.replace(/^\.\//, "components/").replace(/^\.\.\//, ""), tag: source.slice(start, end), body: source.slice(end, at) })
  }
  return found
}

const all = Object.entries(sources).flatMap(([file, source]) => alerts(file, source))

describe("error views", () => {
  it("are found", () => {
    expect(all.length).toBeGreaterThanOrEqual(20)
  })

  it("say an action's error in a plain line, not a tinted box", () => {
    // ErrorBoundary's inline fallback stands in for a control that failed to render: the tint marks its place
    const tinted = all.filter(({ file, tag }) => /\bbg-destructive\/\d+/.test(tag) && file !== "components/ErrorBoundary.tsx")
    expect(tinted.map(({ file }) => file)).toEqual([])
  })

  it("offer Retry for a list that failed through LoadFailed", () => {
    const own = new Set([
      "components/LoadFailed.tsx",
      // a next page that failed under the rows that loaded
      "components/LoadMoreStatus.tsx",
      // the network browser's whole pane, titled with the kind that failed (a page placeholder's look)
      "features/networks/NetworkBrowser.tsx",
      // the image editor's stage when its image failed
      "canvas/ui/EmptyState.tsx",
    ])
    const copies = all.filter(({ file, body }) => /t\("Retry"\)/.test(body) && !own.has(file))
    expect(copies.map(({ file }) => file)).toEqual([])
  })
})
