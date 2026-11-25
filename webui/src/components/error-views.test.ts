/**
 * Error views follow three patterns (docs/remi-webui/frontend.md, "Error views"): a list or panel
 * that could not load is `LoadFailed` (what failed, then Retry), a page that could not load is
 * `PagePlaceholder`, and an action's error is a plain destructive line by the action. Hand-written
 * copies had drifted: a smaller Retry in one list, a muted one in another, tinted boxes for some
 * action errors and plain lines for others.
 *
 * The checks read every class and every Retry, not only elements announced as alerts: the copies
 * that drift are often the ones nobody announced (a failed Extras run, the settings import's
 * skipped values), and a check that started from `role="alert"` never saw them.
 */
import { describe, expect, it } from "vitest"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*", "!../**/*.d.ts"], { query: "?raw", import: "default", eager: true })

/** `src/`-relative paths (the glob gives this folder's files as `./x`). */
const files = Object.entries(sources).map(([file, source]) => ({ file: file.replace(/^\.\//, "components/").replace(/^\.\.\//, ""), source }))

type Hit = { file: string; line: string }

/** The lines of `source` with a match of `pattern` (global). */
function lines(file: string, source: string, pattern: RegExp): Hit[] {
  return [...source.matchAll(pattern)].map((match) => {
    const start = source.lastIndexOf("\n", match.index) + 1
    const end = source.indexOf("\n", match.index)
    return { file, line: source.slice(start, end === -1 ? undefined : end).trim() }
  })
}

/** The bodies of the `<PagePlaceholder>` elements in `source`. */
function placeholders(source: string): string[] {
  return [...source.matchAll(/<PagePlaceholder\b[^>]*>([\s\S]*?)<\/PagePlaceholder>/g)].map((match) => match[1] ?? "")
}

describe("error views", () => {
  it("are found", () => {
    expect(files.length).toBeGreaterThanOrEqual(200)
  })

  it("say an error in a plain line, not a tinted box", () => {
    // a fill, not a state's (`hover:`, `focus:`, `data-[…]:`); the kit's variants (a destructive
    // button, badge or menu item) are controls, not error views
    const fills = files.filter(({ file }) => !file.startsWith("components/ui/")).flatMap(({ file, source }) => lines(file, source, /(?<![\w:/-])bg-destructive\/\d+/g))
    // one tone among others, not an error view of its own
    const tones: [file: string, line: RegExp][] = [
      // a text output's status: success, info, warning and error strips
      ["ui/layout/Outputs.tsx", /^error: \{ icon: CircleX/],
      // workflow advice rows by severity: info, warning and error
      ["pages/generate/WorkflowAdviceBar.tsx", /item\.severity === "error"/],
      // a round tinted icon over a title (the empty-area patterns' glyph)
      ["features/server-control/ServerOverlay.tsx", /AlertDialogMedia/],
      ["canvas/ui/EmptyState.tsx", /rounded-full/],
    ]
    const boxes = fills.filter(({ file, line }) => !tones.some(([tone, pattern]) => tone === file && pattern.test(line)))
    expect(boxes).toEqual([])
  })

  it("offer Retry for a list that failed through LoadFailed, for a page through PagePlaceholder", () => {
    const own = new Set([
      "components/LoadFailed.tsx",
      // a next page that failed under the rows that loaded
      "components/LoadMoreStatus.tsx",
      // the network browser's whole pane, titled with the kind that failed (a page placeholder's look)
      "features/networks/NetworkBrowser.tsx",
      // the image editor's stage when its image failed
      "canvas/ui/EmptyState.tsx",
      // the model picker itself says it could not load its list, and its press loads it again
      "features/model-bar/ModelBar.tsx",
    ])
    const copies = files
      .filter(({ file }) => !own.has(file))
      .flatMap(({ file, source }) => {
        const retries = source.match(/t\("Retry"\)/g)?.length ?? 0
        const inPlaceholders = placeholders(source).reduce((count, body) => count + (body.match(/t\("Retry"\)/g)?.length ?? 0), 0)
        return retries > inPlaceholders ? [file] : []
      })
    expect(copies).toEqual([])
  })
})
