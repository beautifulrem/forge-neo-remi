/**
 * Source checks for rules of docs/remi-webui/frontend.md ("Visual rules") that a class list can break
 * without any test of the component noticing. Each rule is its own test.
 */
import { describe, expect, it } from "vitest"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*", "!../api/gen/**"], { query: "?raw", import: "default", eager: true })

const files = Object.entries(sources).map(([file, source]) => ({ file: file.replace(/^\.\.\//, ""), source }))

/** Every match of `re` in the sources, with the file it is in. */
function occurrences(re: RegExp): { file: string; match: string }[] {
  return files.flatMap(({ file, source }) => [...source.matchAll(re)].map((m) => ({ file, match: m[0] })))
}

describe("type scale", () => {
  // 0.625 rem, 0.6875 rem, text-xs, 0.8 rem, text-sm, 0.9375 rem; text-xl is a theme name.
  const SCALE = new Set(["0.625rem", "0.6875rem", "0.75rem", "0.8rem", "0.875rem", "0.9375rem"])
  // A size off the scale, with the reason frontend.md gives for it.
  const ALLOWED = [
    // Inline code in prose: mono drawn a step smaller than the prose around it, whatever its size.
    { file: "ui/rich-text.tsx", size: "0.85em" },
  ]

  it("keeps arbitrary text sizes on the scale", () => {
    const sizes = occurrences(/\btext-\[(?:length:)?[^\]\s]+\]/g).flatMap(({ file, match }) => {
      const value = match.slice("text-[".length, -1).replace(/^length:/, "")
      // A size scaled by a variable (the JSON Builder canvas's zoom) starts from a size on the scale.
      const size = /^calc\((\d*\.?\d+(?:rem|em|px))\*/.exec(value)?.[1] ?? value
      return /^\d*\.?\d+(?:rem|em|px)$/.test(size) ? [{ file, size }] : []
    })
    expect(sizes.length).toBeGreaterThan(100)
    const off = sizes.filter(({ file, size }) => !SCALE.has(size) && !ALLOWED.some((allowed) => allowed.file === file && allowed.size === size))
    expect(off).toEqual([])
  })
})

describe("pressed toggles", () => {
  const TINT = "PRESSED_TINT_CLASS"
  const toggles = files.filter(({ file, source }) => file !== "components/ui/toggle.tsx" && /\baria-pressed=/.test(source))
  // Pressed state drawn another way, with the reason.
  const OWN_LOOK = new Set([
    // White on the viewer's black backdrop, over the picture: the tint would not read there.
    "features/lightbox/Lightbox.tsx",
  ])

  it("wear the selection tint instead of a hand-written one", () => {
    // A tint written by hand (an aria-pressed: class, a primary wash) instead of the shared one.
    const handWritten = toggles.filter(({ file, source }) => !OWN_LOOK.has(file) && /\baria-pressed:|\bbg-primary\/\d/.test(source) && !source.includes(TINT))
    expect(handWritten.map(({ file }) => file)).toEqual([])
    // Toggles that drew their pressed state with a fill, a variant or a colour of their own.
    for (const file of [
      "features/networks/NetworkBrowser.tsx",
      "pages/generate/SizeTools.tsx",
      "features/prompt-tools/PromptToolbar.tsx",
      "pages/settings/SettingRow.tsx",
      "features/compare/ParamDiffPanel.tsx",
      "features/recipes/RecipeManager.tsx",
      "pages/extensions/DiagnosticsView.tsx",
    ]) {
      expect(sources[`../${file}`], file).toContain(TINT)
    }
  })

  it("keep one variant in both states", () => {
    // A variant swapped on the pressed value draws a filled button once pressed, not the tint.
    const swapped = toggles.flatMap(({ file, source }) =>
      [...source.matchAll(/aria-pressed=\{([^}]+)\}/g)].flatMap((m) => {
        const value = (m[1] ?? "").trim()
        const around = source.slice(Math.max(0, m.index - 300), m.index + 400)
        const escaped = value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
        return new RegExp(`variant=\\{${escaped}\\s*\\?`).test(around) ? [file] : []
      }),
    )
    expect(swapped).toEqual([])
  })
})

describe("letter spacing", () => {
  // Wide tracking is the eyebrow-label look the type rules leave out. A menu's shortcut keys keep
  // shadcn's spacing: a run of key glyphs (⌘⇧K), not words.
  const SHORTCUTS = new Set(["components/ui/command.tsx", "components/ui/dropdown-menu-impl.tsx", "components/ui/dropdown-menu.tsx", "components/ui/context-menu.tsx"])

  it("never widens text", () => {
    const wide = occurrences(/\btracking-wide(?:r|st)?\b/g).filter(({ file }) => !SHORTCUTS.has(file))
    expect(wide).toEqual([])
  })
})
