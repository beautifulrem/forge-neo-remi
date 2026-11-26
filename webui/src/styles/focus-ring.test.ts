/**
 * The focus ring (docs/remi-webui/frontend.md, "Visual rules") is one token, `--focus-ring`:
 * `--ring` mixed with transparency, as strong as each theme needs. Focus rings used to write
 * their own share of `--ring` (40 to 70 %); half of it read at about 2.2:1 on the light theme's
 * paper and under 3:1 on the dark theme's page for some accents, too faint for a focus indicator.
 * Every focus ring takes the token, and each theme's mix reaches 3:1 on the page, on the nav rail and
 * on the image backdrop for every accent.
 */
import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import { ACCENTS } from "@/lib/theme"
import { openingTags } from "@/test/jsx-tags"

const css = readFileSync("src/index.css", "utf8")

const scripts = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*"], { query: "?raw", import: "default", eager: true })
// a stylesheet imported raw is empty under the test config (`css: false`): read them from disk
const stylesheets = Object.fromEntries(Object.keys(import.meta.glob("../**/*.css")).map((file) => [file, readFileSync(`src/styles/${file}`, "utf8")]))
const sources = { ...scripts, ...stylesheets }

/** The declarations of the first top-level rule whose selector is `selector`. */
function block(selector: string): string {
  const start = css.indexOf(`\n${selector} {\n`)
  if (start < 0) throw new Error(`no ${selector} rule in index.css`)
  return css.slice(start, css.indexOf("\n}", start))
}

function token(rule: string, name: string): string | undefined {
  return new RegExp(`^ +--${name}: (.+);$`, "m").exec(rule)?.[1]
}

function oklch(value: string | undefined): [number, number, number] {
  const match = /^oklch\(([\d.]+) ([\d.]+) ([\d.]+)\)$/.exec(value ?? "")
  if (!match) throw new Error(`not a plain oklch() colour: ${value}`)
  return [Number(match[1]), Number(match[2]), Number(match[3])]
}

/** Gamma-encoded sRGB channels (0..1) of an oklch colour, clipped to the gamut. */
function srgb([L, C, h]: [number, number, number]): number[] {
  const a = C * Math.cos((h * Math.PI) / 180)
  const b = C * Math.sin((h * Math.PI) / 180)
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  const linear = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s]
  return linear.map((x) => Math.min(1, Math.max(0, x))).map((x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055))
}

function luminance(channels: number[]): number {
  const [r, g, b] = channels.map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4)) as [number, number, number]
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/** The contrast of `ring` at `alpha`, composited over `surface` the way the browser paints it, against that surface. */
function ringContrast(ring: number[], alpha: number, surface: number[]): number {
  const painted = ring.map((x, i) => alpha * x + (1 - alpha) * surface[i]!)
  const [hi, lo] = [luminance(painted), luminance(surface)].sort((x, y) => y - x) as [number, number]
  return (hi + 0.05) / (lo + 0.05)
}

function alpha(rule: string): number {
  const match = /^(\d+)%$/.exec(token(rule, "focus-ring-alpha") ?? "")
  if (!match) throw new Error("no --focus-ring-alpha percentage")
  return Number(match[1]) / 100
}

const FOCUS_RING = "color-mix(in oklab, var(--ring) var(--focus-ring-alpha), transparent)"

/** Each accent's ring colour in `theme`: its hue and chroma, and the lightness that theme gives it. */
function accentRings(theme: "light" | "dark"): Map<string, [number, number, number]> {
  const accents = [...css.matchAll(/^\[data-accent="(\w+)"\] \{ --brand-h: ([\d.]+); --brand-c: ([\d.]+); --brand-l: ([\d.]+); \}$/gm)]
  // the dark theme sets one lightness for every accent, and its own for a few
  const dark = new Map([...css.matchAll(/^\.dark \[data-accent(?:="(\w+)")?\] \{ --brand-l: ([\d.]+); \}$/gm)].map((m) => [m[1] ?? "", Number(m[2])]))
  expect(dark.has("")).toBe(true)
  return new Map(accents.map((m) => [m[1]!, [theme === "light" ? Number(m[4]) : (dark.get(m[1]!) ?? dark.get(""))!, Number(m[3]), Number(m[2])]]))
}

describe("the focus ring token", () => {
  it("is --ring at each theme's strength, re-derived wherever an accent is set, and a Tailwind colour", () => {
    expect(token(block(":root"), "focus-ring")).toBe(FOCUS_RING)
    expect(token(block("[data-accent]"), "focus-ring")).toBe(FOCUS_RING)
    expect(token(block("[data-accent]"), "ring")).toBe("oklch(var(--brand-l) var(--brand-c) var(--brand-h))")
    expect(css).toMatch(/^ +--color-focus-ring: var\(--focus-ring\);$/m)
  })

  it.each(["light", "dark"] as const)("reaches 3:1 on the %s theme's page, nav rail and image backdrop for every accent", (theme) => {
    const rule = block(theme === "light" ? ":root" : ".dark")
    // the nav rail is --sidebar (--rail): its links take the token too
    const surfaces = { background: srgb(oklch(token(rule, "background"))), sidebar: srgb(oklch(token(rule, "sidebar"))), canvas: srgb(oklch(token(rule, "canvas"))) }
    const accents = accentRings(theme)
    expect([...accents.keys()].sort()).toEqual([...ACCENTS].sort())
    const weak = [...accents].flatMap(([accent, color]) =>
      Object.entries(surfaces)
        .map(([surface, value]) => ({ accent, surface, ratio: Math.round(ringContrast(srgb(color), alpha(rule), value) * 100) / 100 }))
        .filter(({ ratio }) => ratio < 3),
    )
    expect(weak).toEqual([])
  })

  it("is the outline of a focus nothing else marks, in place of the browser's own", () => {
    // the browser's `outline: auto` (its width and colour the browser's) on a link or the skip link;
    // the a11y e2e ("visible focus") measures it on every main page
    const base = css.slice(css.indexOf("@layer base {"))
    expect(base).toMatch(/\n {2}:focus-visible \{\n {4}outline: 3px solid var\(--focus-ring\);\n {4}outline-offset: 0;\n {2}\}/)
  })

  it("comes back as an outline in forced colours, over the controls that take the outline off", () => {
    // forced colours draw no box shadow, so a ring-3 control that takes the outline off
    // (outline-none) showed no focus there. The base layer draws the outline back with an important
    // declaration, which wins over the utilities' normal ones, in the system's highlight colour. The
    // slider's cap shows the focus of the range input nobody sees inside it. The a11y e2e ("visible
    // focus in forced colours") tabs through the main pages under forced colours.
    const base = css.slice(css.indexOf("@layer base {"))
    const forced = /@media \(forced-colors: active\) \{([\s\S]*?)\n {2}\}/.exec(base)?.[1] ?? ""
    const selectors = forced.slice(0, forced.indexOf("{")).replace(/\/\*[\s\S]*?\*\//g, "")
    expect(selectors.split(",").map((selector) => selector.trim())).toEqual([
      ":focus-visible",
      '[data-slot="slider-thumb"]:has(> input:focus-visible)',
    ])
    expect(forced).toMatch(/\{\s*outline: 3px solid Highlight !important;\s*\}/)
    expect(sources["../components/ui/slider.tsx"]).toMatch(/data-slot="slider-thumb"/)
  })

  it("marks the highlighted item of a list whose focus stays in its input, in forced colours", () => {
    // A combobox's and the command palette's highlighted item does not take the focus (that stays in
    // their input) and shows the highlight as a background, which forced colours do not draw: the
    // highlighted item was not told apart there. The forced-colours block outlines it as the focus.
    const base = css.slice(css.indexOf("@layer base {"))
    const forced = /@media \(forced-colors: active\) \{([\s\S]*?)\n {2}\}/.exec(base)?.[1] ?? ""
    const rule = /\n {4}(\[data-slot="combobox-item"\][^{]*)\{([^}]*)\}/.exec(forced.replace(/\/\*[\s\S]*?\*\//g, ""))
    expect(rule?.[1]?.split(",").map((selector) => selector.trim())).toEqual(['[data-slot="combobox-item"][data-highlighted]', '[data-slot="command-item"][data-selected="true"]'])
    // important: the base layer loses to the utilities' outline-none otherwise
    expect(rule?.[2]?.trim()).toBe("outline: 3px solid Highlight !important;")
    expect(sources["../components/ui/combobox.tsx"] ?? "").toMatch(/data-slot="combobox-item"[\s\S]*data-highlighted:bg-accent/)
    expect(sources["../components/ui/command.tsx"] ?? "").toMatch(/data-slot="command-item"[\s\S]*data-selected:bg-muted/)
  })

  it("gives a floating surface an edge in forced colours", () => {
    // Forced colours draw no box shadow: not the ring and shadow of menus, selects and comboboxes, the
    // ring of dialogs or the shadow of sheets; and they replace a tooltip's plain fill with the page's
    // Canvas. A popup sat over the fields under it with no edge at all. The forced-colours block outlines
    // them (not while one has the focus: the focus outline is drawn then), and so the app's floating bars,
    // the networks drawer and the image editor's floating material, whose edge is a shadow too.
    const rule = edgeRule()
    expect(rule.selectors).toEqual([...[...SURFACES, ...Object.keys(APP_SURFACES)].map((slot) => `[data-slot="${slot}"]`), ".material-float"])
    expect(rule.declarations).toBe("outline: 1px solid CanvasText !important;")
    const kit = Object.entries(sources).filter(([file]) => file.startsWith("../components/ui/")).map(([, source]) => source).join("\n")
    for (const slot of SURFACES) expect(kit).toContain(`data-slot="${slot}"`)
    for (const [slot, files] of Object.entries(APP_SURFACES)) {
      for (const file of files) expect(scripts[`../${file}`], file).toContain(` data-slot="${slot}"`)
    }
    // the material's edge is a box shadow: the utility, and the image editor's panels that take it
    expect(css).toContain("\n@utility material-float {\n")
    expect(scripts["../canvas/ui/controls.tsx"]).toMatch(/\bPANEL = "[^"]*\bmaterial-float\b/)
  })

  it("reaches a surface outside the kit that floats on a shadow", () => {
    // frontend.md lets a floating bar and a drawer take the shadow alone, which forced colours do not draw:
    // the settings' apply bar merged with the fields under it. A tag outside the kit that draws a shadow of
    // md or more carries a slot the edge rule lists, or a border of its own, which forced colours draw.
    const listed = edgeRule().selectors.flatMap((selector) => /^\[data-slot="([a-z-]+)"\]$/.exec(selector)?.[1] ?? [])
    const words = (tag: string) => tag.split(/[\s"'`{}(),]+/)
    const lifted = appTags(/./).filter(({ tag }) => words(tag).some((word) => /^shadow-(?:md|lg|xl|2xl)$/.test(word)))
    expect(lifted.length).toBeGreaterThan(5)
    const edgeless = lifted.filter(({ file, tag }) => {
      if (listed.includes(/\sdata-slot="([a-z-]+)"/.exec(tag)?.[1] ?? "")) return false
      if (words(tag).some((word) => /^border(?:-\d+|-\[[^\]]+\])?$/.test(word))) return false
      return !NOT_OVER_THE_PAGE.some((own) => own.file === file && tag.includes(own.look))
    })
    expect(edgeless.map(({ file, name }) => `${file}: <${name}`)).toEqual([])
  })

  it("reaches the popups built outside the kit", () => {
    // The rule reads the kit's slots. A Base UI popup built in app code (the search select's, a network
    // card's menu) carried none, and sat over the page with no edge in forced colours: each takes the
    // slot of the kit popup it stands for. A popover's or hover card's edge there is its bare
    // outline-hidden (lib/design-guards.test.ts keeps that outline to those two).
    const listed = edgeRule().selectors.flatMap((selector) => /^\[data-slot="([a-z-]+)"\]$/.exec(selector)?.[1] ?? [])
    expect(listed.length).toBeGreaterThan(5)
    const popups = appTags(/\.Popup$/)
    expect(popups.length).toBeGreaterThan(2)
    const edgeless = popups.filter(({ file, tag }) => {
      const slot = /\sdata-slot="([a-z-]+)"/.exec(tag)?.[1] ?? ""
      if (listed.includes(slot)) return false
      if (OWN_EDGE.includes(slot) && /[\s"'`]outline-hidden[\s"'`]/.test(tag)) return false
      return !FILLS_THE_SCREEN.some((own) => own.file === file && tag.includes(own.look))
    })
    expect(edgeless.map(({ file, name }) => `${file}: <${name}`)).toEqual([])
  })

  it("marks the highlighted item of a combobox built outside the kit", () => {
    // The search select's items are Base UI's own and carried no slot: its highlighted option, which
    // does not take the focus (that stays in its search field), was not told apart in forced colours.
    const items = appTags(/^Combobox(?:Primitive)?\.Item$/)
    expect(items.length).toBeGreaterThan(0)
    const unmarked = items.filter(({ tag }) => !/\sdata-slot="combobox-item"/.test(tag))
    expect(unmarked.map(({ file, name }) => `${file}: <${name}`)).toEqual([])
  })

  it("is the only focus ring colour drawn from --ring at a share of its own", () => {
    const ownShares = Object.entries(sources)
      .filter(([, source]) => /\b(?:ring|outline)-ring\/\d+\b|var\(--ring\) \d+%/.test(source))
      .map(([file]) => file.replace(/^\.\//, "styles/").replace(/^\.\.\//, ""))
    expect(Object.keys(scripts).length).toBeGreaterThan(100)
    expect(stylesheets["../index.css"]).toContain("@theme inline")
    expect(ownShares).toEqual([])
  })
})

/** The base layer's forced-colours block, its comments left out. */
function forcedColours(): string {
  const base = css.slice(css.indexOf("@layer base {"))
  return (/@media \(forced-colors: active\) \{([\s\S]*?)\n {2}\}/.exec(base)?.[1] ?? "").replace(/\/\*[\s\S]*?\*\//g, "")
}

/** The forced-colours rule that outlines the floating surfaces: its selectors (it applies to them only without the focus) and its declarations. */
function edgeRule(): { selectors: string[]; declarations: string } {
  const rule = /\n {4}:is\((\[data-slot="dialog-content"\][^{]*)\):not\(:focus-visible\) \{([^}]*)\}/.exec(forcedColours())
  return { selectors: (rule?.[1] ?? "").split(",").map((selector) => selector.trim()), declarations: rule?.[2]?.trim() ?? "" }
}

/** The opening tags in the app's own code (outside the kit) whose name matches `name`, with their file. */
function appTags(name: RegExp): { file: string; name: string; tag: string }[] {
  return Object.entries(scripts)
    .filter(([file]) => !file.startsWith("../components/ui/"))
    .flatMap(([file, source]) =>
      openingTags(source)
        .tags.filter((tag) => name.test(tag.name))
        .map((tag) => ({ file: file.replace(/^\.\.\//, ""), name: tag.name, tag: tag.tag })),
    )
}

/** Popup slots whose edge in forced colours is their own bare outline-hidden, not the rule's outline. */
const OWN_EDGE = ["popover-content", "hover-card-content"]

/** Popups with no edge to draw, with the reason: `look` pins the tag to what makes it so. */
const FILLS_THE_SCREEN = [
  // the image viewer covers the whole window: there is no page beside it for an edge to divide it from
  { file: "features/lightbox/Lightbox.tsx", look: "fixed inset-0" },
]

/** The app's floating surfaces whose edge is their shadow alone (frontend.md), by slot, with the files that draw them. */
const APP_SURFACES = {
  "floating-bar": ["pages/gallery/index.tsx", "pages/settings/ApplyBar.tsx", "pages/extensions/InstalledView.tsx"],
  "networks-drawer": ["features/networks/NetworksDrawer.tsx"],
}

/** Shadows on what does not float over the page, with the reason: `look` pins the tag to what makes it so. */
const NOT_OVER_THE_PAGE = [
  // the queue row a drag holds lifts inside its own list for the length of the drag
  { file: "features/queue/PendingList.tsx", look: "border-transparent bg-popover shadow-lg" },
  // the compare slider's handle is a control over the two pictures, whose fill forced colours draw over them
  { file: "features/compare/CompareSlider.tsx", look: "rounded-full bg-white" },
]

/**
 * The kit's floating surfaces, whose edge forced colours do not draw: a ring and a shadow (menus, selects,
 * comboboxes), a ring (dialogs), a shadow (sheets) or a plain fill (tooltips). None of the popover kind,
 * whose outline-hidden is its edge.
 */
const SURFACES =["dialog-content", "alert-dialog-content", "sheet-content", "dropdown-menu-content", "dropdown-menu-sub-content", "context-menu-content", "context-menu-sub-content", "select-content", "combobox-content", "tooltip-content"]

/** The opening JSX tag around `marker` in `source`: from its `<` to the `>` outside braces and strings. */
function tagAround(source: string, marker: string): string {
  const at = source.indexOf(marker)
  if (at < 0) throw new Error(`no ${marker}`)
  const start = source.lastIndexOf("\n", source.lastIndexOf("<", at))
  let depth = 0
  let quote = ""
  for (let i = source.indexOf("<", start); i < source.length; i++) {
    const c = source[i]
    if (quote) {
      if (c === quote) quote = ""
    } else if (c === '"' || c === "`") quote = c
    else if (c === "{") depth++
    else if (c === "}") depth--
    else if (c === ">" && depth === 0 && source[i - 1] !== "=") return source.slice(start, i + 1)
  }
  throw new Error(`no end to the tag around ${marker}`)
}

/**
 * The image viewer's backdrop is near-black in both themes (frontend.md, "Colours"). Its chrome read the
 * page's tokens, so in the light theme its focus ring (80 % of a dark --ring) read about 2.3:1 there for
 * every accent. The chrome carries the `dark` class: the tokens it reads are the dark theme's, and a `dark`
 * scope inside the light theme re-derives the brand colours, which are resolved where they are declared
 * (the root) and would otherwise come down as the light theme's.
 */
describe("a dark scope inside the light theme", () => {
  it("re-derives the brand colours, with the dark theme's lightness for each accent", () => {
    const scoped = [...css.matchAll(/^\.dark \{\n([^}]*)\}/gm)].map((m) => m[1]!)
    const derived = scoped.find((rule) => /--ring:/.test(rule)) ?? ""
    expect(token(derived, "ring")).toBe("oklch(var(--brand-l) var(--brand-c) var(--brand-h))")
    expect(token(derived, "focus-ring")).toBe(FOCUS_RING)
    expect(token(derived, "primary")).toBe("oklch(var(--brand-l) var(--brand-c) var(--brand-h))")
    // each rule that gives the dark theme's accents their lightness names a dark scope under the accent too
    const lightness = [...css.matchAll(/((?:^[^\n{}]+,\n)*^[^\n{}]+) \{ --brand-l: [\d.]+; \}$/gm)].map((m) => m[1]!.split(/,\n/))
    const dark = lightness.filter((selectors) => selectors.some((selector) => selector.startsWith(".dark")))
    expect(dark.length).toBe(3)
    for (const selectors of dark) {
      const accent = /\[data-accent(?:="\w+")?\]/.exec(selectors[0]!)![0]
      expect(selectors).toContain(`${accent} .dark`)
    }
  })

  it("reaches 3:1 for every accent on the image viewer's backdrop, whose chrome is dark", () => {
    const lightbox = scripts["../features/lightbox/Lightbox.tsx"]!
    const backdrop = /data-slot="viewer-backdrop" className="[^"]*\bbg-\[oklch\(([\d.]+)_([\d.]+)_([\d.]+)\)\]/.exec(lightbox)
    expect(backdrop).not.toBeNull()
    const surface = srgb([Number(backdrop![1]), Number(backdrop![2]), Number(backdrop![3])])
    const weak = [...accentRings("dark")]
      .map(([accent, color]) => ({ accent, ratio: Math.round(ringContrast(srgb(color), alpha(block(".dark")), surface) * 100) / 100 }))
      .filter(({ ratio }) => ratio < 3)
    expect(weak).toEqual([])
    // the toolbar, the details panel and the previous / next buttons: what draws a focus ring over the backdrop
    for (const marker of ['@container/actions', '<aside', 't("Previous image")', 't("Next image")']) {
      expect(tagAround(lightbox, marker), marker).toMatch(/className=(?:"|\{cn\(")(?:[^"]* )?dark[ "]/)
    }
  })
})

/** An `oklch(L C H)` or `oklch(L C H / A%)` token: its colour and its alpha (0..1). */
function oklcha(value: string | undefined): { color: [number, number, number]; alpha: number } {
  const match = /^oklch\(([\d.]+) ([\d.]+) ([\d.]+)(?: \/ ([\d.]+)%)?\)$/.exec(value ?? "")
  if (!match) throw new Error(`not an oklch() colour: ${value}`)
  return { color: [Number(match[1]), Number(match[2]), Number(match[3])], alpha: match[4] === undefined ? 1 : Number(match[4]) / 100 }
}

/** `color` at `alpha` painted over `under` (sRGB channels), the way the browser blends it. */
const over = (color: number[], alpha: number, under: number[]) => color.map((x, i) => alpha * x + (1 - alpha) * under[i]!)

function contrast(a: number[], b: number[]): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number]
  return (hi + 0.05) / (lo + 0.05)
}

/**
 * An invalid field's red edge (docs/remi-webui/frontend.md, "Visual rules"): `border-destructive`, or
 * a share of it, under an invalid variant (aria-invalid, data-invalid, has-aria-invalid…). It is a
 * state indicator: 3:1 against what is next to it (WCAG 1.4.11), the surface around the field and
 * the field's fill inside the edge. A border is painted over its element's background (over the
 * surface with bg-clip-padding, a button's): both are checked. The dark theme drew it at half
 * strength, 2.6:1 to 2.8:1 against a field's fill and 1.9:1 against a switch's track. Focused, the
 * edge's outer neighbour is the focus ring, which under the coral accent is nearly the same red: no
 * share of --destructive clears that (nor a checked switch's --primary fill), and the message an
 * invalid field shows covers it (lib/design-guards.test.ts, "invalid fields").
 */
describe("the invalid edge", () => {
  const classes = Object.entries(sources).flatMap(([file, source]) =>
    (source.match(/[^\s"'`{}]+/g) ?? []).map((cls) => ({ file: file.replace(/^\.\.\//, ""), cls })),
  )
  const edges = classes.flatMap(({ file, cls }) => {
    const m = /^(.*):border-destructive(?:\/(\d+))?$/.exec(cls)
    if (!m || !/invalid/.test(m[1]!)) return []
    const dark = m[1]!.split(":").includes("dark")
    return [{ file, cls, share: m[2] === undefined ? 1 : Number(m[2]) / 100, themes: dark ? (["dark"] as const) : (["light", "dark"] as const) }]
  })
  /** The share of a utility a kit file writes, e.g. the switch's unchecked track (`bg-foreground/24`). */
  function share(file: string, cls: RegExp): number {
    const m = cls.exec(sources[`../${file}`] ?? "")
    if (!m) throw new Error(`${cls} not in ${file}`)
    return Number(m[1]) / 100
  }
  /** The tokens an invalid edge and what is around it read: an accent may set none of them. */
  const EDGE_TOKENS = ["destructive", "foreground", "input", "field", "field-hover", "background", "card", "popover", "canvas"]

  it("is drawn by the kit's fields and depends on no accent", () => {
    for (const file of ["components/ui/input.tsx", "components/ui/textarea.tsx", "components/ui/select-styles.ts", "components/ui/combobox-styles.ts", "components/ui/switch.tsx", "components/ui/checkbox.tsx", "components/ui/number-field.tsx", "components/ui/input-group.tsx"]) {
      expect(edges.some((edge) => edge.file === file), file).toBe(true)
    }
    const accentRules = [...css.matchAll(/^([^\n{}]*\[data-accent[^{}]*)\{([^}]*)\}/gm)]
    expect(accentRules.length).toBeGreaterThan(ACCENTS.length)
    const set = accentRules.flatMap((rule) => [...rule[2]!.matchAll(/--([\w-]+):/g)].map((m) => m[1]!)).filter((name) => EDGE_TOKENS.includes(name))
    expect(set).toEqual([])
  })

  /**
   * What an edge is drawn around and where (by default a field's fill, hovered too, on the page, a
   * card or a popover): a switch's unchecked track, a checkbox's or radio's box, nothing (a drop
   * zone, a toggle, a badge); a button's border is painted over the surface (bg-clip-padding), its
   * fill the field's when it acts as one (FIELD_BUTTON_CLASS); the image editor's drop target sits
   * on its stage.
   */
  const AROUND: Record<string, { fills: string[]; overSurface?: boolean; surfaces?: string[] }> = {
    "components/ui/switch.tsx": { fills: ["switch track"] },
    "components/ui/checkbox.tsx": { fills: ["box"] },
    "components/ui/radio-group.tsx": { fills: ["box"] },
    "components/ui/button.tsx": { fills: ["none", "field", "field-hover"], overSurface: true },
    "components/ui/toggle.tsx": { fills: ["none"] },
    "components/ui/badge.tsx": { fills: ["none"] },
    "ui/inputs/DropZone.tsx": { fills: ["none"] },
    "canvas/ui/EmptyState.tsx": { fills: ["none"], surfaces: ["canvas"] },
  }
  const FIELD = { fills: ["field", "field-hover"], surfaces: ["background", "card", "popover"] }

  const cases = (["light", "dark"] as const).flatMap((theme) => ACCENTS.map((accent) => [theme, accent] as const))
  it.each(cases)("reaches 3:1 against the surface and the field's fill in the %s theme with the %s accent", (theme) => {
    const rule = block(theme === "light" ? ":root" : ".dark")
    // a token's value, a comment after it left out
    const tone = (name: string) => oklcha(new RegExp(`^ +--${name}: ([^;]+);`, "m").exec(rule)?.[1])
    const destructive = srgb(tone("destructive").color)
    const surfaces = Object.fromEntries(["background", "card", "popover", "canvas"].map((name) => [name, srgb(tone(name).color)]))
    const fills: Record<string, { color: number[]; alpha: number }> = {
      none: { color: [0, 0, 0], alpha: 0 },
      field: { color: srgb(tone("field").color), alpha: tone("field").alpha },
      "field-hover": { color: srgb(tone("field-hover").color), alpha: tone("field-hover").alpha },
      "switch track": {
        color: srgb(tone("foreground").color),
        alpha: theme === "light" ? share("components/ui/switch.tsx", /\sdata-unchecked:bg-foreground\/(\d+)\s/) : share("components/ui/switch.tsx", /\sdark:data-unchecked:bg-foreground\/(\d+)\s/),
      },
      // no fill in the light theme
      box: theme === "light" ? { color: [0, 0, 0], alpha: 0 } : { color: srgb(tone("input").color), alpha: tone("input").alpha * share("components/ui/checkbox.tsx", /\sdark:bg-input\/(\d+)\s/) },
    }
    const weak = new Map<string, string>()
    for (const edge of edges.filter((edge) => (edge.themes as readonly string[]).includes(theme))) {
      const around = { ...FIELD, ...AROUND[edge.file] }
      let worst = { ratio: Infinity, where: "" }
      for (const surfaceName of around.surfaces) {
        const surface = surfaces[surfaceName]!
        for (const fillName of around.fills) {
          const fill = fills[fillName]!
          const inside = over(fill.color, fill.alpha, surface)
          const painted = over(destructive, edge.share, around.overSurface ? surface : inside)
          for (const [neighbour, next] of [["surface", surface], ["fill", inside]] as const) {
            const ratio = contrast(painted, next)
            if (ratio < worst.ratio) worst = { ratio, where: `${neighbour}: ${fillName} on ${surfaceName}` }
          }
        }
      }
      if (worst.ratio < 3) weak.set(`${edge.file}: ${edge.cls}`, `${worst.ratio.toFixed(2)}:1 against the ${worst.where}`)
    }
    expect(edges.length).toBeGreaterThan(12)
    expect(Object.fromEntries(weak)).toEqual({})
  })
})
