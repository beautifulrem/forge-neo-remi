/**
 * The focus ring (docs/remi-webui/frontend.md, "Visual rules") is one token, `--focus-ring`:
 * `--ring` mixed with transparency, as strong as each theme needs. Half of `--ring` read at about
 * 2.2:1 on the light theme's paper, too faint for a focus indicator (3:1), so every focus ring
 * takes the token, and the light theme's mix reaches 3:1 on the page and on the image backdrop
 * for every accent.
 */
import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import { ACCENTS } from "@/lib/theme"

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

describe("the focus ring token", () => {
  it("is --ring at each theme's strength, re-derived wherever an accent is set, and a Tailwind colour", () => {
    expect(token(block(":root"), "focus-ring")).toBe(FOCUS_RING)
    expect(token(block("[data-accent]"), "focus-ring")).toBe(FOCUS_RING)
    expect(token(block("[data-accent]"), "ring")).toBe("oklch(var(--brand-l) var(--brand-c) var(--brand-h))")
    expect(css).toMatch(/^ +--color-focus-ring: var\(--focus-ring\);$/m)
    // the dark theme keeps the ring it had: half of --ring
    expect(alpha(block(".dark"))).toBe(0.5)
  })

  it("reaches 3:1 on the light theme's page and image backdrop for every accent", () => {
    const root = block(":root")
    const surfaces = { background: srgb(oklch(token(root, "background"))), canvas: srgb(oklch(token(root, "canvas"))) }
    const accents = new Map([...css.matchAll(/^\[data-accent="(\w+)"\] \{ --brand-h: ([\d.]+); --brand-c: ([\d.]+); --brand-l: ([\d.]+); \}$/gm)].map((m) => [m[1]!, [Number(m[4]), Number(m[3]), Number(m[2])] as [number, number, number]]))
    expect([...accents.keys()].sort()).toEqual([...ACCENTS].sort())
    const weak = [...accents].flatMap(([accent, color]) =>
      Object.entries(surfaces)
        .map(([surface, value]) => ({ accent, surface, ratio: Math.round(ringContrast(srgb(color), alpha(root), value) * 100) / 100 }))
        .filter(({ ratio }) => ratio < 3),
    )
    expect(weak).toEqual([])
  })

  it("is what every half-strength ring of the app became", () => {
    const halfRings = Object.entries(sources)
      .filter(([, source]) => /\b(?:ring|outline)-ring\/50\b|var\(--ring\) 50%/.test(source))
      .map(([file]) => file.replace(/^\.\//, "styles/").replace(/^\.\.\//, ""))
    expect(Object.keys(scripts).length).toBeGreaterThan(100)
    expect(stylesheets["../index.css"]).toContain("@theme inline")
    expect(halfRings).toEqual([])
  })
})
