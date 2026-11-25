/**
 * A scroll the app starts itself honours reduced motion (docs/remi-webui/frontend.md, "Motion"):
 * index.css's reduced-motion rule sets `scroll-behavior: auto`, but a `scrollIntoView` or
 * `scrollTo` that passes `behavior: "smooth"` does not read it, so the img2img source panel kept
 * gliding to the editor for a user who asked for no motion. Scrolls take `scrollBehavior()`.
 */
import { describe, expect, it, vi } from "vitest"
import { scrollBehavior } from "./motion"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*"], { query: "?raw", import: "default", eager: true })

function prefersReducedMotion(reduce: boolean) {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) => ({ matches: reduce && query === "(prefers-reduced-motion: reduce)", media: query, addEventListener: () => {}, removeEventListener: () => {} }) as unknown as MediaQueryList,
  )
}

describe("scrollBehavior", () => {
  it("scrolls smoothly by default", () => {
    prefersReducedMotion(false)
    expect(scrollBehavior()).toBe("smooth")
  })

  it("jumps when the system asks for reduced motion", () => {
    prefersReducedMotion(true)
    expect(scrollBehavior()).toBe("auto")
  })

  it("scrolls smoothly where matchMedia is missing", () => {
    vi.stubGlobal("matchMedia", undefined)
    try {
      expect(scrollBehavior()).toBe("smooth")
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("is the only place that asks for a smooth scroll", () => {
    const smooth = Object.entries(sources)
      .filter(([file, source]) => file !== "./motion.ts" && /\bbehavior:[^,}\n]*["'`]smooth["'`]/.test(source))
      .map(([file]) => file.replace(/^\.\//, "lib/").replace(/^\.\.\//, ""))
    expect(Object.keys(sources).length).toBeGreaterThan(100)
    expect(smooth).toEqual([])
  })
})
