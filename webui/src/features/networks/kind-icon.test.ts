import { describe, expect, it } from "vitest"
import { Sparkles } from "@/components/icons"
import { KIND_ICON } from "./kind-icon"

describe("network kind glyphs", () => {
  // Sparkles is the Generate button's wand: a VAE without a preview read as "generate".
  it("never use the Generate button's glyph", () => {
    expect(Object.entries(KIND_ICON).filter(([, Icon]) => Icon === Sparkles)).toEqual([])
    expect(KIND_ICON.vae).toBeDefined()
  })
})
