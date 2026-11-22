import { describe, expect, it } from "vitest"
import { compilePrompt, DraftError } from "./compile"
import { COMPILED } from "./testdata"
import { decompilePrompt } from "./decompile"
import {
  type BuilderDraft,
  buildDraft,
  clampBox,
  defaultBbox,
  finalPrompt,
  fromDraft,
  metaTags,
  missingFields,
  moveBox,
  rescaleBox,
  resizeBox,
  resolutionFor,
  round16,
  unescapeParens,
} from "./draft"

const state = (patch: Partial<BuilderDraft> = {}): BuilderDraft => ({
  width: 1024,
  height: 768,
  caption: "A cat (orange) on a sofa",
  meta: " photo, , soft light ",
  extra: "",
  canvas: "single",
  items: [
    { id: 1, label: "character", caption: "an orange cat", text: "ignored", bbox: [100.4, 200, 500.2, 700] },
    { id: 2, label: "text", caption: "a neon sign", text: "OPEN (24h)", bbox: [600, 50, 1000, 150] },
  ],
  ...patch,
})

describe("resolution", () => {
  it("rounds to multiples of 16, at least 16", () => {
    expect(round16(1000)).toBe(1008)
    expect(round16(3)).toBe(16)
  })

  it("sizes megapixels at an aspect ratio", () => {
    expect(resolutionFor(1, "1:1")).toEqual({ width: 1024, height: 1024 })
    expect(resolutionFor(1, "16:9")).toEqual({ width: 1360, height: 768 })
    expect(resolutionFor(2, "2:3")).toEqual({ width: 1184, height: 1776 })
    expect(resolutionFor(0, "1:1")).toBeNull()
    expect(resolutionFor(1, "wide")).toBeNull()
  })
})

describe("boxes", () => {
  it("places a new box 10 % in, 30 % of the shorter side; each index steps it down-right", () => {
    expect(defaultBbox(1024, 768)).toEqual([77, 77, 307, 307])
    expect(defaultBbox(1024, 1024, 1)).toEqual([153, 153, 460, 460])
    // the offset stops at the canvas edge
    expect(defaultBbox(1024, 1024, 40)).toEqual([717, 717, 1024, 1024])
    // a tiny canvas still leaves a 16 px box
    expect(defaultBbox(16, 16)).toEqual([0, 0, 16, 16])
  })

  it("scales a box between canvas sizes, inside the new one at at least 16 px", () => {
    expect(rescaleBox([800, 800, 1000, 1000], 1024, 1024, 512, 512)).toEqual([400, 400, 500, 500])
    expect(rescaleBox([400, 400, 500, 500], 512, 512, 1024, 1024)).toEqual([800, 800, 1000, 1000])
    // a box on the edge of a tiny canvas stays whole
    expect(rescaleBox([0, 0, 16, 16], 1024, 1024, 16, 16)).toEqual([0, 0, 16, 16])
    // wider than tall: each axis scales on its own
    expect(rescaleBox([100, 200, 500, 700], 1024, 768, 512, 384)).toEqual([50, 100, 250, 350])
  })

  it("moves inside the canvas, keeping the size", () => {
    expect(moveBox([10, 10, 110, 60], 2000, -50, 1024, 768)).toEqual([924, 0, 1024, 50])
    expect(moveBox([10, 10, 110, 60], 4.6, 0.4, 1024, 768)).toEqual([15, 10, 115, 60])
  })

  it("resizes from the bottom-right corner, at least 16 px and inside the canvas", () => {
    expect(resizeBox([10, 10, 110, 60], -500, 30, 1024, 768)).toEqual([10, 10, 26, 90])
    expect(resizeBox([10, 10, 110, 60], 5000, 5000, 1024, 768)).toEqual([10, 10, 1024, 768])
  })

  it("clamps into a smaller canvas, shrinking what does not fit", () => {
    expect(clampBox([900, 600, 1000, 700], 512, 512)).toEqual([412, 412, 512, 512])
    expect(clampBox([0, 0, 1024, 100], 512, 512)).toEqual([0, 0, 512, 100])
  })
})

describe("drafts", () => {
  it("splits the meta tags", () => {
    expect(metaTags(" photo, , soft light ")).toEqual(["photo", "soft light"])
  })

  it("builds the draft in upstream's key order (no meta when empty, text only on text items, rounded boxes)", () => {
    expect(JSON.stringify(buildDraft(state({ meta: "" })))).toBe(
      JSON.stringify({
        width: 1024,
        height: 768,
        bbox_coord_space: "abs",
        bbox_format: "xyxy",
        summary: { canvas: "single", caption: "A cat (orange) on a sofa" },
        items: [
          { label: "character", caption: "an orange cat", bbox: [100, 200, 500, 700] },
          { label: "text", caption: "a neon sign", bbox: [600, 50, 1000, 150], text: "OPEN (24h)" },
        ],
      }),
    )
  })

  it("lists the empty fields the prompt needs", () => {
    const s = state({ caption: " ", items: [{ id: 7, label: "text", caption: "", text: "", bbox: [0, 0, 16, 16] }] })
    expect([...missingFields(s)]).toEqual(["caption", "item:7:caption", "item:7:text"])
    expect(missingFields(state()).size).toBe(0)
  })

  it("sends the compiled prompt, the extra text on its own line", () => {
    const prompt = finalPrompt(state({ extra: "  <lora:moss:0.8>\n more  " }))
    expect(prompt.startsWith('{"summary":{"meta":["photo","soft light"]')).toBe(true)
    expect(prompt.endsWith('"bbox_format":"yxyx"}\n<lora:moss:0.8> more')).toBe(true)
    expect(prompt).not.toContain("\r")
  })

  it("flattens newlines inside the draft's fields; literal backslashes and quotes survive", () => {
    const prompt = finalPrompt(state({ caption: "first line\nsecond line", items: [{ id: 1, label: "text", caption: 'say "hi" \\n literally', text: "A\nB", bbox: [0, 0, 16, 16] }] }))
    const json = prompt.slice(0, prompt.indexOf("bbox_coord_space"))
    expect(json).toContain("first line second line")
    expect(json).toContain('say \\"hi\\" \\\\n literally')
    expect(json).toContain("A B")
    // the literal backslash-n is a JSON escape inside the string, not a newline
    expect(decompilePrompt(prompt, 1024, 768).items[0]!.caption).toBe('say "hi" \\n literally')
  })

  it("throws the compiler's error for an invalid draft", () => {
    expect(() => finalPrompt(state({ width: 1000 }))).toThrow(DraftError)
  })

  it("loads a decompiled prompt with parentheses as typed and fresh ids; the prompt survives the trip", () => {
    const loaded = fromDraft(decompilePrompt(COMPILED, 1024, 768), 5)
    expect(loaded.caption).toBe("A cat (orange) on a sofa")
    expect(loaded.meta).toBe("photo, soft light")
    expect(loaded.items.map((item) => item.id)).toEqual([6, 7])
    expect(loaded.items[1]).toMatchObject({ label: "text", text: "OPEN (24h)" })
    expect(loaded.lastId).toBe(7)
    expect(compilePrompt(buildDraft(loaded))).toBe(COMPILED)
  })

  it("unescapes any run of backslashes before a parenthesis", () => {
    expect(unescapeParens("a \\(b\\) \\\\(c)")).toBe("a (b) (c)")
  })
})
