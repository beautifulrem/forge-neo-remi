import { describe, expect, it } from "vitest"
import { DraftError } from "./compile"
import { type BuilderDraft, finalPrompt } from "./draft"
import { CANVAS, canvasBox, citedCaption, decompileFlux3Prompt, finalFlux3Prompt, flux3Rows, isFlux3Prompt, splitFlux3Prompt } from "./flux3"
import { restore } from "./store"

const ITEMS: BuilderDraft["items"] = [
  { id: 1, label: "text", caption: "a hand-painted chalkboard menu", text: "OPEN\n7-11", bbox: [600, 50, 1000, 150] },
  { id: 2, label: "character", caption: "a barista in a green apron, smiling", text: "", bbox: [100, 200, 500, 700] },
  { id: 3, label: "object", caption: "a steaming latte; ceramic cup", text: "", bbox: [100, 600, 300, 760] },
]
const LATTE = { id: 3, label: "object", caption: "a steaming latte; ceramic cup", text: "", bbox: [100, 600, 300, 760] } as const

const state = (patch: Partial<BuilderDraft> = {}): BuilderDraft => ({
  width: 1024,
  height: 768,
  caption: "Full-color photograph of a cozy cafe counter (morning)",
  meta: " film grain, , soft light ",
  extra: "",
  canvas: "single",
  format: "flux3",
  items: ITEMS,
  ...patch,
})

describe("FLUX.3 layout prompts", () => {
  it("puts pixel boxes on the 0-1000 canvas as [y_min, x_min, y_max, x_max], outward-rounded", () => {
    expect(canvasBox([100, 200, 500, 700], 1024, 768)).toEqual([260, 97, 912, 489])
    expect(canvasBox([0, 0, 1024, 768], 1024, 768)).toEqual([0, 0, CANVAS, CANVAS])
    expect(() => canvasBox([10, 10, 10, 20], 1024, 768)).toThrow(DraftError)
  })

  it("numbers ids per class left to right, then top to bottom, keeping the item order", () => {
    const rows = flux3Rows(state())
    expect(rows.map((row) => row.id)).toEqual(["En_Text_1", "person_1", "object_1"])
    expect(rows[0]?.desc).toBe('Text reading "OPEN\\n7-11", a hand-painted chalkboard menu')
    const two = flux3Rows(state({ items: [{ ...LATTE, bbox: [...LATTE.bbox] }, { ...LATTE, id: 4, bbox: [50, 0, 90, 40] }] }))
    expect(two.map((row) => row.id)).toEqual(["object_2", "object_1"])
  })

  it("cites every element as <id> in the caption and appends the table on one line", () => {
    const prompt = finalFlux3Prompt(state({ extra: " <lora:style:0.8> " }))
    const [body = "", extra] = prompt.split("\n")
    expect(extra).toBe("<lora:style:0.8>")
    const { caption, rows } = splitFlux3Prompt(body)
    expect(caption).toBe(
      "Full-color photograph of a cozy cafe counter (morning). Style: film grain, soft light. Elements: text reading \"OPEN\\n7-11\" <En_Text_1>, a barista in a green apron <person_1>, a steaming latte <object_1>.",
    )
    expect(rows).toEqual(flux3Rows(state()))
    // parentheses stay literal (the FLUX.3 engine reads the text as written)
    expect(body).not.toContain("\\(")
    // an id the caption already cites is not listed again
    expect(citedCaption("A barista <person_1> at work", "", flux3Rows(state()))).not.toContain("barista in a green apron <person_1>")
  })

  it("is what finalPrompt sends in FLUX.3 mode, and MOSS-Image JSON otherwise", () => {
    expect(finalPrompt(state())).toBe(finalFlux3Prompt(state()))
    expect(finalPrompt(state({ format: "moss" })).startsWith("{")).toBe(true)
    expect(() => finalPrompt(state({ caption: " " }))).toThrow(DraftError)
  })

  it("loads a FLUX.3 layout prompt back, including an upsampler-written one", () => {
    const prompt = finalFlux3Prompt(state({ extra: "<lora:x:1>" }))
    expect(isFlux3Prompt(prompt)).toBe(true)
    const { draft, extra } = decompileFlux3Prompt(prompt, 1024, 768)
    expect(extra).toBe("<lora:x:1>")
    expect(draft.summary).toEqual({ caption: "Full-color photograph of a cozy cafe counter (morning).", meta: ["film grain", "soft light"], canvas: "single" })
    expect(draft.items.map((item) => item.label)).toEqual(["text", "character", "object"])
    expect(draft.items[0]).toMatchObject({ text: "OPEN\n7-11", caption: "a hand-painted chalkboard menu" })
    expect(draft.items[1]?.bbox).toEqual([99, 200, 501, 700])

    const upsampled = 'Studio photograph of a wicker table <table_1> holding a sign <En_Text_1>. [{"id": "table_1", "bbox": [500, 100, 950, 900], "desc": "A round wicker table."}, {"id": "En_Text_1", "bbox": [300, 400, 450, 600], "desc": "A card reading \\"SALE\\" in red serif capitals."}]'
    expect(isFlux3Prompt(upsampled)).toBe(true)
    const loaded = decompileFlux3Prompt(upsampled, 1000, 1000).draft
    expect(loaded.summary.caption).toBe("Studio photograph of a wicker table holding a sign.")
    expect(loaded.items[1]).toMatchObject({ label: "text", text: "SALE" })
    expect(isFlux3Prompt('{"summary": {"caption": "x"}, "items": []}')).toBe(false)
    expect(isFlux3Prompt("just a caption")).toBe(false)
  })

  it("persists the prompt format", () => {
    expect(restore({ format: "flux3" }).format).toBe("flux3")
    expect(restore({ format: "other" }).format).toBeUndefined()
  })
})
