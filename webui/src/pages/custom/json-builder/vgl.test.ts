import { describe, expect, it } from "vitest"
import { DraftError } from "./compile"
import { type BuilderDraft, finalPrompt } from "./draft"
import { restore } from "./store"
import { buildVglDocument, decompileVglPrompt, describeBoxLocation, describeRelativeSize, finalVglPrompt, isVglPrompt } from "./vgl"

const draft = (patch: Partial<BuilderDraft> = {}): BuilderDraft => ({
  width: 1024,
  height: 1024,
  caption: "A studio still life of a ceramic teapot and brass sign",
  meta: "editorial photography, film grain",
  extra: "",
  canvas: "single",
  format: "vgl",
  items: [
    { id: 1, label: "object", caption: "a matte celadon ceramic teapot", text: "", bbox: [320, 320, 720, 720] },
    { id: 2, label: "text", caption: "a small embossed brass plaque", text: "TEA ROOM", bbox: [60, 60, 260, 180] },
  ],
  ...patch,
})

describe("Bria Fibo 2.0 VGL JSON mode", () => {
  it("compiles all required VGL 2.0 fields with spatial location and relative size", () => {
    const doc = buildVglDocument(draft({ vgl: { styleMedium: "photograph", cameraAngle: "low-angle", lensFocalLength: "85mm portrait" } }))
    expect(doc.short_description).toBe("A studio still life of a ceramic teapot and brass sign")
    expect(doc.style_medium).toBe("photograph")
    expect(doc.photographic_characteristics.camera_angle).toBe("low-angle")
    expect(doc.photographic_characteristics.lens_focal_length).toBe("85mm portrait")
    expect(doc.artistic_style).toBe("editorial photography, film grain")
    expect(doc.objects).toHaveLength(2)
    expect(doc.objects[0]).toMatchObject({ description: "a matte celadon ceramic teapot", location: "center", relative_size: "medium" })
    expect(doc.text_render).toEqual([{ text: "TEA ROOM", description: "a small embossed brass plaque", location: "top-left" }])
    expect(Object.keys(doc.lighting).sort()).toEqual(["conditions", "direction", "shadows"])
    expect(Object.keys(doc.aesthetics).sort()).toEqual(["color_scheme", "composition", "mood_atmosphere"])
  })

  it("round-trips through finalPrompt, isVglPrompt, and decompileVglPrompt with trailing extra text", () => {
    const prompt = finalPrompt(draft({ extra: "<lora:studio:0.7>" }))
    expect(prompt).toBe(finalVglPrompt(draft({ extra: "<lora:studio:0.7>" })))
    expect(isVglPrompt(prompt)).toBe(true)
    expect(isVglPrompt('{"summary":{"caption":"x"},"items":[]}')).toBe(false)

    const loaded = decompileVglPrompt(prompt, 1024, 1024)
    expect(loaded.extra).toBe("<lora:studio:0.7>")
    expect(loaded.draft.summary.caption).toBe("A studio still life of a ceramic teapot and brass sign")
    expect(loaded.draft.summary.meta).toEqual(["editorial photography", "film grain"])
    expect(loaded.draft.items).toHaveLength(2)
    expect(loaded.draft.items[1]).toMatchObject({ label: "text", text: "TEA ROOM" })
    expect(loaded.vgl.styleMedium).toBe("photograph")
  })

  it("describes box locations and relative sizes accurately", () => {
    expect(describeBoxLocation([0, 0, 200, 200], 1000, 1000)).toBe("top-left")
    expect(describeBoxLocation([400, 400, 600, 600], 1000, 1000)).toBe("center")
    expect(describeBoxLocation([800, 800, 1000, 1000], 1000, 1000)).toBe("bottom-right")
    expect(describeRelativeSize([0, 0, 100, 100], 1000, 1000)).toBe("small")
    expect(describeRelativeSize([0, 0, 900, 900], 1000, 1000)).toBe("dominant")
  })

  it("rejects invalid drafts and persists vgl format in store restore", () => {
    expect(() => buildVglDocument(draft({ caption: "  " }))).toThrow(DraftError)
    expect(restore({ format: "vgl", vgl: { styleMedium: "3D render" } })).toMatchObject({
      format: "vgl",
      vgl: { styleMedium: "3D render" },
    })
  })
})
