import { beforeAll, describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import { noteText } from "./notes"

// what a job ends with: a sentence the processing noted (model advice) as its template and
// values, any other note as {{note}} (remi/jobs/runner.py comment_notes)
const TURBO = "Anima-Turbo checkpoint with CFG {{cfg}} / {{steps}} steps: it is distilled for CFG 1 and 8-12 steps (see the Official Turbo preset); higher values over-cook it."
const EDIT = "Anima Edit model/LoRA detected with an input or reference image while '[Anima] Enable Reference' (anima_do_reference) is off: enable it so Anima stacks the reference frame instead of running plain img2img."

describe("noteText", () => {
  beforeAll(() => {
    i18n.addResourceBundle(
      "zh-CN",
      "server",
      { [TURBO]: "Anima-Turbo 模型配 CFG {{cfg}} / {{steps}} 步：它为 CFG 1、8-12 步蒸馏。", [EDIT]: "检测到 Anima Edit 模型，但参考模式未开启。", "Requires {{capabilities}}": "需要{{capabilities}}", Inpainting: "局部重绘", Edit: "编辑" },
      true,
      true,
    )
  })

  it("translates a sentence the processing noted, with its values", () => {
    const note = { message: TURBO, values: { cfg: "4.5", steps: "36" } }
    expect(noteText(note, "zh-CN")).toBe("Anima-Turbo 模型配 CFG 4.5 / 36 步：它为 CFG 1、8-12 步蒸馏。")
    expect(noteText(note, "en")).toMatch(/^Anima-Turbo checkpoint with CFG 4\.5 \/ 36 steps: /)
  })

  it("translates a plain note the catalogs have word for word, and leaves any other text as it is", () => {
    expect(noteText({ message: "{{note}}", values: { note: EDIT } }, "zh-CN")).toBe("检测到 Anima Edit 模型，但参考模式未开启。")
    expect(noteText({ message: "{{note}}", values: { note: "LoRA {{x}} not found" } }, "zh-CN")).toBe("LoRA {{x}} not found")
  })

  it("translates each name of a list value and joins them the language's way", () => {
    const note = { message: "Requires {{capabilities}}", values: { capabilities: ["Inpainting", "Edit"] } }
    expect(noteText(note, "zh-CN")).toBe("需要局部重绘和编辑")
    expect(noteText(note, "en")).toBe("Requires Inpainting and Edit")
  })
})
