/**
 * The display parse of an infotext runs on every gallery card (its label) and
 * in the compare panel, on text an image carried. Some runs made the
 * parameter regex try every start to the end of the line: a 200,000-character
 * line froze the tab for seconds.
 */
import { describe, expect, it } from "vitest"
import { diffInfotext } from "@/features/compare/param-diff"
import { parseParamLine, splitInfotext } from "./display"

const RUNS = ["word ", "a", "a-", "a:", " ", "\t", "-/", ":", 'k: "', 'k: "\\"', "<lora:a", "<lyco:a:"]

describe("infotext display parse", () => {
  it.each(RUNS)("reads a long run of %j in linear time", (unit) => {
    const run = unit.repeat(Math.ceil(120_000 / unit.length))
    const start = performance.now()
    parseParamLine(run)
    splitInfotext(`${run}\nNegative prompt: ${run}\nSteps: 20, ${run}`)
    diffInfotext(`${run}\nSteps: 20`, "a dog\nSteps: 30")
    expect(performance.now() - start).toBeLessThan(500)
  })

  it("reads ordinary parameter lines as before", () => {
    expect(parseParamLine('Steps: 20, Sampler: Euler a, CFG scale: 7, Size: 832x1216, Lora hashes: "style_a: abc, b: def", ADetailer model 2nd: face_yolov8n.pt, Hires upscaler: R-ESRGAN 4x+, Version: f2.0.1')).toEqual([
      ["Steps", "20"],
      ["Sampler", "Euler a"],
      ["CFG scale", "7"],
      ["Size", "832x1216"],
      ["Lora hashes", "style_a: abc, b: def"],
      ["ADetailer model 2nd", "face_yolov8n.pt"],
      ["Hires upscaler", "R-ESRGAN 4x+"],
      ["Version", "f2.0.1"],
    ])
  })
})
