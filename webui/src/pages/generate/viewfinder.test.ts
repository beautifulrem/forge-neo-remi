import { describe, expect, it } from "vitest"
import { viewfinderMetrics } from "./viewfinder"

describe("viewfinderMetrics", () => {
  it("computes simplified aspect ratio and megapixels for standard resolutions", () => {
    expect(viewfinderMetrics(1024, 1024)).toEqual({ width: 1024, height: 1024, ratio: "1:1", megapixels: "1.05 MP" })
    expect(viewfinderMetrics(1280, 720)).toEqual({ width: 1280, height: 720, ratio: "16:9", megapixels: "0.92 MP" })
    expect(viewfinderMetrics(1024, 1536)).toEqual({ width: 1024, height: 1536, ratio: "2:3", megapixels: "1.57 MP" })
    expect(viewfinderMetrics(1344, 768)).toEqual({ width: 1344, height: 768, ratio: "7:4", megapixels: "1.03 MP" })
  })

  it("falls back to clean decimal ratios when simplified terms are large", () => {
    expect(viewfinderMetrics(1000, 777)).toEqual({ width: 1000, height: 777, ratio: "1.29:1", megapixels: "0.78 MP" })
    expect(viewfinderMetrics(777, 1000)).toEqual({ width: 777, height: 1000, ratio: "1:1.29", megapixels: "0.78 MP" })
  })

  it("clamps and defaults missing or non-positive values safely", () => {
    expect(viewfinderMetrics(undefined, null)).toEqual({ width: 1024, height: 1024, ratio: "1:1", megapixels: "1.05 MP" })
    expect(viewfinderMetrics(-50, 0)).toEqual({ width: 1024, height: 1024, ratio: "1:1", megapixels: "1.05 MP" })
    expect(viewfinderMetrics("832", "1216")).toEqual({ width: 832, height: 1216, ratio: "13:19", megapixels: "1.01 MP" })
  })
})
