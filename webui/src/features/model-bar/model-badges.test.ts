import { describe, expect, it } from "vitest"
import { deriveModelBadges, workersSummary, type WorkersStatus } from "./model-badges"

describe("deriveModelBadges", () => {
  it("combines server capabilities badges with filename and dtype heuristics", () => {
    const badges = deriveModelBadges({
      checkpoint: "qwen-image-2.1-turbo-fp8.safetensors",
      dtype: "int8 (Anima ConvRot)",
      preset: "qwen21",
      capabilities: {
        preset: "qwen21",
        dit_name: "qwen21",
        engine_label: "Qwen-Image-2.1",
        capabilities: ["rgba_output", "multi_reference"],
        max_references: 10,
        reference_slot_format: "<image_{i}>",
        reference_slot_start: 1,
        supports_rgba: true,
        supports_control_blocks: true,
        badges: ["FP8", "Turbo", "RGBA", "BF16"],
      },
    })
    expect(badges.map((b) => b.label)).toEqual(["INT8-ConvRot", "FP8", "Turbo", "RGBA"])
  })

  it("detects Diffusers, GGUF, NVFP4, and SVDQuant formats cleanly", () => {
    expect(deriveModelBadges({ checkpoint: "flux3-klein-diffusers/" }).map((b) => b.label)).toEqual(["Diffusers"])
    expect(deriveModelBadges({ checkpoint: "anima-q4_k.gguf" }).map((b) => b.label)).toEqual(["GGUF"])
    expect(deriveModelBadges({ checkpoint: "model-nvfp4.safetensors" }).map((b) => b.label)).toEqual(["NVFP4"])
    expect(deriveModelBadges({ checkpoint: "model-svdq.safetensors" }).map((b) => b.label)).toEqual(["SVDQuant"])
    expect(deriveModelBadges({ preset: "zit" }).map((b) => b.label)).toEqual(["Turbo"])
  })

  it("returns empty array for plain checkpoints with no special badges", () => {
    expect(deriveModelBadges({ checkpoint: "sd_xl_base.safetensors", dtype: "Automatic", preset: "xl" })).toEqual([])
  })
})

describe("workersSummary", () => {
  it("summarizes single MPS, CUDA, MLX, and multi-GPU pool health", () => {
    const mps: WorkersStatus = { mode: "inprocess", workers: [{ worker_id: "w0", state: "ready", device: "mps" }] }
    expect(workersSummary(mps)).toEqual({ healthy: 1, total: 1, tone: "ok", label: "MPS" })

    const cuda: WorkersStatus = { mode: "subprocess", workers: [{ worker_id: "w0", state: "busy", device: "cuda:0" }] }
    expect(workersSummary(cuda)).toEqual({ healthy: 1, total: 1, tone: "ok", label: "CUDA" })

    const pool: WorkersStatus = {
      mode: "pool",
      workers: [
        { worker_id: "w0", state: "ready", device: "cuda:0" },
        { worker_id: "w1", state: "crashed", device: "cuda:1" },
      ],
    }
    expect(workersSummary(pool)).toEqual({ healthy: 1, total: 2, tone: "warn", label: "1/2 GPU" })

    const down: WorkersStatus = { mode: "subprocess", workers: [{ worker_id: "w0", state: "error", device: "cuda:0" }] }
    expect(workersSummary(down).tone).toBe("error")
  })
})
