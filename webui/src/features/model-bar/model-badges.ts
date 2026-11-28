/**
 * Derives concise Model Bar capability & format metadata badges
 * (`Diffusers`, `INT8-ConvRot`, `FP8`, `NVFP4`, `GGUF`, `SVDQuant`, `Turbo`, `RGBA`)
 * from active `ModelCapabilitySummary.badges` with filename/dtype/preset heuristics as fallback.
 */
import type { MlxStatus, WorkerInfo, WorkersResponse } from "@/api/types"
import type { ModelCapabilitySummary } from "@/ui/schema"

export type { MlxStatus, WorkerInfo, WorkersResponse }

export type ModelBadge = {
  label: string
  /** Semantic category for visual grouping and tooltip context. */
  kind: "format" | "quant" | "variant" | "capability"
}

const KNOWN_BADGE_KIND: Record<string, ModelBadge["kind"]> = {
  Diffusers: "format",
  GGUF: "format",
  "INT8-ConvRot": "quant",
  FP8: "quant",
  NVFP4: "quant",
  SVDQuant: "quant",
  BF16: "quant",
  Turbo: "variant",
  RGBA: "capability",
  Control: "capability",
}

const CANONICAL_ORDER = ["Diffusers", "GGUF", "INT8-ConvRot", "NVFP4", "SVDQuant", "FP8", "Turbo", "RGBA"] as const

export function deriveModelBadges(input: {
  checkpoint?: string | null
  dtype?: string | null
  preset?: string | null
  capabilities?: ModelCapabilitySummary | null
}): ModelBadge[] {
  const found = new Map<string, ModelBadge>()
  const add = (label: string, kind?: ModelBadge["kind"]) => {
    if (!found.has(label)) found.set(label, { label, kind: kind ?? KNOWN_BADGE_KIND[label] ?? "capability" })
  }

  // 1. Server-declared badges from ModelCapabilitySummary (single source of truth)
  for (const raw of input.capabilities?.badges ?? []) {
    const label = raw.trim()
    if (label && label !== "BF16") add(label)
  }

  const ckpt = (input.checkpoint ?? "").toLowerCase()
  const dtype = (input.dtype ?? "").toLowerCase()
  const preset = (input.preset ?? "").toLowerCase()

  // 2. Format & quantization heuristics from checkpoint name / storage dtype
  if (ckpt.includes("diffusers") || ckpt.endsWith("/")) add("Diffusers", "format")
  if (ckpt.endsWith(".gguf") || ckpt.includes(".gguf ") || dtype.includes("gguf")) add("GGUF", "format")

  if (dtype.includes("convrot") || dtype.includes("int8") || ckpt.includes("convrot") || ckpt.includes("int8")) {
    add("INT8-ConvRot", "quant")
  } else if (dtype.includes("nvfp4") || dtype.includes("fp4") || ckpt.includes("nvfp4") || ckpt.includes("fp4")) {
    add("NVFP4", "quant")
  } else if (dtype.includes("svdq") || ckpt.includes("svdq")) {
    add("SVDQuant", "quant")
  } else if (dtype.includes("float8") || dtype.includes("fp8") || ckpt.includes("fp8") || ckpt.includes("e4m3") || ckpt.includes("e5m2")) {
    add("FP8", "quant")
  }

  // 3. Variant & key capability badges
  if (
    preset === "zit" ||
    ckpt.includes("turbo") ||
    ckpt.includes("dmd") ||
    ckpt.includes("lightning") ||
    ckpt.includes("lcm")
  ) {
    add("Turbo", "variant")
  }
  if (input.capabilities?.supports_rgba || preset === "qwen21" || ckpt.includes("rgba")) {
    add("RGBA", "capability")
  }

  // Sort canonically so badge positions stay steady across checkpoint switches
  return [...found.values()].sort((a, b) => {
    const ia = CANONICAL_ORDER.indexOf(a.label as (typeof CANONICAL_ORDER)[number])
    const ib = CANONICAL_ORDER.indexOf(b.label as (typeof CANONICAL_ORDER)[number])
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib)
  })
}

/** Structural input for `workersSummary`: accepts full `WorkersResponse` or partial worker snapshots. */
export type WorkersStatus = Pick<WorkersResponse, "mode"> & {
  workers: readonly (Pick<WorkerInfo, "worker_id"> & {
    state: WorkerInfo["state"] | string
    backend?: WorkerInfo["backend"] | string | null
    device?: WorkerInfo["device"] | string | null
  })[]
  mlx?: Partial<MlxStatus> | null
}

export function workersSummary(status: WorkersStatus): { healthy: number; total: number; tone: "ok" | "warn" | "error"; label: string } {
  const total = status.workers.length
  const healthy = status.workers.filter((w) => w.state === "ready" || w.state === "idle" || w.state === "busy" || w.state === "running").length
  const failed = status.workers.filter((w) => w.state === "error" || w.state === "crashed" || w.state === "stopped").length
  const tone: "ok" | "warn" | "error" = total === 0 || failed === total ? "error" : failed > 0 || healthy < total ? "warn" : "ok"
  if (total > 1) return { healthy, total, tone, label: `${healthy}/${total} GPU` }
  const first = status.workers[0]
  const device = (first?.device ?? "").toLowerCase()
  const short = device.startsWith("cuda") ? "CUDA" : device.startsWith("mps") ? "MPS" : first?.backend === "mlx" ? "MLX" : "GPU"
  return { healthy, total, tone, label: short }
}
