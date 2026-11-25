/**
 * Deterministic client-side stage hashing & cache-hit planner for the linear
 * pipeline builder. Mirrors content-addressed upstream chaining: changing any
 * stage's inputs or parameters invalidates that stage and all enabled stages
 * downstream of it, while unchanged upstream stages keep their cached outputs.
 */
import type { ChainPreviewStage, JobChain, StageKind, StageSpec } from "./types"

export const STAGE_KINDS: readonly StageKind[] = ["prompt_upsample", "generate", "refine", "finish"]

export const STAGE_META: Record<StageKind, { label: string; defaultTab?: "txt2img" | "img2img" | "extras" }> = {
  prompt_upsample: { label: "Prompt Upsample" },
  generate: { label: "Base Generate", defaultTab: "txt2img" },
  refine: { label: "Refine / Inpaint", defaultTab: "img2img" },
  finish: { label: "Finish / Upscale / PiD", defaultTab: "extras" },
}

/** Creates a sensible default StageSpec for a given stage kind. */
export function defaultStage(kind: StageKind): StageSpec {
  switch (kind) {
    case "prompt_upsample":
      return {
        kind,
        name: "Prompt Upsample",
        enabled: true,
        config: { mode: "auto", model_family: "auto" },
        core: { prompt: "" },
      }
    case "generate":
      return {
        kind,
        name: "Base Generate",
        enabled: true,
        tab: "txt2img",
        core: { prompt: "", cfg_scale: 5, width: 1024, height: 1024 },
        scripts: { sampler: { steps: 28, sampler_name: "Euler a" } },
      }
    case "refine":
      return {
        kind,
        name: "Refine / Inpaint",
        enabled: true,
        tab: "img2img",
        core: { denoising_strength: 0.35 },
        scripts: { sampler: { steps: 20 } },
      }
    case "finish":
      return {
        kind,
        name: "Finish / Upscale / PiD",
        enabled: true,
        tab: "extras",
        core: { upscaling_resize: 2, upscaler_1: "R-ESRGAN 4x+" },
      }
  }
}

/** Built-in starter stage sequences (2–4 stages). */
export const STARTER_PIPELINES: { id: string; name: string; stages: StageSpec[] }[] = [
  {
    id: "starter-gen-refine-finish",
    name: "Generate → Refine → Upscale",
    stages: [defaultStage("generate"), defaultStage("refine"), defaultStage("finish")],
  },
  {
    id: "starter-pup-full",
    name: "PUP → Generate → Refine → Finish",
    stages: [defaultStage("prompt_upsample"), defaultStage("generate"), defaultStage("refine"), defaultStage("finish")],
  },
  {
    id: "starter-gen-upscale",
    name: "Generate → Upscale",
    stages: [defaultStage("generate"), defaultStage("finish")],
  },
]

/** Stable canonical JSON serializer (sorted object keys) for deterministic hashing. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(",")}}`
}

/** Fast deterministic FNV-1a 64-bit hex digest (browser-safe, synchronous). */
export function fnvHash(input: string): string {
  let h1 = 0x811c9dc5
  let h2 = 0x01000193
  for (let i = 0; i < input.length; i++) {
    const code = input.charCodeAt(i)
    h1 = Math.imul(h1 ^ code, 0x01000193)
    h2 = Math.imul(h2 ^ (code + i), 0x85ebca6b)
  }
  return `${(h1 >>> 0).toString(16).padStart(8, "0")}${(h2 >>> 0).toString(16).padStart(8, "0")}`
}

/**
 * Computes chained deterministic stage hashes across an ordered list of stages,
 * incorporating each previous enabled stage's hash so any upstream tweak
 * automatically invalidates downstream stages.
 */
export function computeStageHashes(stages: readonly StageSpec[], modelFingerprint = ""): string[] {
  const hashes: string[] = []
  let upstream = modelFingerprint
  for (let i = 0; i < stages.length; i++) {
    const stage = stages[i]!
    const enabled = stage.enabled !== false
    const payload = canonicalJson({
      upstream,
      kind: stage.kind,
      tab: stage.tab ?? STAGE_META[stage.kind].defaultTab ?? null,
      core: stage.core ?? {},
      scripts: stage.scripts ?? {},
      config: stage.config ?? {},
    })
    const hash = fnvHash(payload)
    hashes.push(hash)
    if (enabled) upstream = hash
  }
  return hashes
}

export type PlannedStageCache = ChainPreviewStage & {
  decision: "cached" | "execute" | "skipped"
}

/**
 * Plans per-stage cache reuse against an optional previous run or cached hash
 * index (hash -> outputs/prompt). Any mismatch at stage `i` forces stage `i`
 * and every enabled stage after it to `"execute"`.
 */
export function planPipelineCache(
  stages: readonly StageSpec[],
  options: {
    previousRun?: JobChain | null
    knownCache?: ReadonlyMap<string, { outputs: string[]; prompt?: string | null }>
    serverPreview?: readonly ChainPreviewStage[] | null
    modelFingerprint?: string
  } = {},
): PlannedStageCache[] {
  const hashes = computeStageHashes(stages, options.modelFingerprint)
  const prevByHash = new Map<string, { outputs: string[]; prompt?: string | null }>()

  if (options.previousRun) {
    // Re-hash previous run stages with the same client hash function so hashes align
    // even if backend uses sha256 while local edits recompute via fnvHash.
    const prevClientHashes = computeStageHashes(options.previousRun.stages, options.modelFingerprint)
    options.previousRun.stages.forEach((run, idx) => {
      if (run.status !== "completed" && run.status !== "cached") return
      const entry = { outputs: run.output_file_ids ?? [], prompt: run.output_prompt ?? null }
      if (run.stage_hash) prevByHash.set(run.stage_hash, entry)
      const clientHash = prevClientHashes[idx]
      if (clientHash) prevByHash.set(clientHash, entry)
    })
  }
  if (options.knownCache) {
    for (const [k, v] of options.knownCache) prevByHash.set(k, v)
  }

  let upstreamInvalidated = false
  return stages.map((stage, index) => {
    const enabled = stage.enabled !== false
    const serverStage = options.serverPreview?.[index]
    const stage_hash = serverStage?.stage_hash || hashes[index]!
    const name = stage.name || STAGE_META[stage.kind].label

    if (!enabled) {
      return {
        index,
        kind: stage.kind,
        name,
        enabled: false,
        stage_hash,
        cache_hit: false,
        cached_outputs: [],
        cached_prompt: null,
        decision: "skipped",
      }
    }

    const hitFromServer = Boolean(serverStage && serverStage.kind === stage.kind && serverStage.cache_hit)
    // The server's plan decides; the local estimate only fills in until it arrives
    // (it cannot see random seeds, the loaded model or deleted outputs).
    const localHit = serverStage ? undefined : (prevByHash.get(hashes[index]!) ?? prevByHash.get(stage_hash))
    const cache_hit = !upstreamInvalidated && (hitFromServer || Boolean(localHit))
    if (!cache_hit) upstreamInvalidated = true

    return {
      index,
      kind: stage.kind,
      name,
      enabled: true,
      stage_hash,
      cache_hit,
      cached_outputs: hitFromServer ? serverStage!.cached_outputs : (localHit?.outputs ?? []),
      cached_prompt: hitFromServer ? (serverStage!.cached_prompt ?? null) : (localHit?.prompt ?? null),
      decision: cache_hit ? "cached" : "execute",
    }
  })
}

/** First enabled stage index that will execute (0-based), or `null` if every enabled stage hits cache. */
export function firstExecuteStage(plan: readonly PlannedStageCache[]): number | null {
  const found = plan.find((stage) => stage.decision === "execute")
  return found ? found.index : null
}
