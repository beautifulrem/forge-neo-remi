/**
 * Stateful mock handlers & capability/advice helpers for Forge Neo Remi
 * Workbench APIs (`VITE_MOCK=1` preview & Playwright E2E):
 * - Cross-model parameter Recipes (`/recipes*`)
 * - Linear multi-stage Pipelines with stage caching (`/chains*`)
 * - Gallery Boards, Asset Lineage & Structured Diff (`/boards*`, `/gallery/{id}/lineage`, `/gallery/compare`)
 * - Extension Diagnostics (`/extensions/diagnostics`)
 * - Inference Worker / Multi-GPU / MLX Status (`/workers*`)
 * - Multimodal Prompt Upsampler (`/prompt-tools/upsample`)
 * - Rules / Advice derivation (`/rules/derive`)
 */
import { diffInfotext } from "@/features/compare/param-diff"
import { computeStageHashes, STAGE_META } from "@/features/pipelines/cache-hash"
import type { ChainTemplate, JobChain, StageRun, StageSpec } from "@/features/pipelines/types"
import type { Recipe } from "@/features/recipes/types"
import { mockAdviceFor, mockCapabilitiesForPreset, normalizeMockPreset } from "./capabilities"
import { bootstrap } from "./fixtures"
import { findLibraryItem } from "./library"
import { conflictPatch, MockHttpError } from "./server"
import { broadcast } from "./socket"

export { mockAdviceFor, mockCapabilitiesForPreset, normalizeMockPreset }

type Handler = (match: RegExpExecArray, body: unknown, url: URL) => unknown | Promise<unknown>
type Values = Record<string, unknown>

// ---------------------------------------------------------------- recipes store

let recipeSeq = 0
let recipes: Recipe[] = []

function seedRecipes(): Recipe[] {
  const now = "2026-11-20T10:00:00.000Z"
  return [
    {
      id: "recipe-anima-balanced",
      name: "Anima 2B · Balanced Illustration",
      description: "ER SDE 30 steps, CFG 4.0, crisp anime lighting preset",
      model_family: "anima",
      capabilities: ["anima_hooks"],
      tab: "txt2img",
      core_patch: { cfg_scale: 4, width: 1024, height: 1024 },
      scripts_patch: { sampler: { sampler_name: "ER SDE", steps: 30, scheduler: "Automatic" } },
      tags: ["anima", "illustration"],
      created_at: now,
      updated_at: now,
    },
    {
      id: "recipe-qwen21-turbo",
      name: "Qwen-Image-2.1 · Turbo 8-Step RGBA",
      description: "Fast distilled 8-step setup with CFG 1.0 for transparent cutouts",
      model_family: "qwen21",
      capabilities: ["distilled_cfg1", "rgba_output", "multi_reference"],
      tab: "txt2img",
      core_patch: { cfg_scale: 1, width: 1024, height: 1024 },
      scripts_patch: { sampler: { sampler_name: "Euler", steps: 8, scheduler: "Simple" } },
      tags: ["turbo", "rgba"],
      created_at: now,
      updated_at: now,
    },
    {
      id: "recipe-flux3-inpaint",
      name: "FLUX.3 Klein · Detail Refine",
      description: "Low-denoise img2img polish pass for faces and hands",
      model_family: "flux3",
      capabilities: ["multi_reference"],
      tab: "img2img",
      core_patch: { cfg_scale: 2.5, denoising_strength: 0.35 },
      scripts_patch: { sampler: { sampler_name: "Euler", steps: 20, scheduler: "Simple" } },
      tags: ["refine", "flux3"],
      created_at: now,
      updated_at: now,
    },
  ]
}

// ---------------------------------------------------------------- chains store

let chainSeq = 0
let templateSeq = 0
let chains: JobChain[] = []
let templates: ChainTemplate[] = []
const stageCache = new Map<string, { outputs: string[]; prompt: string | null; infotext: string | null }>()


function executeMockStages(
  specs: readonly StageSpec[],
  options: { modelFingerprint?: string; fromStage?: number | null; forceStages?: readonly number[] } = {},
): StageRun[] {
  const fp = options.modelFingerprint ?? String(bootstrap.options.sd_model_checkpoint ?? "")
  const hashes = computeStageHashes(specs, fp)
  const forced = new Set(options.forceStages ?? [])
  const fromStage = options.fromStage ?? null
  const now = new Date().toISOString()
  let invalidated = false

  return specs.map((spec, index) => {
    const enabled = spec.enabled !== false
    const hash = hashes[index]!
    const name = spec.name?.trim() || STAGE_META[spec.kind].label
    const tab = spec.tab ?? STAGE_META[spec.kind].defaultTab ?? "txt2img"
    if (!enabled) {
      return {
        ...spec,
        index,
        name,
        enabled: false,
        tab,
        core: spec.core ?? {},
        scripts: spec.scripts ?? {},
        config: spec.config ?? {},
        status: "skipped",
        stage_hash: hash,
        cached: false,
        job_id: null,
        output_file_ids: [],
        output_prompt: null,
        infotext: null,
        error: null,
        started_at: now,
        finished_at: now,
      }
    }
    const mustRun = invalidated || forced.has(index) || (fromStage !== null && index >= fromStage)
    const cachedEntry = !mustRun ? stageCache.get(hash) : undefined
    if (cachedEntry) {
      return {
        ...spec,
        index,
        name,
        enabled: true,
        tab,
        core: spec.core ?? {},
        scripts: spec.scripts ?? {},
        config: spec.config ?? {},
        status: "cached",
        stage_hash: hash,
        cached: true,
        job_id: null,
        output_file_ids: [...cachedEntry.outputs],
        output_prompt: cachedEntry.prompt,
        infotext: cachedEntry.infotext,
        error: null,
        started_at: now,
        finished_at: now,
      }
    }
    invalidated = true
    const hue = (index * 73 + chainSeq * 31) % 360
    const outputs = spec.kind === "prompt_upsample" ? [] : [`mock-chain-${chainSeq}-${index}-${hue}`]
    const outputPrompt =
      spec.kind === "prompt_upsample"
        ? `Expanded detailed composition: ${String(spec.core?.prompt ?? "luminous editorial scene")}`
        : null
    const infotext =
      spec.kind === "prompt_upsample"
        ? null
        : `${String(spec.core?.prompt ?? "pipeline stage")}\nSteps: ${String(spec.scripts?.sampler?.steps ?? 20)}, Sampler: Euler a, CFG scale: ${String(spec.core?.cfg_scale ?? 5)}, Seed: ${1000 + index}`
    stageCache.set(hash, { outputs, prompt: outputPrompt, infotext })
    return {
      ...spec,
      index,
      name,
      enabled: true,
      tab,
      core: spec.core ?? {},
      scripts: spec.scripts ?? {},
      config: spec.config ?? {},
      status: "completed",
      stage_hash: hash,
      cached: false,
      job_id: `job-chain-${chainSeq}-${index}`,
      output_file_ids: outputs,
      output_prompt: outputPrompt,
      infotext,
      error: null,
      started_at: now,
      finished_at: now,
    }
  })
}

// ---------------------------------------------------------------- boards store

export type MockBoard = {
  id: string
  name: string
  description: string
  color: string | null
  file_ids: string[]
  created_at: string
  updated_at: string
}

let boardSeq = 0
let boardStoreVersion = 0
let boards: MockBoard[] = []

function seedBoards(): MockBoard[] {
  const now = "2026-11-20T10:00:00.000Z"
  return [
    {
      id: "board-characters",
      name: "Character references",
      description: "Key poses and expressions for multi-reference runs",
      color: null,
      file_ids: ["mock-lib0-0", "mock-lib1-37", "mock-lib2-74"],
      created_at: now,
      updated_at: now,
    },
    {
      id: "board-finals",
      name: "Approved finals",
      description: "Curated outputs ready for upscale",
      color: null,
      file_ids: ["mock-lib3-111", "mock-lib4-148"],
      created_at: now,
      updated_at: now,
    },
  ]
}

function boardResponse(board: MockBoard) {
  return {
    id: board.id,
    name: board.name,
    description: board.description,
    color: board.color,
    item_count: board.file_ids.length,
    cover_file_id: board.file_ids[0] ?? null,
    created_at: board.created_at,
    updated_at: board.updated_at,
  }
}

/** Current board membership version (used by `library.ts` to invalidate gallery cache). */
export function mockBoardsVersion(): number {
  return boardStoreVersion
}

/** File IDs belonging to `boardId` (empty set if the board does not exist). */
export function mockBoardItemIds(boardId: string): Set<string> {
  const found = boards.find((item) => item.id === boardId)
  return new Set(found ? found.file_ids : [])
}

// ---------------------------------------------------------------- workers & diagnostics store

type MockWorker = {
  worker_id: string
  pid: number | null
  state: string
  backend: string
  device: string
  gpu_name: string | null
  vram_total_mb: number | null
  vram_free_mb: number | null
  loaded_model: string | null
  supported_families: string[]
  quant_formats: string[]
  current_job_id: string | null
  jobs_completed: number
  jobs_failed: number
  restarts: number
  uptime_s: number
  last_heartbeat: string
}

let workers: MockWorker[] = []

function seedWorkers(): MockWorker[] {
  return [
    {
      worker_id: "worker-0",
      pid: 48210,
      state: "idle",
      backend: "forge-pytorch",
      device: "mps:0",
      gpu_name: "Apple Silicon Unified GPU",
      vram_total_mb: 32768,
      vram_free_mb: 24120,
      loaded_model: "animagine-xl-4.0.safetensors [a1b2c3]",
      supported_families: ["anima", "flux3", "qwen21", "moss", "zimage", "xl"],
      quant_formats: ["bf16", "fp8", "int8_convrot", "gguf"],
      current_job_id: null,
      jobs_completed: 14,
      jobs_failed: 0,
      restarts: 0,
      uptime_s: 1940.5,
      last_heartbeat: new Date().toISOString(),
    },
  ]
}

const MOCK_MLX = {
  available: false,
  platform_supported: true,
  mlx_version: null,
  mflux_version: null,
  unified_memory_mb: 32768,
  supported_families: ["flux3", "qwen21", "zimage"],
  reason: "Optional mflux sidecar not installed",
}

const MOCK_DIAGNOSTICS = {
  api_version: "1.0",
  schema_version: 1,
  summary: { total: 4, healthy: 2, degraded: 1, error: 0, disabled: 1, with_manifest: 3 },
  items: [
    {
      name: "sd_forge_image_stitch",
      enabled: true,
      builtin: true,
      health: "healthy",
      load_time_ms: 8.4,
      hook_warnings: 0,
      hook_errors: 0,
      action_errors: 0,
      action_timeouts: 0,
      last_error: null,
      last_error_at: null,
      manifest_status: "valid",
      manifest: {
        api_version: "1.0",
        min_remi_version: "1.0.0",
        requires_capabilities: ["multi_reference"],
        requires_widgets: ["image-list"],
        license: "MIT",
        homepage: "https://github.com/forge-neo/remi",
      },
      manifest_errors: [],
      manifest_warnings: [],
      disabled_scripts: [],
      disabled_reason: null,
    },
    {
      name: "sd_forge_anima_dave",
      enabled: true,
      builtin: true,
      health: "healthy",
      load_time_ms: 14.2,
      hook_warnings: 0,
      hook_errors: 0,
      action_errors: 0,
      action_timeouts: 0,
      last_error: null,
      last_error_at: null,
      manifest_status: "valid",
      manifest: {
        api_version: "1.0",
        requires_capabilities: ["anima_hooks"],
        requires_widgets: [],
        license: "Apache-2.0",
      },
      manifest_errors: [],
      manifest_warnings: [],
      disabled_scripts: [],
      disabled_reason: null,
    },
    {
      name: "sd_forge_spectrum",
      enabled: true,
      builtin: true,
      health: "degraded",
      load_time_ms: 21.7,
      hook_warnings: 2,
      hook_errors: 0,
      action_errors: 0,
      action_timeouts: 0,
      last_error: "Spectrum cache bypassed: incompatible with active CFG-distilled 1-step schedule",
      last_error_at: "2026-11-20T10:15:00.000Z",
      manifest_status: "valid",
      manifest: {
        api_version: "1.0",
        requires_capabilities: ["spectrum"],
        requires_widgets: [],
        license: "MIT",
      },
      manifest_errors: [],
      manifest_warnings: [],
      disabled_scripts: [],
      disabled_reason: null,
    },
    {
      name: "legacy_gradio_plugin",
      enabled: false,
      builtin: false,
      health: "disabled",
      load_time_ms: 1.1,
      hook_warnings: 0,
      hook_errors: 0,
      action_errors: 0,
      action_timeouts: 0,
      last_error: null,
      last_error_at: null,
      manifest_status: "missing",
      manifest: null,
      manifest_errors: [],
      manifest_warnings: [],
      disabled_scripts: [{ name: "legacy.py", reason: "declares ui() instead of params()" }],
      disabled_reason: "Declares ui() (Gradio controls) instead of params()",
    },
  ],
}

/** Resets all stateful mock workbench stores (for isolated unit/E2E tests). */
export function resetMockWorkbench() {
  recipeSeq = 0
  recipes = seedRecipes()
  chainSeq = 0
  templateSeq = 0
  stageCache.clear()
  templates = [] // the server lists saved templates only; the built-in starters live in the SPA
  chains = []
  boardSeq = 0
  boardStoreVersion++
  boards = seedBoards()
  workers = seedWorkers()
}

resetMockWorkbench()

// ---------------------------------------------------------------- lineage & compare helpers

function lineageNodeFor(fileId: string, relation: string, depth: number) {
  const item = findLibraryItem(fileId)
  if (!item) return null
  return {
    file_id: item.id,
    relation,
    depth,
    job_id: item.job_id ?? `job-${item.id}`,
    chain_id: null,
    stage_index: null,
    name: item.name,
    width: item.width,
    height: item.height,
    tab: item.tab ?? item.category ?? "txt2img",
    model: item.model ?? null,
    created_at: item.created_at,
    exists: true,
  }
}

function buildAdoptPatch(rawParams: Record<string, string>) {
  const core: Record<string, unknown> = {}
  const scripts: Record<string, Record<string, unknown>> = {}
  const sampler: Record<string, unknown> = {}
  if (rawParams["CFG scale"] !== undefined) core.cfg_scale = Number(rawParams["CFG scale"])
  if (rawParams["Denoising strength"] !== undefined) core.denoising_strength = Number(rawParams["Denoising strength"])
  const size = /^(\d+)x(\d+)$/.exec(rawParams.Size ?? "")
  if (size) {
    core.width = Number(size[1])
    core.height = Number(size[2])
  }
  if (rawParams.Steps !== undefined) sampler.steps = Number(rawParams.Steps)
  if (rawParams.Sampler) sampler.sampler_name = rawParams.Sampler
  if (rawParams["Schedule type"]) sampler.scheduler = rawParams["Schedule type"]
  if (Object.keys(sampler).length > 0) scripts.sampler = sampler
  if (rawParams.Seed !== undefined) scripts.seed = { seed: Number(rawParams.Seed) }
  return { core, scripts, raw_params: rawParams }
}

// ---------------------------------------------------------------- PUP helper

function mockUpsampleResponse(body: Record<string, unknown>) {
  const rawPrompt = String(body.prompt ?? "")
  if (!rawPrompt.trim()) throw new MockHttpError(422, { detail: "prompt must not be empty" })
  const reqFamily = String(body.model_family ?? "auto").toLowerCase()
  const family = reqFamily && reqFamily !== "auto" ? reqFamily : normalizeMockPreset(String(bootstrap.options.forge_preset ?? "flux3"))
  const refIds = Array.isArray(body.reference_file_ids) ? body.reference_file_ids.map(String) : []
  const refUris = Array.isArray(body.reference_images) ? body.reference_images.map(String) : []
  const refCount = refIds.length + refUris.length
  const reqMode = String(body.mode ?? "auto").toLowerCase()
  const mode = reqMode && reqMode !== "auto" ? reqMode : family === "fibo" ? "vgl_json" : refCount > 0 ? "i2i" : "t2i"
  const aspectRatio = typeof body.aspect_ratio === "string" && body.aspect_ratio.trim() ? body.aspect_ratio.trim() : null
  const base = rawPrompt.trim()
  const ratioNote = aspectRatio ? ` framed for a ${aspectRatio} composition` : ""
  const refNote =
    refCount > 0
      ? family === "qwen21" || family === "moss"
        ? ` preserving visual identity from ${Array.from({ length: refCount }, (_, i) => `<image_${i + 1}>`).join(" and ")}`
        : ` guided by ${Array.from({ length: refCount }, (_, i) => `ref_image_${i}`).join(", ")}`
      : ""
  const instructions = typeof body.instructions === "string" && body.instructions.trim() ? ` (${body.instructions.trim()})` : ""

  let vglJson: Record<string, unknown> | null = null
  let prompt: string
  if (mode === "vgl_json") {
    vglJson = {
      short_description: base,
      objects: [{ description: base, location: "center", relationship: "primary subject", relative_size: "large" }],
      background_setting: "clean studio environment",
      lighting: { conditions: "balanced studio lighting", direction: "front-left key light", shadows: "soft diffused shadows" },
      aesthetics: { composition: aspectRatio ?? "1:1", color_scheme: "natural harmonious palette", mood_atmosphere: "calm and focused" },
      style_medium: "photograph",
    }
    prompt = JSON.stringify(vglJson, null, 2)
  } else if (family === "anima") {
    prompt = `${base}, cinematic lighting, finely detailed textures, crisp focal depth, balanced composition${instructions}`
  } else {
    prompt = `A richly detailed studio-lit photograph of ${base}${ratioNote}${refNote}, captured with crisp micro-contrast, natural material textures, and balanced depth of field.${instructions}`
  }

  return {
    prompt,
    original_prompt: rawPrompt,
    model_family: family,
    mode,
    backend: "mock_vlm",
    used_reference_count: refCount,
    aspect_ratio: aspectRatio,
    layout_rows: mode.includes("bbox") ? [{ caption: base, bbox: [0.1, 0.1, 0.9, 0.9] }] : null,
    vgl_json: vglJson,
    warnings: [],
  }
}

// ---------------------------------------------------------------- routes

export const workbenchRoutes: [method: string, pattern: RegExp, handler: Handler][] = [
  // ---- Rules & Advice derivation
  [
    "POST",
    /^\/rules\/derive$/,
    (_m, body) => {
      const req = (body ?? {}) as { core?: Values; scripts?: Record<string, Values>; selected?: string }
      const core = req.core ?? {}
      const scripts = req.scripts ?? {}
      const patch = conflictPatch(core, scripts, req.selected)
      const preset = String(bootstrap.options.forge_preset ?? "xl")
      return {
        ...patch,
        __advice__: mockAdviceFor(preset, core, scripts, bootstrap.options as Values),
      }
    },
  ],

  // ---- Recipes
  [
    "GET",
    /^\/recipes$/,
    (_m, _body, url) => {
      const q = (url.searchParams.get("q") ?? "").trim().toLowerCase()
      const family = url.searchParams.get("model_family")
      const tab = url.searchParams.get("tab")
      const tag = url.searchParams.get("tag")
      const capability = url.searchParams.get("capability")
      const items = recipes.filter((item) => {
        if (family && item.model_family !== family) return false
        if (tab && item.tab !== tab) return false
        if (tag && !item.tags.includes(tag)) return false
        if (capability && !item.capabilities.includes(capability)) return false
        if (!q) return true
        return `${item.name} ${item.description} ${item.model_family} ${item.tags.join(" ")}`.toLowerCase().includes(q)
      })
      return { items, total: items.length }
    },
  ],
  [
    "POST",
    /^\/recipes$/,
    (_m, body) => {
      const b = (body ?? {}) as Partial<Recipe>
      const name = String(b.name ?? "").trim()
      if (!name) throw new MockHttpError(422, { detail: "recipe name must not be empty" })
      const now = new Date().toISOString()
      const created: Recipe = {
        id: `recipe-mock-${++recipeSeq}`,
        name,
        description: String(b.description ?? ""),
        model_family: String(b.model_family ?? ""),
        capabilities: Array.isArray(b.capabilities) ? b.capabilities.map(String) : [],
        tab: b.tab ?? "txt2img",
        core_patch: b.core_patch ?? {},
        scripts_patch: b.scripts_patch ?? {},
        tags: Array.isArray(b.tags) ? b.tags.map(String) : [],
        created_at: now,
        updated_at: now,
      }
      recipes.unshift(created)
      return created
    },
  ],
  ["GET", /^\/recipes\/export$/, () => ({ version: 1, recipes: structuredClone(recipes) })],
  [
    "POST",
    /^\/recipes\/import$/,
    (_m, body) => {
      const b = (body ?? {}) as { recipes?: Partial<Recipe>[]; overwrite?: boolean }
      const incoming = Array.isArray(b.recipes) ? b.recipes : []
      const now = new Date().toISOString()
      let added = 0
      let updated = 0
      let skipped = 0
      const touched: Recipe[] = []
      for (const raw of incoming) {
        const name = String(raw.name ?? "").trim()
        if (!name) {
          skipped++
          continue
        }
        const id = raw.id ? String(raw.id) : `recipe-import-${++recipeSeq}`
        const idx = recipes.findIndex((r) => r.id === id)
        const next: Recipe = {
          id,
          name,
          description: String(raw.description ?? ""),
          model_family: String(raw.model_family ?? ""),
          capabilities: Array.isArray(raw.capabilities) ? raw.capabilities.map(String) : [],
          tab: raw.tab ?? "txt2img",
          core_patch: raw.core_patch ?? {},
          scripts_patch: raw.scripts_patch ?? {},
          tags: Array.isArray(raw.tags) ? raw.tags.map(String) : [],
          created_at: raw.created_at ?? now,
          updated_at: now,
        }
        if (idx >= 0) {
          if (b.overwrite) {
            recipes[idx] = next
            updated++
            touched.push(next)
          } else {
            skipped++
          }
        } else {
          recipes.unshift(next)
          added++
          touched.push(next)
        }
      }
      return { added, updated, skipped, items: touched }
    },
  ],
  [
    "GET",
    /^\/recipes\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = recipes.find((r) => r.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no recipe ${id}` })
      return found
    },
  ],
  [
    "PATCH",
    /^\/recipes\/([^/]+)$/,
    (m, body) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = recipes.find((r) => r.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no recipe ${id}` })
      const patch = (body ?? {}) as Partial<Recipe>
      if (patch.name !== undefined) {
        const trimmed = String(patch.name).trim()
        if (!trimmed) throw new MockHttpError(422, { detail: "recipe name must not be empty" })
        found.name = trimmed
      }
      if (patch.description !== undefined) found.description = String(patch.description)
      if (patch.model_family !== undefined) found.model_family = String(patch.model_family)
      if (patch.capabilities !== undefined) found.capabilities = patch.capabilities.map(String)
      if (patch.tab !== undefined) found.tab = patch.tab
      if (patch.core_patch !== undefined) found.core_patch = patch.core_patch
      if (patch.scripts_patch !== undefined) found.scripts_patch = patch.scripts_patch
      if (patch.tags !== undefined) found.tags = patch.tags.map(String)
      found.updated_at = new Date().toISOString()
      return found
    },
  ],
  [
    "DELETE",
    /^\/recipes\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const idx = recipes.findIndex((r) => r.id === id)
      if (idx < 0) throw new MockHttpError(404, { detail: `no recipe ${id}` })
      recipes.splice(idx, 1)
      return undefined
    },
  ],
  [
    "POST",
    /^\/recipes\/([^/]+)\/apply$/,
    (m, body) => {
      const id = decodeURIComponent(m[1] ?? "")
      const recipe = recipes.find((r) => r.id === id)
      if (!recipe) throw new MockHttpError(404, { detail: `no recipe ${id}` })
      const b = (body ?? {}) as { tab?: Recipe["tab"]; core?: Values; scripts?: Record<string, Values>; check_conflicts?: boolean }
      const mergedCore: Values = { ...(b.core ?? {}), ...recipe.core_patch }
      const mergedScripts: Record<string, Values> = structuredClone(b.scripts ?? {})
      const patch: Record<string, unknown> = {}
      for (const [key, value] of Object.entries(recipe.core_patch)) patch[`core.${key}`] = { value }
      for (const [owner, vals] of Object.entries(recipe.scripts_patch)) {
        mergedScripts[owner] = { ...(mergedScripts[owner] ?? {}), ...vals }
        for (const [key, value] of Object.entries(vals)) patch[`${owner}.${key}`] = { value }
      }
      const conflicts = b.check_conflicts === false ? {} : conflictPatch(mergedCore, mergedScripts, undefined)
      // as the server: the derived values join the patch and the merged form
      for (const [target, entry] of Object.entries(conflicts)) {
        const dot = target.indexOf(".")
        if (target.startsWith("__") || dot <= 0 || !entry || typeof entry !== "object") continue
        patch[target] = { ...(patch[target] as object | undefined), ...entry }
        if ("value" in entry) {
          const owner = target.slice(0, dot)
          const values = owner === "core" ? mergedCore : (mergedScripts[owner] ??= {})
          values[target.slice(dot + 1)] = (entry as { value: unknown }).value
        }
      }
      const activeCaps = mockCapabilitiesForPreset(String(bootstrap.options.forge_preset ?? "xl")).capabilities
      const missingCaps = recipe.capabilities.filter((cap) => !activeCaps.includes(cap))
      const warnings = missingCaps.length > 0 ? [`Active model lacks capabilities: ${missingCaps.join(", ")}`] : []
      return {
        recipe,
        tab: b.tab ?? recipe.tab,
        patch,
        core: mergedCore,
        scripts: mergedScripts,
        conflicts,
        warnings,
      }
    },
  ],

  // ---- Multi-stage Chains
  [
    "GET",
    /^\/chains$/,
    (_m, _body, url) => {
      const status = url.searchParams.get("status")
      const offset = Number(url.searchParams.get("cursor") ?? 0) || 0
      const limit = Math.min(Number(url.searchParams.get("limit") ?? 50) || 50, 200)
      const filtered = status ? chains.filter((c) => c.status === status) : chains
      const items = filtered.slice(offset, offset + limit)
      return { items, next_cursor: offset + limit < filtered.length ? String(offset + limit) : null, total: filtered.length }
    },
  ],
  [
    "POST",
    /^\/chains\/preview$/,
    (_m, body) => {
      const b = (body ?? {}) as { stages?: StageSpec[]; model_fingerprint?: string }
      const specs = Array.isArray(b.stages) ? b.stages : []
      if (specs.length === 0) throw new MockHttpError(422, { detail: "stages must not be empty" })
      const fp = b.model_fingerprint ?? String(bootstrap.options.sd_model_checkpoint ?? "")
      const hashes = computeStageHashes(specs, fp)
      let invalidated = false
      return {
        stages: specs.map((spec, index) => {
          const enabled = spec.enabled !== false
          const hash = hashes[index]!
          const hit = enabled && !invalidated ? stageCache.get(hash) : undefined
          if (enabled && !hit) invalidated = true
          return {
            index,
            kind: spec.kind,
            name: spec.name?.trim() || STAGE_META[spec.kind].label,
            enabled,
            stage_hash: hash,
            cache_hit: Boolean(hit),
            cached_outputs: hit ? [...hit.outputs] : [],
            cached_prompt: hit?.prompt ?? null,
          }
        }),
      }
    },
  ],
  ["GET", /^\/chains\/templates$/, () => ({ items: templates })],
  [
    "POST",
    /^\/chains\/templates$/,
    (_m, body) => {
      const b = (body ?? {}) as Partial<ChainTemplate>
      const name = String(b.name ?? "").trim()
      if (!name || !Array.isArray(b.stages) || b.stages.length === 0) {
        throw new MockHttpError(422, { detail: "template requires name and at least one stage" })
      }
      const now = new Date().toISOString()
      const created: ChainTemplate = {
        id: `tpl-mock-${++templateSeq}`,
        name,
        description: String(b.description ?? ""),
        model_family: String(b.model_family ?? ""),
        stages: structuredClone(b.stages),
        tags: Array.isArray(b.tags) ? b.tags.map(String) : [],
        created_at: now,
        updated_at: now,
      }
      templates.unshift(created)
      return created
    },
  ],
  [
    "GET",
    /^\/chains\/templates\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = templates.find((t) => t.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no chain template ${id}` })
      return found
    },
  ],
  [
    "DELETE",
    /^\/chains\/templates\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const idx = templates.findIndex((t) => t.id === id)
      if (idx < 0) throw new MockHttpError(404, { detail: `no chain template ${id}` })
      templates.splice(idx, 1)
      return undefined
    },
  ],
  [
    "POST",
    /^\/chains$/,
    (_m, body) => {
      const b = (body ?? {}) as { name?: string; stages?: StageSpec[]; run_now?: boolean; model_fingerprint?: string }
      const specs = Array.isArray(b.stages) ? b.stages : []
      if (specs.length === 0) throw new MockHttpError(422, { detail: "stages must not be empty" })
      ++chainSeq
      const now = new Date().toISOString()
      const stages = executeMockStages(specs, { modelFingerprint: b.model_fingerprint })
      const chain: JobChain = {
        id: `chain-mock-${chainSeq}`,
        name: b.name?.trim() || `Pipeline #${chainSeq}`,
        status: b.run_now === false ? "paused" : "completed",
        current_stage: stages.length - 1,
        total_stages: stages.length,
        stages,
        created_at: now,
        updated_at: now,
        started_at: now,
        finished_at: b.run_now === false ? null : now,
        error: null,
      }
      chains.unshift(chain)
      for (const stage of stages) {
        if (stage.status === "completed" || stage.status === "cached") {
          broadcast({
            type: "chain.stage_completed",
            chain_id: chain.id,
            stage_index: stage.index,
            kind: stage.kind,
            status: stage.status,
            cached: stage.cached,
            output_file_ids: stage.output_file_ids,
          })
        }
      }
      broadcast({ type: "chain.updated", chain_id: chain.id, ...structuredClone(chain) })
      return chain
    },
  ],
  [
    "GET",
    /^\/chains\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = chains.find((c) => c.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no chain ${id}` })
      return found
    },
  ],
  [
    "POST",
    /^\/chains\/([^/]+)\/rerun$/,
    (m, body) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = chains.find((c) => c.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no chain ${id}` })
      const b = (body ?? {}) as { stages?: StageSpec[]; from_stage?: number | null; force_stages?: number[]; model_fingerprint?: string }
      const specs: StageSpec[] = b.stages ?? found.stages
      const now = new Date().toISOString()
      found.stages = executeMockStages(specs, {
        modelFingerprint: b.model_fingerprint,
        fromStage: b.from_stage ?? null,
        forceStages: b.force_stages ?? [],
      })
      found.total_stages = found.stages.length
      found.current_stage = found.stages.length - 1
      found.status = "completed"
      found.updated_at = now
      found.finished_at = now
      found.error = null
      broadcast({ type: "chain.updated", chain_id: found.id, ...structuredClone(found) })
      return found
    },
  ],
  [
    "POST",
    /^\/chains\/([^/]+)\/(pause|resume|cancel)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const action = m[2]
      const found = chains.find((c) => c.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no chain ${id}` })
      if (action === "cancel" && ["completed", "failed", "cancelled"].includes(found.status)) {
        throw new MockHttpError(409, { detail: `chain is already ${found.status}` })
      }
      found.status = action === "pause" ? "paused" : action === "cancel" ? "cancelled" : "completed"
      found.updated_at = new Date().toISOString()
      broadcast({ type: "chain.updated", chain_id: found.id, ...structuredClone(found) })
      return found
    },
  ],
  [
    "DELETE",
    /^\/chains\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const idx = chains.findIndex((c) => c.id === id)
      if (idx < 0) throw new MockHttpError(404, { detail: `no chain ${id}` })
      chains.splice(idx, 1)
      return undefined
    },
  ],

  // ---- Gallery Boards, Lineage & Compare
  ["GET", /^\/boards$/, () => ({ items: boards.map(boardResponse) })],
  [
    "POST",
    /^\/boards$/,
    (_m, body) => {
      const b = (body ?? {}) as { name?: string; description?: string; color?: string | null }
      const name = String(b.name ?? "").trim()
      if (!name) throw new MockHttpError(422, { detail: "board name must not be empty" })
      if (boards.some((item) => item.name.toLowerCase() === name.toLowerCase())) {
        throw new MockHttpError(409, { detail: `board ${name} already exists` })
      }
      const now = new Date().toISOString()
      const created: MockBoard = {
        id: `board-mock-${++boardSeq}`,
        name,
        description: String(b.description ?? ""),
        color: b.color ?? null,
        file_ids: [],
        created_at: now,
        updated_at: now,
      }
      boards.unshift(created)
      boardStoreVersion++
      return boardResponse(created)
    },
  ],
  [
    "GET",
    /^\/boards\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = boards.find((b) => b.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no board ${id}` })
      return boardResponse(found)
    },
  ],
  [
    "PATCH",
    /^\/boards\/([^/]+)$/,
    (m, body) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = boards.find((b) => b.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no board ${id}` })
      const b = (body ?? {}) as { name?: string; description?: string; color?: string | null }
      if (b.name !== undefined) {
        const name = String(b.name).trim()
        if (!name) throw new MockHttpError(422, { detail: "board name must not be empty" })
        if (boards.some((item) => item.id !== id && item.name.toLowerCase() === name.toLowerCase())) {
          throw new MockHttpError(409, { detail: `board ${name} already exists` })
        }
        found.name = name
      }
      if (b.description !== undefined) found.description = String(b.description)
      if (b.color !== undefined) found.color = b.color
      found.updated_at = new Date().toISOString()
      boardStoreVersion++
      return boardResponse(found)
    },
  ],
  [
    "DELETE",
    /^\/boards\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const idx = boards.findIndex((b) => b.id === id)
      if (idx < 0) throw new MockHttpError(404, { detail: `no board ${id}` })
      boards.splice(idx, 1)
      boardStoreVersion++
      return undefined
    },
  ],
  [
    "POST",
    /^\/boards\/([^/]+)\/items$/,
    (m, body) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = boards.find((b) => b.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no board ${id}` })
      const fileIds = Array.isArray((body as { file_ids?: unknown })?.file_ids) ? (body as { file_ids: unknown[] }).file_ids.map(String) : []
      if (fileIds.length === 0) throw new MockHttpError(422, { detail: "file_ids must not be empty" })
      found.file_ids = Array.from(new Set([...found.file_ids, ...fileIds]))
      found.updated_at = new Date().toISOString()
      boardStoreVersion++
      return { board_id: id, count: found.file_ids.length }
    },
  ],
  [
    "POST",
    /^\/boards\/([^/]+)\/items\/remove$/,
    (m, body) => {
      const id = decodeURIComponent(m[1] ?? "")
      const found = boards.find((b) => b.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no board ${id}` })
      const drop = new Set(Array.isArray((body as { file_ids?: unknown })?.file_ids) ? (body as { file_ids: unknown[] }).file_ids.map(String) : [])
      found.file_ids = found.file_ids.filter((fid) => !drop.has(fid))
      found.updated_at = new Date().toISOString()
      boardStoreVersion++
      return { board_id: id, count: found.file_ids.length }
    },
  ],
  [
    "DELETE",
    /^\/boards\/([^/]+)\/items\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const fileId = decodeURIComponent(m[2] ?? "")
      const found = boards.find((b) => b.id === id)
      if (!found) throw new MockHttpError(404, { detail: `no board ${id}` })
      found.file_ids = found.file_ids.filter((fid) => fid !== fileId)
      found.updated_at = new Date().toISOString()
      boardStoreVersion++
      return undefined
    },
  ],
  [
    "GET",
    /^\/gallery\/([^/]+)\/lineage$/,
    (m) => {
      const fileId = decodeURIComponent(m[1] ?? "")
      const self = lineageNodeFor(fileId, "self", 0)
      if (!self) throw new MockHttpError(404, { detail: `no output ${fileId}` })
      const match = /^mock-lib(\d+)-(\d+)$/.exec(fileId)
      const ancestors: NonNullable<ReturnType<typeof lineageNodeFor>>[] = []
      const siblings: NonNullable<ReturnType<typeof lineageNodeFor>>[] = []
      const descendants: NonNullable<ReturnType<typeof lineageNodeFor>>[] = []
      if (match) {
        const idx = Number(match[1])
        if (idx > 0) {
          const prev = lineageNodeFor(`mock-lib${idx - 1}-${((idx - 1) * 37) % 360}`, "init_img", -1)
          if (prev) ancestors.push(prev)
        }
        const sib = lineageNodeFor(`mock-lib${idx + 2}-${((idx + 2) * 37) % 360}`, "sibling", 0)
        if (sib) siblings.push(sib)
        const next = lineageNodeFor(`mock-lib${idx + 1}-${((idx + 1) * 37) % 360}`, "upscale", 1)
        if (next) descendants.push(next)
      }
      return { file_id: fileId, item: self, ancestors, siblings, descendants }
    },
  ],
  [
    "POST",
    /^\/gallery\/compare$/,
    (_m, body) => {
      const b = (body ?? {}) as { id_a?: string | null; id_b?: string | null; infotext_a?: string | null; infotext_b?: string | null }
      if (!b.id_a && !b.id_b && !b.infotext_a && !b.infotext_b) {
        throw new MockHttpError(422, { detail: "provide id_a/id_b or infotext_a/infotext_b" })
      }
      let textA = b.infotext_a ?? ""
      let textB = b.infotext_b ?? ""
      if (b.id_a) {
        const item = findLibraryItem(b.id_a)
        if (!item) throw new MockHttpError(404, { detail: `no output ${b.id_a}` })
        if (!textA) textA = item.infotext ?? ""
      }
      if (b.id_b) {
        const item = findLibraryItem(b.id_b)
        if (!item) throw new MockHttpError(404, { detail: `no output ${b.id_b}` })
        if (!textB) textB = item.infotext ?? ""
      }
      const diff = diffInfotext(textA, textB)
      return {
        ...diff,
        id_a: b.id_a ?? null,
        id_b: b.id_b ?? null,
        infotext_a: textA,
        infotext_b: textB,
        adopt_a_patch: buildAdoptPatch(diff.adopt_a_patch?.raw_params ?? {}),
        adopt_b_patch: buildAdoptPatch(diff.adopt_b_patch?.raw_params ?? {}),
      }
    },
  ],

  // ---- Extension Diagnostics
  ["GET", /^\/extensions\/diagnostics$/, () => structuredClone(MOCK_DIAGNOSTICS)],

  // ---- Workers
  [
    "GET",
    /^\/workers$/,
    () => ({
      mode: "subprocess",
      workers: workers.map((w) => ({
        ...w,
        loaded_model: String(bootstrap.options.sd_model_checkpoint ?? w.loaded_model ?? ""),
        last_heartbeat: new Date().toISOString(),
      })),
      mlx: MOCK_MLX,
    }),
  ],
  [
    "GET",
    /^\/workers\/([^/]+)$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const worker = workers.find((w) => w.worker_id === id)
      if (!worker) throw new MockHttpError(404, { detail: `no worker ${id}` })
      return { ...worker, last_heartbeat: new Date().toISOString() }
    },
  ],
  [
    "POST",
    /^\/workers\/([^/]+)\/restart$/,
    (m) => {
      const id = decodeURIComponent(m[1] ?? "")
      const worker = workers.find((w) => w.worker_id === id)
      if (!worker) throw new MockHttpError(404, { detail: `no worker ${id}` })
      worker.restarts += 1
      worker.uptime_s = 0.1
      worker.state = "idle"
      worker.current_job_id = null
      worker.last_heartbeat = new Date().toISOString()
      return { ok: true, worker: { ...worker } }
    },
  ],

  // ---- Multimodal Prompt Upsampler (PUP)
  ["POST", /^\/prompt-tools\/upsample$/, (_m, body) => mockUpsampleResponse((body ?? {}) as Record<string, unknown>)],
]
