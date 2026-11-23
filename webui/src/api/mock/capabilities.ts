/**
 * Pure mock capability summaries & pre-submit workflow advice helpers
 * (leaf module with zero circular imports into `fixtures.ts` or `server.ts`).
 */
import type { AdviceItem, ModelCapabilitySummary } from "@/ui/schema"

type Values = Record<string, unknown>

type PresetSpec = {
  dit_name: string | null
  engine_label: string
  capabilities: string[]
  max_references: number
  reference_slot_format: string | null
  reference_slot_start: number
  supports_rgba: boolean
  supports_control_blocks: boolean
  attention_backends: string[]
  quant_formats: string[]
  compile_preset: string | null
  is_video: boolean
  frames: number
  use_distill: boolean
  use_shift: boolean
  defaults: { sampler: string; scheduler: string; steps: number; cfg_scale: number }
}

const PRESET_ALIASES: Record<string, string> = {
  sd15: "sd",
  sd1: "sd",
  sdxl: "xl",
  flux2: "klein",
  zimage: "zit",
  "z-image": "zit",
  krea2: "krea",
  "qwen-2.1": "qwen21",
  "qwen2.1": "qwen21",
  "moss-image": "moss",
  kromoss: "moss",
  "flux-3": "flux3",
  "flux.3": "flux3",
}

const STD_ATTN = ["sdpa", "flash", "sage"]
const DIT_QUANTS = ["bf16", "fp8", "scaled_fp8", "int8_convrot", "gguf"]

const PRESET_SPECS: Record<string, PresetSpec> = {
  sd: {
    dit_name: null,
    engine_label: "SD 1.5",
    capabilities: [],
    max_references: 0,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: [...STD_ATTN, "xformers"],
    quant_formats: ["fp16", "bf16", "fp8", "gguf"],
    compile_preset: null,
    is_video: false,
    frames: 1,
    use_distill: false,
    use_shift: false,
    defaults: { sampler: "Euler a", scheduler: "Automatic", steps: 20, cfg_scale: 7 },
  },
  xl: {
    dit_name: null,
    engine_label: "SDXL",
    capabilities: [],
    max_references: 0,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: [...STD_ATTN, "xformers"],
    quant_formats: ["fp16", "bf16", "fp8", "gguf"],
    compile_preset: null,
    is_video: false,
    frames: 1,
    use_distill: false,
    use_shift: false,
    defaults: { sampler: "Euler a", scheduler: "Automatic", steps: 28, cfg_scale: 5 },
  },
  flux: {
    dit_name: null,
    engine_label: "FLUX.1",
    capabilities: ["distilled_cfg1", "multi_reference"],
    max_references: 1,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: ["bf16", "fp8", "gguf", "svdq", "nvfp4"],
    compile_preset: null,
    is_video: false,
    frames: 1,
    use_distill: true,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 20, cfg_scale: 1 },
  },
  klein: {
    dit_name: null,
    engine_label: "FLUX.2 Klein",
    capabilities: ["distilled_cfg1", "multi_reference"],
    max_references: 4,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: ["bf16", "fp8", "gguf"],
    compile_preset: null,
    is_video: false,
    frames: 1,
    use_distill: true,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 20, cfg_scale: 1 },
  },
  qwen: {
    dit_name: null,
    engine_label: "Qwen-Image",
    capabilities: ["distilled_cfg1", "multi_reference"],
    max_references: 4,
    reference_slot_format: "Picture {i}",
    reference_slot_start: 0,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: ["bf16", "fp8", "gguf"],
    compile_preset: null,
    is_video: false,
    frames: 1,
    use_distill: false,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 24, cfg_scale: 4 },
  },
  lumina: {
    dit_name: null,
    engine_label: "Lumina-Image-2.0",
    capabilities: [],
    max_references: 0,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: ["bf16", "fp8", "gguf"],
    compile_preset: null,
    is_video: false,
    frames: 1,
    use_distill: false,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 24, cfg_scale: 4 },
  },
  zit: {
    dit_name: "zimage",
    engine_label: "Z-Image",
    capabilities: ["control_blocks", "distilled_cfg1", "multi_reference", "per_block_compile", "spectrum"],
    max_references: 4,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: true,
    attention_backends: STD_ATTN,
    quant_formats: DIT_QUANTS,
    compile_preset: "Z-Image per-block",
    is_video: false,
    frames: 1,
    use_distill: true,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 8, cfg_scale: 1 },
  },
  wan: {
    dit_name: null,
    engine_label: "Wan 2.2",
    capabilities: ["multi_reference", "video"],
    max_references: 1,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: ["bf16", "fp8", "gguf"],
    compile_preset: null,
    is_video: true,
    frames: 81,
    use_distill: false,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 20, cfg_scale: 5 },
  },
  anima: {
    dit_name: "anima",
    engine_label: "Anima",
    capabilities: ["anima_hooks", "control_blocks", "multi_reference", "per_block_compile", "rgba_output", "spectrum"],
    max_references: 4,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: true,
    supports_control_blocks: true,
    attention_backends: [...STD_ATTN, "masked_sdpa"],
    quant_formats: DIT_QUANTS,
    compile_preset: "Anima per-block",
    is_video: false,
    frames: 1,
    use_distill: false,
    use_shift: true,
    defaults: { sampler: "ER SDE", scheduler: "Automatic", steps: 30, cfg_scale: 4 },
  },
  ernie: {
    dit_name: "ernie",
    engine_label: "ERNIE-Image",
    capabilities: ["distilled_cfg1", "per_block_compile", "spectrum"],
    max_references: 0,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: DIT_QUANTS,
    compile_preset: "ERNIE-Image per-block",
    is_video: false,
    frames: 1,
    use_distill: true,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 8, cfg_scale: 1 },
  },
  pid: {
    dit_name: null,
    engine_label: "PiD",
    capabilities: ["distilled_cfg1"],
    max_references: 0,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: ["bf16", "fp8"],
    compile_preset: null,
    is_video: false,
    frames: 1,
    use_distill: true,
    use_shift: false,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 1, cfg_scale: 1 },
  },
  krea: {
    dit_name: "krea2",
    engine_label: "Krea 2",
    capabilities: ["distilled_cfg1", "multi_reference", "per_block_compile", "spectrum"],
    max_references: 4,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: DIT_QUANTS,
    compile_preset: "Krea 2 per-block",
    is_video: false,
    frames: 1,
    use_distill: false,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 24, cfg_scale: 3.5 },
  },
  qwen21: {
    dit_name: "qwen21",
    engine_label: "Qwen-Image-2.1",
    capabilities: ["control_blocks", "distilled_cfg1", "multi_reference", "per_block_compile", "prefix_kv_cache", "rgba_output", "spectrum"],
    max_references: 10,
    reference_slot_format: "<image_{i}>",
    reference_slot_start: 1,
    supports_rgba: true,
    supports_control_blocks: true,
    attention_backends: STD_ATTN,
    quant_formats: DIT_QUANTS,
    compile_preset: "Qwen-Image-2.1 per-block",
    is_video: false,
    frames: 1,
    use_distill: true,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 8, cfg_scale: 1 },
  },
  moss: {
    dit_name: "moss",
    engine_label: "MOSS-Image",
    capabilities: ["distilled_cfg1", "multi_reference", "per_block_compile", "spectrum", "structured_layout"],
    max_references: 4,
    reference_slot_format: null,
    reference_slot_start: 1,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: DIT_QUANTS,
    compile_preset: "MOSS-Image per-block",
    is_video: false,
    frames: 1,
    use_distill: false,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 24, cfg_scale: 3.5 },
  },
  flux3: {
    dit_name: "flux3",
    engine_label: "FLUX.3",
    capabilities: ["distilled_cfg1", "multi_reference", "per_block_compile", "spectrum", "structured_layout"],
    max_references: 10,
    reference_slot_format: "ref_image_{i}",
    reference_slot_start: 0,
    supports_rgba: false,
    supports_control_blocks: false,
    attention_backends: STD_ATTN,
    quant_formats: DIT_QUANTS,
    compile_preset: "FLUX.3 per-block",
    is_video: false,
    frames: 1,
    use_distill: true,
    use_shift: true,
    defaults: { sampler: "Euler", scheduler: "Simple", steps: 8, cfg_scale: 1 },
  },
}

export function normalizeMockPreset(preset: string | null | undefined): string {
  const key = String(preset ?? "xl").trim().toLowerCase() || "xl"
  return PRESET_ALIASES[key] ?? key
}

/** Builds a `ModelCapabilitySummary` matching `backend/nn/model_capabilities.py`. */
export function mockCapabilitiesForPreset(
  preset: string | null | undefined,
  checkpoint?: string | null,
  dtype?: string | null,
): ModelCapabilitySummary {
  const norm = normalizeMockPreset(preset)
  const spec = PRESET_SPECS[norm] ?? PRESET_SPECS.xl!
  const ckpt = String(checkpoint ?? "").toLowerCase()
  const dt = String(dtype ?? "").toLowerCase()
  const combined = `${ckpt} ${dt}`
  const badges: string[] = []
  if (ckpt.includes("diffusers") || ckpt.includes("model_index.json")) badges.push("Diffusers")
  if (combined.includes("gguf")) badges.push("GGUF")
  else if (combined.includes("nvfp4") || /\bfp4\b/.test(combined)) badges.push("NVFP4")
  else if (combined.includes("svdq") || combined.includes("nunchaku")) badges.push("SVDQuant")
  else if (combined.includes("convrot") || combined.includes("int8")) badges.push("INT8-ConvRot")
  else if (combined.includes("fp8") || combined.includes("float8") || combined.includes("e4m3") || combined.includes("e5m2")) badges.push("FP8")
  if (spec.use_distill || /turbo|distill|lightning|dmd/.test(ckpt)) badges.push("Turbo")
  if (spec.supports_rgba) badges.push("RGBA")
  if (spec.supports_control_blocks) badges.push("Control")

  return {
    preset: norm,
    dit_name: spec.dit_name,
    engine_label: spec.engine_label,
    capabilities: [...spec.capabilities],
    max_references: spec.max_references,
    reference_slot_format: spec.reference_slot_format,
    reference_slot_start: spec.reference_slot_start,
    supports_rgba: spec.supports_rgba,
    supports_control_blocks: spec.supports_control_blocks,
    supports_cfg_guidance: true,
    supports_negative_prompt: true,
    attention_backends: [...spec.attention_backends],
    quant_formats: [...spec.quant_formats],
    compile_preset: spec.compile_preset,
    is_video: spec.is_video,
    frames: spec.frames,
    use_distill: spec.use_distill,
    use_shift: spec.use_shift,
    defaults: { ...spec.defaults },
    badges,
  }
}

/** Derives pre-submit workflow advice items matching `backend/nn/dit_registry.py`. */
export function mockAdviceFor(
  preset: string | null | undefined,
  core: Values = {},
  scripts: Record<string, Values> = {},
  options: Values = {},
): AdviceItem[] {
  const norm = normalizeMockPreset(preset)
  const items: AdviceItem[] = []
  const cfg = Number(core.cfg_scale ?? 5)
  const steps = Number(scripts.sampler?.steps ?? 28)
  const samplerName = String(scripts.sampler?.sampler_name ?? "")

  if (norm === "qwen21") {
    if (steps > 12 || cfg > 1.5) {
      items.push({
        code: "qwen21.turbo_mismatch",
        severity: "warning",
        message: "Qwen-Image-2.1 Turbo works best at 8 steps and CFG 1.0.",
        patch: { "sampler.steps": { value: 8 }, cfg_scale: { value: 1 } },
        action_label: "Set Turbo (8 steps · CFG 1.0)",
      })
    }
    const fmt = String(options.samples_format ?? "png").toLowerCase()
    if (fmt === "jpg" || fmt === "jpeg") {
      items.push({
        code: "qwen21.rgba_jpeg",
        severity: "warning",
        message: "Output format is JPEG; switch to PNG to keep alpha transparency.",
        patch: { __options__: { samples_format: "png" } },
        action_label: "Switch Output to PNG",
      })
    }
  } else if (norm === "flux3" || norm === "flux" || norm === "klein") {
    if (cfg > 1.5) {
      items.push({
        code: `${norm}.distilled_cfg`,
        severity: "warning",
        message: "Distilled FLUX works best with CFG 1.0.",
        patch: { cfg_scale: { value: 1 } },
        action_label: "Set CFG 1.0",
      })
    }
  } else if (norm === "anima") {
    if (steps < 20 || cfg > 6) {
      items.push({
        code: "anima.sampling_range",
        severity: "info",
        message: "Anima recommends 28–36 steps with CFG 3.5–5.0.",
        patch: { "sampler.steps": { value: 30 }, cfg_scale: { value: 4 } },
        action_label: "Use Anima defaults",
      })
    }
  } else if (norm === "moss" || norm === "krea") {
    if (cfg > 4.5) {
      items.push({
        code: `${norm}.cfg_hint`,
        severity: "info",
        message: "Lower CFG (3.0–4.0) preserves color balance on this DiT.",
        patch: { cfg_scale: { value: 3.5 } },
        action_label: "Set CFG 3.5",
      })
    }
  }

  if (samplerName === "LCM" && cfg > 2.5) {
    items.push({
      code: "sampler.lcm_cfg",
      severity: "warning",
      message: "LCM works best at low CFG (1.0–2.0) and 4–8 steps.",
      patch: { cfg_scale: { value: 1.5 }, "sampler.steps": { value: 6 } },
      action_label: "Tune for LCM",
    })
  }
  return items
}
