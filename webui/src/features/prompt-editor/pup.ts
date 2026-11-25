/**
 * Multimodal Prompt Upsampler (PUP) client and form-context helpers.
 * Talks to `POST /api/v2/prompt-tools/upsample` (with an offline mock fallback
 * when `VITE_MOCK=1`).
 */
import { MOCK } from "@/api/http"
import { workbenchRequest } from "@/features/pipelines/workbench-http"
import type { FormState } from "@/ui/form-store"

export const PUP_FAMILIES = ["auto", "flux3", "qwen21", "moss", "fibo", "zimage", "ernie", "anima"] as const
export type PupFamily = (typeof PUP_FAMILIES)[number]

/**
 * Model names, shown as they are (not translated). No "auto": that is the picker's own choice,
 * shown as `t("Auto")`, and a result always names the family the server picked.
 */
export const PUP_FAMILY_LABELS: Record<Exclude<PupFamily, "auto">, string> = {
  flux3: "FLUX.3",
  qwen21: "Qwen-2.1",
  moss: "MOSS",
  fibo: "Fibo 2.0",
  zimage: "Z-Image",
  ernie: "ERNIE",
  anima: "Anima",
}

export const PUP_MODES = ["auto", "t2i", "t2i_bbox", "i2i", "i2i_bbox", "vgl_json"] as const
export type PupMode = (typeof PUP_MODES)[number]

export const PUP_MODE_LABELS: Record<PupMode, string> = {
  auto: "Auto",
  t2i: "Text to image",
  t2i_bbox: "Layout (BBox)",
  i2i: "Reference edit",
  i2i_bbox: "Reference + BBox",
  vgl_json: "VGL JSON",
}

export type PupRequest = {
  prompt: string
  model_family?: PupFamily | string
  reference_file_ids?: string[]
  aspect_ratio?: string | null
  mode?: PupMode
  instructions?: string
  negative?: boolean
}

export type PupResponse = {
  prompt: string
  original_prompt: string
  model_family: string
  mode: string
  backend: string
  used_reference_count: number
  aspect_ratio: string | null
  layout_rows: unknown[] | null
  vgl_json: Record<string, unknown> | null
  warnings: string[]
}

/** Detects the best PUP model family from the active preset, checkpoint name, or schema capability family. */
export function detectPupFamily(preset?: string | null, checkpoint?: string | null, capabilityFamily?: string | null): PupFamily {
  const candidates = [capabilityFamily, preset, checkpoint].filter((v): v is string => typeof v === "string" && v.trim() !== "")
  for (const raw of candidates) {
    const s = raw.toLowerCase()
    if (s === "flux3" || s.includes("flux3") || s.includes("flux-3") || s.includes("flux.3")) return "flux3"
    if (s === "qwen21" || s.includes("qwen21") || s.includes("qwen-image-2.1") || s.includes("qwen_image_2.1")) return "qwen21"
    if (s === "moss" || s.includes("moss") || s.includes("kromoss")) return "moss"
    if (s === "fibo" || s.includes("fibo")) return "fibo"
    if (s === "zit" || s === "zimage" || s.includes("z-image") || s.includes("zimage")) return "zimage"
    if (s === "ernie" || s.includes("ernie")) return "ernie"
    if (s === "anima" || s.includes("anima")) return "anima"
  }
  return "auto"
}

function gcd(a: number, b: number): number {
  let x = Math.abs(Math.round(a))
  let y = Math.abs(Math.round(b))
  while (y !== 0) {
    const t = x % y
    x = y
    y = t
  }
  return x || 1
}

/** Turns form dimensions into a concise aspect ratio (`16:9`, `1:1`, or `WxH` for unusual sizes). */
export function formatAspectRatio(width: unknown, height: unknown): string | null {
  const w = Number(width)
  const h = Number(height)
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null
  const d = gcd(w, h)
  const rw = Math.round(w / d)
  const rh = Math.round(h / d)
  if (rw <= 21 && rh <= 21) return `${rw}:${rh}`
  return `${Math.round(w)}x${Math.round(h)}`
}

/** Collects active reference image upload IDs from the form (ImageStitch / any `image_list` + img2img `init_img`). */
export function collectFormReferenceIds(state: Pick<FormState, "values" | "index">): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  const push = (id: unknown) => {
    if (typeof id === "string" && id.trim() !== "" && !seen.has(id)) {
      seen.add(id)
      out.push(id)
    }
  }
  // img2img source image first if present
  push(state.values.core?.init_img)
  for (const [owner, nodes] of state.index.inputs) {
    const ownerValues = state.values[owner]
    if (!ownerValues) continue
    for (const [name, node] of nodes) {
      if (node.type === "image_list") {
        const list = ownerValues[name]
        if (Array.isArray(list)) for (const id of list) push(id)
      }
    }
  }
  return out
}

function mockUpsample(body: PupRequest): PupResponse {
  const family = body.model_family && body.model_family !== "auto" ? body.model_family : "flux3"
  const refs = body.reference_file_ids ?? []
  const mode = body.mode && body.mode !== "auto" ? body.mode : family === "fibo" ? "vgl_json" : refs.length > 0 ? "i2i" : "t2i"
  const base = body.prompt.trim()
  const ratioNote = body.aspect_ratio ? ` framed for a ${body.aspect_ratio} composition` : ""
  const refNote =
    refs.length > 0
      ? family === "qwen21" || family === "moss"
        ? ` preserving visual identity from ${refs.map((_, i) => `<image_${i + 1}>`).join(" and ")}`
        : ` guided by ${refs.map((_, i) => `ref_image_${i}`).join(", ")}`
      : ""
  const note = body.instructions?.trim() ? ` (${body.instructions.trim()})` : ""
  const prompt =
    mode === "vgl_json"
      ? JSON.stringify(
          {
            short_description: base,
            objects: [{ description: base, location: "center", relationship: "primary subject", relative_size: "large" }],
            background_setting: "clean studio environment",
            lighting: { conditions: "balanced studio lighting", direction: "front-left key light", shadows: "soft diffused shadows" },
            aesthetics: { composition: body.aspect_ratio ?? "1:1", color_scheme: "natural harmonious palette", mood_atmosphere: "calm and focused" },
            photographic_characteristics: { depth_of_field: "moderate depth of field", focus: "sharp focus on primary subject", camera_angle: "eye-level", lens_focal_length: "50mm standard" },
            style_medium: "photograph",
          },
          null,
          2,
        )
      : family === "anima"
        ? `${base}, cinematic lighting, finely detailed textures, crisp focal depth, balanced studio composition${note}`
        : `A richly detailed studio-lit photograph of ${base}${ratioNote}${refNote}, captured with crisp micro-contrast, natural material textures, and balanced depth of field.${note}`
  return {
    prompt,
    original_prompt: body.prompt,
    model_family: family,
    mode,
    backend: "mock_vlm",
    used_reference_count: refs.length,
    aspect_ratio: body.aspect_ratio ?? null,
    layout_rows: null,
    vgl_json: mode === "vgl_json" ? (JSON.parse(prompt) as Record<string, unknown>) : null,
    warnings: [],
  }
}

/** Calls `POST /api/v2/prompt-tools/upsample`. */
export async function upsamplePromptRequest(body: PupRequest, signal?: AbortSignal): Promise<PupResponse> {
  if (MOCK) return mockUpsample(body)
  // the shared transport: an expired session opens the sign-in like every other request
  return workbenchRequest<PupResponse>("/prompt-tools/upsample", { method: "POST", body, signal })
}
