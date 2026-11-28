/**
 * Multimodal Prompt Upsampler (PUP) client and form-context helpers.
 * Talks to `POST /api/v2/prompt-tools/upsample` via the typed OpenAPI client
 * (`api.promptTools.upsample`).
 */
import { api } from "@/api/endpoints"
import type { PromptUpsampled, PromptUpsampleRequest, PromptUpsampleWarning } from "@/api/types"
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

export type PupRequest = PromptUpsampleRequest
export type PupResponse = PromptUpsampled
export type PupWarning = PromptUpsampleWarning

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

/** Calls `POST /api/v2/prompt-tools/upsample` via the typed OpenAPI client. */
export async function upsamplePromptRequest(body: PupRequest, signal?: AbortSignal): Promise<PupResponse> {
  return api.promptTools.upsample(body, signal)
}
