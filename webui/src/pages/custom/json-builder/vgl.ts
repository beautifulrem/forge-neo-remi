/**
 * Bria Fibo 2.0 VGL 2.0 JSON structured prompt compiler & decompiler for the
 * JSON Builder (matches `backend/nn/fibo.py` `VGL_SCHEMA_VERSION = "2.0"`).
 *
 * Bria Fibo 2.0 weights are licensed under CC BY-NC 4.0 (non-commercial use
 * only without a commercial license from Bria).
 */
import { check, DraftError, isObject } from "./compile"
import type { Box, Label, LayoutDraft } from "./decompile"

export const VGL_SCHEMA_VERSION = "2.0"
export const VGL_LICENSE = "CC BY-NC 4.0"

export const VGL_STYLE_MEDIUMS = [
  "photograph",
  "digital illustration",
  "3D render",
  "anime illustration",
  "oil painting",
] as const

export type VglFields = {
  styleMedium: string
  backgroundSetting: string
  lightingConditions: string
  lightingDirection: string
  lightingShadows: string
  composition: string
  colorScheme: string
  moodAtmosphere: string
  depthOfField: string
  focus: string
  cameraAngle: string
  lensFocalLength: string
}

export const DEFAULT_VGL: VglFields = {
  styleMedium: "photograph",
  backgroundSetting: "studio environment with clean backdrop",
  lightingConditions: "soft balanced studio lighting",
  lightingDirection: "front-left key light",
  lightingShadows: "soft diffused shadows",
  composition: "balanced rule of thirds",
  colorScheme: "natural harmonious palette",
  moodAtmosphere: "calm and focused",
  depthOfField: "moderate depth of field",
  focus: "sharp focus on primary subject",
  cameraAngle: "eye-level",
  lensFocalLength: "50mm standard",
}

export type VglObject = {
  description: string
  location?: string
  relationship?: string
  relative_size?: string
  shape_and_color?: string
}

export type VglDocument = {
  short_description: string
  objects: VglObject[]
  background_setting: string
  lighting: { conditions: string; direction: string; shadows: string }
  aesthetics: { composition: string; color_scheme: string; mood_atmosphere: string }
  photographic_characteristics: { depth_of_field: string; focus: string; camera_angle: string; lens_focal_length: string }
  style_medium: string
  artistic_style?: string
  text_render?: { text: string; description: string; location: string }[]
}

type DraftLike = {
  width: number
  height: number
  caption: string
  meta: string
  extra: string
  items: { label: Label; caption: string; text: string; bbox: Box }[]
  vgl?: Partial<VglFields>
}

const flat = (text: string) => text.replace(/\s+/g, " ").trim()

/** Maps a pixel box center to a canonical spatial phrase (`top-left`, `center`, `bottom-right`, …). */
export function describeBoxLocation(bbox: Box, width: number, height: number): string {
  const cx = (bbox[0] + bbox[2]) / 2 / Math.max(width, 1)
  const cy = (bbox[1] + bbox[3]) / 2 / Math.max(height, 1)
  const v = cy < 0.34 ? "top" : cy > 0.66 ? "bottom" : "middle"
  const h = cx < 0.34 ? "left" : cx > 0.66 ? "right" : "center"
  if (v === "middle" && h === "center") return "center"
  if (v === "middle") return `center-${h}`
  if (h === "center") return `${v}-center`
  return `${v}-${h}`
}

/** Maps a pixel box area ratio to a relative size label (`small`, `medium`, `large`, `dominant`). */
export function describeRelativeSize(bbox: Box, width: number, height: number): string {
  const area = Math.max(0, bbox[2] - bbox[0]) * Math.max(0, bbox[3] - bbox[1])
  const ratio = area / Math.max(width * height, 1)
  if (ratio < 0.08) return "small"
  if (ratio < 0.25) return "medium"
  if (ratio < 0.55) return "large"
  return "dominant"
}

/** Reconstructs an approximate pixel box from a VGL location + relative size pair. */
function boxFromLocation(location: string | undefined, size: string | undefined, width: number, height: number, index: number): Box {
  const loc = (location ?? "").toLowerCase()
  const sz = (size ?? "medium").toLowerCase()
  const frac = sz.includes("small") ? 0.24 : sz.includes("dominant") ? 0.68 : sz.includes("large") ? 0.48 : 0.34
  const bw = Math.max(16, Math.round(width * frac))
  const bh = Math.max(16, Math.round(height * frac))
  let cx = 0.5
  let cy = 0.5
  if (loc.includes("left")) cx = 0.26
  else if (loc.includes("right")) cx = 0.74
  if (loc.includes("top") || loc.includes("upper")) cy = 0.26
  else if (loc.includes("bottom") || loc.includes("lower")) cy = 0.74
  // Offset fallback boxes slightly when no location was provided
  if (!loc) {
    cx = Math.min(0.8, 0.28 + (index % 3) * 0.22)
    cy = Math.min(0.8, 0.32 + Math.floor(index / 3) * 0.24)
  }
  const x1 = Math.max(0, Math.min(width - bw, Math.round(cx * width - bw / 2)))
  const y1 = Math.max(0, Math.min(height - bh, Math.round(cy * height - bh / 2)))
  return [x1, y1, x1 + bw, y1 + bh]
}

/** Builds a schema-valid Bria Fibo 2.0 VGL 2.0 document from the builder draft. */
export function buildVglDocument(state: DraftLike): VglDocument {
  const shortDescription = flat(state.caption)
  check(shortDescription.length > 0, "The summary caption must be nonempty text")
  check(state.items.length > 0, "Add at least one item")
  const vgl: VglFields = { ...DEFAULT_VGL, ...state.vgl }

  const objects: VglObject[] = []
  const textRender: { text: string; description: string; location: string }[] = []

  for (let i = 0; i < state.items.length; i++) {
    const item = state.items[i]!
    const desc = flat(item.caption)
    check(desc.length > 0, `Item ${i + 1} needs a caption`)
    const location = describeBoxLocation(item.bbox, state.width, state.height)
    const relative_size = describeRelativeSize(item.bbox, state.width, state.height)
    if (item.label === "text") {
      const wording = item.text.trim()
      check(wording.length > 0, `Item ${i + 1} needs the text to render`)
      textRender.push({ text: wording, description: desc, location })
      objects.push({
        description: `Text reading "${wording}", ${desc}`,
        location,
        relationship: i === 0 ? "primary subject" : "scene element",
        relative_size,
      })
    } else {
      objects.push({
        description: desc,
        location,
        relationship: item.label === "character" ? "primary character" : i === 0 ? "primary subject" : "supporting object",
        relative_size,
      })
    }
  }

  const meta = state.meta
    .split(",")
    .map((tag) => flat(tag))
    .filter(Boolean)

  const doc: VglDocument = {
    short_description: shortDescription,
    objects,
    background_setting: flat(vgl.backgroundSetting) || DEFAULT_VGL.backgroundSetting,
    lighting: {
      conditions: flat(vgl.lightingConditions) || DEFAULT_VGL.lightingConditions,
      direction: flat(vgl.lightingDirection) || DEFAULT_VGL.lightingDirection,
      shadows: flat(vgl.lightingShadows) || DEFAULT_VGL.lightingShadows,
    },
    aesthetics: {
      composition: flat(vgl.composition) || DEFAULT_VGL.composition,
      color_scheme: flat(vgl.colorScheme) || DEFAULT_VGL.colorScheme,
      mood_atmosphere: flat(vgl.moodAtmosphere) || DEFAULT_VGL.moodAtmosphere,
    },
    photographic_characteristics: {
      depth_of_field: flat(vgl.depthOfField) || DEFAULT_VGL.depthOfField,
      focus: flat(vgl.focus) || DEFAULT_VGL.focus,
      camera_angle: flat(vgl.cameraAngle) || DEFAULT_VGL.cameraAngle,
      lens_focal_length: flat(vgl.lensFocalLength) || DEFAULT_VGL.lensFocalLength,
    },
    style_medium: flat(vgl.styleMedium) || DEFAULT_VGL.styleMedium,
  }

  if (meta.length > 0) doc.artistic_style = meta.join(", ")
  if (textRender.length > 0) doc.text_render = textRender
  return doc
}

/** Compiles the draft to a VGL 2.0 JSON prompt string (plus optional extra line). */
export function finalVglPrompt(state: DraftLike): string {
  const json = JSON.stringify(buildVglDocument(state))
  const extra = state.extra.replace(/\s+/g, " ").trim()
  return extra ? `${json}\n${extra}` : json
}

function splitVglJson(text: string): { doc: Record<string, unknown> | null; extra: string } {
  const trimmed = text.trim()
  if (!trimmed.startsWith("{")) return { doc: null, extra: "" }
  // Fast path: whole text is JSON
  try {
    const parsed: unknown = JSON.parse(trimmed)
    if (isObject(parsed)) return { doc: parsed, extra: "" }
  } catch {
    // Try separating trailing extra line(s)
  }
  const lines = trimmed.split(/\r?\n/)
  for (let cut = lines.length - 1; cut > 0; cut--) {
    const body = lines.slice(0, cut).join("\n").trim()
    const extra = lines.slice(cut).join(" ").trim()
    try {
      const parsed: unknown = JSON.parse(body)
      if (isObject(parsed)) return { doc: parsed, extra }
    } catch {
      // keep searching
    }
  }
  return { doc: null, extra: "" }
}

/** True when `text` is a Bria Fibo 2.0 VGL JSON prompt. */
export function isVglPrompt(text: string): boolean {
  const { doc } = splitVglJson(text)
  if (!doc) return false
  return typeof doc.short_description === "string" && Array.isArray(doc.objects) && ("style_medium" in doc || "lighting" in doc || "aesthetics" in doc)
}

/** Parses a VGL 2.0 JSON prompt back into a JSON Builder draft + VGL field overrides. */
export function decompileVglPrompt(
  text: string,
  width: number,
  height: number,
): { draft: LayoutDraft; vgl: Partial<VglFields>; extra: string } {
  check(Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0, "width and height must be positive numbers")
  const { doc, extra } = splitVglJson(text)
  if (!doc || typeof doc.short_description !== "string" || !Array.isArray(doc.objects)) {
    throw new DraftError("Not a valid Bria Fibo 2.0 VGL JSON prompt")
  }
  const caption = flat(doc.short_description)
  check(caption.length > 0, "VGL short_description must be nonempty text")

  const objects = doc.objects.filter(isObject)
  check(objects.length > 0, "VGL objects array must contain at least one item")

  const items = objects.map((obj, index) => {
    const rawDesc = typeof obj.description === "string" ? flat(obj.description) : ""
    check(rawDesc.length > 0, `Object ${index + 1} needs a description`)
    const location = typeof obj.location === "string" ? obj.location : undefined
    const size = typeof obj.relative_size === "string" ? obj.relative_size : undefined
    const relationship = typeof obj.relationship === "string" ? obj.relationship.toLowerCase() : ""
    const bbox = boxFromLocation(location, size, width, height, index)
    const quoted = /^Text reading "((?:[^"\\]|\\.)*)",\s*/.exec(rawDesc)
    if (quoted) {
      return {
        label: "text" as const,
        text: (quoted[1] ?? "").replace(/\\n/g, "\n"),
        caption: rawDesc.slice(quoted[0].length) || rawDesc,
        bbox,
      }
    }
    const label: Label = relationship.includes("character") || relationship.includes("person") ? "character" : "object"
    return { label, caption: rawDesc, bbox }
  })

  const meta =
    typeof doc.artistic_style === "string"
      ? doc.artistic_style
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean)
      : []

  const lighting = isObject(doc.lighting) ? doc.lighting : {}
  const aesthetics = isObject(doc.aesthetics) ? doc.aesthetics : {}
  const photo = isObject(doc.photographic_characteristics) ? doc.photographic_characteristics : {}
  const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined)

  const vgl: Partial<VglFields> = {
    ...(str(doc.style_medium) ? { styleMedium: str(doc.style_medium) } : {}),
    ...(str(doc.background_setting) ? { backgroundSetting: str(doc.background_setting) } : {}),
    ...(str(lighting.conditions) ? { lightingConditions: str(lighting.conditions) } : {}),
    ...(str(lighting.direction) ? { lightingDirection: str(lighting.direction) } : {}),
    ...(str(lighting.shadows) ? { lightingShadows: str(lighting.shadows) } : {}),
    ...(str(aesthetics.composition) ? { composition: str(aesthetics.composition) } : {}),
    ...(str(aesthetics.color_scheme) ? { colorScheme: str(aesthetics.color_scheme) } : {}),
    ...(str(aesthetics.mood_atmosphere) ? { moodAtmosphere: str(aesthetics.mood_atmosphere) } : {}),
    ...(str(photo.depth_of_field) ? { depthOfField: str(photo.depth_of_field) } : {}),
    ...(str(photo.focus) ? { focus: str(photo.focus) } : {}),
    ...(str(photo.camera_angle) ? { cameraAngle: str(photo.camera_angle) } : {}),
    ...(str(photo.lens_focal_length) ? { lensFocalLength: str(photo.lens_focal_length) } : {}),
  }

  return {
    draft: { width, height, summary: { caption, meta, canvas: "single" }, items },
    vgl,
    extra,
  }
}
