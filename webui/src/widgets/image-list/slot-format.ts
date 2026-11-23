/**
 * Model-aware reference slot badge formatting for multi-reference image lists
 * (e.g. `<image_1>` for Qwen-Image-2.1 / MOSS, `ref_image_0` for FLUX.3 / Fibo,
 * `Picture 0` for Qwen-Image, `Image 1` for generic models).
 */

export type SlotFormatSpec = {
  template: string
  /** Starting index for `{i}` placeholders (`1` for `<image_1>`, `0` for `ref_image_0`). */
  start?: number
}

export function formatSlotToken(index: number, spec?: string | SlotFormatSpec | null): string {
  const template = typeof spec === "string" ? spec.trim() : spec?.template?.trim()
  const explicitStart = typeof spec === "object" && spec !== null ? spec.start : undefined
  if (!template) return `Image ${index + 1}`
  if (template.includes("{i0}")) return template.replace(/\{i0\}/g, String(index))
  if (template.includes("{n}")) return template.replace(/\{n\}/g, String(index + 1))
  if (template.includes("{i1}")) return template.replace(/\{i1\}/g, String(index + 1))
  if (template.includes("{i}")) {
    const start = explicitStart ?? (/(?:^|_)ref(?:_image)?_\{i\}|picture \{i\}/i.test(template) ? 0 : 1)
    return template.replace(/\{i\}/g, String(index + start))
  }
  return `${template} ${index + (explicitStart ?? 1)}`
}

/** Resolves the slot format spec from widget props, schema capability metadata, or active preset/checkpoint. */
export function resolveSlotFormat(options: {
  propFormat?: string | null
  schemaFormat?: string | null
  schemaStart?: number | null
  preset?: string | null
  checkpoint?: string | null
}): SlotFormatSpec {
  if (options.propFormat?.trim()) return { template: options.propFormat.trim(), start: options.schemaStart ?? undefined }
  if (options.schemaFormat?.trim()) return { template: options.schemaFormat.trim(), start: options.schemaStart ?? undefined }
  const candidates = [options.preset, options.checkpoint].filter((v): v is string => typeof v === "string" && v.trim() !== "")
  for (const raw of candidates) {
    const s = raw.toLowerCase()
    if (s === "qwen21" || s.includes("qwen21") || s.includes("qwen-image-2.1") || s.includes("qwen_image_2.1")) {
      return { template: "<image_{i}>", start: 1 }
    }
    if (s === "qwen" || s.includes("qwen-image")) return { template: "Picture {i}", start: 0 }
    if (s === "moss" || s.includes("moss") || s.includes("kromoss")) return { template: "<image_{i}>", start: 1 }
    if (s === "flux3" || s === "klein" || s.includes("flux3") || s.includes("flux-3") || s.includes("flux.3")) {
      return { template: "ref_image_{i}", start: 0 }
    }
    if (s === "fibo" || s.includes("fibo")) return { template: "ref_image_{i}", start: 0 }
  }
  return { template: "Image {n}", start: 1 }
}
