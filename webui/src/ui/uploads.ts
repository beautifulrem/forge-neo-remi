/**
 * Uploads (POST /api/v2/uploads) and their local previews. Images are shown
 * from a local object URL right after upload; ids restored from elsewhere
 * (patches, infotext paste) fall back to the server's /files/{id}.
 *
 * Object URLs pin the file's bytes in memory, so they are revoked when no
 * form references the id any more (checked when an input drops an id and
 * before every new upload), and at most `MAX_PREVIEWS` are kept.
 */
import { useState } from "react"
import { create } from "zustand"
import { api } from "@/api/endpoints"
import type { Upload } from "@/api/types"
import { t } from "@/i18n"
import { showToast } from "@/lib/toast"
import { allFormStores } from "./form-store"

export type UploadInfo = Pick<Upload, "id"> &
  Partial<Omit<Upload, "id">> & {
    url?: string
    /** True when the uploaded image carries an alpha / transparency channel. */
    hasAlpha?: boolean
    /** ms timestamp of the upload */
    at?: number
  }

/**
 * Fast header inspection for PNG (color types 4/6 or tRNS chunk) and WebP
 * (VP8X alpha bit / VP8L lossless) without decoding pixels onto a canvas.
 */
export function detectAlphaFromBytes(bytes: Uint8Array): boolean {
  // PNG signature: 89 50 4E 47 0D 0A 1A 0A; IHDR color type is at byte 25
  if (
    bytes.length >= 26 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  ) {
    const colorType = bytes[25]
    if (colorType === 4 || colorType === 6) return true
    // Scan early chunks for tRNS (indexed transparency)
    for (let i = 33; i + 8 <= bytes.length; ) {
      const length = ((bytes[i]! << 24) | (bytes[i + 1]! << 16) | (bytes[i + 2]! << 8) | bytes[i + 3]!) >>> 0
      const type = String.fromCharCode(bytes[i + 4]!, bytes[i + 5]!, bytes[i + 6]!, bytes[i + 7]!)
      if (type === "tRNS") return true
      if (type === "IDAT" || type === "IEND") break
      i += 12 + length
    }
    return false
  }
  // WebP: RIFF....WEBP + VP8X (flags byte at offset 20, bit 4 = alpha) or VP8L (offset 24 bit 4 = alpha)
  if (
    bytes.length >= 21 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    const chunk = String.fromCharCode(bytes[12]!, bytes[13]!, bytes[14]!, bytes[15]!)
    if (chunk === "VP8X") return Boolean(bytes[20]! & 0x10)
    if (chunk === "VP8L" && bytes.length >= 25) return Boolean(bytes[24]! & 0x10)
  }
  return false
}

async function detectFileAlpha(file: File): Promise<boolean> {
  if (file.type !== "image/png" && file.type !== "image/webp" && !/\.(png|webp)$/i.test(file.name)) return false
  try {
    const slice = file.slice(0, 256)
    const buffer = typeof slice.arrayBuffer === "function" ? await slice.arrayBuffer() : null
    return buffer ? detectAlphaFromBytes(new Uint8Array(buffer)) : false
  } catch {
    return false
  }
}

type UploadState = { byId: Record<string, UploadInfo> }

const useUploads = create<UploadState>()(() => ({ byId: {} }))

/** Local previews kept at most; older ones fall back to the server URL. */
export const MAX_PREVIEWS = 24
/** A fresh upload is not yet in a form (its input stores the id after the whole batch). */
const GRACE_MS = 10_000

/** Whether any value of any form store contains the upload id (values are JSON). */
function referenced(id: string): boolean {
  const contains = (value: unknown): boolean =>
    value === id ||
    (Array.isArray(value) ? value.some(contains) : value !== null && typeof value === "object" && Object.values(value).some(contains))
  for (const store of allFormStores()) {
    if (Object.values(store.getState().values).some((owner) => Object.values(owner).some(contains))) return true
  }
  return false
}

/**
 * Revokes the previews of `ids` that no form references (all previews older
 * than the grace period when omitted), then caps the rest.
 */
function sweep(ids?: readonly string[]) {
  const byId = useUploads.getState().byId
  const now = Date.now()
  const candidates = ids ?? Object.keys(byId).filter((id) => now - (byId[id]?.at ?? 0) > GRACE_MS)
  const revoked = new Set(candidates.filter((id) => byId[id]?.url && !referenced(id)))
  // Insertion order is upload order: the oldest previews go first.
  const live = Object.keys(byId).filter((id) => byId[id]?.url && !revoked.has(id))
  for (const id of live.slice(0, Math.max(0, live.length - MAX_PREVIEWS))) revoked.add(id)
  if (revoked.size === 0) return
  const next = { ...byId }
  for (const id of revoked) {
    const info = next[id]
    if (!info?.url) continue
    URL.revokeObjectURL(info.url)
    // Keep the metadata (file name); the preview falls back to /files/{id}.
    next[id] = { ...info, url: undefined }
  }
  useUploads.setState({ byId: next })
}

/** Drops the local preview of an upload once no form uses it (after the caller's state update). */
export function releaseUpload(id: string) {
  queueMicrotask(() => sweep([id]))
}

export async function uploadFile(file: File): Promise<UploadInfo> {
  const [result, hasAlpha] = await Promise.all([api.upload(file), detectFileAlpha(file)])
  sweep()
  const serverAlpha = (result as { has_alpha?: boolean; mode?: string }).has_alpha === true || (result as { mode?: string }).mode === "RGBA"
  const info: UploadInfo = {
    ...result,
    name: result.name ?? file.name,
    url: file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
    hasAlpha: hasAlpha || serverAlpha || undefined,
    at: Date.now(),
  }
  useUploads.setState((s) => ({ byId: { ...s.byId, [info.id]: info } }))
  return info
}

/** Records alpha-channel presence on an upload entry (e.g. from image metadata or server info). */
export function markUploadAlpha(id: string, hasAlpha: boolean) {
  const current = useUploads.getState().byId[id]
  if (current?.hasAlpha === hasAlpha) return
  useUploads.setState((s) => ({ byId: { ...s.byId, [id]: { ...s.byId[id], id, hasAlpha } } }))
}

/**
 * Records the pixel size of a file this tab did not upload (a result or
 * gallery image sent to a form), from the output's metadata: helpers that
 * need the size (the "Resize by" result size) then know it without loading
 * the image.
 */
export function describeFile(id: string, size: { width?: number | null; height?: number | null }) {
  const width = size.width ?? 0
  const height = size.height ?? 0
  if (!(width > 0 && height > 0)) return
  const current = useUploads.getState().byId[id]
  if (current?.width === width && current.height === height) return
  useUploads.setState((s) => ({ byId: { ...s.byId, [id]: { ...s.byId[id], id, width, height } } }))
}

export function useUploadInfo(id: string | null | undefined): UploadInfo | null {
  const info = useUploads((s) => (id ? s.byId[id] : undefined))
  if (!id) return null
  return info?.url ? info : { ...info, id, url: api.fileUrl(id) }
}

export function imageFiles(list: FileList | readonly File[] | null | undefined): File[] {
  return Array.from(list ?? []).filter((file) => file.type.startsWith("image/"))
}

/** Uploads files, tracking busy state and reporting failures as toasts. */
export function useUploader() {
  const [busy, setBusy] = useState(0)
  const upload = async (files: readonly File[]): Promise<UploadInfo[]> => {
    if (files.length === 0) return []
    setBusy((n) => n + 1)
    try {
      const settled = await Promise.allSettled(files.map(uploadFile))
      const failed = settled.filter((r) => r.status === "rejected")
      if (failed.length > 0) {
        const reason = (failed[0] as PromiseRejectedResult).reason
        showToast("error", t("Upload failed"), reason instanceof Error ? reason.message : String(reason))
      }
      return settled.flatMap((r) => (r.status === "fulfilled" ? [r.value] : []))
    } finally {
      setBusy((n) => n - 1)
    }
  }
  return { upload, busy: busy > 0 }
}

/** Test hook: the preview URL currently held for an id. */
export const previewUrlOf = (id: string) => useUploads.getState().byId[id]?.url
