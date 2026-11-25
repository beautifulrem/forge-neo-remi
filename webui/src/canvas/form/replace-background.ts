/**
 * Replacing (or removing) the image a form's layers are painted over: the
 * painted layers go with it (a lock on either keeps both, and the write
 * names it). Their last strokes are uploaded first, so the
 * change can be undone from a toast (the old image and its layers come back
 * from their uploads).
 */
import { toast } from "sonner"
import { t } from "@/i18n"
import { runChangeEffects } from "@/ui/actions"
import type { FormStore, WriteOptions } from "@/ui/form-store"
import { flushPendingWrites } from "@/ui/pending-writes"
import { releaseUpload } from "@/ui/uploads"

export type BackgroundTarget = {
  store: FormStore
  /** Form id (its pending writes are flushed first, its change effects run). */
  formId: string
  owner: string
  /** The background image param. */
  background: string
  /** The params painted over it (cleared with it). */
  painted: readonly string[]
}

/**
 * Writes params of `owner` for the user (change effects run for what was written); returns the
 * names a lock kept, which the write names in a toast.
 */
export function writeValues({ store, formId, owner }: Pick<BackgroundTarget, "store" | "formId" | "owner">, values: Record<string, unknown>, options?: WriteOptions): string[] {
  const kept = store.getState().setValues(owner, values, options)
  const written = kept.length > 0 && options?.whole ? [] : Object.keys(values).filter((name) => !kept.includes(name))
  if (written.length > 0) runChangeEffects({ store, tab: formId }, owner, written)
  return kept
}

/**
 * Sets the background to `id` (null: removes it), clearing the painted layers; undoable when they
 * had paint. False when a lock kept the background as it was: it is not `id`.
 */
export async function replaceBackground(target: BackgroundTarget, id: string | null): Promise<boolean> {
  const { store, formId, owner, background, painted } = target
  // the layers' last strokes only: an upload replacing the image is a pending write itself
  await flushPendingWrites(formId, { values: false }).catch(() => {})
  const current = store.getState().values[owner] ?? {}
  const old = current[background] ?? null
  if (old === id) return true
  const values: Record<string, unknown> = { [background]: id }
  const previous: Record<string, unknown> = { [background]: old }
  let hadPaint = false
  for (const param of painted) {
    values[param] = null
    previous[param] = current[param] ?? null
    if (previous[param]) hadPaint = true
  }
  // a lock on the image (or a layer) keeps all of it: paint stays with the image it was painted over
  if (writeValues(target, values, { whole: true }).length > 0) return false
  const release = () => {
    if (typeof old === "string" && store.getState().values[owner]?.[background] !== old) releaseUpload(old)
  }
  if (!hadPaint) {
    release()
    return true
  }
  // The old image stays referenced by the undo until the toast goes.
  toast(id ? t("Image replaced") : t("Image removed"), {
    description: t("The painted layers were cleared with it."),
    action: { label: t("Undo"), onClick: () => void writeValues(target, previous, { whole: true }) },
    onAutoClose: release,
    onDismiss: release,
  })
  return true
}
