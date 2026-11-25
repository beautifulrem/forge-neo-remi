/** What the result actions do, without UI: reuse seed, use parameters, send to. */
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import type { FormTab } from "@/api/types"
import { downloadFile } from "@/features/gallery/files"
import { t } from "@/i18n"
import { runChangeEffects } from "@/ui/actions"
import type { FormIndex } from "@/ui/form-index"
import { getFormStore } from "@/ui/form-store"
import { infotextValue } from "@/ui/infotext"
import { lockNote } from "@/ui/locks"
import type { PasteResult } from "@/features/infotext/paste"
import { formOf, type SendItem, type SendTarget, sendTo, sendToReference } from "./send-to"
import { failureText } from "@/features/errors/error-text"

/** The `[owner, name]` of the input that reads infotext key `key` ("Seed" → the seed script's `seed`). */
export function findByInfotext(index: FormIndex, key: string): [owner: string, name: string] | null {
  for (const [owner, nodes] of index.inputs) {
    for (const [name, node] of nodes) if (node.infotext === key) return [owner, name]
  }
  return null
}

/**
 * Sets the form's seed to the one recorded in `infotext`: "missing" when either is missing,
 * "kept" when a lock kept the seed (the write names it).
 */
export function reuseSeed(tab: FormTab, infotext: string | null | undefined): "reused" | "missing" | "kept" {
  const seed = infotextValue(infotext, "Seed")
  const store = getFormStore(tab)
  const target = findByInfotext(store.getState().index, "Seed")
  if (seed === null || !target) return "missing"
  const [owner, name] = target
  if (store.getState().setValues(owner, { [name]: seed }).length > 0) return "kept"
  runChangeEffects({ store, tab }, owner, [name])
  return "reused"
}

const TARGET_LABEL: Record<SendTarget, string> = {
  txt2img: "txt2img",
  img2img: "img2img",
  inpaint: "Inpaint",
  extras: "Extras",
}

/** The values a lock kept from a send, as a toast description (none: nothing to add). */
function lockDescription(target: SendTarget, { locked }: PasteResult): { description?: string } {
  const note = lockNote(getFormStore(formOf(target)).getState().index, locked)
  return note ? { description: note } : {}
}

/** Pastes a result's parameters (all of them, seed and size included) into form `tab`. */
export function applyParameters(tab: SendTarget, infotext: string) {
  const id = toast.loading(t("Reading parameters…"))
  sendTo(tab, { infotext }, {})
    .then((result) => toast.success(t("Parameters applied: {{count}}", { count: result.count }), { id, ...lockDescription(tab, result) }))
    .catch((error: unknown) =>
      toast.error(t("Could not read the parameters"), { id, description: failureText(error) }),
    )
}

/** Runs a send with progress and result toasts. */
export function sendWithFeedback(target: SendTarget, item: SendItem) {
  const where = t(TARGET_LABEL[target])
  const id = toast.loading(t("Sending to {{target}}…", { target: where }))
  sendTo(target, item)
    .then((result) => toast.success(t("Sent to {{target}}", { target: where }), { id, ...lockDescription(target, result) }))
    .catch((error: unknown) =>
      toast.error(t("Could not send to {{target}}", { target: where }), { id, description: failureText(error) }),
    )
}

/** Appends one or more images to the active form's ImageStitch reference slots with toast feedback. */
export function sendToReferenceWithFeedback(items: readonly SendItem[], preferredTab?: "txt2img" | "img2img") {
  sendToReference(items, preferredTab)
    .then((result) => {
      // the values a lock kept (the list: nothing was added)
      const note = result.tab ? lockNote(getFormStore(result.tab).getState().index, result.locked) : undefined
      if (!result.owner) {
        toast.warning(t("No ImageStitch reference slot found on this form"))
      } else if (result.added === 0) {
        if (note) toast.info(note)
        else toast.info(t("All selected images are already in Reference Images ({{total}} total)", { total: result.total }))
      } else {
        toast.success(t("Added {{count}} images to Reference Images ({{total}} total)", { count: result.added, total: result.total }), note ? { description: note } : undefined)
      }
    })
    .catch((error: unknown) => {
      toast.error(t("Could not send to Reference Images"), { description: failureText(error) })
    })
}

/** Copies outputs to the Save folder (`outdir_save`); with `zip`, also downloads them packed. */
export async function saveResults(fileIds: readonly string[], zip = false) {
  const id = toast.loading(t("Saving…"))
  try {
    const result = await api.saveOutputs({ ids: [...fileIds], zip })
    if (zip && result.zip) downloadFile(result.zip)
    toast.success(t("Saved: {{count}}", { count: result.saved.length }), { id })
  } catch (error) {
    toast.error(t("Could not save"), { id, description: failureText(error) })
  }
}
