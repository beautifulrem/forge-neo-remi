/**
 * Lightweight registry of mounted main positive prompt editors by form id
 * (`txt2img`, `img2img`, …) so widgets (e.g. `image-list` reference slot badges)
 * can insert tokens directly at the prompt cursor without importing CodeMirror.
 */
import type { FormStore } from "@/ui/form-store"
import { lockedNames } from "@/ui/locks"
import type { PromptEditorHandle } from "./PromptEditor"

const handles = new Map<string, PromptEditorHandle>()

export function registerActivePrompt(formId: string, handle: PromptEditorHandle | null) {
  if (handle) handles.set(formId, handle)
  else handles.delete(formId)
}

/**
 * Inserts `token` cleanly into the active prompt editor (or appends to the store's `core.prompt`).
 * A lock on the prompt keeps it: the append goes to the store, which keeps it out and names the
 * lock, and the result is false.
 */
export function insertPromptToken(formId: string, token: string, store?: FormStore): boolean {
  const trimmed = token.trim()
  if (!trimmed) return false
  const handle = handles.get(formId)
  const view = handle?.view()
  const locked = store !== undefined && lockedNames(store.getState().locks, store.getState().values, "core").has("prompt")
  if (handle && view && !locked) {
    const { from, to } = view.state.selection.main
    const doc = view.state.doc.toString()
    const before = from > 0 ? doc[from - 1] : ""
    const after = to < doc.length ? doc[to] : ""
    const lead = before && !/[\s([{"']/.test(before) ? " " : ""
    const trail = after && !/[\s,.;:!?)\]}"']/.test(after) ? " " : ""
    handle.insertText(`${lead}${trimmed}${trail}`)
    return true
  }
  if (!store) return false
  const current = store.getState().values.core?.prompt
  const text = typeof current === "string" ? current : ""
  const sep = text.length === 0 || /\s$/.test(text) ? "" : " "
  return store.getState().setValues("core", { prompt: `${text}${sep}${trimmed}` }).length === 0
}
