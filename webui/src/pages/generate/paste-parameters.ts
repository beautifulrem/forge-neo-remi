/** Pasting generation parameters into the form on screen, with feedback toasts. */
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { sentenceParams } from "@/api/http"
import type { FormTab, InfotextParsed } from "@/api/types"
import { applyPaste, pastePlan } from "@/features/infotext/paste"
import { useForm } from "@/ui/hooks"
import { lockNote } from "@/ui/locks"
import { failureText, validationMessage } from "@/features/errors/error-text"
import { noteLines } from "@/lib/toast"

/** Params of other forms or scripts (`"owner.param"`), core ones bare as in the server's patches. */
const refNames = (refs: readonly string[], language: string) =>
  new Intl.ListFormat(language, { type: "conjunction" }).format(refs.map((ref) => (ref.startsWith("core.") ? ref.slice("core.".length) : ref)))

/**
 * Parameters into the form: `text` is parsed by the server for this form
 * (empty text = the last generation's); `image` is read in the browser, or
 * uploaded and read by the server (`/infotext/image`) when the browser finds
 * nothing (formats it does not parse).
 */
export function usePasteParameters() {
  const { t, i18n } = useTranslation()
  const { store, formId } = useForm()
  const tab = formId as FormTab

  const apply = (parsed: InfotextParsed, id: string | number) => {
    const plan = pastePlan(store.getState().index, parsed)
    const { count, locked } = applyPaste({ store, tab }, plan.entries)
    // values a toggle's lock kept as they were, then the pasted params this form does not have
    const note = noteLines(
      lockNote(store.getState().index, locked, i18n.language),
      plan.skipped.length > 0 && t("Not in this form: {{names}}", { names: refNames(plan.skipped, i18n.language) }),
    )
    const kept = note ? { description: note } : {}
    if (count === 0) toast.info(t("No parameters to apply"), { id, ...kept })
    else toast.success(t("Parameters applied: {{count}}", { count }), { id, ...kept })
    const problems = parsed.problems ?? []
    if (problems.length > 0) {
      // each value's message in the UI language, by its template where it names the value
      const text = (p: (typeof problems)[number]) => validationMessage({ message: p.message, template: p.template ?? undefined, params: sentenceParams(p.params) })
      toast.warning(t("Some values were skipped"), { description: problems.slice(0, 4).map((p) => `${p.param ?? p.owner ?? ""}: ${text(p)}`).join("\n") })
    }
  }
  const fail = (id: string | number, error: unknown) =>
    toast.error(t("Could not read the parameters"), { id, description: failureText(error) })

  const fromText = async (text: string) => {
    const id = toast.loading(t("Reading parameters…"))
    try {
      apply(await api.parseInfotext(text, tab), id)
    } catch (error) {
      fail(id, error)
    }
  }

  const fromImage = async (file: File) => {
    const id = toast.loading(t("Reading parameters…"))
    try {
      // A file the browser cannot parse (corrupt EXIF, …) is still worth the server's look.
      // The image parsers load with the first dropped image (not with the form).
      const text = await import("@/features/infotext/read-image-info").then(({ readImageParameters }) => readImageParameters(file)).catch(() => null)
      if (text) return apply(await api.parseInfotext(text, tab), id)
      const upload = await api.upload(file)
      const info = await api.imageInfo(upload.id, tab)
      if (info.parsed) apply(info.parsed, id)
      else toast.info(t("This image has no generation parameters"), { id })
    } catch (error) {
      fail(id, error)
    }
  }

  return { fromText, fromImage }
}
