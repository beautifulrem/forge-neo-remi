/**
 * Importer notes and the notes a job ends with: English messages with `{{name}}` placeholders,
 * translated like the chrome strings. `{{note}}` carries text the server passes on as it is (a
 * note the processing left), translated only where a catalog has it word for word; a list value
 * holds names the catalogs have (capability labels), each translated and joined the language's way.
 */
import type { JobNote } from "@/api/types"
import { i18n, t } from "@/i18n"
import { messageText } from "@/ui/messages"

/** `language`: the render's (`useTranslation().i18n.language`), so a language switch re-renders it. */
export function noteText(note: JobNote | string, language = i18n.language): string {
  if (typeof note === "string") return t(note, { lng: language })
  const values = note.values ?? {}
  if (note.message === "{{note}}" && typeof values.note === "string") return messageText(i18n, values.note, language)
  const list = new Intl.ListFormat(language, { type: "conjunction" })
  const filled = Object.fromEntries(
    Object.entries(values).map(([name, value]) => [name, Array.isArray(value) ? list.format(value.map((item) => messageText(i18n, item, language))) : value]),
  )
  return t(note.message, { ...filled, lng: language })
}
