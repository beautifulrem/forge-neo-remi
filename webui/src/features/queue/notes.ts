/** Importer notes are English messages with `{{name}}` placeholders, translated like the chrome strings. */
import type { JobNote } from "@/api/types"
import { i18n, t } from "@/i18n"

/** `language`: the render's (`useTranslation().i18n.language`), so a language switch re-renders it. */
export function noteText(note: JobNote | string, language = i18n.language): string {
  if (typeof note === "string") return t(note, { lng: language })
  return t(note.message, { ...((note.values ?? {}) as Record<string, string | number>), lng: language })
}
