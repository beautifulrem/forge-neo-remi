/**
 * Server-made texts shown to the user (text outputs, action toasts, rejected
 * values, model advice). A plain string is translated only when a catalog has
 * it verbatim (arbitrary text is never run through interpolation); a message
 * `{ text, params }` (`remi.ui.message`) is a catalog key whose `{{name}}`
 * placeholders are filled from `params` after translation; a sentence the
 * server filled in (`remi.ui.Sentence`) comes as its English text with
 * `template` and `params`.
 */
import type { i18n as I18n } from "i18next"
import type { SentenceParams } from "./schema"

export type Message = { text: string; params?: Record<string, unknown> }

export function isMessage(value: unknown): value is Message {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false
  const { text, params } = value as Record<string, unknown>
  return typeof text === "string" && (params === undefined || (typeof params === "object" && params !== null && !Array.isArray(params)))
}

/**
 * The display text of a string or message in `language` (`i18n`'s current one when left out). A
 * render passes the language it shows (`useTranslation().i18n.language`): the React Compiler
 * memoizes a render on what the render itself reads, so text made here without it would stay in
 * the old language after a switch.
 */
export function messageText(i18n: Pick<I18n, "t" | "exists" | "language">, value: string | Message, language = i18n.language): string {
  if (typeof value === "string") return value && i18n.exists(value, { lng: language }) ? i18n.t(value, { lng: language }) : value
  // Missing keys fall back to the English text, interpolated all the same; the values go in
  // `replace`, so a param named like an option (`count`, `context`) is only a value
  return i18n.t(value.text, { lng: language, replace: value.params ?? {}, defaultValue: value.text })
}

/**
 * A sentence the server filled in, in `language`: its `template` translated and filled with
 * `params` (a list param's catalog names each translated, joined the language's way); the
 * server's English `text` where the catalogs lack the template (English itself); without a
 * template, `text` by its exact key.
 */
export function sentenceText(
  i18n: Pick<I18n, "t" | "exists" | "language">,
  text: string,
  template?: string | null,
  params?: SentenceParams | null,
  language = i18n.language,
): string {
  if (!template || !params || !i18n.exists(template, { lng: language })) return messageText(i18n, text, language)
  const list = new Intl.ListFormat(language, { type: "conjunction" })
  const values = Object.fromEntries(Object.entries(params).map(([name, value]) => [name, typeof value === "string" ? value : list.format(value.map((item) => messageText(i18n, item, language)))]))
  return messageText(i18n, { text: template, params: values }, language)
}
