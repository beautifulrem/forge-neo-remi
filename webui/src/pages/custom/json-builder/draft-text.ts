/**
 * Why the JSON Builder refuses a draft or a prompt to load, in the UI language
 * (docs/remi-webui/frontend.md, "Failure text").
 */
import { failureText } from "@/features/errors/error-text"
import { i18n } from "@/i18n"
import { DraftError } from "./compile"

/**
 * `error` in `language`: a {@link DraftError}'s template translated and filled with its values
 * (paths such as `items[2].bbox` as written, a `word` translated); a text that is not JSON in one
 * sentence (the browser's reason is English); any other failure as `failureText` says it. A render
 * passes the language it shows (`useTranslation().i18n.language`).
 */
export function draftErrorText(error: unknown, language = i18n.language): string {
  const lng = language
  if (error instanceof DraftError) {
    const replace = Object.fromEntries(Object.entries(error.values).map(([name, value]) => [name, typeof value === "object" ? i18n.t(value.word, { lng }) : value]))
    // the values go in `replace`: a value is data, never an option of t()
    return i18n.t(error.template, { lng, replace, defaultValue: error.template })
  }
  if (error instanceof SyntaxError) return i18n.t("The text is not valid JSON.", { lng })
  return failureText(error, {}, lng)
}
