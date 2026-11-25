/**
 * A failed request as text for the user, in the UI language. The server's
 * error details are English (they are for logs and API clients), so the
 * pages show a sentence chosen by status code instead; callers pass their
 * own (already translated) sentence for the statuses that mean something
 * specific to them. A refused queue, pipeline or recipe operation says why
 * with the sentence its body carries (`template`, a catalog key, and
 * `params`): it comes before the caller's sentence for its status, except
 * for a 404, whose sentence names what is gone by its id ("no recipe 'r1'")
 * where the caller knows it in the user's words. A 422 that lists the
 * refused values says why: those messages are catalog keys
 * (`validationText`). A text shown by a render takes the render's language
 * (`useTranslation().i18n.language`): the React Compiler keeps a render's
 * text until the values it read change.
 */
import { toast } from "sonner"
import { ApiError, fieldMessages, RequestError, sentenceParams, validationErrors } from "@/api/http"
import { i18n, t } from "@/i18n"
import { sentenceText } from "@/ui/messages"
import type { FormIndex } from "@/ui/form-index"
import type { FieldMessage, RefusedValue, ValidationError } from "@/ui/schema"

export type StatusTexts = Partial<Record<number, string>>

export function errorText(error: unknown, texts: StatusTexts = {}, language = i18n.language): string {
  const lng = language
  if (!(error instanceof ApiError)) return t("The server cannot be reached.", { lng })
  const own = texts[error.status]
  // what is gone, in the caller's words: the body names it by its id
  if (own && error.status === 404) return own
  const sentence = bodySentence(error, language)
  if (sentence !== null) return sentence
  if (own) return own
  switch (error.status) {
    case 401:
      return t("Sign in again to continue.", { lng })
    case 403:
      return t("The server does not allow this.", { lng })
    case 404:
      return t("It is no longer there.", { lng })
    case 409:
      return t("It changed meanwhile; refresh and try again.", { lng })
    case 413:
      return t("The file is too large.", { lng })
    case 415:
      return t("This file cannot be read as an image.", { lng })
    case 422: {
      const errors = validationErrors(error)
      return errors ? validationText(errors, language) : t("The server rejected the request.", { lng })
    }
    case 503:
      return t("The server is busy; try again shortly.", { lng })
    default:
      return error.status >= 500 ? t("The server ran into an error ({{status}}).", { status: error.status, lng }) : t("The request failed ({{status}}).", { status: error.status, lng })
  }
}

/**
 * The sentence a refused request's body carries (`remi.ui.Sentence`: `{detail, template, params}`),
 * in the UI language; null for any other body. It says more than any reading of the status: a
 * recipe name already taken is a 409, as a job that started meanwhile is. A 404's names the thing
 * by its id, so a caller's own 404 sentence comes first (`errorText`).
 */
function bodySentence(error: ApiError, language: string): string | null {
  const body = error.detail
  if (!body || typeof body !== "object") return null
  const { detail, template, params } = body as { detail?: unknown; template?: unknown; params?: unknown }
  if (typeof template !== "string" || !template) return null
  // the English sentence where the catalogs lack the template (English itself)
  return sentenceText(i18n, typeof detail === "string" && detail ? detail : error.message, template, sentenceParams(params) ?? {}, language)
}

/**
 * A failure that is either a request (explained by its status, as `errorText`) or the app's own
 * error, whose message is already a sentence in the UI language. A request that got no answer is a
 * `RequestError` (marked where it is made, `api/http.ts`); any other `TypeError` is a bug and keeps
 * its message, as a bug does in an error boundary.
 */
export function failureText(error: unknown, texts: StatusTexts = {}, language = i18n.language): string {
  return error instanceof Error && !(error instanceof ApiError) && !(error instanceof RequestError) ? error.message : errorText(error, texts, language)
}

/**
 * A failed prompt tool (translating, writing or upsampling a prompt). A service that failed (502:
 * the translation service, the chat model, a local model) is explained only by its own answer, so
 * that answer follows a sentence in the UI language. A 413 here is text that is too long, a 409 a
 * service that is not set up.
 */
export function serviceFailureText(error: unknown, texts: StatusTexts = {}, language = i18n.language): string {
  const lng = language
  if (error instanceof ApiError && error.status === 502) return t("The service failed: {{detail}}", { detail: error.message, lng })
  return failureText(error, { 409: t("Set up the service in Settings first.", { lng }), 413: t("The text is too long.", { lng }), ...texts }, language)
}

/**
 * What a component that failed to render says: a request by its status (the server's detail and the
 * browser's reason are English): one the server answered (`ApiError`) or one that got no answer
 * (`RequestError`: the server down, a chunk that could not be fetched). Anything else is a bug and
 * keeps its own message.
 */
export function renderFailureText(error: Error, language = i18n.language): string {
  return error instanceof ApiError || error instanceof RequestError ? errorText(error, {}, language) : error.message
}

/**
 * A rejected value's message in the UI language: a catalog key (`core.required`, a setting's
 * reason), or the template of one that names the value ("must be at most {{maximum}}, got
 * {{value}}"). Text the server only passes on (a script's own exception) stays as it is.
 */
export function validationMessage(error: FieldMessage, language = i18n.language): string {
  return sentenceText(i18n, error.message, error.template, error.params, language)
}

/** Rejected values in the UI language: each distinct message once, as under its field. */
export function validationText(errors: readonly FieldMessage[], language = i18n.language): string {
  return [...new Set(errors.map((error) => validationMessage(error, language)))].join(" · ")
}

/**
 * The messages under one field (none: undefined) in `language`, the one the render shows
 * (`useTranslation().i18n.language`; see `messageText`).
 */
export function fieldErrorText(errors: readonly FieldMessage[] | undefined, language: string): string | undefined {
  return errors?.length ? validationText(errors, language) : undefined
}

/**
 * The "Some parameters are invalid" toast for rejected or missing values (the fields show them
 * inline). Each message is named by its field's label in `index` (the form's), since the field
 * may be in a closed section; one the form does not show (a script no longer loaded) is the message
 * alone.
 */
export function toastInvalid(errors: readonly ValidationError[], index?: FormIndex): void {
  const language = i18n.language
  const lines = errors.map((error) => fieldMessage(error, index, language))
  toast.error(t("Some parameters are invalid"), { description: [...new Set(lines)].join(" · ") })
}

/**
 * A refused value's message in `language`, named by its field's label in `index` ("Sampling
 * steps: must be at most 150, got 200"); the message alone when the entry names no field or the
 * form does not show it (a script no longer loaded).
 */
function fieldMessage(error: RefusedValue, index: FormIndex | undefined, language: string): string {
  const message = validationMessage(error, language)
  const label = error.owner !== undefined && error.param !== undefined ? index?.inputs.get(error.owner)?.get(error.param)?.label : undefined
  return label ? t("{{field}}: {{message}}", { field: t(label, { lng: language }), message, lng: language }) : message
}

/**
 * A text the server stored or sent as it is: a fixed sentence (how a job or a model load ended,
 * `remi/jobs/texts.py`) in the UI language, found by its exact key in the catalogs; any other text
 * (a job's own exception, a diagnostic) as it is, never read as a template.
 */
export function serverText(text: string, language = i18n.language): string {
  return i18n.exists(text, { lng: language }) ? t(text, { lng: language }) : text
}

/**
 * Why a job failed, in `language`: the values it refused when it ran (`errors`, as the server
 * stores and sends them: their templates translate), each named by its field's label in `index`
 * (the job's form) as the toast of a refused submission names them, where it has them; else its
 * error text (`serverText`). A value can be refused only then: a style deleted, an image removed
 * or a model gone while the job waited.
 */
export function jobErrorText(error: string, errors: unknown, language = i18n.language, index?: FormIndex): string {
  const messages = fieldMessages(errors)
  return messages.length > 0 ? [...new Set(messages.map((message) => fieldMessage(message, index, language)))].join(" · ") : serverText(error, language)
}
