/**
 * A failed request as text for the user, in the UI language. The server's
 * error details are English (they are for logs and API clients), so the
 * pages show a sentence chosen by status code instead; callers pass their
 * own (already translated) sentence for the statuses that mean something
 * specific to them. A 422 that lists the refused values says why: those
 * messages are catalog keys (`validationText`).
 */
import { toast } from "sonner"
import { ApiError, fieldMessages, RequestError, validationErrors } from "@/api/http"
import { i18n, t } from "@/i18n"
import { sentenceText } from "@/ui/messages"
import type { FieldMessage } from "@/ui/schema"

export type StatusTexts = Partial<Record<number, string>>

export function errorText(error: unknown, texts: StatusTexts = {}): string {
  if (!(error instanceof ApiError)) return t("The server cannot be reached.")
  const own = texts[error.status]
  if (own) return own
  switch (error.status) {
    case 401:
      return t("Sign in again to continue.")
    case 403:
      return t("The server does not allow this.")
    case 404:
      return t("It is no longer there.")
    case 409:
      return t("It changed meanwhile; refresh and try again.")
    case 413:
      return t("The file is too large.")
    case 415:
      return t("This file cannot be read as an image.")
    case 422: {
      const errors = validationErrors(error)
      return errors ? validationText(errors) : t("The server rejected the request.")
    }
    case 503:
      return t("The server is busy; try again shortly.")
    default:
      return error.status >= 500 ? t("The server ran into an error ({{status}}).", { status: error.status }) : t("The request failed ({{status}}).", { status: error.status })
  }
}

/**
 * A failure that is either a request (explained by its status, as `errorText`) or the app's own
 * error, whose message is already a sentence in the UI language. A `TypeError` is `fetch` failing
 * to reach the server.
 */
export function failureText(error: unknown, texts: StatusTexts = {}): string {
  return error instanceof Error && !(error instanceof ApiError) && !(error instanceof TypeError) ? error.message : errorText(error, texts)
}

/**
 * A failed prompt tool (translating, writing or upsampling a prompt). A service that failed (502:
 * the translation service, the chat model, a local model) is explained only by its own answer, so
 * that answer follows a sentence in the UI language. A 413 here is text that is too long, a 409 a
 * service that is not set up.
 */
export function serviceFailureText(error: unknown, texts: StatusTexts = {}): string {
  if (error instanceof ApiError && error.status === 502) return t("The service failed: {{detail}}", { detail: error.message })
  return failureText(error, { 409: t("Set up the service in Settings first."), 413: t("The text is too long."), ...texts })
}

/**
 * What a component that failed to render says: a request by its status (the server's detail and the
 * browser's reason are English): one the server answered (`ApiError`) or one that got no answer
 * (`RequestError`: the server down, a chunk that could not be fetched). Anything else is a bug and
 * keeps its own message.
 */
export function renderFailureText(error: Error): string {
  return error instanceof ApiError || error instanceof RequestError ? errorText(error) : error.message
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

/** The "Some parameters are invalid" toast for rejected or missing values (the fields show them inline). */
export function toastInvalid(errors: readonly FieldMessage[]): void {
  toast.error(t("Some parameters are invalid"), { description: validationText(errors) })
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
 * stores and sends them: their templates translate) where it has them, else its error text
 * (`serverText`). A value can be refused only then: a style deleted, an image removed or a model
 * gone while the job waited.
 */
export function jobErrorText(error: string, errors: unknown, language = i18n.language): string {
  const messages = fieldMessages(errors)
  return messages.length > 0 ? validationText(messages, language) : serverText(error, language)
}
