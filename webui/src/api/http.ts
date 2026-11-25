/**
 * HTTP transport for /api/v2: the typed client generated from the server's
 * /openapi.json (`gen/schema.d.ts`, `pnpm gen:api`) and the error mapping.
 * Callers use the functions in `endpoints.ts`, never URLs.
 */
import createClient from "openapi-fetch"
import type { FieldMessage, SentenceParams, ValidationError } from "@/ui/schema"
import type { paths } from "./gen/schema"
import { sessionRefused } from "./session"

declare global {
  interface Window {
    /** The server's root path under `--subpath` (index.html sets it; absent at the root). */
    __REMI_ROOT__?: string
  }
}

/** The app's mount path ("" at the server root, "/sd" under `--subpath /sd`). */
export const ROOT = (typeof window === "undefined" ? "" : (window.__REMI_ROOT__ ?? "")).replace(/\/+$/, "")

/** Path prefix of every API request (also the WebSocket's). */
export const API_BASE = `${ROOT}/api/v2`

/** Set at build time: `VITE_MOCK=1 pnpm dev` serves fixtures instead of the server. */
export const MOCK = import.meta.env.VITE_MOCK === "1"

export class ApiError extends Error {
  readonly status: number
  /** The decoded error body (see `validationErrors`). */
  readonly detail: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
  }
}

/**
 * A request that got no answer the app can read: the server could not be reached (`fetch` failed:
 * stopped, restarting, offline) or answered with a body that is not JSON (a proxy's error page). A
 * `TypeError`, as `fetch` failing is, so every check for an unreachable server holds; error views
 * explain it by `errorText`, never by its (browser's English) message.
 */
export class RequestError extends TypeError {
  constructor(cause: unknown) {
    super("the request got no answer", { cause })
    this.name = "RequestError"
  }
}

/**
 * For a request that removes something (`.catch(alreadyGone)`): a 404 means it is gone already (a
 * second press before the list caught up, another tab), which is what was asked, not a failure.
 * Any other failure is thrown on.
 */
export function alreadyGone(error: unknown): void {
  if (!(error instanceof ApiError && error.status === 404)) throw error
}

/** The caller gave up on the request (an `AbortSignal`): not a failure to explain. */
export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError"
}

/** `error`, marked as a {@link RequestError} unless it is an answer (`ApiError`), already marked or an abort. */
export function requestFailure(error: unknown): unknown {
  return error instanceof ApiError || error instanceof RequestError || isAbort(error) ? error : new RequestError(error)
}

/** `fetch`, with a failure to get an answer marked as a {@link RequestError}. */
export async function request(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init)
  } catch (error) {
    throw requestFailure(error)
  }
}

export type Query = Record<string, string | number | boolean | null | undefined>

/** An API URL for `<img>`, `<a>` and `<audio>` (`path` is relative to /api/v2). */
export function apiUrl(path: string, query?: Query): string {
  let url = `${API_BASE}${path}`
  const qs = query ? new URLSearchParams(compactQuery(query)).toString() : ""
  if (qs) url += `?${qs}`
  return url
}

/** Drops unset query values (`undefined`, `null`, `""`) and stringifies the rest. */
function compactQuery(query: Query): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== "") out[key] = String(value)
  }
  return out
}

async function transport(sent: Request): Promise<Response> {
  let response: Response
  if (MOCK) {
    const { handleMockRequest } = await import("./mock/server")
    response = await handleMockRequest(sent)
  } else response = await request(sent)
  // The session ended or the server now requires a login: the app asks for it (features/auth).
  if (response.status === 401 && !new URL(sent.url).pathname.startsWith(`${API_BASE}/auth/`)) sessionRefused()
  return response
}

/** The generated, path-typed client. Paths are the server's (`/api/v2/...`); `ROOT` is the base URL. */
export const client = createClient<paths>({
  baseUrl: typeof location === "undefined" ? ROOT : `${location.origin}${ROOT}`,
  headers: { Accept: "application/json" },
  fetch: transport,
  querySerializer: (query) => new URLSearchParams(compactQuery((query ?? {}) as Query)).toString(),
})

type Result<T> = { data?: T; error?: unknown; response: Response }

/** The data of a generated-client call, or an `ApiError` for any non-2xx answer. */
export async function unwrap<T>(call: Promise<Result<T>>): Promise<T> {
  let result: Result<T>
  try {
    result = await call
  } catch (error) {
    throw requestFailure(error) // no answer, or a body that is not JSON
  }
  const { data, error, response } = result
  if (!response.ok) throw new ApiError(response.status, errorMessage(response.status, error), error)
  return data as T
}

/** A rejected value as the server sends it: `owner` and `param` are null when it names no single field. */
type SentError = Omit<ValidationError, "owner" | "param"> & { owner: string | null; param: string | null }

function isValidationError(value: unknown): value is SentError {
  if (!value || typeof value !== "object") return false
  const { owner, param, message } = value as Record<string, unknown>
  const name = (part: unknown) => part === null || part === undefined || typeof part === "string"
  return name(owner) && name(param) && typeof message === "string"
}

/** A sentence's values as the server sends them (`remi.ui.Sentence`): text, or a list of names. */
export function sentenceParams(value: unknown): SentenceParams | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined
  const entries = Object.entries(value)
  const valid = entries.every(([, item]) => typeof item === "string" || (Array.isArray(item) && item.every((name) => typeof name === "string")))
  return valid ? (Object.fromEntries(entries) as SentenceParams) : undefined
}

/**
 * The messages of a job's refused values as the server stores and sends them (`errors` of a job
 * or of `job.failed`; an entry may name no field), with a template only with its values. Entries
 * of another shape (stored by hand) are left out.
 */
export function fieldMessages(value: unknown): FieldMessage[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((entry: unknown): FieldMessage[] => {
    if (!entry || typeof entry !== "object") return []
    const { message, template, params } = entry as Record<string, unknown>
    if (typeof message !== "string") return []
    const values = sentenceParams(params)
    return [typeof template === "string" && values ? { message, template, params: values } : { message }]
  })
}

/**
 * A rejected value with only the fields the client reads (a template only with its values). One
 * that names no single field (a script no longer loaded) gets "" there: it marks no field, and
 * its message still says why.
 */
function validationEntry({ owner: sentOwner, param: sentParam, message, template, params }: SentError): ValidationError {
  const owner = sentOwner ?? ""
  const param = sentParam ?? ""
  const values = sentenceParams(params)
  return typeof template === "string" && values ? { owner, param, message, template, params: values } : { owner, param, message }
}

/** The rejected values of a `{"errors": [{owner, param, message}]}` body (schema-format.md §3), or null. */
export function validationErrors(error: unknown): ValidationError[] | null {
  const payload = error instanceof ApiError ? error.detail : error
  if (!payload || typeof payload !== "object" || !("errors" in payload)) return null
  const errors = (payload as { errors: unknown }).errors
  if (!Array.isArray(errors)) return null
  const valid = errors.filter(isValidationError).map(validationEntry)
  return valid.length > 0 ? valid : null
}

/**
 * Something readable from an error body: Remi's validation `errors`, or
 * FastAPI's `{detail: string | [{msg, loc}]}`.
 */
function errorMessage(status: number, payload: unknown): string {
  const errors = validationErrors(payload)
  if (errors)
    return errors
      .map((e) => {
        const name = [e.owner === "core" ? "" : e.owner, e.param].filter(Boolean).join(".")
        return name ? `${name}: ${e.message}` : e.message
      })
      .join("; ")
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === "string") return detail
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d)))
        .join("; ")
    }
  }
  if (typeof payload === "string" && payload) return payload
  return `HTTP ${status}`
}
