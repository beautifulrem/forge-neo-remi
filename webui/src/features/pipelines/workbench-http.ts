/**
 * Thin typed JSON transport for workbench `/api/v2/*` endpoints.
 * Built strictly on `@/api/http`'s `apiUrl`, `ApiError` and `MOCK` (routing
 * mock-mode requests through `@/api/mock/server`'s `handleMockRequest` via a
 * dynamic import so production bundles tree-shake the mock server cleanly).
 */
import { ApiError, apiUrl, MOCK, type Query } from "@/api/http"

export { ApiError, MOCK }

type RequestOptions = {
  method?: "GET" | "POST" | "PATCH" | "PUT" | "DELETE"
  query?: Query
  body?: unknown
  signal?: AbortSignal
}

function messageFrom(status: number, payload: unknown): string {
  if (payload && typeof payload === "object") {
    if ("detail" in payload) {
      const detail = (payload as { detail: unknown }).detail
      if (typeof detail === "string" && detail) return detail
      if (Array.isArray(detail) && detail.length > 0) {
        return detail
          .map((item) => (item && typeof item === "object" && "msg" in item ? String((item as { msg: unknown }).msg) : String(item)))
          .join("; ")
      }
    }
    if ("message" in payload && typeof (payload as { message: unknown }).message === "string") {
      return (payload as { message: string }).message
    }
  }
  if (typeof payload === "string" && payload) return payload
  return `HTTP ${status}`
}

/** Sends a JSON request to `/api/v2<path>` and returns the parsed body (or `undefined` on 204). */
export async function workbenchRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", query, body, signal } = options
  const headers: Record<string, string> = { Accept: "application/json" }
  const init: RequestInit = { method, headers, signal }
  if (body !== undefined) {
    headers["Content-Type"] = "application/json"
    init.body = JSON.stringify(body)
  }
  const url = apiUrl(path, query)
  let response: Response
  if (MOCK) {
    const { handleMockRequest } = await import("@/api/mock/server")
    const fullUrl = new URL(url, typeof location === "undefined" ? "http://localhost" : location.origin)
    response = await handleMockRequest(new Request(fullUrl, init))
  } else {
    response = await fetch(url, init)
  }
  if (response.status === 204) return undefined as T
  const text = await response.text()
  let payload: unknown = undefined
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }
  if (!response.ok) throw new ApiError(response.status, messageFrom(response.status, payload), payload)
  return payload as T
}
