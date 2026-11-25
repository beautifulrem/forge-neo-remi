import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "./endpoints"
import { ApiError, apiUrl, RequestError, validationErrors } from "./http"

const reply = (body: unknown, status: number) =>
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }))

const submit = () => api.jobs.create({ tab: "txt2img", core: {}, scripts: {} }).catch((e: unknown) => e)

describe("http errors", () => {
  afterEach(() => vi.restoreAllMocks())

  it("maps a rejected request's validation errors (schema-format.md §3)", async () => {
    const errors = [
      { owner: "core", param: "width", message: "expected a number, got 'abc'" },
      { owner: "sampler", param: "steps", message: "must be at most 150" },
    ]
    reply({ errors }, 422)
    const error = await submit()
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(422)
    expect((error as ApiError).message).toBe("width: expected a number, got 'abc'; sampler.steps: must be at most 150")
    expect(validationErrors(error)).toEqual(errors)
  })

  it("keeps FastAPI details readable and ignores malformed error lists", async () => {
    reply({ detail: [{ msg: "field required", loc: ["body", "tab"] }] }, 422)
    const error = await submit()
    expect((error as ApiError).message).toBe("field required")
    expect(validationErrors(error)).toBeNull()
    expect(validationErrors(new ApiError(400, "x", { errors: [{ owner: 1 }] }))).toBeNull()
  })
})

describe("generated client requests", () => {
  afterEach(() => vi.restoreAllMocks())

  it("sends the defaulted job body and encodes path params", async () => {
    const fetch = reply({ job_id: "a", job_ids: ["a"], seq: 1 }, 201)
    await api.jobs.create({ tab: "img2img", core: { prompt: "x" }, scripts: {} })
    const request = fetch.mock.calls[0]![0] as Request
    expect(new URL(request.url).pathname).toBe("/api/v2/jobs")
    expect(await request.json()).toEqual({ tab: "img2img", core: { prompt: "x" }, scripts: {}, priority: 0, repeat: 1, forever: false })

    const metaFetch = reply({}, 200)
    await api.networks.meta("lora", "sub/a b")
    const meta = (metaFetch.mock.calls.at(-1)![0] as Request).url
    expect(new URL(meta).pathname).toBe("/api/v2/networks/lora/sub%2Fa%20b/meta")
  })

  it("drops unset query values", async () => {
    const fetch = reply({ items: [] }, 200)
    await api.jobs.list({ cursor: null, limit: 50 })
    expect(new URL((fetch.mock.calls[0]![0] as Request).url).search).toBe("?limit=50")
    expect(apiUrl("/thumbs/x", { w: undefined })).toBe("/api/v2/thumbs/x")
  })

  // A failed fetch reached the page error view and the inline boundary as the browser's English
  // ("Failed to fetch"): they could not tell it from a bug. It is marked where it happens.
  it("marks a request that got no answer, and lets the caller's own abort through", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"))
    const offline = await submit()
    expect(offline).toBeInstanceOf(RequestError)
    expect(offline).toBeInstanceOf(TypeError)

    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("<html>proxy error</html>", { status: 200, headers: { "Content-Type": "application/json" } }))
    expect(await submit()).toBeInstanceOf(RequestError)

    const abort = new DOMException("The operation was aborted.", "AbortError")
    vi.spyOn(globalThis, "fetch").mockRejectedValue(abort)
    expect(await submit()).toBe(abort)
  })
})
