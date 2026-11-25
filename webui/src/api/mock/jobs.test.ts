/** The mock's `POST /jobs` keeps the server's contract: one job per `Idempotency-Key`. */
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { handleMockRequest } from "./server"

async function call(method: string, path: string, body?: unknown, headers: Record<string, string> = {}) {
  const init: RequestInit = { method, headers: { Accept: "application/json", ...headers } }
  if (body !== undefined) {
    init.headers = { ...init.headers, "Content-Type": "application/json" }
    init.body = JSON.stringify(body)
  }
  const response = await handleMockRequest(new Request(`http://localhost/api/v2${path}`, init))
  return { status: response.status, data: (await response.json()) as Record<string, unknown> }
}

const job = { tab: "txt2img", core: { prompt: "a cat" }, scripts: {} }

describe("mock jobs", () => {
  beforeEach(() => call("POST", "/queue/pause"))
  afterEach(() => call("POST", "/queue/resume"))

  it("answers a repeated Idempotency-Key with the first submission's jobs", async () => {
    const before = (await call("GET", "/queue")).data.size as number
    const first = await call("POST", "/jobs", job, { "Idempotency-Key": "press-1" })
    const again = await call("POST", "/jobs", job, { "Idempotency-Key": "press-1" })
    expect(again.data).toEqual(first.data)
    const other = await call("POST", "/jobs", job, { "Idempotency-Key": "press-2" })
    expect(other.data.job_id).not.toBe(first.data.job_id)
    expect((await call("GET", "/queue")).data.size).toBe(before + 2)
  })
})
