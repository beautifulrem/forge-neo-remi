import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { extrasSchema } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { getFormStore } from "@/ui/form-store"
import { registerPendingWrite } from "@/ui/pending-writes"
import { tabOwners } from "@/ui/tab"
import { finishExtras, interruptExtras, resetExtrasRuns, resyncExtras, runExtras, singleSource, useExtrasRuns, wireExtrasRuns } from "./run"

function mockFetch(answer: (method: string, path: string) => [number, unknown]) {
  const calls: { method: string; path: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const text = request.method === "GET" ? "" : await request.text()
    calls.push({ method: request.method, path, body: text ? JSON.parse(text) : undefined })
    const [status, data] = answer(request.method, path)
    return new Response(status === 204 ? null : JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

describe("extras runs", () => {
  beforeEach(() => {
    resetExtrasRuns()
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
    queryClient.setQueryData(keys.schema("extras"), extrasSchema)
    getFormStore("extras").getState().load(tabOwners(extrasSchema))
  })
  afterEach(() => vi.restoreAllMocks())

  it("knows the single source image", () => {
    expect(singleSource({ mode: "Single Image", image: "up1" })).toBe("up1")
    expect(singleSource({ image: "up1" })).toBe("up1")
    expect(singleSource({ mode: "Batch Process", image: "up1" })).toBeNull()
    expect(singleSource({ mode: "Single Image", image: null })).toBeNull()
  })

  it("submits the form, follows the job and keeps its source for the comparison", async () => {
    wireExtrasRuns()
    const calls = mockFetch(() => [201, { job_id: "x1", job_ids: ["x1"], seq: 3 }])
    getFormStore("extras").getState().setValue("core", "image", "up9")
    await runExtras()
    const body = calls[0]!.body as { tab: string; core: { image: string } }
    expect(body.tab).toBe("extras")
    expect(body.core.image).toBe("up9")
    expect(useExtrasRuns.getState().jobs).toEqual(["x1"])
    expect(useProgress.getState().owned).toEqual({ x1: "extras" })

    // The WebSocket bridge reports the result through the progress store.
    useProgress.getState().completed({ job_id: "x1", outputs: [{ id: "out1", kind: "image" }], infotext: "Postprocess upscale by: 4" })
    const [result] = useExtrasRuns.getState().results
    expect(result).toMatchObject({ jobId: "x1", source: "up9", info: "Postprocess upscale by: 4" })
    expect(result!.outputs.map((output) => output.id)).toEqual(["out1"])
    expect(useExtrasRuns.getState().jobs).toEqual([])
  })

  it("sends one request for a second press while the first has no answer", async () => {
    const calls = mockFetch(() => [201, { job_id: "x1", job_ids: ["x1"], seq: 3 }])
    getFormStore("extras").getState().setValue("core", "image", "up9")
    await Promise.all([runExtras(), runExtras()])
    expect(calls.filter((call) => call.method === "POST")).toHaveLength(1)
    expect(useExtrasRuns.getState().jobs).toEqual(["x1"])
  })


  // Process pressed while the source image was still uploading said "an image is required": the
  // run checked the form before the upload had set it.
  it("waits for a value still being written (an upload) before it checks and sends the form", async () => {
    const calls = mockFetch(() => [201, { job_id: "x2", job_ids: ["x2"], seq: 4 }])
    let land!: () => void
    const upload = new Promise<void>((resolve) => (land = resolve))
    const unregister = registerPendingWrite("extras", () => upload)
    try {
      const first = runExtras()
      const second = runExtras() // a second press while it waits queues nothing more
      await new Promise((resolve) => setTimeout(resolve, 10))
      expect(calls).toEqual([])
      getFormStore("extras").getState().setValue("core", "image", "up7")
      land()
      await Promise.all([first, second])
    } finally {
      unregister()
    }
    expect(calls.filter((call) => call.method === "POST" && call.path === "/jobs")).toHaveLength(1)
    expect((calls[0]!.body as { core: { image: string } }).core.image).toBe("up7")
    expect(getFormStore("extras").getState().errors.core?.image).toBeUndefined()
  })

  it("marks rejected values on the form", async () => {
    mockFetch(() => [422, { errors: [{ owner: "core", param: "image", message: "an image is required" }] }])
    await runExtras()
    expect(getFormStore("extras").getState().errors.core?.image).toEqual([{ message: "an image is required" }])
    expect(useExtrasRuns.getState().jobs).toEqual([])
  })

  it("interrupts the running job and cancels the queued ones", async () => {
    const calls = mockFetch(() => [200, {}])
    useExtrasRuns.setState({ jobs: ["a", "b"] })
    useProgress.setState({ running: { jobId: "a", step: 0, steps: 0 } })
    interruptExtras()
    await vi.waitFor(() => expect(calls.map((call) => `${call.method} ${call.path}`).sort()).toEqual(["DELETE /jobs/b", "POST /jobs/a/interrupt"]))
  })

  it("keeps following a job while the server is unreachable and settles it once it answers", async () => {
    useExtrasRuns.setState({ jobs: ["j1"], sources: { j1: "up1" } })
    mockFetch(() => [502, { detail: "restarting" }])
    await resyncExtras()
    expect(useExtrasRuns.getState().jobs).toEqual(["j1"])

    vi.restoreAllMocks()
    mockFetch(() => [200, { id: "j1", status: "completed", outputs: [{ file_id: "o1", url: "/files/o1" }], infotext: "x" }])
    await resyncExtras()
    expect(useExtrasRuns.getState().jobs).toEqual([])
    expect(useExtrasRuns.getState().results[0]).toMatchObject({ jobId: "j1", source: "up1", info: "x" })
  })

  it("keeps the values a failed run refused, with their templates", async () => {
    // the page shows them in the UI language; the error text alone is English
    wireExtrasRuns()
    useExtrasRuns.setState({ jobs: ["x2"], sources: { x2: null } })
    useProgress.getState().track("x2", "extras")
    const refused = { owner: "core", param: "upscaler_1", message: "'Gone' is not one of the choices ('None')", template: "{{value}} is not one of the choices ({{choices}})", params: { value: "'Gone'", choices: "'None'" } }
    useProgress.getState().failed({ job_id: "x2", error: "core.upscaler_1: 'Gone' is not one of the choices ('None')", errors: [refused, { message: { a: 1 } }] })
    expect(useExtrasRuns.getState().results[0]).toMatchObject({ jobId: "x2", errors: [{ message: refused.message, template: refused.template, params: refused.params }] })
    // stored by hand: entries the page cannot read are dropped, the run is kept
    sessionStorage.setItem("remi-extras-runs", JSON.stringify({ state: { results: [{ jobId: "s", outputs: [], source: null, info: null, error: "x", errors: [{ message: 5 }, { message: "ok" }], at: 1 }], jobs: [], sources: {} }, version: 1 }))
    await useExtrasRuns.persist.rehydrate()
    expect(useExtrasRuns.getState().results.map((result) => result.errors)).toEqual([[{ message: "ok" }]])
    sessionStorage.clear()
  })

  it("reports a job the server no longer knows as gone", async () => {
    useExtrasRuns.setState({ jobs: ["j2"], sources: { j2: null } })
    mockFetch(() => [404, { detail: "no such job" }])
    await resyncExtras()
    expect(useExtrasRuns.getState().jobs).toEqual([])
    expect(useExtrasRuns.getState().results[0]?.error).toBeTruthy()
  })

  it("records a finished run when the session storage is full", () => {
    useExtrasRuns.setState({ jobs: ["job-1"], sources: { "job-1": null } })
    const write = Storage.prototype.setItem
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (this: Storage, key: string, value: string) {
      if (this === sessionStorage) throw new DOMException("The quota has been exceeded.", "QuotaExceededError")
      write.call(this, key, value)
    })
    expect(() => finishExtras({ jobId: "job-1", outputs: [] })).not.toThrow()
    expect(useExtrasRuns.getState().results.map((result) => result.jobId)).toEqual(["job-1"])
  })

  it("ignores jobs it did not queue", () => {
    finishExtras({ jobId: "other", outputs: [] })
    expect(useExtrasRuns.getState().results).toEqual([])
  })

  // sessionStorage edited by hand or by another app: an object where text or a number belongs
  // reached React as a child and took the Extras page down, on every reload of the tab
  it("drops stored runs the page cannot show", async () => {
    const run = { jobId: "j", outputs: [{ id: "o", kind: "image", width: 512, height: 512, infotext: "Steps: 2" }], source: null, info: null, at: 1 }
    const bad = [
      { ...run, jobId: "error", error: { detail: "x" } },
      { ...run, jobId: "size", outputs: [{ id: "o", kind: "image", width: { a: 1 }, height: 5 }] },
      { ...run, jobId: "fraction", outputs: [{ id: "o", kind: "image", width: 1.5 }] },
      { ...run, jobId: "infotext", outputs: [{ id: "o", kind: "image", infotext: { a: 1 } }] },
      { ...run, jobId: "role", outputs: [{ id: "o", kind: "image", role: ["x"] }] },
    ]
    sessionStorage.setItem("remi-extras-runs", JSON.stringify({ state: { results: [run, ...bad, { ...run, jobId: "plain", error: "failed", outputs: [{ id: "o", kind: "image" }] }], jobs: [], sources: {} }, version: 1 }))
    await useExtrasRuns.persist.rehydrate()
    expect(useExtrasRuns.getState().results.map((result) => result.jobId)).toEqual(["j", "plain"])
    sessionStorage.clear()
  })

  it("restores the session's runs when the page reloads", async () => {
    // rehydration runs inside create(), before the rest of the module is evaluated
    const result = { jobId: "j1", at: 1, source: null, info: "", error: "", outputs: [{ id: "o1", kind: "image", width: 64, height: 48, infotext: "", role: null, missing: false }] }
    sessionStorage.setItem("remi-extras-runs", JSON.stringify({ state: { results: [result], jobs: [], sources: {} }, version: 1 }))
    try {
      vi.resetModules()
      const fresh = (await import("./run")).useExtrasRuns.getState()
      expect(fresh.results.map((run) => run.jobId)).toEqual(["j1"])
    } finally {
      sessionStorage.removeItem("remi-extras-runs")
    }
  })
})
