import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import type { JobDetail } from "@/api/types"
import { useWs } from "@/api/ws"
import { tabSchema } from "@/api/mock/fixtures"
import { useSession } from "@/features/gallery/session-store"
import { FakeSocket } from "@/test/fake-socket"
import { resetRestore, restoreRuns, wireRunSync } from "./restore"
import { dispatchRun, MAX_RUNS, persistedJobs, persistedRun, persistKey, resetRunControllers, useRuns, wireRunControllers } from "./run-controller"
import { phaseOf } from "./run-machine"
import { created, jobDetail } from "@/test/api-fixtures"

const job = jobDetail

/** `GET /jobs/{id}` answering from `jobs` (404 for unknown ids). */
function serverJobs(jobs: Record<string, JobDetail>) {
  return vi.spyOn(api.jobs, "get").mockImplementation(async (id) => {
    const found = jobs[id]
    if (!found) throw new ApiError(404, "not found")
    return found
  })
}

async function socket() {
  useWs.getState().connect()
  await vi.waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0))
  const ws = FakeSocket.instances.at(-1)!
  ws.open()
  return ws
}

const send = (ws: FakeSocket, event: object) => ws.receive(JSON.stringify(event))

beforeEach(() => {
  FakeSocket.instances = []
  vi.stubGlobal("WebSocket", FakeSocket)
  resetRunControllers()
  resetRestore()
  wireRunControllers()
  wireRunSync()
  useSession.getState().clear()
  useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
  queryClient.setQueryData(keys.schema("txt2img"), tabSchema("txt2img"))
})
afterEach(() => {
  useWs.getState().disconnect()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe("restoreRuns", () => {
  it("re-attaches queued/running jobs and files finished ones", async () => {
    // A previous page load left three jobs in flight.
    dispatchRun("txt2img", { type: "submitted", jobIds: ["a", "b", "c"] })
    expect(persistedJobs("txt2img")).toEqual(["a", "b", "c"])
    resetRunControllers()
    sessionStorage.setItem(persistKey("txt2img"), JSON.stringify(["a", "b", "c"]))

    const get = serverJobs({
      a: job("a", "completed", [{ id: "out-a", kind: "image", infotext: "a\nSteps: 1, Seed: 1, Size: 8x8" }]),
      b: job("b", "running"),
      c: job("c", "queued"),
    })
    await restoreRuns()
    // Each persisted id is looked up on its own (no history page to fall off).
    expect(get.mock.calls.map(([id]) => id).sort()).toEqual(["a", "b", "c"])
    expect(useRuns.getState().runs.txt2img).toMatchObject({ jobs: ["b", "c"], running: "b" })
    expect(useProgress.getState().owned).toMatchObject({ b: "txt2img", c: "txt2img" })
    expect(useSession.getState().items.map((item) => item.fileId)).toEqual(["out-a"])
    expect(persistedJobs("txt2img")).toEqual(["b", "c"])
  })

  it("keeps Generate forever going across a reload while its job is in flight", async () => {
    // A previous page load was generating forever: its job is still running.
    vi.spyOn(api.jobs, "create").mockResolvedValue(created("f1"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(persistedRun("txt2img")).toEqual({ jobs: ["f1"], forever: true, repeatLeft: 0 }))
    resetRunControllers()
    sessionStorage.setItem(persistKey("txt2img"), JSON.stringify({ jobs: ["f1"], forever: true, repeatLeft: 0 }))

    const ws = await socket()
    serverJobs({ f1: job("f1", "running") })
    await restoreRuns()
    // The menu's "Generate forever" is checked again, and the run after this one is queued.
    expect(useRuns.getState().runs.txt2img).toMatchObject({ jobs: ["f1"], running: "f1", forever: true })
    expect(persistedRun("txt2img").forever).toBe(true)
    const create = vi.spyOn(api.jobs, "create").mockResolvedValue(created("f2"))
    send(ws, { type: "job.completed", job_id: "f1", seq: 1, status: "completed", outputs: [] })
    await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(1))
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["f2"]))
  })

  it("leaves a loop another tab is running to that tab", async () => {
    // This tab is generating forever: its job is in flight.
    vi.spyOn(api.jobs, "create").mockResolvedValue(created("f1"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(persistedRun("txt2img")).toEqual({ jobs: ["f1"], forever: true, repeatLeft: 0 }))
    // A second tab opens: nothing in memory, its own session storage is empty, local storage is shared.
    resetRunControllers()
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
    sessionStorage.clear()
    const ws = await socket()
    serverJobs({ f1: job("f1", "running") })
    await restoreRuns()
    expect(useRuns.getState().runs.txt2img).toMatchObject({ jobs: [], forever: false })
    // The loop's next job is queued by its own tab only, not once more by this one.
    // (the spy of this tab's first submit, counted from here)
    const create = vi.spyOn(api.jobs, "create").mockResolvedValue(created("f2"))
    create.mockClear()
    send(ws, { type: "job.completed", job_id: "f1", seq: 1, status: "completed", outputs: [] })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(create).not.toHaveBeenCalled()
  })

  it("leaves a loop to the tab it was duplicated from, which still runs it", async () => {
    // A duplicated tab starts with a copy of the original's session storage: the loop's job in flight.
    sessionStorage.setItem(persistKey("txt2img"), JSON.stringify({ jobs: ["f1"], forever: true, repeatLeft: 0 }))
    // The original tab is alive and still looping txt2img.
    const original = new BroadcastChannel("remi-run-loops")
    original.onmessage = (event: MessageEvent<{ type: string; tab: string; id: string; jobs: string[] }>) => {
      if (event.data.type === "who-loops" && event.data.tab === "txt2img" && event.data.jobs.includes("f1")) original.postMessage({ type: "loops", tab: "txt2img", id: event.data.id })
    }
    try {
      const ws = await socket()
      serverJobs({ f1: job("f1", "running") })
      await restoreRuns()
      expect(useRuns.getState().runs.txt2img).toMatchObject({ forever: false, repeatLeft: 0 })
      const create = vi.spyOn(api.jobs, "create").mockResolvedValue(created("f2"))
      send(ws, { type: "job.completed", job_id: "f1", seq: 1, status: "completed", outputs: [] })
      await new Promise((resolve) => setTimeout(resolve, 50))
      expect(create).not.toHaveBeenCalled()
    } finally {
      original.close()
    }
  })

  it("answers a duplicated tab that asks who loops a form", async () => {
    vi.spyOn(api.jobs, "create").mockResolvedValue(created("f1"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img).toMatchObject({ forever: true, jobs: ["f1"] }))
    const asker = new BroadcastChannel("remi-run-loops")
    try {
      const answer = new Promise<unknown>((resolve) => {
        asker.onmessage = (event: MessageEvent) => resolve(event.data)
      })
      // another tab's own loop of the same form (other jobs) gets no answer
      asker.postMessage({ type: "who-loops", tab: "txt2img", id: "q0", jobs: ["elsewhere"] })
      asker.postMessage({ type: "who-loops", tab: "txt2img", id: "q1", jobs: ["f1"] })
      expect(await answer).toEqual({ type: "loops", tab: "txt2img", id: "q1" })
    } finally {
      asker.close()
    }
  })

  it("does not start Generate forever again when its job ended while the page was closed", async () => {
    sessionStorage.setItem(persistKey("txt2img"), JSON.stringify({ jobs: ["f1"], forever: true, repeatLeft: 0 }))
    serverJobs({ f1: job("f1", "completed") })
    const create = vi.spyOn(api.jobs, "create")
    await restoreRuns()
    expect(create).not.toHaveBeenCalled()
    expect(useRuns.getState().runs.txt2img).toMatchObject({ jobs: [], forever: false })
    expect(sessionStorage.getItem(persistKey("txt2img"))).toBeNull()
  })

  it("reads the bare job list an older version persisted", () => {
    sessionStorage.setItem(persistKey("img2img"), JSON.stringify(["x", 3, "y"]))
    expect(persistedRun("img2img")).toEqual({ jobs: ["x", "y"], forever: false, repeatLeft: 0 })
  })

  it("forgets jobs the server no longer knows", async () => {
    sessionStorage.setItem(persistKey("img2img"), JSON.stringify(["gone"]))
    serverJobs({})
    await restoreRuns()
    expect(sessionStorage.getItem(persistKey("img2img"))).toBeNull()
    expect(useRuns.getState().runs.img2img.jobs).toEqual([])
  })

  it("applies a job event that arrives while its lookup is pending", async () => {
    const ws = await socket()
    sessionStorage.setItem(persistKey("txt2img"), JSON.stringify(["r"]))
    let answer!: (value: JobDetail) => void
    vi.spyOn(api.jobs, "get").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    const restored = restoreRuns()
    // The job finishes before the (stale) lookup answer arrives.
    send(ws, { type: "job.completed", job_id: "r", seq: 1, status: "completed", outputs: [] })
    answer(job("r", "running"))
    await restored
    expect(phaseOf(useRuns.getState().runs.txt2img)).toBe("idle")
    expect(sessionStorage.getItem(persistKey("txt2img"))).toBeNull()
  })

  it("does not drop a job submitted while the restore is pending", async () => {
    sessionStorage.setItem(persistKey("txt2img"), JSON.stringify(["old"]))
    let answer!: (value: JobDetail) => void
    vi.spyOn(api.jobs, "get").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    const restored = restoreRuns()
    dispatchRun("txt2img", { type: "submitted", jobIds: ["new"] })
    answer(job("old", "completed"))
    await restored
    expect(useRuns.getState().runs.txt2img.jobs).toEqual(["new"])
    expect(persistedJobs("txt2img")).toEqual(["new"])
  })
})

describe("resync after missed events", () => {
  it("goes idle when the job completed while the socket was down", async () => {
    let ws = await socket()
    vi.spyOn(api.jobs, "create").mockResolvedValue(created("j1"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["j1"]))
    send(ws, { type: "job.started", job_id: "j1", seq: 1 })
    useProgress.getState().progress({ job_id: "j1", seq: 1, step: 3, steps: 8 })

    // Laptop sleeps: the socket drops and the job fails meanwhile (so forever stops).
    ws.drop()
    serverJobs({ j1: job("j1", "failed") })
    ws = await socket()
    send(ws, { type: "hello", features: [] })
    send(ws, { type: "queue.changed", size: 0, paused: false, running: null })

    await vi.waitFor(() => expect(phaseOf(useRuns.getState().runs.txt2img)).toBe("idle"))
    expect(useRuns.getState().runs.txt2img.forever).toBe(false)
    expect(useProgress.getState().running).toBeNull()
    expect(sessionStorage.getItem(persistKey("txt2img"))).toBeNull()
  })

  it("files the results of a job that completed unseen and keeps generating forever", async () => {
    const ws = await socket()
    const create = vi.spyOn(api.jobs, "create").mockResolvedValueOnce(created("f1")).mockResolvedValueOnce(created("f2"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["f1"]))
    send(ws, { type: "job.started", job_id: "f1", seq: 1 })
    serverJobs({ f1: job("f1", "completed", [{ id: "out-f1", kind: "image" }]) })
    // A queue change naming another running job reveals the missed completion.
    send(ws, { type: "queue.changed", size: 0, running: "someone-else" })
    await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(2))
    expect(useSession.getState().items.map((item) => item.fileId)).toEqual(["out-f1"])
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["f2"]))
  })

  it("marks a queued job running when the queue says it runs", async () => {
    const ws = await socket()
    dispatchRun("img2img", { type: "submitted", jobIds: ["q1"] })
    send(ws, { type: "queue.changed", size: 0, running: "q1" })
    expect(phaseOf(useRuns.getState().runs.img2img)).toBe("running")
  })
})

// A stored repeat count is one the Repeat field can give (it takes at most 999 runs): a hand-edited
// 1e9 kept generating for good after a reload.
describe("a stored repeat count", () => {
  it("is within what the Repeat field gives", () => {
    sessionStorage.setItem(persistKey("txt2img"), JSON.stringify({ jobs: ["j"], forever: false, repeatLeft: 1e9 }))
    expect(persistedRun("txt2img").repeatLeft).toBe(MAX_RUNS - 1)
    sessionStorage.removeItem(persistKey("txt2img"))
  })
})

// A stored run is read back on every load: a damaged one (cut short, edited by hand, another app on
// this origin) must restore no run rather than throw before the page has started.
describe("a damaged stored run", () => {
  afterEach(() => sessionStorage.removeItem(persistKey("txt2img")))

  it.each(["{not json", "null", "5", '"j1"', '{"jobs": "j1", "forever": "yes", "repeatLeft": "3"}', '{"jobs": [1, null], "repeatLeft": null}'])("restores no run from %s", (raw) => {
    sessionStorage.setItem(persistKey("txt2img"), raw)
    expect(persistedRun("txt2img")).toEqual({ jobs: [], forever: false, repeatLeft: 0 })
  })
})
