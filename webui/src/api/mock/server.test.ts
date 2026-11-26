/**
 * The mock's refused values read as the server's: the server's English sentence, its template and
 * its values (`remi.ui.Sentence`), so the zh mock preview shows them in Chinese through the
 * catalog sample, as the product does.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { sentenceParams } from "@/api/http"
import { validationMessage } from "@/features/errors/error-text"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import type { ValidationError } from "@/ui/schema"
import { bootstrap, serverCatalogZh } from "./fixtures"
import { actionPatch, handleMockRequest, validateJob } from "./server"
import { MockSocket } from "./socket"

const HAN = /\p{Script=Han}/u

beforeAll(() => {
  i18n.addResourceBundle("zh-CN", "server", serverCatalogZh, true, true)
})

afterAll(() => {
  i18n.removeResourceBundle("zh-CN", "server")
})

describe("the mock's job validation", () => {
  const errors = validateJob({
    tab: "txt2img",
    core: { prompt: "a cat #invalid" },
    scripts: { anima: { shift: 12 }, seed: { seed: 1.5, subseed: true }, sampler: { sampler_name: "Nope", steps: "many" } },
  } as never)
  const by = (owner: string, param: string) => errors.find((error) => error.owner === owner && error.param === param)

  it("refuses values with the server's sentences, their templates and their values", () => {
    expect(by("anima", "shift")).toEqual({
      owner: "anima",
      param: "shift",
      message: "must be at most 10, got 12",
      template: "must be at most {{maximum}}, got {{value}}",
      params: { maximum: "10", value: "12" },
    })
    expect(by("seed", "seed")).toMatchObject({ message: "a seed is a whole number, got 1.5", template: "a seed is a whole number, got {{value}}" })
    expect(by("seed", "subseed")).toMatchObject({ message: "expected a number, got True", params: { value: "True" } })
    expect(by("sampler", "sampler_name")).toMatchObject({ template: "{{value}} is not one of the choices ({{choices}})" })
    expect(by("sampler", "sampler_name")?.message).toMatch(/^'Nope' is not one of the choices \('Euler a', 'Euler', .*, \.\.\.\)$/)
    expect(by("sampler", "steps")).toMatchObject({ message: "expected a number, got 'many'" })
    for (const error of errors.filter((e) => e.template)) {
      // the client keeps a template only with its values as text (`sentenceParams`)
      expect(sentenceParams(error.params)).toEqual(error.params)
    }
  })

  it("is said in Chinese in the zh preview", () => {
    expect(errors.length).toBeGreaterThanOrEqual(6)
    for (const error of errors) {
      const zh = validationMessage(error, "zh-CN")
      expect(zh, error.message).toMatch(HAN)
      expect(zh).not.toBe(error.message)
    }
    expect(validationMessage(by("anima", "shift")!, "zh-CN")).toBe("不能大于 10，当前为 12")
  })
})

describe("the mock's action toasts", () => {
  it("name their values as the server's messages do, so they translate", () => {
    const patch = actionPatch("anima", "apply_preset", { values: { preset: "detail" } } as never) as { __toast__?: { message: unknown } }
    expect(patch.__toast__?.message).toEqual({ text: "Preset “{{preset}}” applied", params: { preset: "detail" } })
    expect(serverCatalogZh["Preset “{{preset}}” applied"]).toMatch(HAN)
  })
})

describe("the mock's refused settings", () => {
  async function patch(values: Record<string, unknown>) {
    const response = await handleMockRequest(
      new Request("http://localhost/api/v2/options/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) }),
    )
    return { status: response.status, errors: ((await response.json()) as { errors: ValidationError[] }).errors }
  }

  it("say why in the server's words, translatable in the zh preview", async () => {
    const { status, errors } = await patch({ samples_filename_pattern: "[seed] #invalid", no_such_setting: 1 })
    expect(status).toBe(422)
    const [rejected, unknown] = errors
    // the demo rule names the value as the server's sentences do: a template and its values
    expect(rejected).toEqual({
      owner: "settings",
      param: "samples_filename_pattern",
      message: "'[seed] #invalid' contains #invalid (mock validation demo)",
      template: "{{value}} contains #invalid (mock validation demo)",
      params: { value: "'[seed] #invalid'" },
    })
    // the server's own reason (remi/server/options.py, frozen_reason): a catalog key
    expect(unknown).toEqual({ owner: "settings", param: "no_such_setting", message: "unknown setting" })
    expect(validationMessage(rejected!, "zh-CN")).toBe("'[seed] #invalid' 含有 #invalid（mock 校验演示）")
    // a reason of the server's own is in the SPA's catalog, as in the product
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    expect(validationMessage(unknown!, "zh-CN")).toBe("未知设置")
  })
})

describe("the mock's model load", () => {
  const initial = { ...bootstrap.options }
  afterEach(() => {
    vi.useRealTimers()
    Object.assign(bootstrap.options, initial)
  })

  async function call(method: string, path: string, body?: unknown) {
    const init: RequestInit = { method, headers: { Accept: "application/json" } }
    if (body !== undefined) Object.assign(init, { headers: { Accept: "application/json", "Content-Type": "application/json" }, body: JSON.stringify(body) })
    return handleMockRequest(new Request(`http://localhost/api/v2${path}`, init))
  }

  // As on the server (remi/server/options.py, remi/core/presets.py): a model change only selects
  // it, and the next job loads it, announcing model.loading at its start and model.loaded once loaded
  // (remi/jobs/queue.py, `_note_model_loading` / `poll`).
  it("loads a changed model at the next job, not at the change", async () => {
    vi.useFakeTimers()
    const socket = new MockSocket()
    const events: Record<string, unknown>[] = []
    socket.onmessage = (event) => {
      if (typeof event.data === "string") events.push(JSON.parse(event.data) as Record<string, unknown>)
    }
    await vi.advanceTimersByTimeAsync(200)
    const changed = call("PATCH", "/options", { sd_model_checkpoint: "flux1-dev.safetensors [f1e2d3]" })
    await vi.advanceTimersByTimeAsync(1000)
    expect((await changed).status).toBe(200)
    actionPatch("core", "__forge_preset__", { preset: "flux" } as never)
    await vi.advanceTimersByTimeAsync(3000)
    const model = (event: Record<string, unknown>) => String(event.type).startsWith("model.")
    expect(events.filter(model)).toEqual([])
    expect(bootstrap.models.checkpoint).toBe("flux1-dev.safetensors [f1e2d3]")

    const queued = call("POST", "/jobs", { tab: "txt2img", core: { prompt: "a cat", batch_size: 1 }, scripts: { sampler: { steps: 3 } } })
    await vi.advanceTimersByTimeAsync(10_000)
    expect((await queued).status).toBeLessThan(300)
    const order = events.filter((event) => model(event) || event.type === "job.started" || event.type === "job.progress").map((event) => String(event.type))
    expect(order.slice(0, 4)).toEqual(["job.started", "model.loading", "model.loaded", "job.progress"])
    expect(events.filter(model)).toEqual([
      { type: "model.loading", stage: "loading", model: "flux1-dev.safetensors [f1e2d3]" },
      { type: "model.loaded", model: "flux1-dev.safetensors [f1e2d3]" },
    ])

    // loaded now: the next job starts at once
    events.length = 0
    void call("POST", "/jobs", { tab: "txt2img", core: { prompt: "a dog", batch_size: 1 }, scripts: { sampler: { steps: 3 } } })
    await vi.advanceTimersByTimeAsync(10_000)
    socket.close()
    expect(events.filter(model)).toEqual([])
  })
})

// Routes the SPA calls that the mock answered 404 ("Mock: … not found"): the network preview dialog's
// Gallery tab (GET /outputs) said it could not load the gallery, and the queue page's "Stop generate
// forever" and "Clear the queue" (POST /queue/stop-forever, POST /interrupt) failed.
describe("the mock's output list and queue-wide actions", () => {
  async function call(method: string, path: string, body?: unknown) {
    const init: RequestInit = { method, headers: { Accept: "application/json" } }
    if (body !== undefined) Object.assign(init, { headers: { Accept: "application/json", "Content-Type": "application/json" }, body: JSON.stringify(body) })
    return handleMockRequest(new Request(`http://localhost/api/v2${path}`, init))
  }
  type Page = { items: { id: string; kind: string; name: string; infotext?: string | null; created_at: string; tab?: string | null }[]; next_cursor: string | null; total: number }

  it("lists the outputs newest first, a page at a time, as GET /outputs does", async () => {
    const response = await call("GET", "/outputs?limit=5")
    expect(response.status).toBe(200)
    const first = (await response.json()) as Page
    expect(first.items).toHaveLength(5)
    expect(first.total).toBeGreaterThan(5)
    for (const item of first.items) {
      expect(Object.keys(item).sort()).toEqual(["created_at", "height", "id", "infotext", "job_id", "kind", "name", "tab", "width"])
      expect(item.kind).toBe("image")
    }
    const times = first.items.map((item) => Date.parse(item.created_at))
    expect(times).toEqual([...times].sort((a, b) => b - a))
    const next = (await (await call("GET", `/outputs?limit=5&cursor=${first.next_cursor}`)).json()) as Page
    expect(next.items.map((item) => item.id).filter((id) => first.items.some((item) => item.id === id))).toEqual([])
    // `q` searches the infotext and the names; `tab` keeps one tab's
    const found = (await (await call("GET", "/outputs?q=LOWRES&tab=img2img&limit=200")).json()) as Page
    expect(found.items.length).toBeGreaterThan(0)
    expect(found.items.every((item) => item.tab === "img2img" && /lowres/i.test(`${item.name}\n${item.infotext}`))).toBe(true)
  })

  it("stops generate forever, and clears the queue on an interrupt that asks", async () => {
    await call("POST", "/queue/pause")
    const queued = (await (await call("POST", "/jobs", { tab: "txt2img", core: { prompt: "a fox" }, scripts: {}, forever: true })).json()) as { job_ids?: string[]; job_id?: string }
    const id = queued.job_ids?.[0] ?? queued.job_id ?? ""
    expect((await (await call("GET", "/queue/items")).json()) as { id: string; forever: boolean }[]).toContainEqual(expect.objectContaining({ id, forever: true }))
    const stopped = await call("POST", "/queue/stop-forever")
    expect(stopped.status).toBe(200)
    expect(await stopped.json()).toEqual(expect.objectContaining({ size: 1, paused: true }))
    expect((await (await call("GET", "/queue/items")).json()) as { id: string; forever: boolean }[]).toContainEqual(expect.objectContaining({ id, forever: false }))
    const cleared = await call("POST", "/interrupt", { clear_queue: true })
    expect(cleared.status).toBe(200)
    expect(await cleared.json()).toEqual({ size: 0, paused: true, running: null })
    expect((await (await call("GET", `/jobs/${id}`)).json()) as { status: string }).toEqual(expect.objectContaining({ status: "cancelled" }))
    await call("POST", "/queue/resume")
  })
})
