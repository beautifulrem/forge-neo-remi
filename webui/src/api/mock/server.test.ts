/**
 * The mock's refused values read as the server's: the server's English sentence, its template and
 * its values (`remi.ui.Sentence`), so the zh mock preview shows them in Chinese through the
 * catalog sample, as the product does.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { sentenceParams } from "@/api/http"
import { validationMessage } from "@/features/errors/error-text"
import { i18n } from "@/i18n"
import { serverCatalogZh } from "./fixtures"
import { actionPatch, validateJob } from "./server"
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

describe("the mock's model load", () => {
  afterEach(() => vi.useRealTimers())

  it("sends what the server sends: one model.loading with its stage and model, then model.loaded", async () => {
    vi.useFakeTimers()
    const socket = new MockSocket()
    const events: Record<string, unknown>[] = []
    socket.onmessage = (event) => {
      if (typeof event.data === "string") events.push(JSON.parse(event.data) as Record<string, unknown>)
    }
    await vi.advanceTimersByTimeAsync(200)
    actionPatch("core", "__forge_preset__", { preset: "flux" } as never)
    await vi.advanceTimersByTimeAsync(5000)
    socket.close()
    expect(events.filter((event) => event.type === "model.loading")).toEqual([{ type: "model.loading", stage: "loading", model: "flux preset" }])
    expect(events.filter((event) => event.type === "model.loaded")).toEqual([{ type: "model.loaded", model: "flux preset" }])
  })
})
