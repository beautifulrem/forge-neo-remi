import * as sonner from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { i18n } from "@/i18n"
import { ADVICE_ACTION, applyAdvicePatch, CONFLICTS_ACTION, hasModifiedAdviceParam, runAction, runBulkChangeEffects, runChangeEffects, setAdviceParams, splitResponse } from "./actions"
import { createFormStore } from "./form-store"
import type { AdviceItem } from "./schema"

describe("splitResponse", () => {
  it("separates reserved keys and resolves cross-owner params", () => {
    const split = splitResponse("anima", {
      preset: { value: "detail", label: "Preset", bogus: 1 },
      "core.steps": { value: 40, maximum: 80 },
      __toast__: { level: "success", message: "ok" },
      __outputs__: { status: "done" },
      __settings__: { sd_vae: "Automatic" },
      __future__: { anything: true },
      broken: "not an object",
    })
    expect(split.params).toEqual([
      ["anima", "preset", { value: "detail", label: "Preset" }],
      ["core", "steps", { value: 40, maximum: 80 }],
    ])
    expect(split.toast).toEqual({ level: "success", message: "ok" })
    expect(split.outputs).toEqual({ status: "done" })
    expect(split.settings).toEqual({ sd_vae: "Automatic" })
  })

  it("takes a toast message with params (remi.ui.message)", () => {
    const message = { text: "Saved template: {{name}}", params: { name: "mine" } }
    expect(splitResponse("x", { __toast__: { level: "success", message } }).toast).toEqual({ level: "success", message })
    expect(splitResponse("x", { __toast__: { message: { text: 3 } } }).toast).toBeUndefined()
  })

  it("takes the parts of a merged toast (remi.ui.merge_patches), or none when one is no message", () => {
    const parts = ["Done.", { text: "Saved template: {{name}}", params: { name: "mine" } }]
    const toast = { level: "success", message: "Done.; Saved template: mine", parts }
    expect(splitResponse("x", { __toast__: toast }).toast).toEqual({ level: "success", message: "Done.; Saved template: mine", parts })
    expect(splitResponse("x", { __toast__: { ...toast, parts: ["Done.", 3] } }).toast).toEqual({ level: "success", message: "Done.; Saved template: mine" })
    expect(splitResponse("x", { __toast__: { ...toast, parts: "Done." } }).toast).toEqual({ level: "success", message: "Done.; Saved template: mine" })
  })

  it("tolerates empty and malformed responses", () => {
    expect(splitResponse("x", null)).toEqual({ params: [] })
    expect(splitResponse("x", { __toast__: { message: 3 } })).toEqual({ params: [] })
  })

  it("parses __advice__ and __capabilities__ reserved keys", () => {
    const split = splitResponse("core", {
      __advice__: [
        {
          code: "qwen21.turbo",
          severity: "warning",
          message: "Use 8 steps for Turbo",
          patch: { "sampler.steps": { value: 8 } },
          action_label: "Set 8 steps",
        },
        { bad: true },
      ],
      __capabilities__: {
        preset: "qwen21",
        dit_name: "qwen21",
        engine_label: "Qwen-Image-2.1",
        capabilities: ["rgba_output"],
        max_references: 10,
        reference_slot_format: "<image_{i}>",
        supports_rgba: true,
        supports_control_blocks: true,
      },
    })
    expect(split.advice).toEqual([
      {
        code: "qwen21.turbo",
        severity: "warning",
        message: "Use 8 steps for Turbo",
        patch: { "sampler.steps": { value: 8 } },
        action_label: "Set 8 steps",
      },
    ])
    expect(split.capabilities?.preset).toBe("qwen21")
  })
})

describe("action patches a lock keeps", () => {
  afterEach(() => vi.restoreAllMocks())

  function lockedForm() {
    const store = createFormStore("locked-action", { persistDrafts: false })
    store.getState().load([
      { owner: "core", layout: { type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1 }, defaults: { steps: 20 } },
      { owner: "pin", layout: { type: "toggle", name: "on", label: "Pin steps", locks: "core.steps" }, defaults: { on: true } },
    ])
    return store
  }

  it("names them in the action's toast, or in a toast of their own", async () => {
    const info = vi.spyOn(sonner.toast, "info").mockImplementation((() => "") as never)
    const success = vi.spyOn(sonner.toast, "success").mockImplementation((() => "") as never)
    const store = lockedForm()
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({ steps: { value: 8 }, __toast__: { level: "success", message: "Preset applied" } }))
    await runAction({ store, tab: "txt2img" }, "core", "__preset__")
    expect(store.getState().values.core?.steps).toBe(20)
    expect(success).toHaveBeenCalledWith("Preset applied", { description: "Overridden by a lock: Steps" })
    await Promise.resolve()
    expect(info).not.toHaveBeenCalled() // named once, in the action's toast

    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({ steps: { value: 8 } }))
    await runAction({ store, tab: "txt2img" }, "core", "turbo")
    expect(info).toHaveBeenCalledExactlyOnceWith("Overridden by a lock: Steps", { id: "locks:locked-action" })

    // a patch that leaves the locked value as it is says nothing
    info.mockClear()
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({ steps: { value: 20, enabled: true } }))
    await runAction({ store, tab: "txt2img" }, "core", "turbo")
    expect(info).not.toHaveBeenCalled()
  })

  // the conflicts pass runs on load, after every paste and on conflict param edits: the lock is
  // the user's own, and the paste or recipe that started the pass already named what it kept
  it("leave the conflicts pass quiet", async () => {
    const info = vi.spyOn(sonner.toast, "info").mockImplementation((() => "") as never)
    const store = lockedForm()
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({ steps: { value: 8, enabled: false } }))
    await runAction({ store, tab: "txt2img" }, "core", CONFLICTS_ACTION)
    await runAction({ store, tab: "txt2img" }, "core", CONFLICTS_ACTION)
    expect(store.getState().values.core?.steps).toBe(20)
    expect(info).not.toHaveBeenCalled()
  })

  it("names them for a one-click advice fix", async () => {
    const info = vi.spyOn(sonner.toast, "info").mockImplementation((() => "") as never)
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({}))
    const store = lockedForm()
    await applyAdvicePatch({ store, tab: "txt2img" }, { code: "turbo", severity: "warning", message: "Use 8 steps", patch: { "core.steps": { value: 8 } } })
    expect(store.getState().values.core?.steps).toBe(20)
    expect(info).toHaveBeenCalledExactlyOnceWith("Overridden by a lock: Steps", { id: "locks:locked-action" })
  })
})

// A merged toast's English message no catalog has: each part is translated on its own
describe("a merged action toast", () => {
  afterEach(async () => {
    vi.restoreAllMocks()
    await i18n.changeLanguage("en")
  })

  it("shows each part in the UI language", async () => {
    i18n.addResource("zh-CN", "server", "Preset applied", "已应用预设")
    i18n.addResource("zh-CN", "server", "Steps clamped to {{maximum}}", "步数已限制为 {{maximum}}")
    const success = vi.spyOn(sonner.toast, "success").mockImplementation((() => "") as never)
    const store = createFormStore("merged-toast", { persistDrafts: false })
    const toast = {
      level: "success",
      message: "Preset applied; Steps clamped to 80",
      parts: ["Preset applied", { text: "Steps clamped to {{maximum}}", params: { maximum: "80" } }],
    }
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({ __toast__: toast }))
    await i18n.changeLanguage("zh-CN")
    await runAction({ store, tab: "txt2img" }, "core", "__preset__")
    expect(success).toHaveBeenLastCalledWith("已应用预设 · 步数已限制为 80", undefined)

    // in English (and for parts no catalog has) each part is its English text
    await i18n.changeLanguage("en")
    await runAction({ store, tab: "txt2img" }, "core", "__preset__")
    expect(success).toHaveBeenLastCalledWith("Preset applied · Steps clamped to 80", undefined)
  })
})

// The server's error details are English, for logs: the toast says what failed in the UI language
describe("runAction failures", () => {
  afterEach(() => vi.restoreAllMocks())

  const cases: [string, () => Promise<Response>, string][] = [
    [
      "a server error",
      async () => Response.json({ error: "Internal Server Error", detail: "Internal server error (id 0a1b2c3d4e5f); the details are in the server log" }, { status: 500 }),
      "The server ran into an error (500).",
    ],
    ["no answer", async () => Promise.reject(new TypeError("Failed to fetch")), "The server cannot be reached."],
    ["an unknown action", async () => Response.json({ detail: "core has no action 'x'" }, { status: 404 }), "It is no longer there."],
  ]
  it.each(cases)("explains %s in the UI language", async (_name, answer, text) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(answer)
    const shown: unknown[][] = []
    vi.spyOn(sonner.toast, "error").mockImplementation(((...args: unknown[]) => shown.push(args)) as never)
    const store = createFormStore("probe", { persistDrafts: false })
    expect(await runAction({ store, tab: "txt2img" }, "core", "swap_dimensions")).toBeNull()
    expect(shown).toEqual([["Action failed", { description: text }]])
  })
})

// The fix wrote the form's values first and then saved the server options: when saving failed the
// toast said the fix could not be applied, yet half of it (the steps) already was.
describe("applyAdvicePatch", () => {
  afterEach(() => vi.restoreAllMocks())

  it("leaves the form as it was when the server options cannot be saved", async () => {
    vi.spyOn(api, "patchOptions").mockRejectedValue(new ApiError(422, "frozen"))
    const store = createFormStore("advice-probe", { persistDrafts: false })
    store.getState().load([
      { owner: "core", layout: { type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1, integer: true }, defaults: { steps: 30 } },
    ])
    const item: AdviceItem = { code: "turbo", severity: "warning", message: "Use 8 steps and PNG", patch: { steps: { value: 8 }, __options__: { samples_format: "png" } } }
    await expect(applyAdvicePatch({ store, tab: "txt2img" }, item)).rejects.toThrow()
    expect(store.getState().values.core?.steps).toBe(30)
  })
})

describe("reactive advice refresh", () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  function adviceForm(adviceParams?: string[]) {
    const store = createFormStore("reactive-advice", { persistDrafts: false })
    store.getState().load(
      [
        {
          owner: "core",
          layout: {
            type: "column",
            children: [
              { type: "text", name: "prompt", label: "Prompt", prompt: true },
              { type: "slider", name: "cfg_scale", label: "CFG Scale", minimum: 1, maximum: 30, step: 0.5 },
            ],
          },
          defaults: { prompt: "", cfg_scale: 1 },
        },
        {
          owner: "sampler",
          layout: { type: "slider", name: "steps", label: "Sampling steps", minimum: 1, maximum: 150, step: 1, integer: true },
          defaults: { steps: 8 },
        },
        {
          owner: "imagestitch",
          layout: { type: "image_list", name: "references", label: "References" },
          defaults: { references: [] },
        },
      ],
      {
        capabilities: {
          preset: "qwen21",
          dit_name: "qwen21",
          engine_label: "Qwen-Image-2.1",
          capabilities: ["multi_reference"],
          max_references: 2,
          reference_slot_format: "<image_{i}>",
          supports_rgba: true,
          supports_control_blocks: true,
        },
      },
    )
    setAdviceParams(store, adviceParams)
    return store
  }

  it("triggers debounced core/__advice__ when cfg_scale, steps, prompt, or image_list changes and clears advice on manual fix", async () => {
    vi.useFakeTimers()
    const calls: { url: string; body: Record<string, unknown> }[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const req = input as Request
      const body = (await req.json()) as Record<string, unknown>
      calls.push({ url: new URL(req.url).pathname, body })
      const cfg = Number((body.values as Record<string, unknown>)?.cfg_scale ?? 1)
      const steps = Number(((body.scripts as Record<string, Record<string, unknown>>)?.sampler?.steps) ?? 8)
      const advice: AdviceItem[] =
        cfg > 1.5 || steps > 12
          ? [{ code: "qwen21.turbo_mismatch", severity: "warning", message: `steps=${steps}, cfg=${cfg}` }]
          : []
      return Response.json({ __advice__: advice })
    })

    const store = adviceForm()
    const ctx = { store, tab: "txt2img" }

    // Rapid edits coalesce into a single debounced core/__advice__ request.
    store.getState().setValue("core", "cfg_scale", 7)
    runChangeEffects(ctx, "core", ["cfg_scale"])
    store.getState().setValue("sampler", "steps", 30)
    runChangeEffects(ctx, "sampler", ["steps"])
    expect(calls).toHaveLength(0)

    await vi.advanceTimersByTimeAsync(200)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.url).toBe("/api/v2/actions/core/__advice__")
    expect(store.getState().advice).toHaveLength(1)
    expect(store.getState().advice[0]!.code).toBe("qwen21.turbo_mismatch")

    // Manually fixing both parameters refreshes advice and clears the warning.
    store.getState().setValue("core", "cfg_scale", 1)
    store.getState().setValue("sampler", "steps", 8)
    runBulkChangeEffects(ctx, [
      { owner: "core", names: ["cfg_scale"] },
      { owner: "sampler", names: ["steps"] },
    ])
    await vi.advanceTimersByTimeAsync(200)
    expect(calls).toHaveLength(2)
    expect(store.getState().advice).toEqual([])

    // Prompt and image_list edits also trigger advice refresh.
    store.getState().setValue("core", "prompt", "<lora:turbo:1> cat")
    runChangeEffects(ctx, "core", ["prompt"])
    await vi.advanceTimersByTimeAsync(200)
    expect(calls).toHaveLength(3)

    store.getState().setValue("imagestitch", "references", ["up-1", "up-2", "up-3"])
    runChangeEffects(ctx, "imagestitch", ["references"])
    await vi.advanceTimersByTimeAsync(200)
    expect(calls).toHaveLength(4)
  })

  it("respects server-declared TabSchema.advice_params", async () => {
    vi.useFakeTimers()
    const calls: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      calls.push(new URL((input as Request).url).pathname)
      return Response.json({ __advice__: [] })
    })

    const store = adviceForm(["core.cfg_scale", "imagestitch.references"])
    expect(hasModifiedAdviceParam(store, { tab: "txt2img", advice_params: ["core.cfg_scale"] })).toBe(false)
    store.getState().setValue("core", "cfg_scale", 5)
    expect(hasModifiedAdviceParam(store, { tab: "txt2img", advice_params: ["core.cfg_scale"] })).toBe(true)

    runChangeEffects({ store, tab: "txt2img" }, "core", ["cfg_scale"])
    await vi.advanceTimersByTimeAsync(200)
    expect(calls).toEqual(["/api/v2/actions/core/__advice__"])
  })

  it("applies __advice__ immediately when __forge_preset__ returns it", async () => {
    const store = adviceForm()
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({
        cfg_scale: { value: 1 },
        "sampler.steps": { value: 8 },
        __advice__: [{ code: "qwen21.rgba_jpeg", severity: "warning", message: "Switch Output to PNG" }],
      }),
    )

    await runAction({ store, tab: "txt2img" }, "core", "__forge_preset__", { preset: "qwen21" })
    expect(store.getState().values.core?.cfg_scale).toBe(1)
    expect(store.getState().values.sampler?.steps).toBe(8)
    expect(store.getState().advice).toEqual([{ code: "qwen21.rgba_jpeg", severity: "warning", message: "Switch Output to PNG", patch: null, action_label: null }])
  })

  it("discards stale in-flight __advice__ and __conflicts__ responses after __forge_preset__ starts", async () => {
    const store = adviceForm()
    const ctx = { store, tab: "txt2img" }

    let resolveStaleAdvice!: (value: Response) => void
    let resolveStaleConflicts!: (value: Response) => void
    const staleAdvicePromise = new Promise<Response>((resolve) => (resolveStaleAdvice = resolve))
    const staleConflictsPromise = new Promise<Response>((resolve) => (resolveStaleConflicts = resolve))

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL((input as Request).url).pathname
      if (url.endsWith(`/${ADVICE_ACTION}`)) return staleAdvicePromise
      if (url.endsWith(`/${CONFLICTS_ACTION}`)) return staleConflictsPromise
      if (url.endsWith("/__forge_preset__")) {
        return Response.json({
          __capabilities__: {
            preset: "flux3",
            dit_name: "flux3",
            engine_label: "FLUX.3",
            capabilities: [],
            max_references: 0,
            reference_slot_format: null,
            supports_rgba: false,
            supports_control_blocks: false,
          },
          __advice__: [{ code: "flux3.fresh", severity: "info", message: "Fresh Flux 3 advice" }],
        })
      }
      return Response.json({})
    })

    const adviceReq = runAction(ctx, "core", ADVICE_ACTION)
    const conflictsReq = runAction(ctx, "core", CONFLICTS_ACTION)

    await runAction(ctx, "core", "__forge_preset__", { preset: "flux3" })
    expect(store.getState().capabilities?.preset).toBe("flux3")
    expect(store.getState().advice.map((a) => a.code)).toEqual(["flux3.fresh"])

    // Resolve older in-flight responses after the preset switch completed: both must be discarded.
    resolveStaleAdvice(
      Response.json({
        __capabilities__: {
          preset: "qwen21",
          dit_name: "qwen21",
          engine_label: "Qwen-Image-2.1",
          capabilities: ["multi_reference"],
          max_references: 2,
          reference_slot_format: "<image_{i}>",
          supports_rgba: true,
          supports_control_blocks: true,
        },
        __advice__: [{ code: "qwen21.stale", severity: "warning", message: "Stale Qwen advice" }],
      }),
    )
    resolveStaleConflicts(Response.json({ cfg_scale: { value: 9 } }))

    expect(await adviceReq).toBeNull()
    expect(await conflictsReq).toBeNull()
    expect(store.getState().capabilities?.preset).toBe("flux3")
    expect(store.getState().advice.map((a) => a.code)).toEqual(["flux3.fresh"])
    expect(store.getState().values.core?.cfg_scale).toBe(1)
  })

  it("discards stale in-flight __advice__ started before applyAdvicePatch", async () => {
    const store = adviceForm()
    const ctx = { store, tab: "txt2img" }
    const adviceItem: AdviceItem = {
      code: "qwen21.turbo_mismatch",
      severity: "warning",
      message: "Use CFG 1",
      patch: { cfg_scale: { value: 1 } },
    }
    store.getState().setAdvice([adviceItem])

    let resolveStaleAdvice!: (value: Response) => void
    const staleAdvicePromise = new Promise<Response>((resolve) => (resolveStaleAdvice = resolve))
    let adviceCalls = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL((input as Request).url).pathname
      if (url.endsWith(`/${ADVICE_ACTION}`)) {
        adviceCalls++
        if (adviceCalls === 1) return staleAdvicePromise
        return Response.json({ __advice__: [] })
      }
      return Response.json({})
    })

    const staleReq = runAction(ctx, "core", ADVICE_ACTION)
    await applyAdvicePatch(ctx, adviceItem)

    resolveStaleAdvice(Response.json({ __advice__: [adviceItem] }))
    expect(await staleReq).toBeNull()
    expect(store.getState().advice).toEqual([])
  })

  it("discards __advice__ started while __forge_preset__ is in flight when __forge_preset__ settles first", async () => {
    const store = adviceForm()
    const ctx = { store, tab: "txt2img" }

    let resolvePreset!: (value: Response) => void
    let resolveInFlightAdvice!: (value: Response) => void
    const presetPromise = new Promise<Response>((resolve) => (resolvePreset = resolve))
    const inFlightAdvicePromise = new Promise<Response>((resolve) => (resolveInFlightAdvice = resolve))

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL((input as Request).url).pathname
      if (url.endsWith("/__forge_preset__")) return presetPromise
      if (url.endsWith(`/${ADVICE_ACTION}`)) return inFlightAdvicePromise
      return Response.json({})
    })

    const presetReq = runAction(ctx, "core", "__forge_preset__", { preset: "flux3" })
    const adviceReq = runAction(ctx, "core", ADVICE_ACTION)

    resolvePreset(
      Response.json({
        __capabilities__: {
          preset: "flux3",
          dit_name: "flux3",
          engine_label: "FLUX.3",
          capabilities: [],
          max_references: 0,
          reference_slot_format: null,
          supports_rgba: false,
          supports_control_blocks: false,
        },
        __advice__: [{ code: "flux3.fresh", severity: "info", message: "Fresh Flux 3 advice" }],
      }),
    )
    await presetReq
    expect(store.getState().capabilities?.preset).toBe("flux3")
    expect(store.getState().advice.map((a) => a.code)).toEqual(["flux3.fresh"])

    resolveInFlightAdvice(
      Response.json({
        __capabilities__: {
          preset: "qwen21",
          dit_name: "qwen21",
          engine_label: "Qwen-Image-2.1",
          capabilities: ["multi_reference"],
          max_references: 2,
          reference_slot_format: "<image_{i}>",
          supports_rgba: true,
          supports_control_blocks: true,
        },
        __advice__: [{ code: "qwen21.stale", severity: "warning", message: "Stale Qwen advice" }],
      }),
    )

    expect(await adviceReq).toBeNull()
    expect(store.getState().capabilities?.preset).toBe("flux3")
    expect(store.getState().advice.map((a) => a.code)).toEqual(["flux3.fresh"])
  })

  it("aborts stale applyAdvicePatch across patchOptions when __forge_preset__ or newer patch resolves first", async () => {
    const store = adviceForm()
    const ctx = { store, tab: "txt2img" }
    type PatchResult = Awaited<ReturnType<typeof api.patchOptions>>

    let resolvePatchOptions!: (value: PatchResult) => void
    const patchPromise = new Promise<PatchResult>((resolve) => (resolvePatchOptions = resolve))
    vi.spyOn(api, "patchOptions").mockReturnValue(patchPromise)
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL((input as Request).url).pathname
      if (url.endsWith("/__forge_preset__")) {
        return Response.json({
          cfg_scale: { value: 1 },
          __capabilities__: {
            preset: "flux3",
            dit_name: "flux3",
            engine_label: "FLUX.3",
            capabilities: [],
            max_references: 0,
            reference_slot_format: null,
            supports_rgba: false,
            supports_control_blocks: false,
          },
          __advice__: [{ code: "flux3.fresh", severity: "info", message: "Fresh Flux 3 advice" }],
        })
      }
      return Response.json({ __advice__: [] })
    })

    const staleAdvice: AdviceItem = {
      code: "qwen21.old_fix",
      severity: "warning",
      message: "Old advice",
      patch: { cfg_scale: { value: 99 }, __options__: { samples_format: "png" } },
    }
    const pendingPatch = applyAdvicePatch(ctx, staleAdvice)

    await runAction(ctx, "core", "__forge_preset__", { preset: "flux3" })
    expect(store.getState().values.core?.cfg_scale).toBe(1)

    resolvePatchOptions({} as PatchResult)
    expect(await pendingPatch).toBe(false)
    expect(store.getState().values.core?.cfg_scale).toBe(1)
    expect(store.getState().capabilities?.preset).toBe("flux3")
    expect(store.getState().advice.map((a) => a.code)).toEqual(["flux3.fresh"])

    // Also verify racing two applyAdvicePatch calls discards the older one.
    let resolveFirst!: (value: PatchResult) => void
    let resolveSecond!: (value: PatchResult) => void
    vi.spyOn(api, "patchOptions")
      .mockReturnValueOnce(new Promise<PatchResult>((r) => (resolveFirst = r)))
      .mockReturnValueOnce(new Promise<PatchResult>((r) => (resolveSecond = r)))
    const first = applyAdvicePatch(ctx, { code: "first", severity: "info", message: "1", patch: { cfg_scale: { value: 11 }, __options__: { a: 1 } } })
    const second = applyAdvicePatch(ctx, { code: "second", severity: "info", message: "2", patch: { cfg_scale: { value: 22 }, __options__: { a: 2 } } })
    resolveSecond({} as PatchResult)
    expect(await second).toBe(true)
    expect(store.getState().values.core?.cfg_scale).toBe(22)
    resolveFirst({} as PatchResult)
    expect(await first).toBe(false)
    expect(store.getState().values.core?.cfg_scale).toBe(22)
  })
})
