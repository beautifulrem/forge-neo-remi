import * as sonner from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { i18n } from "@/i18n"
import { applyAdvicePatch, CONFLICTS_ACTION, runAction, splitResponse } from "./actions"
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
