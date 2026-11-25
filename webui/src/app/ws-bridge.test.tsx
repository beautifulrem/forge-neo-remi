import { renderHook } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useProgress } from "@/api/progress"
import { queryClient } from "@/api/queries"
import { useWs } from "@/api/ws"
import { FakeSocket } from "@/test/fake-socket"
import { MODEL_LOADING_TIMEOUT_MS, useWsBridge } from "./ws-bridge"

async function connected() {
  const hook = renderHook(() => useWsBridge())
  await vi.waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0))
  const socket = FakeSocket.instances.at(-1)!
  socket.open()
  const send = (event: object) => socket.receive(JSON.stringify(event))
  send({ type: "hello", features: [] })
  return { hook, socket, send }
}

describe("ws-bridge", () => {
  beforeEach(() => {
    FakeSocket.instances = []
    vi.stubGlobal("WebSocket", FakeSocket)
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {}, modelLoading: null })
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it("warns about failed script steps of this client's jobs only", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const warning = vi.spyOn(toast, "warning").mockReturnValue(1)
    const { hook, send } = await connected()
    const warnings = [{ message: "{{script}} failed ({{hook}}): {{error}}", values: { script: "adetailer.py", hook: "postprocess_image", error: "boom" } }]
    send({ type: "job.completed", job_id: "theirs", seq: 1, status: "completed", outputs: [], warnings })
    expect(warning).not.toHaveBeenCalled()
    useProgress.setState({ owned: { mine: "txt2img" } })
    send({ type: "job.completed", job_id: "mine", seq: 2, status: "completed", outputs: [], warnings })
    expect(warning).toHaveBeenCalledWith(expect.any(String), { description: "adetailer.py failed (postprocess_image): boom" })
    hook.unmount()
  })

  it("refreshes schemas and pages when the options change", async () => {
    const invalidate = vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    send({ type: "options.changed", keys: ["ui_reorder_list"] })
    const keys = invalidate.mock.calls.map(([filters]) => filters?.queryKey)
    expect(keys).toEqual(expect.arrayContaining([["schema"], ["page"], ["options"], ["app"], ["models"], ["styles"]]))
    hook.unmount()
  })

  it("refetches the network lists when an option they follow changes", async () => {
    const invalidate = vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    const networks = () => invalidate.mock.calls.filter(([filters]) => filters?.queryKey?.[0] === "networks").length
    send({ type: "options.changed", keys: ["ui_reorder_list"] })
    expect(networks()).toBe(0)
    send({ type: "options.changed", keys: ["forge_preset"] })
    expect(networks()).toBe(1)
    send({ type: "options.changed", keys: ["extra_networks_default_multiplier", "samples_format"] })
    expect(networks()).toBe(2)
    hook.unmount()
  })

  it("refetches networks on a rescan and everything after a resync", async () => {
    const invalidate = vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    send({ type: "networks.changed", kind: "lora" })
    expect(invalidate.mock.calls.map(([filters]) => filters?.queryKey)).toEqual(expect.arrayContaining([["networks"], ["models"]]))
    invalidate.mockClear()
    send({ type: "pong" })
    expect(invalidate).not.toHaveBeenCalled()
    send({ type: "resync", dropped: 300 })
    expect(invalidate).toHaveBeenCalledWith()
    hook.unmount()
  })

  it("clears the previous job's preview when a job starts and keeps results per form", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    useWs.setState({ preview: { jobSeq: 1, step: 20, url: "blob:old" } })
    useProgress.getState().track("j2", "img2img")
    send({ type: "job.started", job_id: "j2", seq: 2, steps: 20 })
    expect(useWs.getState().preview).toBeNull()
    send({ type: "job.completed", job_id: "j2", seq: 2, outputs: [{ id: "f1" }] })
    expect(useProgress.getState().last).toEqual({ img2img: { jobId: "j2", outputs: [{ id: "f1" }], infotext: undefined } })
    // Another client's job never becomes "last".
    send({ type: "job.completed", job_id: "other", outputs: [{ id: "f2" }] })
    expect(useProgress.getState().last.txt2img).toBeUndefined()
    // A job that finishes before its POST answer arrives is attached when tracked.
    send({ type: "job.completed", job_id: "fast", outputs: [{ id: "f3" }] })
    useProgress.getState().track("fast", "txt2img")
    expect(useProgress.getState().last.txt2img?.jobId).toBe("fast")
    hook.unmount()
  })

  it("ends model loading on a terminal stage, model.loaded, a reconnect, or a timeout", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const { hook, send } = await connected()
    const loading = () => useProgress.getState().modelLoading
    send({ type: "model.loading", stage: "Loading weights" })
    expect(loading()?.stage).toBe("Loading weights")
    send({ type: "model.loading", stage: "done" })
    expect(loading()).toBeNull()
    send({ type: "model.loading", stage: "Loading weights" })
    send({ type: "model.loaded", model: "x" })
    expect(loading()).toBeNull()
    send({ type: "model.loading", stage: "Loading weights" })
    send({ type: "hello", features: [] })
    expect(loading()).toBeNull()
    send({ type: "model.loading", stage: "Loading weights" })
    vi.advanceTimersByTime(MODEL_LOADING_TIMEOUT_MS + 1)
    expect(loading()).toBeNull()
    hook.unmount()
  })

  it("drops the progress of a job the server no longer runs (missed events)", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    send({ type: "job.progress", job_id: "j1", seq: 1, step: 3, steps: 8 })
    send({ type: "queue.changed", size: 0, running: "j1" })
    expect(useProgress.getState().running?.jobId).toBe("j1")
    // After a reconnect the snapshot says nothing runs any more.
    send({ type: "hello", features: [] })
    send({ type: "queue.changed", size: 0, paused: false, running: null })
    expect(useProgress.getState().running).toBeNull()
    hook.unmount()
  })

  it("does not report a cancelled or interrupted job as failed", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const error = vi.spyOn(toast, "error").mockReturnValue(0)
    const { hook, send } = await connected()
    useProgress.getState().track("c1", "txt2img")
    useProgress.getState().track("f1", "txt2img")
    send({ type: "job.failed", job_id: "c1", status: "cancelled", error: "Cancelled" })
    expect(error).not.toHaveBeenCalled()
    send({ type: "job.failed", job_id: "f1", status: "failed", error: "CUDA out of memory" })
    expect(error).toHaveBeenCalledTimes(1)
    hook.unmount()
  })

  it("says how a job or a model load ended in the UI language", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const error = vi.spyOn(toast, "error").mockReturnValue(0)
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "Out of memory: the model was unloaded", "内存不足：模型已卸载")
    i18n.addResource("zh-CN", "server", "the model was not loaded", "模型未能加载")
    await i18n.changeLanguage("zh-CN")
    try {
      const { hook, send } = await connected()
      useProgress.getState().track("f2", "txt2img")
      send({ type: "job.failed", job_id: "f2", status: "failed", error: "Out of memory: the model was unloaded" })
      expect(error.mock.calls[0]![1]).toMatchObject({ description: "内存不足：模型已卸载" })
      send({ type: "model.loaded", model: "x", error: "the model was not loaded" })
      expect(error.mock.calls[1]![1]).toMatchObject({ description: "模型未能加载" })
      hook.unmount()
    } finally {
      await i18n.changeLanguage("en")
    }
  })

  it("says what a job noted and which values it refused in the UI language", async () => {
    // the server sends both with their templates: the English text alone left the toasts in English
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const warning = vi.spyOn(toast, "warning").mockReturnValue(1)
    const error = vi.spyOn(toast, "error").mockReturnValue(0)
    const { i18n } = await import("@/i18n")
    const turbo = "Anima-Turbo checkpoint with CFG {{cfg}} / {{steps}} steps: it is distilled for CFG 1 and 8-12 steps (see the Official Turbo preset); higher values over-cook it."
    const choice = "{{value}} is not one of the choices ({{choices}})"
    i18n.addResource("zh-CN", "server", turbo, "Anima-Turbo 模型配 CFG {{cfg}} / {{steps}} 步")
    i18n.addResource("zh-CN", "server", choice, "{{value}} 不在可选项（{{choices}}）中")
    await i18n.changeLanguage("zh-CN")
    try {
      const { hook, send } = await connected()
      useProgress.getState().track("w1", "txt2img")
      useProgress.getState().track("f3", "txt2img")
      send({ type: "job.completed", job_id: "w1", seq: 1, status: "completed", outputs: [], warnings: [{ message: turbo, values: { cfg: "4.5", steps: "36" } }] })
      expect(warning.mock.calls[0]![1]).toEqual({ description: "Anima-Turbo 模型配 CFG 4.5 / 36 步" })
      const refused = { owner: "core", param: "styles", message: "'Watercolor' is not one of the choices ('Cinematic')", template: choice, params: { value: "'Watercolor'", choices: "'Cinematic'" } }
      send({ type: "job.failed", job_id: "f3", status: "failed", error: "core.styles: 'Watercolor' is not one of the choices ('Cinematic')", errors: [refused] })
      expect(error.mock.calls[0]![1]).toEqual({ description: "'Watercolor' 不在可选项（'Cinematic'）中" })
      hook.unmount()
    } finally {
      await i18n.changeLanguage("en")
    }
  })

  // The toast listed the refused values' messages without the fields they are about.
  it("names the field of each value a failed job refused, by its form's label", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const error = vi.spyOn(toast, "error").mockReturnValue(0)
    const { tabSchema } = await import("@/api/mock/fixtures")
    const { keys } = await import("@/api/queries")
    queryClient.setQueryData(keys.schema("txt2img"), tabSchema("txt2img"))
    const { i18n } = await import("@/i18n")
    const choice = "{{value}} is not one of the choices ({{choices}})"
    i18n.addResource("zh-CN", "server", choice, "{{value}} 不在可选项（{{choices}}）中")
    i18n.addResourceBundle("zh-CN", "chrome", { Styles: "样式", "{{field}}: {{message}}": "{{field}}：{{message}}" }, true, true)
    await i18n.changeLanguage("zh-CN")
    try {
      const { hook, send } = await connected()
      useProgress.getState().track("f4", "txt2img")
      const refused = { owner: "core", param: "styles", message: "'Watercolor' is not one of the choices ('Cinematic')", template: choice, params: { value: "'Watercolor'", choices: "'Cinematic'" } }
      send({ type: "job.failed", job_id: "f4", tab: "txt2img", status: "failed", error: "core.styles: 'Watercolor' is not one of the choices ('Cinematic')", errors: [refused] })
      expect(error.mock.calls[0]![1]).toEqual({ description: "样式：'Watercolor' 不在可选项（'Cinematic'）中" })
      hook.unmount()
    } finally {
      queryClient.removeQueries({ queryKey: keys.schema("txt2img") })
      await i18n.changeLanguage("en")
    }
  })
})
