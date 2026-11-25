import { toast } from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createFormStore } from "@/ui/form-store"
import { insertPromptToken, registerActivePrompt } from "./active-editor"
import type { PromptEditorHandle } from "./PromptEditor"

const promptForm = (formId: string, locked: boolean) => {
  const store = createFormStore(formId, { persistDrafts: false })
  store.getState().load([
    { owner: "core", layout: { type: "text", name: "prompt", label: "Prompt" }, defaults: { prompt: "portrait" } },
    { owner: "keep", layout: { type: "toggle", name: "prompt", label: "Keep prompt", locks: "core.prompt" }, defaults: { prompt: locked } },
  ])
  return store
}

const editor = (doc: string): PromptEditorHandle => {
  const view = { state: { doc: { toString: () => doc, length: doc.length }, selection: { main: { from: doc.length, to: doc.length } } } }
  return { focus: vi.fn(), insertText: vi.fn(), view: () => view as never }
}

describe("insertPromptToken", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    registerActivePrompt("ins-editor", null)
  })

  it("appends the token to the prompt when no editor is mounted", () => {
    const store = promptForm("ins-plain", false)
    expect(insertPromptToken("ins-plain", "<image_1>", store)).toBe(true)
    expect(store.getState().values.core?.prompt).toBe("portrait <image_1>")
  })

  it("leaves a locked prompt alone and names the lock, without an editor", async () => {
    const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
    const store = promptForm("ins-locked", true)
    expect(insertPromptToken("ins-locked", "<image_1>", store)).toBe(false)
    expect(store.getState().values.core?.prompt).toBe("portrait")
    await Promise.resolve()
    expect(info).toHaveBeenCalledExactlyOnceWith("Overridden by a lock: Prompt", { id: "locks:ins-locked" })
  })

  it("leaves a locked prompt alone and names the lock, with an editor", async () => {
    const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
    const store = promptForm("ins-editor", true)
    const handle = editor("portrait")
    registerActivePrompt("ins-editor", handle)
    expect(insertPromptToken("ins-editor", "<image_1>", store)).toBe(false)
    expect(handle.insertText).not.toHaveBeenCalled()
    await Promise.resolve()
    expect(info).toHaveBeenCalledExactlyOnceWith("Overridden by a lock: Prompt", { id: "locks:ins-editor" })
  })

  it("inserts at the cursor of a mounted editor", () => {
    const store = promptForm("ins-editor", false)
    const handle = editor("portrait")
    registerActivePrompt("ins-editor", handle)
    expect(insertPromptToken("ins-editor", "<image_1>", store)).toBe(true)
    expect(handle.insertText).toHaveBeenCalledWith(" <image_1>")
  })
})
