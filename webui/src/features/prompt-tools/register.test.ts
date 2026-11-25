/**
 * The completion engine loads at the first completion (typing asks for it): when its chunk cannot
 * be fetched, there are no completions, and the failure is told to the chunk reload (a new build on
 * the server has no such chunk any more), as every other load the user asked for is.
 */
import type { CompletionContext } from "@codemirror/autocomplete"
import { afterEach, describe, expect, it, vi } from "vitest"

afterEach(() => {
  vi.doUnmock("@/features/prompt-tools/completion/source")
  vi.resetModules()
})

describe("the prompt tools' completion source", () => {
  it("tells the chunk reload when the completion engine could not load", async () => {
    vi.resetModules()
    vi.doMock("@/features/prompt-tools/completion/source", () => {
      throw new TypeError("Failed to fetch dynamically imported module")
    })
    const { onLoadFailed } = await import("@/lib/retryable-lazy")
    const told = vi.fn()
    const stop = onLoadFailed(told)
    await import("./register")
    const { completionSources } = await import("@/features/prompt-editor/completion-registry")
    const entry = completionSources().find((source) => source.id === "prompt-tools")
    expect(entry).toBeDefined()
    const result = await entry!.source({} as CompletionContext)
    expect(result).toBeNull()
    expect(told).toHaveBeenCalledTimes(1)
    stop()
  })
})
