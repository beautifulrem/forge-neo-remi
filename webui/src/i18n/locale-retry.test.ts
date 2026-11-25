/**
 * A language whose chrome catalog (its own lazy chunk) could not be fetched is asked for again: the
 * app stayed in English until a reload, and the rejection went unhandled.
 */
import { afterEach, describe, expect, it, vi } from "vitest"

const chunk = vi.hoisted(() => ({ reachable: false, loads: 0 }))

vi.mock("../locales/zh-CN.json", () => {
  chunk.loads += 1
  if (!chunk.reachable) throw new TypeError("Failed to fetch dynamically imported module")
  return { default: { "Load more": "加载更多" } }
})

const { activateLocale, i18n } = await import("./index")
const { onLoadFailed, retryFailedLoads } = await import("@/lib/retryable-lazy")

describe("a language whose catalog could not be fetched", () => {
  afterEach(async () => {
    vi.restoreAllMocks()
    chunk.reachable = false
    await activateLocale("en")
  })

  it("is switched to once the browser is back online", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response("{}", { headers: { "Content-Type": "application/json" } }))
    const failed = vi.fn()
    const stop = onLoadFailed(failed)
    try {
      await expect(activateLocale("zh_CN")).resolves.toBeUndefined()
      expect(i18n.language).toBe("en")
      expect(failed).toHaveBeenCalled()
      chunk.reachable = true
      window.dispatchEvent(new Event("online"))
      await vi.waitFor(() => expect(i18n.language).toBe("zh-CN"))
      expect(i18n.t("Load more")).toBe("加载更多")
    } finally {
      stop()
    }
  })

  it("is not switched to when another language was chosen meanwhile", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response("{}", { headers: { "Content-Type": "application/json" } }))
    await activateLocale("zh_CN")
    await activateLocale("en")
    chunk.reachable = true
    retryFailedLoads()
    window.dispatchEvent(new Event("online"))
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(i18n.language).toBe("en")
  })
})
