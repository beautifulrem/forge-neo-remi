import { renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import { notifyResults, resultCount, useProgressTitle } from "./effects"

function setApp(options: Record<string, unknown>, userAssets: string[] = []) {
  queryClient.setQueryData(keys.app, { ...structuredClone(bootstrap), options: { ...bootstrap.options, ...options }, user_assets: userAssets })
}

describe("result notifications", () => {
  afterEach(() => {
    queryClient.clear()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("does not count a batch's grid", () => {
    setApp({ return_grid: true })
    expect(resultCount(5)).toBe(4)
    expect(resultCount(1)).toBe(1)
    setApp({ return_grid: false })
    expect(resultCount(5)).toBe(5)
  })

  it("plays the user's notification.mp3 only when it exists", () => {
    const play = vi.fn(() => Promise.resolve())
    const created: string[] = []
    vi.stubGlobal(
      "Audio",
      class {
        volume = 1
        currentTime = 0
        constructor(src: string) {
          created.push(src)
        }
        play = play
      },
    )
    vi.spyOn(document, "hasFocus").mockReturnValue(false)
    setApp({ notification_audio: true, notification_volume: 50 })
    notifyResults(1)
    expect(play).not.toHaveBeenCalled()
    setApp({ notification_audio: true, notification_volume: 50 }, ["notification.mp3"])
    notifyResults(1)
    expect(created).toEqual(["/api/v2/user-assets/notification.mp3"])
    expect(play).toHaveBeenCalledTimes(1)
  })
})

describe("the progress title", () => {
  it("ends on the newest page title when the app renamed the page during the job", () => {
    document.title = "Generate · Remi"
    const job = (step: number) => ({ jobId: "j", step, steps: 10 })
    const { rerender, unmount } = renderHook(({ step }) => useProgressTitle(job(step), true), { initialProps: { step: 4 } })
    expect(document.title).toBe("[40%] · Generate · Remi")
    document.title = "生成 · Remi" // the language changed mid-job
    rerender({ step: 5 })
    expect(document.title).toBe("[50%] · 生成 · Remi")
    document.title = "图生图 · Remi" // and again, then the job ends
    unmount()
    expect(document.title).toBe("图生图 · Remi")
  })
})
