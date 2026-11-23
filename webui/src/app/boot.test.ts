import { afterEach, describe, expect, it, vi } from "vitest"
import { settleWithin } from "./boot"

describe("settleWithin", () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it("ends when the work settles, resolved or rejected", async () => {
    await expect(settleWithin(Promise.resolve(1), 1000)).resolves.toBeUndefined()
    await expect(settleWithin(Promise.reject(new Error("down")), 1000)).resolves.toBeUndefined()
  })

  it("ends after the limit when the server never answers", async () => {
    vi.useFakeTimers()
    let done = false
    void settleWithin(new Promise(() => {}), 4000).then(() => {
      done = true
    })
    await vi.advanceTimersByTimeAsync(3999)
    expect(done).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    expect(done).toBe(true)
  })
})
