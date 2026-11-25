import { describe, expect, it } from "vitest"
import { flushPendingWrites, pendingWrite, registerPendingWrite } from "./pending-writes"

const tick = () => new Promise((resolve) => setTimeout(resolve, 10))

describe("pending writes", () => {
  it("a flush waits for the registered flushes and the values on their way", async () => {
    let release!: () => void
    const write = pendingWrite("form-a", () => new Promise<void>((resolve) => (release = resolve)))
    let done = false
    const flush = flushPendingWrites("form-a").then(() => (done = true))
    await tick()
    expect(done).toBe(false)
    release()
    await flush
    await write
    expect(done).toBe(true)
  })

  // Replacing a background flushes the layers' last strokes; it runs inside an upload's write, which
  // it would otherwise wait for: the write never ended.
  it("a write that flushes the registered writes only does not wait for itself", { timeout: 1000 }, async () => {
    let layers = 0
    const unregister = registerPendingWrite("form-b", async () => void layers++)
    let written = false
    await pendingWrite("form-b", async () => {
      await tick() // the upload
      await flushPendingWrites("form-b", { values: false })
      written = true
    })
    expect(written).toBe(true)
    expect(layers).toBe(1)
    unregister()
  })
})
