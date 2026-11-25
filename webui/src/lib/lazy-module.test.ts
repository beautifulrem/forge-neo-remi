/**
 * A module fetched on intent loads again after a failed fetch (offline, the server restarting): a
 * promise kept after it was rejected would keep the control broken for the rest of the session.
 */
import { describe, expect, it, vi } from "vitest"
import { lazyModule, loadForUse, prefetch } from "./lazy-module"
import { onLoadFailed } from "./retryable-lazy"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*", "!../lib/lazy-module.ts"], {
  query: "?raw",
  import: "default",
  eager: true,
})

describe("lazyModule", () => {
  it("fetches again on the next intent after a failed fetch", async () => {
    const module = { value: 1 }
    let fail = true
    const importer = vi.fn(() => (fail ? Promise.reject(new TypeError("Failed to fetch dynamically imported module")) : Promise.resolve(module)))
    const lazy = lazyModule(importer)
    await expect(lazy.load()).rejects.toThrow(TypeError)
    expect(lazy.get()).toBeNull()
    fail = false
    await expect(lazy.load()).resolves.toBe(module)
    expect(lazy.get()).toBe(module)
    await lazy.load()
    expect(importer).toHaveBeenCalledTimes(2)
  })

  it("lets a prefetch fail without an unhandled rejection", async () => {
    const lazy = lazyModule(() => Promise.reject(new TypeError("offline")))
    prefetch(lazy)
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(lazy.get()).toBeNull()
  })

  it("tells the chunk reload when a module someone asked for did not load, never for a hover", async () => {
    const told = vi.fn()
    const stop = onLoadFailed(told)
    const closed = vi.fn()
    const lazy = lazyModule(() => Promise.reject(new TypeError("Failed to fetch dynamically imported module")))
    prefetch(lazy)
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(told).not.toHaveBeenCalled()
    loadForUse(lazy, closed)
    await vi.waitFor(() => expect(told).toHaveBeenCalledTimes(1))
    expect(closed).toHaveBeenCalledTimes(1)
    stop()
  })

  it("is how the app keeps a module it fetched on intent", () => {
    // `x ??= import(...)` keeps a rejected promise for good; lazyModule forgets it
    const cached = Object.entries(sources)
      .filter(([, source]) => /\?\?=\s*import\(/.test(source))
      .map(([file]) => file.replace(/^\.\.\//, ""))
    expect(cached).toEqual([])
  })
})
