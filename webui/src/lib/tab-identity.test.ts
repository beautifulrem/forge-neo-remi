/** Two browser tabs simulated with their own session storage and BroadcastChannel. */
import { afterEach, describe, expect, it } from "vitest"
import { claimTabId, TAB_ID_KEY, type TabClaim } from "./tab-identity"

// a channel of its own: the app's tabs in other test files never hear these
const CHANNEL = "remi-tabs-identity-test"

function memoryStorage(entries: Record<string, string> = {}) {
  const map = new Map(Object.entries(entries))
  return { getItem: (key: string) => map.get(key) ?? null, setItem: (key: string, value: string) => void map.set(key, value) }
}

const claims: TabClaim[] = []

/** A tab starting with `storage` (a copy of another's, for a duplicated tab). */
function openTab(storage: ReturnType<typeof memoryStorage>, wait = 30) {
  const claim = claimTabId({ storage, channel: new BroadcastChannel(CHANNEL), wait })
  claims.push(claim)
  return claim
}

afterEach(() => {
  for (const claim of claims.splice(0)) claim.close()
})

describe("a tab's identity", () => {
  it("is new in a tab without one: its storage is its own at once", async () => {
    const storage = memoryStorage()
    expect(await openTab(storage, 10_000).owned).toBe(true)
    expect(storage.getItem(TAB_ID_KEY)).toEqual(expect.any(String))
  })

  // its own question must not come back to it as another tab's answer
  it("is kept by a reload when no other tab has it", async () => {
    const storage = memoryStorage({ [TAB_ID_KEY]: "tab-reloaded" })
    expect(await openTab(storage).owned).toBe(true)
    expect(storage.getItem(TAB_ID_KEY)).toBe("tab-reloaded")
  })

  it("is left to the open tab a duplicated tab copied it from: the copy takes a new one", async () => {
    const original = memoryStorage({ [TAB_ID_KEY]: "tab-original" })
    expect(await openTab(original).owned).toBe(true)
    const copy = memoryStorage({ [TAB_ID_KEY]: "tab-original" })
    const copyTab = openTab(copy)
    expect(await copyTab.owned).toBe(false)
    expect(copy.getItem(TAB_ID_KEY)).not.toBe("tab-original")
    expect(original.getItem(TAB_ID_KEY)).toBe("tab-original")
    // the copy's next reload keeps its new id: the original does not answer for it
    copyTab.close()
    const reloaded = memoryStorage({ [TAB_ID_KEY]: copy.getItem(TAB_ID_KEY)! })
    expect(await openTab(reloaded).owned).toBe(true)
  })

  it("is the original's again when the copy asks after the original has answered once", async () => {
    const original = memoryStorage({ [TAB_ID_KEY]: "tab-twice" })
    await openTab(original).owned
    expect(await openTab(memoryStorage({ [TAB_ID_KEY]: "tab-twice" })).owned).toBe(false)
    expect(await openTab(memoryStorage({ [TAB_ID_KEY]: "tab-twice" })).owned).toBe(false)
  })

  // a tab duplicated while the original reloads: both ask at once
  it("is kept by exactly one of two tabs that ask for it at the same time", async () => {
    // the page loads' random names sort either way: a few rounds take both turns
    for (let round = 0; round < 8; round++) {
      const first = openTab(memoryStorage({ [TAB_ID_KEY]: `tab-race-${round}` }))
      const second = openTab(memoryStorage({ [TAB_ID_KEY]: `tab-race-${round}` }))
      const owned = await Promise.all([first.owned, second.owned])
      expect(owned.filter(Boolean)).toHaveLength(1)
    }
  })

  it("is the tab's own without a channel to ask on", async () => {
    const storage = memoryStorage({ [TAB_ID_KEY]: "tab-alone" })
    expect(await claimTabId({ storage, channel: null }).owned).toBe(true)
  })
})
