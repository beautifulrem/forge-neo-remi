import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { timeAgo } from "./helpers"

const at = (iso: string) => new Date(iso).getTime() / 1000

describe("timeAgo", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-03-11T13:00:00"))
  })
  afterEach(() => vi.useRealTimers())

  it("moves to the next unit when the rounded value reaches the limit", () => {
    const now = Date.now() / 1000
    expect(timeAgo(now - 3570, "en")).toBe("1 hour ago")
    expect(timeAgo(now - 86000, "en")).toBe("yesterday")
    expect(timeAgo(now - 59.6, "en")).toBe("1 minute ago")
  })

  it("counts days by the calendar", () => {
    expect(timeAgo(at("2026-03-10T01:00:00"), "en")).toBe("yesterday") // 36 hours ago
    expect(timeAgo(at("2026-03-09T23:00:00"), "en")).toBe("2 days ago") // 38 hours, two midnights
    expect(timeAgo(at("2026-03-10T01:00:00"), "zh-CN")).toBe("昨天")
    vi.setSystemTime(new Date("2026-03-11T00:30:00"))
    expect(timeAgo(at("2026-03-09T22:00:00"), "en")).toBe("2 days ago") // 26.5 hours, but two midnights
  })

  it("counts from the time it is given (the time a render read)", () => {
    const now = at("2026-03-11T13:00:00")
    expect(timeAgo(now - 30, "en", (now + 120) * 1000)).toBe("3 minutes ago")
    expect(timeAgo(now - 30, "zh-CN", (now + 120) * 1000)).toBe("3分钟前")
  })

  it("uses weeks, months and years further back", () => {
    expect(timeAgo(at("2026-02-25T13:00:00"), "en")).toBe("2 weeks ago")
    expect(timeAgo(at("2025-12-11T13:00:00"), "en")).toBe("3 months ago")
    expect(timeAgo(at("2024-03-11T13:00:00"), "en")).toBe("2 years ago")
  })
})
