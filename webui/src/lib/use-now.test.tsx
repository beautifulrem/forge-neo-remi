/**
 * The clock relative times read: one timer for every component that shows one, running only while
 * one is on screen.
 */
import { act, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { NOW_TICK_MS, useNow } from "./use-now"

afterEach(() => vi.useRealTimers())

function Clock({ label }: { label: string }) {
  const now = useNow()
  return <p>{`${label} ${new Date(now).toISOString()}`}</p>
}

describe("useNow", () => {
  it("advances every component on one shared timer, stopped once none is left", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-03-11T13:00:00Z"))
    const started = vi.spyOn(globalThis, "setInterval")
    const stopped = vi.spyOn(globalThis, "clearInterval")
    const view = render(
      <>
        <Clock label="a" />
        <Clock label="b" />
      </>,
    )
    expect(view.container.textContent).toBe("a 2026-03-11T13:00:00.000Zb 2026-03-11T13:00:00.000Z")
    expect(started).toHaveBeenCalledTimes(1)
    act(() => vi.advanceTimersByTime(NOW_TICK_MS))
    expect(view.container.textContent).toBe("a 2026-03-11T13:00:30.000Zb 2026-03-11T13:00:30.000Z")
    view.unmount()
    expect(stopped).toHaveBeenCalledTimes(1)
  })

  it("starts from the current time after a while with nobody reading it", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-03-11T13:00:00Z"))
    render(<Clock label="a" />).unmount()
    vi.setSystemTime(new Date("2026-03-11T15:00:00Z"))
    const view = render(<Clock label="a" />)
    expect(view.container.textContent).toBe("a 2026-03-11T15:00:00.000Z")
    view.unmount()
  })
})
