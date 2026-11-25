/**
 * A pinned bar's depth goes to its scroller twice: as scroll padding, so a control scrolled into
 * view stops clear of the bar, and as `--pinned-top` / `--pinned-bottom`, so the page can keep
 * what it lays out at the scroller's edge (JSON Builder's canvas, whose zoom bar sat under the send
 * bar on a phone held sideways) above the bar.
 */
import { render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { type PinnedEdge, usePinnedBar } from "./use-pinned-bar"

afterEach(() => vi.restoreAllMocks())

function Bar({ edge }: { edge: PinnedEdge }) {
  return <div ref={usePinnedBar(edge)} data-testid="bar" />
}

function Scroller({ bar }: { bar: PinnedEdge | null }) {
  return (
    <div data-testid="scroller" style={{ overflowY: "auto" }}>
      {bar ? <Bar edge={bar} /> : null}
    </div>
  )
}

describe("a pinned bar", () => {
  it("gives its scroller its depth as scroll padding and as a custom property, until it goes", () => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(49)
    const { getByTestId, rerender } = render(<Scroller bar="bottom" />)
    const scroller = getByTestId("scroller")
    // the bar's height and 8 px of clearance
    expect(scroller.style.scrollPaddingBottom).toBe("57px")
    expect(scroller.style.getPropertyValue("--pinned-bottom")).toBe("57px")
    expect(scroller.style.getPropertyValue("--pinned-top")).toBe("")

    rerender(<Scroller bar="top" />)
    expect(scroller.style.getPropertyValue("--pinned-top")).toBe("57px")
    expect(scroller.style.getPropertyValue("--pinned-bottom")).toBe("")

    rerender(<Scroller bar={null} />)
    expect(scroller.style.scrollPaddingTop).toBe("")
    expect(scroller.style.getPropertyValue("--pinned-top")).toBe("")
  })
})
