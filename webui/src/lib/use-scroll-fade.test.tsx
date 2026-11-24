import { act, render, screen, waitFor } from "@testing-library/react"
import { useRef } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useScrollFade } from "./use-scroll-fade"

// jsdom has no layout: each strip says how long it is through `data-length` (its content) and
// `data-room` (what it shows), on the axis it scrolls.
function length(element: HTMLElement) {
  return Number(element.dataset.length ?? element.dataset.room ?? 0)
}
function room(element: HTMLElement) {
  return Number(element.dataset.room ?? 0)
}

/** The strip's mask as written, though jsdom writes black as rgb() and drops the default "to bottom". */
function mask() {
  return screen.getByTestId("strip").style.maskImage.replaceAll("rgb(0, 0, 0)", "#000").replace("linear-gradient(transparent", "linear-gradient(to bottom, transparent")
}

function Strip({ items, axis }: { items: string[]; axis?: "x" | "y" }) {
  const ref = useRef<HTMLDivElement>(null)
  const fade = useScrollFade(ref, null, axis)
  return (
    <div ref={ref} data-testid="strip" data-room="100" data-length={items.length * 40} style={fade}>
      {items.map((item) => (
        <span key={item}>{item}</span>
      ))}
    </div>
  )
}

describe("useScrollFade", () => {
  afterEach(() => vi.restoreAllMocks())

  function measure() {
    vi.spyOn(HTMLElement.prototype, "scrollWidth", "get").mockImplementation(function (this: HTMLElement) {
      return length(this)
    })
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function (this: HTMLElement) {
      return room(this)
    })
    vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockImplementation(function (this: HTMLElement) {
      return length(this)
    })
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
      return room(this)
    })
  }

  // The gallery's boards arrive after the strip mounts: the strip overflowed with no fade, because
  // only the strip's own size and scrolling were watched.
  it("fades the end once items arriving after it mounts overflow it", async () => {
    measure()
    const view = render(<Strip items={["a"]} />)
    expect(mask()).toBe("")
    await act(async () => view.rerender(<Strip items={["a", "b", "c", "d"]} />))
    await waitFor(() => expect(mask()).toBe("linear-gradient(to right, transparent, #000 0rem, #000 calc(100% - 1.5rem), transparent)"))
  })

  // The navigation rail scrolls up and down on a phone held sideways.
  it("fades the bottom of a strip that scrolls up and down", () => {
    measure()
    render(<Strip items={["a", "b", "c", "d"]} axis="y" />)
    expect(mask()).toBe("linear-gradient(to bottom, transparent, #000 0rem, #000 calc(100% - 1.5rem), transparent)")
  })
})
