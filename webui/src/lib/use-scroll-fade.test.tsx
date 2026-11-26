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

  // The editor's image toolbar shows once there is an image, after the editor that holds the hook has
  // mounted: the hook watched the element its ref held at mount, none, and the toolbar never faded.
  it("fades a strip that mounts after the component holding the hook", async () => {
    measure()
    function Late({ shown }: { shown: boolean }) {
      const ref = useRef<HTMLDivElement>(null)
      const fade = useScrollFade(ref, null)
      return shown ? (
        <div ref={ref} data-testid="strip" data-room="100" data-length="300" style={fade}>
          <span>a</span>
        </div>
      ) : null
    }
    const view = render(<Late shown={false} />)
    await act(async () => view.rerender(<Late shown />))
    await waitFor(() => expect(mask()).toBe("linear-gradient(to right, transparent, #000 0rem, #000 calc(100% - 1.5rem), transparent)"))
  })

  // The navigation rail scrolls up and down on a phone held sideways.
  it("fades the bottom of a strip that scrolls up and down", () => {
    measure()
    render(<Strip items={["a", "b", "c", "d"]} axis="y" />)
    expect(mask()).toBe("linear-gradient(to bottom, transparent, #000 0rem, #000 calc(100% - 1.5rem), transparent)")
  })
})

// The fade is a mask on the element it is given, and a mask cuts everything drawn outside that element's
// border box: on an editor toolbar that was the floating panel (`PANEL`, `material-float`), its edge and
// shadow went while it scrolled, and in forced colours its 1 px outline too. The panel wraps the strip.
describe("the strip a fade is on", () => {
  const sources = import.meta.glob<string>(["../**/*.tsx", "!../**/*.test.*"], { query: "?raw", import: "default", eager: true })

  it("is not a floating panel", () => {
    const strips = Object.entries(sources).flatMap(([file, source]) => [
      // the element a fade is on (`style={fade}`, or spread into its style)
      ...[...source.matchAll(/\bconst (\w+) = useScrollFade\(/g)].map(([, name]) => {
        const at = source.search(new RegExp(`style=\\{(?:${name}\\}|\\{[^}]*\\.\\.\\.${name}\\b)`))
        const tag = at < 0 ? "" : source.slice(source.lastIndexOf("<", at), source.indexOf(">", at) + 1)
        return { file, name, tag }
      }),
      // FadeStrip takes its classes from where it is used
      ...[...source.matchAll(/<FadeStrip\b[^>]*>/g)].map(([tag]) => ({ file, name: "FadeStrip", tag })),
    ])
    expect(strips.length).toBeGreaterThanOrEqual(6)
    expect(strips.filter(({ tag }) => !tag)).toEqual([])
    expect(strips.filter(({ tag }) => /\bPANEL\b|material-float|\bshadow-/.test(tag)).map(({ file, name }) => `${file}: ${name}`)).toEqual([])
  })
})

