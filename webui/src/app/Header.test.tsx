import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useWs } from "@/api/ws"
import { ConnectionDot, Header } from "./Header"
import { HeaderSlot } from "./header-slot"

// The connection's state was a coloured dot alone, its name in a hover tooltip: a touch screen
// showed no words for it.
describe("ConnectionDot", () => {
  afterEach(() => act(() => useWs.setState({ status: "idle" })))

  it("says Offline beside the dot while the connection is lost", () => {
    useWs.setState({ status: "closed" })
    render(<ConnectionDot />)
    expect(screen.getByText("Offline")).toBeVisible()
    act(() => useWs.setState({ status: "reconnecting" }))
    expect(screen.getByText("Offline")).toBeVisible()
  })

  it("shows the dot alone while connected or first connecting", () => {
    useWs.setState({ status: "open" })
    render(<ConnectionDot />)
    expect(screen.queryByText("Offline")).toBeNull()
    act(() => useWs.setState({ status: "connecting" }))
    expect(screen.queryByText("Offline")).toBeNull()
  })
})

// On a 390 px phone the header had a few pixels to spare: the word pushed the appearance menu
// past the window. The word takes the dot's own room first, then the page title yields (the
// bottom bar names the page too); the controls keep their size.
describe("the header on a phone while offline", () => {
  afterEach(() => act(() => useWs.setState({ status: "idle" })))

  it("draws the word in the dot's room, without padding of its own", () => {
    useWs.setState({ status: "closed" })
    render(<ConnectionDot />)
    const box = screen.getByText("Offline").parentElement!
    expect(box).toHaveClass("gap-1")
    expect(box).not.toHaveClass("px-1.5")
  })

  it("lets the page title shrink below sm, with its full text as a title, and keeps the controls whole", () => {
    render(<Header title="Generate" locales={[]} />)
    const title = screen.getByRole("heading", { level: 1, name: "Generate" })
    expect(title).toHaveAttribute("title", "Generate")
    expect(title).toHaveClass("min-w-0", "truncate", "sm:shrink-0")
    expect(title).not.toHaveClass("shrink-0")
    const controls = screen.getAllByRole("button", { name: "Search" })[0]!.parentElement!
    expect(controls).toHaveClass("shrink-0", "sm:shrink")
  })
})

// On a phone Generate's title was cut to a stub ("G…" at 360 px in English) beside its txt2img /
// img2img tabs, which name the page already, as the bottom bar does. A page whose own tabs are in
// the header gives the title's room to them below sm: visually hidden, still the page's heading.
describe("the header on a phone, on a page with its own tabs", () => {
  it("hides the title visually below sm, keeping it as the page's heading", () => {
    render(
      <>
        <Header title="Generate" locales={[]} />
        <HeaderSlot>
          <nav aria-label="Mode" data-page-tabs="">
            <a href="#/txt2img">txt2img</a>
          </nav>
        </HeaderSlot>
      </>,
    )
    const header = screen.getByRole("banner")
    const title = screen.getByRole("heading", { level: 1, name: "Generate" })
    // read out, never hidden from screen readers
    expect(title).not.toHaveAttribute("aria-hidden")
    expect(title).not.toHaveClass("hidden")
    expect(header).toHaveClass("group/header")
    expect(title).toHaveClass("max-sm:group-has-data-page-tabs/header:sr-only")
    expect(header.querySelector("[data-page-tabs]")).toBe(screen.getByRole("navigation", { name: "Mode" }))
  })

  // At 320 px in English with a quick setting pinned and the connection lost, the word "Offline"
  // took the last 13 px: the appearance menu left the window. There the word gives way to the dot
  // (wraps to a second line its box clips), its state still read out; wherever it fits it shows.
  it("lets the word Offline give way to the dot where even the page's tabs leave no room", () => {
    useWs.setState({ status: "closed" })
    try {
      render(<Header title="Generate" locales={[]} />)
      const word = screen.getByText("Offline")
      const box = word.parentElement!
      expect(box).toHaveClass("h-7", "min-w-7", "flex-wrap", "content-start", "overflow-hidden", "shrink-0", "max-sm:shrink")
      // the word and the dot each a line 28 px tall: a wrapped word is below the box, not beside the dot
      expect(word).toHaveClass("h-7")
      expect(box.firstElementChild).toHaveClass("h-7")
      expect(screen.getByRole("status")).toHaveTextContent("Disconnected")
      // only on a phone, and only beside a page's tabs, does the row shrink the box (no title to cut first)
      const controls = screen.getAllByRole("button", { name: "Search" })[0]!.parentElement!
      expect(controls).toHaveClass("shrink-0", "max-sm:group-has-data-page-tabs/header:shrink")
    } finally {
      act(() => useWs.setState({ status: "idle" }))
    }
  })

  // At 320 px in English Generate's tabs, the networks button and four 28 px controls (a quick
  // setting pinned) left the appearance menu past the window: the controls sit 4 px apart there.
  it("sets the controls 4 px apart below sm", () => {
    render(<Header title="Generate" locales={[]} />)
    const controls = screen.getAllByRole("button", { name: "Search" })[0]!.parentElement!
    expect(controls).toHaveClass("gap-1.5", "max-sm:gap-1")
  })
})

describe("header control ErrorBoundary recovery", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.resetModules()
    vi.doUnmock("./ModelQuickBar")
  })

  it("shows a compact retry control when a header widget throws and recovers on retry", async () => {
    let shouldThrow = true
    vi.resetModules()
    vi.doMock("./ModelQuickBar", () => ({
      default: () => {
        if (shouldThrow) throw new Error("model bar boom")
        return <div data-testid="recovered-model-bar">Model ready</div>
      },
    }))
    vi.spyOn(console, "error").mockImplementation(() => {})
    const { Header: MockedHeader } = await import("./Header")
    render(<MockedHeader title="Generate" locales={[]} />)

    const retry = await screen.findByRole("button", { name: "Model (Try again)" })
    expect(retry).toHaveClass("h-7")
    shouldThrow = false
    fireEvent.click(retry)
    expect(await screen.findByTestId("recovered-model-bar")).toHaveTextContent("Model ready")
  })
})
