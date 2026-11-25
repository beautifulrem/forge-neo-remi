import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
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
})
