import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { useWs } from "@/api/ws"
import { ConnectionDot, Header } from "./Header"

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
