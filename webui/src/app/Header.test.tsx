import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { useWs } from "@/api/ws"
import { ConnectionDot } from "./Header"

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
