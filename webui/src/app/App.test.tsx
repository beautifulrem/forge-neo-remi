import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { FakeSocket } from "@/test/fake-socket"

describe("App ErrorBoundary resilience", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.resetModules()
    vi.doUnmock("./NavRail")
    vi.doUnmock("./Header")
    vi.unstubAllGlobals()
    queryClient.clear()
  })

  it("catches outer shell failures with Try again recovery", async () => {
    let failNav = true
    vi.resetModules()
    vi.stubGlobal("WebSocket", FakeSocket)
    vi.spyOn(console, "error").mockImplementation(() => {})
    vi.doMock("./NavRail", () => ({
      NavRail: () => {
        if (failNav) throw new Error("nav crashed")
        return <nav aria-label="Primary">Nav ok</nav>
      },
      BottomBar: () => null,
    }))

    const { App } = await import("./App")
    render(
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>,
    )

    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong")
    expect(screen.getByText("nav crashed")).toBeVisible()
    failNav = false
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    expect(await screen.findByRole("navigation", { name: "Primary" })).toHaveTextContent("Nav ok")
  })

  it("isolates Header crashes without bringing down the main page", async () => {
    vi.resetModules()
    vi.stubGlobal("WebSocket", FakeSocket)
    vi.spyOn(console, "error").mockImplementation(() => {})
    vi.doMock("./Header", () => ({
      Header: () => {
        throw new Error("header crashed")
      },
    }))

    const { App } = await import("./App")
    render(
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>,
    )

    expect(await screen.findByRole("alert")).toHaveTextContent("header crashed")
    expect(screen.getByRole("main")).toBeInTheDocument()
  })
})
