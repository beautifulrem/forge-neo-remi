import { QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, waitFor, within } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import { useHeaderSlotTarget } from "@/app/header-slot-store"
import { NetworksDrawerHost } from "./NetworksDrawerHost"
import { useNetworksUi } from "./store"

const drawer = vi.hoisted(() => ({ error: new Error("the drawer broke") as Error | null }))

vi.mock("./NetworksDrawer", () => ({
  default: () => {
    if (drawer.error) throw drawer.error
    return <p>the drawer</p>
  },
}))

describe("networks drawer host", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    drawer.error = new Error("the drawer broke")
    useNetworksUi.setState({ drawerOpen: false })
  })

  describe("in the header", () => {
    let header: HTMLElement
    beforeEach(() => {
      header = document.createElement("div")
      document.body.append(header)
      act(() => useHeaderSlotTarget.getState().setElement(header))
      render(
        <QueryClientProvider client={queryClient}>
          <NetworksDrawerHost tab="txt2img" />
        </QueryClientProvider>,
      )
    })
    afterEach(() => {
      act(() => useHeaderSlotTarget.getState().setElement(null))
      header.remove()
    })

    // Below lg, where its label is hidden, the button was 36 px wide beside the header's 28 px icon
    // buttons, which a phone's header has no room for. The label is still read out there: hidden
    // (display: none), the toggle took its name from its title, "Show networks" or "Hide networks".
    it("is a 28 px square where its label is hidden, keeping its one name", () => {
      const label = within(header).getByText("Networks")
      expect(label).toHaveClass("max-lg:sr-only")
      expect(label).not.toHaveClass("hidden")
      expect(label.closest("button")).toHaveClass("max-lg:size-7", "max-lg:px-0")
    })

    // On a 320 px phone the hairline after the tabs and its margins were the room a pinned quick
    // setting needed; there the button's icon sets it apart from the tabs' words.
    it("leaves out the hairline after the tabs on a phone", () => {
      expect(header.querySelector(".w-px")).toHaveClass("max-sm:hidden")
    })
  })

  // The drawer stays open across reloads: one that fails to render took Generate down with it on
  // every load, the prompts and the Generate button included.
  it("closes a drawer that fails to render and keeps the page", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const failed = vi.spyOn(toast, "error")
    useNetworksUi.setState({ drawerOpen: true })
    render(
      <QueryClientProvider client={queryClient}>
        <p>the prompts</p>
        <NetworksDrawerHost tab="txt2img" />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(useNetworksUi.getState().drawerOpen).toBe(false))
    expect(screen.getByText("the prompts")).toBeInTheDocument()
    expect(failed).toHaveBeenCalledWith("The networks panel could not be shown", expect.objectContaining({ description: "the drawer broke" }))
  })

  // The toast said the panel failed and offered nothing: opening it again meant finding its button.
  it("offers to open the panel again from the failure toast", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const failed = vi.spyOn(toast, "error")
    useNetworksUi.setState({ drawerOpen: true })
    render(
      <QueryClientProvider client={queryClient}>
        <NetworksDrawerHost tab="txt2img" />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(failed).toHaveBeenCalled())
    const action = (failed.mock.calls[0]?.[1] as { action?: { label: string; onClick: () => void } } | undefined)?.action
    expect(action?.label).toBe("Retry")
    drawer.error = null
    act(() => action?.onClick())
    expect(useNetworksUi.getState().drawerOpen).toBe(true)
    expect(await screen.findByText("the drawer")).toBeInTheDocument()
  })

  // A list the drawer needs failed to load: the toast said the server's English detail.
  it("explains a request that failed in the UI language", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const failed = vi.spyOn(toast, "error")
    drawer.error = new ApiError(503, "the model list is being refreshed")
    useNetworksUi.setState({ drawerOpen: true })
    render(
      <QueryClientProvider client={queryClient}>
        <NetworksDrawerHost tab="txt2img" />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(failed).toHaveBeenCalledWith("The networks panel could not be shown", expect.objectContaining({ description: "The server is busy; try again shortly." })))
  })
})
