import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import { NetworksDrawerHost } from "./NetworksDrawerHost"
import { useNetworksUi } from "./store"

const drawer = vi.hoisted(() => ({ error: new Error("the drawer broke") as Error }))

vi.mock("./NetworksDrawer", () => ({
  default: () => {
    throw drawer.error
  },
}))

describe("networks drawer host", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    drawer.error = new Error("the drawer broke")
    useNetworksUi.setState({ drawerOpen: false })
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
    expect(failed).toHaveBeenCalledWith("The networks panel could not be shown", { description: "the drawer broke" })
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
    await waitFor(() => expect(failed).toHaveBeenCalledWith("The networks panel could not be shown", { description: "The server is busy; try again shortly." }))
  })
})
