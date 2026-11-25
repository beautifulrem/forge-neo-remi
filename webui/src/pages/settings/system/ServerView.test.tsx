import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import { ServerView } from "./ServerView"

const STATUS = { restartable: true, extension_access: true, settings_frozen: false, model_loaded: "anima.safetensors [0123456789]", model_hash: "0123456789abcdef", version: "1.0.0" }

function renderView() {
  vi.spyOn(api.system, "hashStatus").mockReturnValue(new Promise(() => {}))
  render(
    <QueryClientProvider client={queryClient}>
      <ServerView onImport={() => {}} onExport={() => {}} />
    </QueryClientProvider>,
  )
}

// While /server/status loaded, or when only it failed, the page said "No model loaded", offered
// "Load model" and called restarting "Not available" (start with webui.sh): none of it was known.
describe("ServerView", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("claims nothing about the model or restarting before the status arrives", () => {
    vi.spyOn(api.system, "status").mockReturnValue(new Promise(() => {}))
    renderView()
    expect(screen.queryByText("No model loaded")).toBeNull()
    expect(screen.queryByText("Not available")).toBeNull()
    expect(screen.queryByText(/webui\.sh/)).toBeNull()
    expect(screen.getByRole("button", { name: "Load model" })).toBeDisabled()
  })

  it("says the status could not be read, with a Retry, instead of a false state", async () => {
    const status = vi.spyOn(api.system, "status").mockRejectedValueOnce(new ApiError(404, "boom", { detail: "boom" })) // a 4xx is not retried
    status.mockResolvedValueOnce(STATUS as never)
    renderView()
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Could not read the server status")
    // in the version's place on the header line, so the panels below stay where they were
    expect(alert.closest("header")).not.toBeNull()
    expect(alert).toHaveClass("-my-1")
    expect(screen.queryByText(/^Version /)).toBeNull()
    expect(screen.queryByText("No model loaded")).toBeNull()
    expect(screen.queryByText("Not available")).toBeNull()
    expect(screen.queryByText(/webui\.sh/)).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("anima.safetensors")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Restart server" })).toBeEnabled()
  })

  it("gives the weights in memory's lone refresh button a 40 px target", async () => {
    vi.spyOn(api.system, "status").mockResolvedValue(STATUS as never)
    vi.spyOn(api.system, "loadedModels").mockReturnValue(new Promise(() => {}))
    renderView()
    fireEvent.click(screen.getByRole("button", { name: "Show the weights in memory" }))
    // 24 px and 8 px on each side; nothing else in its row takes taps
    expect(await screen.findByRole("button", { name: "Refresh" })).toHaveClass("size-6", "relative", "after:absolute", "after:-inset-2")
  })
})
