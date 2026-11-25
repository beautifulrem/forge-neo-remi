import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import { ServerView } from "./ServerView"

const STATUS = { restartable: true, extension_access: true, settings_frozen: false, model_loaded: "anima.safetensors [0123456789]", model_hash: "0123456789abcdef", version: "1.0.0" }

function renderView(hashes: Awaited<ReturnType<typeof api.system.hashStatus>> | null = null) {
  vi.spyOn(api.system, "hashStatus").mockReturnValue(hashes ? Promise.resolve(hashes) : new Promise(() => {}))
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
    // the model's place says it is not known rather than staying empty
    expect(screen.getByText("Unknown")).toBeInTheDocument()
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

  it("keeps the weights shown when refreshing them fails, and says the refresh failed", async () => {
    vi.spyOn(api.system, "status").mockResolvedValue(STATUS as never)
    const weights = vi.spyOn(api.system, "loadedModels").mockResolvedValueOnce({ items: [{ name: "anima.safetensors", memory_mb: 6144, device: "cuda:0" }] } as never)
    weights.mockRejectedValueOnce(new ApiError(404, "boom", { detail: "boom" })) // a 4xx is not retried
    renderView()
    fireEvent.click(screen.getByRole("button", { name: "Show the weights in memory" }))
    const table = await screen.findByRole("table", { name: "Weights in memory" })
    expect(within(table).getByText("anima.safetensors")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))
    expect(await screen.findByRole("alert")).toBeInTheDocument()
    expect(within(screen.getByRole("table", { name: "Weights in memory" })).getByText("anima.safetensors")).toBeInTheDocument()
  })

  // "Finished: 1 checkpoints hashed"
  it("counts one hashed checkpoint in the singular", async () => {
    vi.spyOn(api.system, "status").mockResolvedValue(STATUS as never)
    renderView({ running: false, done: 1, total: 1, failed: [], current: [], finished_at: "2026-11-26T00:00:00Z", threads: 1 })
    expect(await screen.findByText("Finished: 1 checkpoint hashed")).toBeInTheDocument()
  })
})
