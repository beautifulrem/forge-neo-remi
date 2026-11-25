import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { bootstrap, models } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import ModelBar, { MODULES_HELP_URL } from "./ModelBar"
import { WorkerHealthPill } from "./WorkerHealthPill"

function renderBar() {
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelBar />
    </QueryClientProvider>,
  )
}

/** The picker's own trigger (the narrow-header popover holds a second one once open). */
const modulesTriggers = () => screen.getAllByRole("button", { name: "VAE / Text encoders" })

describe("model bar modules picker", () => {
  let patch: ReturnType<typeof vi.spyOn>
  beforeEach(() => {
    queryClient.setQueryData(keys.app, structuredClone(bootstrap))
    queryClient.setQueryData(keys.models, models)
    patch = vi.spyOn(api, "patchOptions").mockResolvedValue({} as never)
    vi.spyOn(api, "app").mockResolvedValue(structuredClone(bootstrap))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("applies the selection once, when the picker closes", async () => {
    renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    fireEvent.click(await screen.findByRole("checkbox", { name: "ae.safetensors" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "clip_l.safetensors" }))
    expect(patch).not.toHaveBeenCalled()
    fireEvent.keyDown(screen.getByRole("checkbox", { name: "ae.safetensors" }), { key: "Escape" })
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1))
    expect(patch).toHaveBeenCalledWith({ forge_additional_modules: ["ae.safetensors", "clip_l.safetensors"] })
  })

  it("keeps a selection made in the narrow header's popover when everything closes at once", async () => {
    const { unmount } = renderBar()
    fireEvent.click(screen.getByRole("button", { name: "Model settings" }))
    // The nested picker inside the settings popover.
    await waitFor(() => expect(modulesTriggers()).toHaveLength(2))
    fireEvent.click(modulesTriggers()[1]!)
    fireEvent.click(await screen.findByRole("checkbox", { name: "qwen_image_vae.safetensors" }))
    // The page goes away (or the outer popover unmounts its content) without the picker closing itself.
    unmount()
    expect(patch).toHaveBeenCalledTimes(1)
    expect(patch).toHaveBeenCalledWith({ forge_additional_modules: ["qwen_image_vae.safetensors"] })
  })

  it("leaves room under the last row for its checkbox's hit box, so a list that fits does not scroll", async () => {
    renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    const list = (await screen.findByRole("checkbox", { name: "ae.safetensors" })).closest("[role=group]")!
    // a 28 px row, the checkbox's 32 px after: box centred on it: 2 px past the row
    expect(list).toHaveClass("pb-0.5")
  })

  it("keeps the list at least 96 px tall when the popover is short, but no taller than a short list", async () => {
    renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    const list = (await screen.findByRole("checkbox", { name: "ae.safetensors" })).closest("[role=group]")!
    // the help text below scrolls with the popover instead of squeezing the list to a row and a half
    expect(list).toHaveClass("min-h-24")
    fireEvent.change(screen.getByRole("textbox", { name: "Search modules" }), { target: { value: "ae.safe" } })
    expect(list).not.toHaveClass("min-h-24")
  })

  it("writes nothing when the selection did not change", async () => {
    const { unmount } = renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    const box = await screen.findByRole("checkbox", { name: "ae.safetensors" })
    fireEvent.click(box)
    fireEvent.click(box)
    unmount()
    expect(patch).not.toHaveBeenCalled()
  })

  it("says what the modules are for and links to where to get them", async () => {
    renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    const link = await screen.findByRole("link", { name: "Where to get them" })
    expect(link).toHaveAttribute("href", MODULES_HELP_URL)
    expect(link).toHaveAttribute("target", "_blank")
    expect(link.getAttribute("rel")).toContain("noopener")
    expect(screen.getByText(/Checkpoints without a built-in VAE or text encoder/)).toBeVisible()
  })
})

describe("model bar when the model list does not load", () => {
  beforeEach(() => {
    queryClient.setQueryData(keys.app, structuredClone(bootstrap))
    vi.spyOn(api, "app").mockResolvedValue(structuredClone(bootstrap))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("says the models could not load and loads them again on retry, instead of an empty picker", async () => {
    const list = vi.spyOn(api, "models").mockRejectedValue(new ApiError(404, "boom"))
    renderBar()
    const retry = await screen.findByRole("button", { name: /Could not load the models/ })
    list.mockResolvedValue(models)
    fireEvent.click(retry)
    await waitFor(() => expect(screen.queryByRole("button", { name: /Could not load the models/ })).toBeNull())
    expect(screen.getByRole("combobox", { name: "Checkpoint" })).toBeTruthy()
  })

  it("keeps the current model in view and names the failure in the retry's title", async () => {
    vi.spyOn(api, "models").mockRejectedValue(new ApiError(404, "boom")) // a 4xx is not retried
    renderBar()
    const retry = await screen.findByRole("button", { name: /Could not load the models/ })
    expect(retry).toHaveAttribute("title", "Could not load the models · It is no longer there. · Retry")
    expect(retry).toHaveTextContent("animagine-xl-4.0")
  })

  it("shows the failure in the modules picker too, with a retry", async () => {
    vi.spyOn(api, "models").mockRejectedValue(new ApiError(404, "boom"))
    renderBar()
    await screen.findByRole("button", { name: /Could not load the models/ })
    fireEvent.click(modulesTriggers()[0]!)
    const alert = await screen.findByRole("alert")
    expect(alert.textContent).toContain("Retry")
    expect(screen.queryByText("No modules found")).toBeNull()
  })

  it("says the modules are loading while the model list is on its way, not that there are none", async () => {
    vi.spyOn(api, "models").mockImplementation(() => new Promise(() => {})) // a slow server (or a retry after a failure)
    renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    expect(await screen.findByText("Loading…")).toBeInTheDocument()
    expect(screen.queryByText("No modules found")).toBeNull()
  })
})

describe("worker health pill single-status-dot rule", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("stays quiet monochrome when healthy and only renders an alert dot when degraded", async () => {
    queryClient.setQueryData(["workers"], {
      mode: "inprocess",
      workers: [{ worker_id: "w0", state: "ready", device: "mps" }],
    })
    const { container, rerender } = render(
      <QueryClientProvider client={queryClient}>
        <WorkerHealthPill />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("button", { name: "Worker status" })).toHaveTextContent("MPS")
    expect(container.querySelector('[data-slot="worker-alert-dot"]')).toBeNull()

    queryClient.setQueryData(["workers"], {
      mode: "pool",
      workers: [
        { worker_id: "w0", state: "ready", device: "cuda:0" },
        { worker_id: "w1", state: "crashed", device: "cuda:1" },
      ],
    })
    rerender(
      <QueryClientProvider client={queryClient}>
        <WorkerHealthPill />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("button", { name: "Worker status" })).toHaveTextContent("1/2 GPU")
    expect(container.querySelector('[data-slot="worker-alert-dot"]')).not.toBeNull()
  })
})

// The server's only stage is "loading": the zh UI showed lowercase English "loading" in the pill
// and in both live regions on every model switch.
describe("model bar loading stage", () => {
  afterEach(() => {
    useProgress.setState({ modelLoading: null })
    queryClient.clear()
  })

  it("names the stage in the UI language", async () => {
    queryClient.setQueryData(keys.app, structuredClone(bootstrap))
    queryClient.setQueryData(keys.models, models)
    useProgress.setState({ modelLoading: { stage: "loading", model: "x" } })
    renderBar()
    const statuses = await screen.findAllByRole("status")
    expect(statuses.some((status) => status.textContent === "Loading")).toBe(true)
    expect(statuses.some((status) => status.textContent === "loading")).toBe(false)
  })
})
