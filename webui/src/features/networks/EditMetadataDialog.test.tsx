import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { bootstrap, models } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import { PreferredModules } from "./EditMetadataDialog"

function renderModules() {
  render(
    <QueryClientProvider client={queryClient}>
      <PreferredModules id="preferred" value={[]} onChange={() => {}} />
    </QueryClientProvider>,
  )
}

// A checkpoint's preferred VAE / text encoders, in its metadata editor (the phone has no model bar).
describe("preferred modules", () => {
  beforeEach(() => queryClient.setQueryData(keys.app, structuredClone(bootstrap)))
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("says the modules are on their way while the model list loads, not that there are none", async () => {
    vi.spyOn(api, "models").mockReturnValue(new Promise(() => {}))
    renderModules()
    expect(await screen.findByRole("status")).toHaveTextContent("Loading…")
    expect(screen.queryByText("No modules found")).not.toBeInTheDocument()
  })

  it("offers a retry when the model list does not load", async () => {
    const list = vi.spyOn(api, "models").mockRejectedValue(new ApiError(404, "boom"))
    renderModules()
    const retry = await screen.findByRole("button", { name: "Retry" })
    expect(screen.queryByText("No modules found")).not.toBeInTheDocument()
    // centred in the list, as its loading and empty states are
    expect(screen.getByRole("alert")).toHaveClass("justify-center")
    list.mockResolvedValue(models)
    fireEvent.click(retry)
    expect(await screen.findByRole("checkbox", { name: models.modules?.[0]?.title ?? models.modules?.[0]?.name ?? "" })).toBeInTheDocument()
  })
})
