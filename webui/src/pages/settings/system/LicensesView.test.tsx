import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import { LicensesView } from "./LicensesView"

describe("LicensesView", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("links the notices of the libraries bundled into the web interface", () => {
    vi.spyOn(api.system, "licenses").mockReturnValue(new Promise(() => {}))
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("link", { name: /Web interface libraries/ })).toHaveAttribute("href", "/THIRD-PARTY-NOTICES.txt")
  })

  it("says the list did not load in the UI language, with a Retry", async () => {
    const licenses = vi.spyOn(api.system, "licenses").mockRejectedValueOnce(new ApiError(404, "boom", { detail: "boom" }))
    licenses.mockResolvedValueOnce({ groups: [{ title: "", items: [{ id: "a", name: "Some library" }] }] } as never)
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("It is no longer there.")
    expect(alert).not.toHaveTextContent("boom")
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(screen.getByText("Some library")).toBeInTheDocument())
  })
})
