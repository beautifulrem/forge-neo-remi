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

  it("gives the one-line notices link a 40 px tall target, like the Show license toggles", () => {
    vi.spyOn(api.system, "licenses").mockReturnValue(new Promise(() => {}))
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("link", { name: /Web interface libraries/ })).toHaveClass("relative", "after:absolute", "after:inset-x-0", "after:-inset-y-2.5")
  })

  // An entry's name links to its project: a 20 px line of text with no target around it. Its box
  // reaches the panel's hairline above and stops where the Show license toggle's box starts.
  it("gives an entry's name link a target that stops short of its Show license toggle", async () => {
    vi.spyOn(api.system, "licenses").mockResolvedValue({
      groups: [
        {
          title: "Libraries",
          items: [
            { id: "a", name: "Described", url: "https://example.com/a", description: "A library", license_url: "https://example.com/a/LICENSE" },
            { id: "b", name: "Bare", url: "https://example.com/b", license_url: "https://example.com/b/LICENSE" },
          ],
        },
      ],
    } as never)
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    const described = await screen.findByRole("link", { name: "Described" })
    expect(described).toHaveClass("relative", "after:absolute", "after:inset-x-0", "after:-top-2.5", "after:-bottom-2.5")
    // the description keeps 20 px between the name and the toggle: each box takes 10 of it
    const toggles = screen.getAllByRole("button", { name: "Show license" })
    expect(toggles[0]).toHaveClass("after:-top-2.5", "after:-bottom-2.5")
    // without one the toggle sits 4 px under the name: 2 px each, 32 px boxes that do not overlap
    expect(screen.getByRole("link", { name: "Bare" })).toHaveClass("after:-top-2.5", "after:-bottom-0.5")
    expect(toggles[1]).toHaveClass("after:-top-0.5", "after:-bottom-2.5")
    expect(toggles[1]).not.toHaveClass("after:-inset-y-2.5")
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
