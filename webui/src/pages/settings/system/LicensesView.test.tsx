import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
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
})
