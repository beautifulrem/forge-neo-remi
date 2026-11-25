import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import QueuePage from "./index"

// The three modes were a strip of buttons written by hand: on a narrow phone, or with longer
// translated labels, nothing kept them on one line.
describe("the queue's modes", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("are a tab strip that switches the view", () => {
    vi.spyOn(globalThis, "fetch").mockReturnValue(new Promise(() => {}))
    render(
      <QueryClientProvider client={queryClient}>
        <QueuePage />
      </QueryClientProvider>,
    )
    const strip = screen.getByRole("tablist", { name: "Queue" })
    const tabs = within(strip).getAllByRole("tab")
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Queue & History", "Pipelines", "Recipes"])
    expect(tabs[0]).toHaveAttribute("aria-selected", "true")
    fireEvent.click(within(strip).getByRole("tab", { name: "Recipes" }))
    expect(within(strip).getByRole("tab", { name: "Recipes" })).toHaveAttribute("aria-selected", "true")
    expect(tabs[0]).toHaveAttribute("aria-selected", "false")
  })
})
