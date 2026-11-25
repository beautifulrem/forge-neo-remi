/**
 * The queue's mode tabs sit at the top of `main`, which clips what it holds (`overflow-clip`): a
 * focus ring drawn outside a tab lost its top there. The tab strip keeps 4 px of room above the
 * tabs (frontend.md: 4 px of room around controls at a clipping edge), and the tabs are as much
 * shorter, so the bar keeps its height (h-9) and nothing below it moves.
 */
import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import QueuePage from "./index"

beforeEach(() => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 404 }))
})

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe("the queue's mode tabs", () => {
  it("keep room for their focus ring under main's top edge, in a bar of the same height", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <QueuePage />
      </QueryClientProvider>,
    )
    const strip = screen.getByRole("tablist", { name: "Queue" })
    expect(strip.className.split(" ")).toContain("pt-1")
    for (const tab of within(strip).getAllByRole("tab")) {
      expect(tab.className.split(" ")).toContain("h-8")
    }
  })
})
