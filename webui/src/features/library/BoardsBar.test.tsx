import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { AssignBoardPopover } from "./BoardsBar"

function renderPopover() {
  render(
    <QueryClientProvider client={queryClient}>
      <AssignBoardPopover fileIds={["file-1"]} />
    </QueryClientProvider>,
  )
  fireEvent.click(screen.getByRole("button", { name: "Add to board" }))
}

describe("Add to board", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("says the boards are on their way while they load, not that there are none", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise<Response>(() => {}))
    renderPopover()
    expect(await screen.findByRole("status")).toHaveTextContent("Loading…")
    expect(screen.queryByText("No boards yet")).not.toBeInTheDocument()
  })

  it("offers a retry when the boards do not load", async () => {
    let fail = true
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      // a client error: the query client does not retry it, so the test needs no backoff
      fail ? new Response(JSON.stringify({ detail: "database is locked" }), { status: 400 }) : new Response(JSON.stringify([]), { status: 200 }),
    )
    renderPopover()
    const retry = await screen.findByRole("button", { name: "Retry" })
    expect(screen.queryByText("No boards yet")).not.toBeInTheDocument()
    fail = false
    fireEvent.click(retry)
    expect(await screen.findByText("No boards yet")).toBeInTheDocument()
  })
})
