import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { AssignBoardPopover, BoardsBar } from "./BoardsBar"
import { useGalleryView } from "./filters"

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

  // It read "Removed from “Portraits”", a done state, though nothing had been removed yet.
  it("names the removal from the board being viewed as an action", async () => {
    const board = { id: "b1", name: "Portraits", description: "", item_count: 1, cover_file_id: null, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify([board]), { status: 200 }))
    render(
      <QueryClientProvider client={queryClient}>
        <AssignBoardPopover fileIds={["file-1"]} activeBoardId="b1" />
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole("button", { name: "Add to board" }))
    expect(await screen.findByRole("button", { name: "Remove from “Portraits”" })).toBeInTheDocument()
    expect(screen.queryByText("Removed from “Portraits”")).not.toBeInTheDocument()
  })
})

describe("Boards bar", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
    useGalleryView.setState({ boardId: null })
  })

  it("goes back to All boards when the remembered board no longer exists", async () => {
    const board = { id: "b1", name: "Portraits", description: "", item_count: 1, cover_file_id: null, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify([board]), { status: 200 }))
    // deleted in another tab, or on another device
    useGalleryView.setState({ boardId: "gone" })
    render(
      <QueryClientProvider client={queryClient}>
        <BoardsBar activeItems={[]} />
      </QueryClientProvider>,
    )
    await screen.findByRole("tab", { name: /Portraits/ })
    await waitFor(() => expect(useGalleryView.getState().boardId).toBeNull())
    expect(screen.getByRole("tab", { name: "All boards" })).toHaveAttribute("aria-selected", "true")
  })

  it("selects a board it has just created", async () => {
    const board = (id: string, name: string) => ({ id, name, description: "", item_count: 0, cover_file_id: null, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" })
    const boards = [board("b1", "Portraits")]
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      if (init?.method === "POST") {
        const created = board("b2", "Night")
        boards.push(created)
        return new Response(JSON.stringify(created), { status: 200 })
      }
      return new Response(JSON.stringify(boards), { status: 200 })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <BoardsBar activeItems={[]} />
      </QueryClientProvider>,
    )
    await screen.findByRole("tab", { name: /Portraits/ })
    fireEvent.click(screen.getByRole("button", { name: "New board" }))
    fireEvent.change(await screen.findByPlaceholderText("Board name"), { target: { value: "Night" } })
    fireEvent.click(screen.getByRole("button", { name: "Create board" }))
    expect(await screen.findByRole("tab", { name: /Night/ })).toHaveAttribute("aria-selected", "true")
    expect(useGalleryView.getState().boardId).toBe("b2")
  })

  it("keeps a board that exists", async () => {
    const board = { id: "b1", name: "Portraits", description: "", item_count: 1, cover_file_id: null, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify([board]), { status: 200 }))
    useGalleryView.setState({ boardId: "b1" })
    render(
      <QueryClientProvider client={queryClient}>
        <BoardsBar activeItems={[]} />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole("tab", { name: /Portraits/ })).toHaveAttribute("aria-selected", "true")
    expect(useGalleryView.getState().boardId).toBe("b1")
  })
})
