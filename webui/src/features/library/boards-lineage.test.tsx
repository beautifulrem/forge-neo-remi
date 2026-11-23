import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type { GalleryItem } from "@/api/types"
import type { ResultItem } from "@/features/gallery/session-store"
import { Lightbox } from "@/features/lightbox/Lightbox"
import { type Board } from "./boards-api"
import { BoardsBar } from "./BoardsBar"
import { useGalleryView } from "./filters"
import { LineageStrip } from "./LineageStrip"

const ITEM: GalleryItem = {
  id: "cur-1",
  name: "current.png",
  width: 1024,
  height: 1024,
  size: 1024,
  mime: "image/png",
  created_at: "2026-11-22T10:00:00Z",
  job_id: "job-77",
  tab: "img2img",
  category: "img2img",
  model: "anima",
  starred: false,
  infotext: null,
}

describe("Gallery Boards, Lineage & Lightbox RGBA 4-backdrop", () => {
  beforeEach(() => {
    useGalleryView.setState({ boardId: null })
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("lists boards, creates a board, and switches active board filter", async () => {
    const boards: Board[] = [
      { id: "b1", name: "Heroes", description: "", color: null, item_count: 4, cover_file_id: "img1", created_at: "2026-11-22T00:00:00Z", updated_at: "2026-11-22T00:00:00Z" },
    ]
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      const method = init?.method ?? (typeof input === "string" ? "GET" : (input as Request).method)
      if (url.endsWith("/boards") && method === "GET") {
        return new Response(JSON.stringify(boards), { status: 200, headers: { "Content-Type": "application/json" } })
      }
      if (url.endsWith("/boards") && method === "POST") {
        const body = JSON.parse(String(init?.body ?? "{}")) as { name: string }
        const created = { id: "b2", name: body.name, description: "", color: null, item_count: 0, cover_file_id: null, created_at: "2026-11-22T00:00:00Z", updated_at: "2026-11-22T00:00:00Z" }
        boards.push(created)
        return new Response(JSON.stringify(created), { status: 201, headers: { "Content-Type": "application/json" } })
      }
      return new Response("{}", { status: 200 })
    })

    render(
      <QueryClientProvider client={queryClient}>
        <BoardsBar activeItems={[{ fileId: "img1" }]} />
      </QueryClientProvider>,
    )

    const heroTab = await screen.findByRole("tab", { name: /Heroes/ })
    fireEvent.click(heroTab)
    expect(useGalleryView.getState().boardId).toBe("b1")
    expect(screen.getByRole("button", { name: /Send board to Reference Images/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "New board" }))
    fireEvent.change(await screen.findByPlaceholderText("Board name"), { target: { value: "Bosses" } })
    fireEvent.click(screen.getByRole("button", { name: "Create board" }))
    await waitFor(() => expect(useGalleryView.getState().boardId).toBe("b2"))
  })

  it("renders Asset Lineage chain (parent -> current -> child) and jumps on click", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          file_id: "cur-1",
          item: { file_id: "cur-1", relation: "self", depth: 0, job_id: "job-77", name: "current.png" },
          ancestors: [{ file_id: "parent-1", relation: "img2img", depth: -1, name: "parent.png" }],
          siblings: [],
          descendants: [{ file_id: "child-1", relation: "upscale", depth: 1, name: "child.png" }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )
    const onSelect = vi.fn()
    const onClose = vi.fn()

    render(
      <QueryClientProvider client={queryClient}>
        <LineageStrip item={ITEM} onSelectImage={onSelect} onClose={onClose} />
      </QueryClientProvider>,
    )

    expect(await screen.findByText("Asset lineage")).toBeInTheDocument()
    fireEvent.click(await screen.findByTitle("Jump to parent.png"))
    expect(onSelect).toHaveBeenCalledWith("parent-1")
    fireEvent.click(screen.getByTitle("Jump to child.png"))
    expect(onSelect).toHaveBeenCalledWith("child-1")

    const getJob = vi.spyOn(api.jobs, "get").mockResolvedValue({
      id: "job-77",
      seq: 77,
      tab: "img2img",
      status: "completed",
      created_at: "",
      outputs: [],
      priority: 0,
      forever: false,
      starred: false,
      replayable: true,
      notes: [],
      warnings: [],
      core: { prompt: "from job" },
      scripts: {},
      layout_hash: "h",
      schema_version: 1,
    })
    fireEvent.click(screen.getByRole("button", { name: "Load source job" }))
    await waitFor(() => expect(getJob).toHaveBeenCalledWith("job-77"))
  })

  it("toggles all 4 RGBA backdrops in Lightbox via picker and B shortcut", () => {
    const items: ResultItem[] = [{ key: "j:0", fileId: "f0", jobId: "j", tab: "txt2img", index: 0, infotext: null, createdAt: 0 }]
    render(
      <QueryClientProvider client={queryClient}>
        <Lightbox open onOpenChange={() => {}} items={items} current="j:0" onNavigate={() => {}} tab="txt2img" />
      </QueryClientProvider>,
    )
    const image = screen.getByRole("img", { name: "Generated image" })
    Object.defineProperties(image, { naturalWidth: { value: 400 }, naturalHeight: { value: 400 } })
    fireEvent.load(image)

    expect(document.querySelector('[data-backdrop="checker"]')).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Green screen" }))
    expect(document.querySelector('[data-backdrop="green"]')).toBeInTheDocument()

    const dialog = screen.getByRole("dialog")
    fireEvent.keyDown(dialog, { key: "b" })
    expect(document.querySelector('[data-backdrop="checker"]')).toBeInTheDocument()
    fireEvent.keyDown(dialog, { key: "B" })
    expect(document.querySelector('[data-backdrop="black"]')).toBeInTheDocument()
  })
})
