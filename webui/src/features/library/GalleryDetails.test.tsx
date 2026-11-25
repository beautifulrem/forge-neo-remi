import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { keys } from "@/api/queries"
import type { GalleryItem } from "@/api/types"
import { bootstrapWith } from "@/test/api-fixtures"

const actions = vi.hoisted(() => ({ starOutputs: vi.fn(() => Promise.resolve()), openInPngInfo: vi.fn(), revealOutput: vi.fn(() => Promise.resolve()) }))
vi.mock("./queries", () => actions)

const { GalleryDetails } = await import("./GalleryDetails")

const ITEM: GalleryItem = {
  id: "file-1",
  name: "00012-1234.png",
  width: 832,
  height: 1216,
  size: 1536 * 1024,
  mime: "image/png",
  created_at: "2026-10-19T08:30:00Z",
  category: "txt2img",
  model: "animagine-xl-4.0",
  starred: false,
  infotext: "1girl, lantern\nSteps: 28, Seed: 1234, Model: animagine-xl-4.0",
}

function renderDetails(item: GalleryItem, features: string[] = []) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  client.setQueryData(keys.app, { ...bootstrapWith({}), features })
  const onDelete = vi.fn()
  const onClose = vi.fn()
  render(
    <QueryClientProvider client={client}>
      <GalleryDetails item={item} onDelete={onDelete} onClose={onClose} />
    </QueryClientProvider>,
  )
  return { onDelete, onClose }
}

/** A file fact (the first list; the parameters' chips are another). */
const fact = (term: string) => within(document.querySelector("dl")!).getByText(term, { selector: "dt" }).nextElementSibling?.textContent

describe("GalleryDetails", () => {
  beforeEach(() => {
    for (const action of Object.values(actions)) action.mockClear()
  })

  it("lists the file facts and the generation parameters", () => {
    renderDetails(ITEM)
    expect(fact("File")).toBe("00012-1234.png")
    expect(fact("Size")).toBe("832 × 1216")
    expect(fact("File size")).toBe("1.5 MB")
    expect(fact("Created")).toBe(new Date(ITEM.created_at).toLocaleString("en"))
    expect(fact("From")).toBe("txt2img")
    expect(fact("Model")).toBe("animagine-xl-4.0")
    expect(screen.getByText("1girl, lantern")).toBeInTheDocument()
  })

  it("falls back for unknown sizes, unreadable dates and categories, and leaves out a missing model", () => {
    renderDetails({ ...ITEM, width: null, height: undefined, size: 300, created_at: "yesterday", category: "other", model: null, infotext: null })
    expect(fact("Size")).toBe("? × ?")
    expect(fact("File size")).toBe("300 B")
    expect(fact("Created")).toBe("yesterday")
    expect(fact("From")).toBe("Other")
    expect(within(document.querySelector("dl")!).queryByText("Model", { selector: "dt" })).toBeNull()
    expect(screen.getByText("No generation parameters")).toBeInTheDocument()
  })

  // The Remove button turned Tailwind's red-300 on hover: a palette shade, not the destructive token.
  // The viewer's backdrop is black in both themes, so the button reads the dark theme's tokens.
  it("turns the Remove button the destructive colour of a dark surface on hover", () => {
    renderDetails(ITEM)
    const remove = screen.getByRole("button", { name: "Remove…" })
    expect(remove).toHaveClass("dark", "hover:text-destructive")
    expect(remove.className).not.toMatch(/red-\d/)
  })

  it("stars and unstars", () => {
    renderDetails(ITEM)
    const star = screen.getByRole("button", { name: "Star" })
    expect(star).toHaveAttribute("aria-pressed", "false")
    fireEvent.click(star)
    expect(actions.starOutputs).toHaveBeenCalledWith(["file-1"], true)
  })

  it("shows a starred image as starred and unstars it", () => {
    renderDetails({ ...ITEM, starred: true })
    // the state is the pressed state; the name stays the action's (a toggle's name does not change)
    const starred = screen.getByRole("button", { name: "Star" })
    expect(starred).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(starred)
    expect(actions.starOutputs).toHaveBeenCalledWith(["file-1"], false)
  })

  // The trash button opens the same dialog as the gallery's "Remove…" (which removes, and deletes the
  // files only when asked): it was called "Delete" here.
  it("opens PNG Info after closing the viewer, and asks the page to remove the image", () => {
    const { onClose, onDelete } = renderDetails(ITEM)
    fireEvent.click(screen.getByRole("button", { name: "PNG Info" }))
    expect(onClose).toHaveBeenCalledOnce()
    expect(actions.openInPngInfo).toHaveBeenCalledWith("file-1")
    expect(onClose.mock.invocationCallOrder[0]).toBeLessThan(actions.openInPngInfo.mock.invocationCallOrder[0]!)
    const remove = screen.getByRole("button", { name: "Remove…" })
    expect(remove).toHaveAttribute("title", "Remove…")
    fireEvent.click(remove)
    expect(onDelete).toHaveBeenCalledWith(ITEM)
  })

  it("offers Show in folder only where the server lets folders be opened", () => {
    renderDetails(ITEM)
    expect(screen.queryByRole("button", { name: "Show in folder" })).toBeNull()
  })

  it("reveals the file in its folder", async () => {
    renderDetails(ITEM, ["folders.open"])
    fireEvent.click(await screen.findByRole("button", { name: "Show in folder" }))
    expect(actions.revealOutput).toHaveBeenCalledWith("file-1")
  })
})
