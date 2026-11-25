import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import type { ResultItem } from "@/features/gallery/session-store"
import { Lightbox } from "./Lightbox"

const item = (n: number): ResultItem => ({ key: `job:${n}`, fileId: `file-${n}`, jobId: "job", tab: "txt2img", index: n, infotext: null, createdAt: 0 })
const items = [item(0), item(1), item(2)]

function renderViewer(current = "job:0") {
  const onNavigate = vi.fn()
  const onOpenChange = vi.fn()
  render(
    <QueryClientProvider client={queryClient}>
      <Lightbox open onOpenChange={onOpenChange} items={items} current={current} onNavigate={onNavigate} tab="txt2img" />
    </QueryClientProvider>,
  )
  const image = screen.getByRole("img", { name: "Generated image" })
  Object.defineProperties(image, { naturalWidth: { value: 400 }, naturalHeight: { value: 300 } })
  fireEvent.load(image)
  return { onNavigate, onOpenChange, dialog: screen.getByRole("dialog"), image }
}

// the size and the zoom are two facts side by side ("400 × 300", "125%")
const zoomText = () => {
  expect(screen.getByText("400 × 300")).toBeInTheDocument()
  return screen.getByText(/^\d+%$/).textContent
}

describe("lightbox", () => {
  afterEach(() => vi.restoreAllMocks())

  it("navigates, downloads and closes from the keyboard", () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    const { onNavigate, onOpenChange, dialog } = renderViewer()
    fireEvent.keyDown(dialog, { key: "ArrowRight" })
    expect(onNavigate).toHaveBeenLastCalledWith("job:1")
    fireEvent.keyDown(dialog, { key: "ArrowLeft" })
    expect(onNavigate).toHaveBeenLastCalledWith("job:2") // wraps around
    fireEvent.keyDown(dialog, { key: "End" })
    expect(onNavigate).toHaveBeenLastCalledWith("job:2")

    fireEvent.keyDown(dialog, { key: "s" })
    expect(click).toHaveBeenCalledTimes(1)
    const link = click.mock.contexts[0] as HTMLAnchorElement
    expect(link.getAttribute("href")).toBe("/api/v2/files/file-0?download=1")
    expect(link.getAttribute("download")).toBe("")

    fireEvent.keyDown(dialog, { key: "Escape" })
    expect(onOpenChange).toHaveBeenCalledWith(false, expect.anything())
  })

  // Opened from the keyboard, the focus went to the first toolbar button (Download) with its tooltip:
  // Escape closed the tooltip first, and Enter downloaded the image
  it("takes the focus on the image, not on a toolbar button", async () => {
    const { dialog, image } = renderViewer()
    await waitFor(() => expect(dialog).toContainElement(document.activeElement as HTMLElement | null))
    expect(document.activeElement).not.toBe(dialog)
    expect(document.activeElement).toContainElement(image)
  })

  // The arrows in the Save menu turned the viewer to the next image.
  it("leaves the arrows to a menu opened from its toolbar", async () => {
    const { onNavigate } = renderViewer()
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    const entry = await screen.findByRole("menuitem", { name: "Save to the Save folder" })
    fireEvent.keyDown(entry, { key: "ArrowRight" })
    fireEvent.keyDown(entry, { key: "End" })
    expect(onNavigate).not.toHaveBeenCalled()
  })

  it("turns the page on a sideways wheel (once per swipe) and zooms on a vertical one", () => {
    const { onNavigate, image } = renderViewer()
    const viewport = image.parentElement!
    fireEvent.wheel(viewport, { deltaX: 40, deltaY: 2 })
    fireEvent.wheel(viewport, { deltaX: 40, deltaY: 2 }) // the same swipe
    expect(onNavigate).toHaveBeenCalledTimes(1)
    expect(onNavigate).toHaveBeenLastCalledWith("job:1")
    expect(zoomText()).toContain("100%")
    fireEvent.wheel(viewport, { deltaY: -100 })
    expect(zoomText()).not.toBe("100%")
  })

  it("zooms with + / − and resets with 1", () => {
    const { dialog } = renderViewer()
    expect(zoomText()).toContain("100%")
    fireEvent.keyDown(dialog, { key: "+" })
    expect(zoomText()).toContain("125%")
    fireEvent.keyDown(dialog, { key: "1" })
    expect(zoomText()).toContain("100%")
  })

  it("pinch-zooms with two fingers", () => {
    const { image } = renderViewer()
    const viewport = image.parentElement!
    const touch = (type: "pointerDown" | "pointerMove" | "pointerUp", pointerId: number, clientX: number) =>
      fireEvent[type](viewport, { pointerId, clientX, clientY: 0, button: 0, pointerType: "touch" })
    touch("pointerDown", 1, 0)
    touch("pointerDown", 2, 100)
    touch("pointerMove", 1, -50)
    touch("pointerMove", 2, 150)
    expect(zoomText()).toContain("200%")
    touch("pointerUp", 1, -50)
    touch("pointerUp", 2, 150)
    expect(zoomText()).toContain("200%")
  })
})
