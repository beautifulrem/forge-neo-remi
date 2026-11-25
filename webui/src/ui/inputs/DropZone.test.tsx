import { act, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { DropZone } from "./DropZone"

function paste(file: File) {
  const event = new Event("paste", { bubbles: true, cancelable: true }) as ClipboardEvent
  Object.defineProperty(event, "clipboardData", { value: { files: [file] } })
  act(() => void document.body.dispatchEvent(event))
  return event
}

describe("drop zone", () => {
  afterEach(() => vi.restoreAllMocks())

  // A caller's inline onFiles is a new function on every render: the zone took its paste listener
  // off the window and put it back each time.
  it("listens for pastes once, and hands them to the onFiles of the latest render", () => {
    const listen = vi.spyOn(window, "addEventListener")
    const pastes = () => listen.mock.calls.filter(([type]) => type === "paste").length
    const first = vi.fn()
    const { container, rerender } = render(
      <DropZone aria-label="Image" onFiles={first}>
        drop
      </DropZone>,
    )
    expect(pastes()).toBe(1)
    const latest = vi.fn()
    for (const onFiles of [vi.fn(), vi.fn(), latest]) {
      rerender(
        <DropZone aria-label="Image" onFiles={onFiles}>
          drop
        </DropZone>,
      )
    }
    expect(pastes()).toBe(1)

    const zone = container.querySelector<HTMLElement>("[aria-label=Image]")!
    zone.focus()
    const file = new File([new Uint8Array([1])], "p.png", { type: "image/png" })
    expect(paste(file).defaultPrevented).toBe(true)
    expect(latest).toHaveBeenCalledWith([file])
    expect(first).not.toHaveBeenCalled()
  })
})
