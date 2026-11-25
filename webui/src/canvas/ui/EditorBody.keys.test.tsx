/**
 * The editor's keys while candidates are staged: Enter and Escape act on the
 * candidate only when nothing else in the editor has the key, whichever of the
 * editor and the generate page started listening first.
 */
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { dispatchRun, useRuns } from "@/features/progress/run-controller"
import { IDLE } from "@/features/progress/run-machine"
import { TOUCH_ONLY_QUERY } from "@/lib/use-media-query"
import { useGenerateShortcuts } from "@/pages/generate/shortcuts"
import { EditorDocument } from "../engine/document"
import { createMemoryRaster, testImage } from "../engine/memory-raster"
import { useCanvasStaging } from "../state/staging"
import { EditorBody } from "./EditorBody"

vi.mock("@/features/progress/run-controller", async (original) => ({ ...(await original<typeof import("@/features/progress/run-controller")>()), dispatchRun: vi.fn() }))
// No canvas in jsdom: the renderer and the pointer interaction are inert stand-ins.
vi.mock("../engine/renderer", () => ({
  Renderer: class {
    element = document.createElement("div")
    attach() {}
    setView() {}
    setSize() {}
    setOverlays() {}
    setLayer() {}
    setAccent() {}
    setStyle() {}
    destroy() {}
  },
}))
vi.mock("../engine/interaction", () => ({
  Interaction: class {
    busy = false
    setSpace() {}
    setShift() {}
    destroy() {}
    cancel() {}
  },
}))

const DOC = "img2img"

function newDoc() {
  const doc = new EditorDocument({ createRaster: createMemoryRaster, sync: { encode: async () => new Blob(), upload: async () => "x", delay: 5 } })
  doc.setBackground(testImage(8, 6), { width: 8, height: 6 }, "bg")
  return doc
}

/** The generate page's shortcuts, and the editor shown with it or later (the Source view after "Stage latest result"). */
function Page({ editorFirst = false, busy = false, withMask = false, onAcceptStaged = () => {} }: { editorFirst?: boolean; busy?: boolean; withMask?: boolean; onAcceptStaged?: (candidate: unknown) => void }) {
  useGenerateShortcuts("img2img")
  const [shown, setShown] = useState(editorFirst)
  const [doc] = useState(newDoc)
  return (
    <>
      <button type="button" onClick={() => setShown(true)}>
        show source
      </button>
      {shown ? (
        <EditorBody
          doc={doc}
          editor={{ docKey: DOC, image: "bg", onImageChange: () => {}, ...(withMask ? { layers: { mask: { kind: "mask", value: null } }, activeLayer: "mask" } : {}) } as never}
          screenBrush={false}
          plainColor={null}
          toolbarAlways
          maximized={false}
          onMaximize={() => {}}
          busy={busy}
          ready
          loadFailed={false}
          onRetry={() => {}}
          onFiles={() => {}}
          onBlank={() => {}}
          onRemove={() => {}}
          onAcceptStaged={onAcceptStaged}
        />
      ) : null}
    </>
  )
}

const stage = () => act(() => useCanvasStaging.getState().stageCandidates(DOC, [{ fileId: "c1" }, { fileId: "c2" }]))
const candidates = () => useCanvasStaging.getState().sessions[DOC]?.candidates.length ?? 0
const settle = () => act(() => new Promise((resolve) => setTimeout(resolve)))

afterEach(() => {
  act(() => useCanvasStaging.getState().discardAll(DOC))
  useRuns.setState({ runs: { ...useRuns.getState().runs, img2img: IDLE } })
  vi.mocked(dispatchRun).mockClear()
})

describe("Enter with a staged candidate", () => {
  // Enter on "Discard candidate" (or any editor button) accepted the candidate: the source image
  // was replaced and the other candidates dropped, and the button did nothing.
  it("leaves a focused button to press itself", () => {
    stage()
    const accept = vi.fn()
    render(<Page editorFirst onAcceptStaged={accept} />)
    const discard = screen.getByRole("button", { name: /Discard candidate/ })
    discard.focus()
    expect(fireEvent.keyDown(discard, { key: "Enter", code: "Enter" })).toBe(true) // not cancelled: the browser clicks it
    expect(accept).not.toHaveBeenCalled()
  })

  it("accepts the candidate from the staging bar itself", () => {
    stage()
    const accept = vi.fn()
    render(<Page editorFirst onAcceptStaged={accept} />)
    const bar = screen.getByRole("toolbar", { name: "Canvas staging" })
    bar.focus()
    fireEvent.keyDown(bar, { key: "Enter", code: "Enter" })
    expect(accept).toHaveBeenCalledWith(expect.objectContaining({ fileId: "c1" }))
  })

  // The bar's Accept is disabled while the editor loads or uploads; Enter accepted anyway.
  it("waits while the editor is busy, as the bar's Accept does", () => {
    stage()
    const accept = vi.fn()
    render(<Page editorFirst busy onAcceptStaged={accept} />)
    const bar = screen.getByRole("toolbar", { name: "Canvas staging" })
    bar.focus()
    fireEvent.keyDown(bar, { key: "Enter", code: "Enter" })
    expect(accept).not.toHaveBeenCalled()
  })
})

describe("Escape with a staged candidate while a job of the form runs", () => {
  const running = () => useRuns.setState({ runs: { ...useRuns.getState().runs, img2img: { ...IDLE, jobs: ["job-1"], running: "job-1" } } })

  // The page listened first and interrupted the job; the candidate stayed.
  it("discards the candidate, not the job, when the editor appeared after the page", async () => {
    running()
    stage()
    render(<Page />)
    fireEvent.click(screen.getByText("show source"))
    const bar = screen.getByRole("toolbar", { name: "Canvas staging" })
    bar.focus()
    fireEvent.keyDown(bar, { key: "Escape", code: "Escape" })
    await settle()
    expect(dispatchRun).not.toHaveBeenCalled()
    expect(candidates()).toBe(1)
  })

  it("discards the candidate, not the job, when the editor appeared with the page", async () => {
    running()
    stage()
    render(<Page editorFirst />)
    const bar = screen.getByRole("toolbar", { name: "Canvas staging" })
    bar.focus()
    fireEvent.keyDown(bar, { key: "Escape", code: "Escape" })
    await settle()
    expect(dispatchRun).not.toHaveBeenCalled()
    expect(candidates()).toBe(1)
  })

  it("interrupts the job when nothing on the page uses Escape", async () => {
    running()
    render(<Page />)
    fireEvent.keyDown(document.body, { key: "Escape", code: "Escape" })
    await settle()
    expect(dispatchRun).toHaveBeenCalledWith("img2img", { type: "interrupt" })
  })
})

describe("the shortcuts button", () => {
  afterEach(() => vi.restoreAllMocks())

  /** Every ancestor of `element` up to the body, the element too. */
  const chain = (element: HTMLElement) => {
    const out: HTMLElement[] = []
    for (let node: HTMLElement | null = element; node && node !== document.body; node = node.parentElement) out.push(node)
    return out
  }

  // It was hidden in an editor narrower than @md, by width: a narrow editor on a desktop (a pane of
  // the generate view) lost the list of its keys, while "?" still opened it, from a hidden button.
  it("shows wherever there is a keyboard, at any width", () => {
    render(<Page editorFirst />)
    const button = screen.getByRole("button", { name: "Keyboard shortcuts" })
    expect(chain(button).filter((node) => node.classList.contains("hidden"))).toEqual([])
  })

  // A touch screen without a keyboard has no keys to list.
  it("is left out on a touch screen without a keyboard", () => {
    const matchMedia = window.matchMedia
    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({ ...matchMedia(query), matches: query === TOUCH_ONLY_QUERY }))
    render(<Page editorFirst />)
    expect(screen.queryByRole("button", { name: "Keyboard shortcuts" })).toBeNull()
  })
})

// With the pointer resting on the editor (hover gives it the keys), the keys pressed in the Layers
// popover, which is portaled outside the editor, reached the editor too: an arrow on one of its
// buttons stepped through the staged candidates, and Escape (closing the popover) discarded one.
describe("keys in a popup the editor opened", () => {
  const index = () => useCanvasStaging.getState().sessions[DOC]?.index

  async function openLayers() {
    stage()
    const { container } = render(<Page editorFirst withMask />)
    fireEvent.pointerMove(container.querySelector("[data-slot=image-editor]")!)
    fireEvent.click(screen.getByRole("button", { name: "Layers and view" }))
    return screen.findByRole("dialog", { name: "Layers and view" })
  }

  it("stay with the popup", async () => {
    const popup = await openLayers()
    const slider = popup.querySelector<HTMLInputElement>('input[type="range"]')!
    expect(slider).toHaveAttribute("aria-label", "Opacity")
    slider.focus()
    fireEvent.keyDown(slider, { key: "ArrowRight", code: "ArrowRight" })
    expect(index()).toBe(0)
    const hide = within(popup).getByRole("button", { name: "Hide Mask" })
    hide.focus()
    fireEvent.keyDown(hide, { key: "ArrowRight", code: "ArrowRight" })
    expect(index()).toBe(0)
    fireEvent.keyDown(hide, { key: "Escape", code: "Escape" })
    expect(candidates()).toBe(2)
  })

  it("reach the editor from the dialog that holds it (maximised)", () => {
    stage()
    render(
      <div role="dialog" aria-label="Image editor" tabIndex={-1}>
        <Page editorFirst />
      </div>,
    )
    const dialog = screen.getByRole("dialog", { name: "Image editor" })
    fireEvent.pointerMove(dialog.querySelector("[data-slot=image-editor]")!)
    dialog.focus()
    fireEvent.keyDown(dialog, { key: "ArrowRight", code: "ArrowRight" })
    expect(index()).toBe(1)
  })

  it("still reach the editor from elsewhere", async () => {
    await openLayers()
    fireEvent.keyDown(document.body, { key: "ArrowRight", code: "ArrowRight" })
    expect(index()).toBe(1)
  })
})
