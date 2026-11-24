/**
 * The editor's keys while candidates are staged: Enter and Escape act on the
 * candidate only when nothing else in the editor has the key, whichever of the
 * editor and the generate page started listening first.
 */
import { act, fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { dispatchRun, useRuns } from "@/features/progress/run-controller"
import { IDLE } from "@/features/progress/run-machine"
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
function Page({ editorFirst = false, onAcceptStaged = () => {} }: { editorFirst?: boolean; onAcceptStaged?: (candidate: unknown) => void }) {
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
          editor={{ docKey: DOC, image: "bg", onImageChange: () => {} } as never}
          screenBrush={false}
          plainColor={null}
          toolbarAlways
          maximized={false}
          onMaximize={() => {}}
          busy={false}
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
