import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useProgress } from "@/api/progress"
import { MAX_STAGED_CANDIDATES, requestStagingFocus, takeStagingFocus, useCanvasStaging, wireImg2imgStaging } from "./staging"
import { StagingBar } from "../ui/StagingBar"

describe("canvas in-place staging store", () => {
  beforeEach(() => {
    useCanvasStaging.setState({ sessions: {}, stayOnCanvas: false })
  })

  it("stages up to 4 candidates, navigates with wrap-around, adjusts opacity and split", () => {
    const store = useCanvasStaging.getState()
    store.stageCandidates("img2img", [
      { fileId: "c1" },
      { fileId: "c2" },
      { fileId: "c3" },
      { fileId: "c4" },
      { fileId: "c5" }, // capped at MAX_STAGED_CANDIDATES
    ])
    expect(useCanvasStaging.getState().sessions.img2img?.candidates).toHaveLength(MAX_STAGED_CANDIDATES)
    expect(useCanvasStaging.getState().sessions.img2img?.index).toBe(0)

    store.prev("img2img")
    expect(useCanvasStaging.getState().sessions.img2img?.index).toBe(3)
    store.next("img2img")
    expect(useCanvasStaging.getState().sessions.img2img?.index).toBe(0)

    store.setOpacity("img2img", 1.5)
    expect(useCanvasStaging.getState().sessions.img2img?.opacity).toBe(1)
    store.setOpacity("img2img", 0.4)
    expect(useCanvasStaging.getState().sessions.img2img?.opacity).toBe(0.4)

    store.toggleSplitCompare("img2img")
    expect(useCanvasStaging.getState().sessions.img2img?.splitCompare).toBe(true)
    store.setSplitPosition("img2img", 0.25)
    expect(useCanvasStaging.getState().sessions.img2img?.splitPosition).toBe(0.25)
  })

  it("discards individual candidates and clears session when last candidate is discarded", () => {
    const store = useCanvasStaging.getState()
    store.stageCandidates("doc-1", [{ fileId: "a" }, { fileId: "b" }])
    store.select("doc-1", 1)
    store.discardCurrent("doc-1")
    expect(useCanvasStaging.getState().sessions["doc-1"]?.candidates.map((c) => c.fileId)).toEqual(["a"])
    expect(useCanvasStaging.getState().sessions["doc-1"]?.index).toBe(0)

    store.discardCurrent("doc-1")
    expect(useCanvasStaging.getState().sessions["doc-1"]).toBeUndefined()
  })

  it("auto-stages completed img2img job outputs when stayOnCanvas is enabled", () => {
    wireImg2imgStaging()
    useCanvasStaging.getState().setStayOnCanvas(true)
    useProgress.setState((s) => ({
      ...s,
      last: {
        ...s.last,
        img2img: {
          jobId: "job-stage-1",
          outputs: [
            { id: "out-1", kind: "image", width: 512, height: 512 },
            { id: "out-2", kind: "image", width: 512, height: 512 },
            { id: "grid-0", kind: "image", role: "grid", width: 1024, height: 512 },
          ],
          infotext: "refined",
        },
      },
    }))
    const staged = useCanvasStaging.getState().sessions.img2img
    expect(staged?.candidates.map((c) => c.fileId)).toEqual(["out-1", "out-2"])
  })

  it("renders StagingBar controls and dispatches navigation, accept and discard actions", () => {
    const handlers = {
      onPrev: vi.fn(),
      onNext: vi.fn(),
      onSelect: vi.fn(),
      onOpacityChange: vi.fn(),
      onCompareOriginalChange: vi.fn(),
      onToggleSplit: vi.fn(),
      onAccept: vi.fn(),
      onDiscardCurrent: vi.fn(),
      onDiscardAll: vi.fn(),
    }
    render(
      <StagingBar
        session={{
          candidates: [{ fileId: "cand-1" }, { fileId: "cand-2" }],
          index: 0,
          opacity: 0.8,
          compareOriginal: false,
          splitCompare: false,
          splitPosition: 0.5,
        }}
        docKey="img2img"
        {...handlers}
      />,
    )
    expect(screen.getByText("Candidate 1 of 2")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Next candidate" }))
    expect(handlers.onNext).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole("button", { name: "Accept as base layer" }))
    expect(handlers.onAccept).toHaveBeenCalledWith({ fileId: "cand-1" })
    fireEvent.click(screen.getByRole("button", { name: "Discard candidate" }))
    expect(handlers.onDiscardCurrent).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole("button", { name: "Split compare" }))
    expect(handlers.onToggleSplit).toHaveBeenCalledOnce()
  })

  // "Hold to view original" was a click toggle: holding showed nothing, and the original stayed
  // after letting go. The label says what the button does.
  it("names the comparison with the original as the toggle it is", () => {
    const session = { candidates: [{ fileId: "cand-1" }], index: 0, opacity: 0.8, compareOriginal: false, splitCompare: false, splitPosition: 0.5 }
    const handlers = { onPrev: vi.fn(), onNext: vi.fn(), onSelect: vi.fn(), onOpacityChange: vi.fn(), onCompareOriginalChange: vi.fn(), onToggleSplit: vi.fn(), onAccept: vi.fn(), onDiscardCurrent: vi.fn(), onDiscardAll: vi.fn() }
    render(<StagingBar session={session} docKey="img2img" {...handlers} />)
    const toggle = screen.getByRole("button", { name: "Show the original" })
    expect(toggle).toHaveAttribute("aria-pressed", "false")
    fireEvent.click(toggle)
    expect(handlers.onCompareOriginalChange).toHaveBeenCalledWith(true)
  })

  it("takes the focus when staged by a button that goes away, and only then", () => {
    const session = { candidates: [{ fileId: "cand-1" }], index: 0, opacity: 0.8, compareOriginal: false, splitCompare: false, splitPosition: 0.5 }
    const handlers = { onPrev: vi.fn(), onNext: vi.fn(), onSelect: vi.fn(), onOpacityChange: vi.fn(), onCompareOriginalChange: vi.fn(), onToggleSplit: vi.fn(), onAccept: vi.fn(), onDiscardCurrent: vi.fn(), onDiscardAll: vi.fn() }
    // staged by a finished run: the focus stays where the user is
    const first = render(<StagingBar session={session} docKey="img2img" {...handlers} />)
    expect(document.activeElement).toBe(document.body)
    first.unmount()
    requestStagingFocus("img2img")
    render(<StagingBar session={session} docKey="img2img" {...handlers} />)
    // the bar, not Accept or Discard: a held Enter must not decide
    expect(document.activeElement).toBe(screen.getByRole("toolbar", { name: "Canvas staging" }))
    expect(takeStagingFocus("img2img")).toBe(false) // asked once, taken once
  })

  // The other candidates' dots were muted-foreground at 45 %: 2.1:1 (light) against the bar, too
  // faint for the only mark of a button. The current one is told apart by its width and colour.
  it("draws the other candidates' dots in a solid tone", () => {
    const session = { candidates: [{ fileId: "a" }, { fileId: "b" }, { fileId: "c" }], index: 1, opacity: 0.8, compareOriginal: false, splitCompare: false, splitPosition: 0.5 }
    const handlers = { onPrev: vi.fn(), onNext: vi.fn(), onSelect: vi.fn(), onOpacityChange: vi.fn(), onCompareOriginalChange: vi.fn(), onToggleSplit: vi.fn(), onAccept: vi.fn(), onDiscardCurrent: vi.fn(), onDiscardAll: vi.fn() }
    render(<StagingBar session={session} docKey="img2img" {...handlers} />)
    const dot = (index: number) => screen.getByRole("button", { name: `Candidate ${index} of 3` }).querySelector("span")!.className.split(" ")
    for (const other of [1, 3]) {
      expect(dot(other)).toContain("bg-muted-foreground")
      expect(dot(other).filter((name) => name.startsWith("bg-muted-foreground/"))).toEqual([])
    }
    expect(dot(2)).toContain("bg-foreground")
  })
})
