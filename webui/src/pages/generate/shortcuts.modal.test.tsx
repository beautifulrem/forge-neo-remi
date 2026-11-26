import { act, fireEvent, render, renderHook, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { dispatchRun, useRuns } from "@/features/progress/run-controller"
import { IDLE } from "@/features/progress/run-machine"
import { useGenerateShortcuts } from "./shortcuts"

vi.mock("@/features/progress/run-controller", async (original) => ({ ...(await original<typeof import("@/features/progress/run-controller")>()), dispatchRun: vi.fn() }))
vi.mock("@/features/queue/edit-session", () => ({ enqueueForm: vi.fn() }))

// the first open of a popover loads its implementation
const slow = { timeout: 15_000 }

describe("the generate keys", { timeout: 30_000 }, () => {
  afterEach(() => {
    vi.mocked(dispatchRun).mockClear()
    useRuns.setState({ runs: { ...useRuns.getState().runs, txt2img: IDLE } })
  })

  // A popover (aspect ratio, "Generate several times") is a role="dialog" too, but not modal:
  // Ctrl+Enter pressed in one was held back as if a dialog covered the page.
  it("generate and skip while a popover is open", async () => {
    renderHook(() => useGenerateShortcuts("txt2img"))
    render(
      <Popover defaultOpen>
        <PopoverTrigger>Aspect ratio</PopoverTrigger>
        <PopoverContent aria-labelledby="ratio-title">
          <PopoverTitle id="ratio-title">Aspect ratio</PopoverTitle>
        </PopoverContent>
      </Popover>,
    )
    const popup = await screen.findByRole("dialog", { name: "Aspect ratio" }, slow)
    fireEvent.keyDown(popup, { key: "Enter", ctrlKey: true })
    expect(dispatchRun).toHaveBeenCalledWith("txt2img", { type: "shortcut" })
    fireEvent.keyDown(popup, { key: "Enter", altKey: true })
    expect(dispatchRun).toHaveBeenCalledWith("txt2img", { type: "skip" })
  })

  it("stay with a modal dialog while one is open", async () => {
    renderHook(() => useGenerateShortcuts("txt2img"))
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Edit style</DialogTitle>
        </DialogContent>
      </Dialog>,
    )
    const dialog = await screen.findByRole("dialog", { name: "Edit style" })
    fireEvent.keyDown(dialog, { key: "Enter", ctrlKey: true })
    fireEvent.keyDown(dialog, { key: "Enter", altKey: true })
    expect(dispatchRun).not.toHaveBeenCalled()
  })

  // The form's schema tabs (ui/layout/Tabs.tsx) are the kit's tabs: their panel is no popup.
  it("interrupt the run on Escape from a button in a form tab", async () => {
    useRuns.setState({ runs: { ...useRuns.getState().runs, txt2img: { ...IDLE, jobs: ["job-1"], running: "job-1" } } })
    renderHook(() => useGenerateShortcuts("txt2img"))
    render(
      <Tabs defaultValue="hires">
        <TabsList>
          <TabsTrigger value="hires">Hires. fix</TabsTrigger>
        </TabsList>
        <TabsContent value="hires">
          <button type="button">Upscale</button>
        </TabsContent>
      </Tabs>,
    )
    const button = screen.getByRole("button", { name: "Upscale" })
    button.focus()
    fireEvent.keyDown(button, { key: "Escape" })
    // the interrupt is decided once the event has run its course
    await act(() => new Promise((resolve) => setTimeout(resolve)))
    expect(dispatchRun).toHaveBeenCalledWith("txt2img", { type: "interrupt" })
  })
})
