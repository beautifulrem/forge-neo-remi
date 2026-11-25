import { StrictMode } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { tabSchema } from "@/api/mock/fixtures"
import type { PageInfo } from "@/api/types"
import { navigate } from "@/app/router"
import { type ResultItem, useSession } from "@/features/gallery/session-store"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { toast } from "sonner"
import { decompilePrompt } from "./decompile"
import { finalPrompt } from "./draft"
import JsonBuilderPage from "./index"
import { sizeLimit } from "./size-limit"
import { useBuilder } from "./store"
import { COMPILED, DRAFT } from "./testdata"

// The real editor is CodeMirror: a plain textarea stands in for it here.
vi.mock("@/features/prompt-editor", () => ({
  LazyPromptEditor: ({ value, onChange, ariaLabel, disabled }: { value: string; onChange: (v: string) => void; ariaLabel?: string; disabled?: boolean }) => (
    <textarea aria-label={ariaLabel} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
  ),
}))

// The styles combobox is a Base UI combobox: a plain toggle stands in for it.
vi.mock("@/ui/inputs/ChipsSelect", () => ({
  ChipsSelect: ({ selected, onChange, disabled, ariaLabel }: { selected: string[]; onChange: (v: string[]) => void; disabled?: boolean; ariaLabel?: string }) => (
    <button type="button" aria-label={ariaLabel} disabled={disabled} onClick={() => onChange(["style-a"])}>
      {selected.join(",") || "none"}
    </button>
  ),
}))

const PAGE = { id: "json-builder", title: "JSON Builder", icon: "braces", widget: "json-builder" } as PageInfo
const initial = useBuilder.getState()
const sessionInitial = useSession.getState()

function renderPage() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <JsonBuilderPage page={PAGE} />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  useBuilder.setState(initial, true)
  useSession.setState(sessionInitial, true)
  navigate({ name: "page", id: "json-builder" }, { replace: true })
  // on_change actions of the forms (hires resolution) answer nothing
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response("{}", { headers: { "Content-Type": "application/json" } }))
})
afterEach(() => vi.restoreAllMocks())

describe("JSON Builder page", () => {
  it("shows the draft's items on the canvas and in the list", () => {
    renderPage()
    const canvas = screen.getByTestId("json-builder-canvas")
    expect(within(canvas).getAllByRole("group")).toHaveLength(1)
    fireEvent.click(screen.getByRole("button", { name: "Add item" }))
    expect(within(canvas).getAllByRole("group")).toHaveLength(2)
    expect(within(screen.getByRole("list", { name: "Items" })).getAllByRole("listitem")).toHaveLength(2)
    fireEvent.click(screen.getByRole("button", { name: "Remove item 1" }))
    expect(within(canvas).getAllByRole("group")).toHaveLength(1)
  })

  // On a phone held sideways (852×393) the stacked canvas took 75vh and ran under the send bar
  // pinned to the page's bottom, with its zoom bar half hidden there. The page is a size container
  // and the canvas ends the send bar's depth (`--pinned-bottom`, use-pinned-bar.ts) above its foot.
  it("keeps the stacked canvas clear of the send bar pinned to the page's foot", () => {
    renderPage()
    const canvas = screen.getByTestId("json-builder-canvas")
    const page = canvas.closest<HTMLElement>("[data-scroll-root]")!
    expect(page.className.split(" ")).toContain("max-lg:[container-type:size]")
    expect(canvas.className.split(" ")).toContain("max-h-[min(75vh,calc(100cqh-var(--pinned-bottom,0px)-var(--spacing)*4))]")
    // its 16rem floor gives way down to 10rem: a phone held sideways with the browser's bars has less room
    expect(canvas.className.split(" ")).toContain("min-h-[min(16rem,max(10rem,calc(100cqh-var(--pinned-bottom,0px)-var(--spacing)*4)))]")
  })

  it("marks the empty fields instead of sending", () => {
    const load = getFormStore("txt2img").getState()
    load.load(tabOwners(tabSchema("txt2img")))
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Send to txt2img" }))
    expect(screen.getByLabelText("Caption")).toHaveAttribute("aria-invalid", "true")
    expect(screen.getByLabelText("Caption of item 1")).toHaveAttribute("aria-invalid", "true")
    expect(location.hash).toBe("#/ext/json-builder")
    // a filled field is no longer marked
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "a room" } })
    expect(screen.getByLabelText("Caption")).not.toHaveAttribute("aria-invalid")
  })

  it("sends the prompt, width and height to txt2img and shows it", async () => {
    getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
    renderPage()
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "A cat (orange)" } })
    fireEvent.change(screen.getByLabelText("Caption of item 1"), { target: { value: "a cat" } })
    fireEvent.change(screen.getByLabelText("Extra"), { target: { value: "<lora:moss:1>" } })
    fireEvent.click(screen.getByRole("button", { name: "Send to txt2img" }))
    await waitFor(() => expect(location.hash).toBe("#/txt2img"))
    const core = getFormStore("txt2img").getState().values.core
    expect(core?.prompt).toBe(finalPrompt(useBuilder.getState()))
    expect(core?.prompt).toMatch(/^\{"summary":\{"canvas":"single","caption":"A cat \\\(orange\\\)"\}.*"bbox_format":"yxyx"\}\n<lora:moss:1>$/)
    expect([core?.width, core?.height]).toEqual([1024, 1024])
  })

  it("names the values a lock kept from the sent prompt and size", async () => {
    const success = vi.spyOn(toast, "success")
    const store = getFormStore("txt2img")
    store.getState().load(tabOwners(tabSchema("txt2img")))
    // a toggle that locks the width (`locks: "core.width"`) is on, over a width of the user's own
    store.setState((s) => ({ locks: [{ owner: "keep", name: "size", target: ["core", "width"] }], values: { ...s.values, core: { ...s.values.core, width: 768 }, keep: { size: true } } }))
    renderPage()
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "a room" } })
    fireEvent.change(screen.getByLabelText("Caption of item 1"), { target: { value: "a lamp" } })
    fireEvent.click(screen.getByRole("button", { name: "Send to txt2img" }))
    await waitFor(() => expect(success).toHaveBeenCalledWith("Sent to txt2img", { description: "Overridden by a lock: Width" }))
    expect(store.getState().values.core?.width).toBe(768)
  })

  it("warns when the size is past what the target form accepts", async () => {
    const warning = vi.spyOn(toast, "warning")
    const success = vi.spyOn(toast, "success")
    for (const tab of ["txt2img", "img2img"] as const) getFormStore(tab).getState().load(tabOwners(tabSchema(tab)))
    useBuilder.getState().setSize(3072, 1024)
    renderPage()
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "a room" } })
    fireEvent.change(screen.getByLabelText("Caption of item 1"), { target: { value: "a lamp" } })
    fireEvent.click(screen.getByRole("button", { name: "Send to img2img" }))
    await waitFor(() => expect(warning).toHaveBeenCalledOnce())
    expect(warning.mock.calls[0]![0]).toMatch(/at most 2048 px/)
    expect(success).not.toHaveBeenCalled()
    // the values still arrive: the form shows them and its check names the field
    expect(getFormStore("img2img").getState().values.core?.width).toBe(3072)
    // txt2img takes up to 4096
    expect(sizeLimit("txt2img", 3072, 1024)).toBeNull()
    expect(sizeLimit("img2img", 1024, 2048)).toBeNull()
    expect(sizeLimit("img2img", 1024, 2064)).toBe(2048)
  })

  it("loads a compiled prompt (and the text after it) for the size given", async () => {
    useBuilder.getState().setSize(1024, 768)
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Load prompt…" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.change(within(dialog).getByLabelText("Compiled prompt or layout draft"), { target: { value: "not json" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Load" }))
    expect(within(dialog).getByRole("alert")).toHaveTextContent(/JSON/)

    fireEvent.change(within(dialog).getByLabelText("Compiled prompt or layout draft"), { target: { value: `${COMPILED} <lora:x:1>` } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Load" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    const s = useBuilder.getState()
    expect(s.items.map((item) => [item.label, item.caption, item.bbox])).toEqual([
      ["character", "an orange cat", [100, 200, 501, 700]],
      ["text", "a neon sign", [600, 50, 1000, 150]],
    ])
    expect([s.caption, s.extra]).toEqual(["A cat (orange) on a sofa", "<lora:x:1>"])
    expect(screen.getByLabelText("Caption of item 2")).toHaveValue("a neon sign")
  })

  it("loads a saved layout file", async () => {
    renderPage()
    const file = new File([JSON.stringify(DRAFT)], "layout.json", { type: "application/json" })
    fireEvent.change(screen.getByTestId("json-builder-file"), { target: { files: [file] } })
    await waitFor(() => expect(useBuilder.getState().items).toHaveLength(2))
    expect([useBuilder.getState().width, useBuilder.getState().height]).toEqual([1024, 768])
  })

  it("drags a box and its corner in canvas pixels, within the canvas", () => {
    // the stage is drawn at half size: one client pixel is two canvas pixels
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(DOMRect.fromRect({ x: 0, y: 0, width: 512, height: 512 }))
    renderPage()
    const box = within(screen.getByTestId("json-builder-canvas")).getByRole("group")
    fireEvent.pointerDown(box, { button: 0, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(window, { clientX: 101, clientY: 100 }) // under the threshold
    expect(useBuilder.getState().items[0]!.bbox).toEqual([102, 102, 409, 409])
    fireEvent.pointerMove(window, { clientX: 150, clientY: 90 })
    fireEvent.pointerUp(window)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([202, 82, 509, 389])
    fireEvent.pointerMove(window, { clientX: 400, clientY: 400 }) // released: no effect
    expect(useBuilder.getState().items[0]!.bbox).toEqual([202, 82, 509, 389])

    const handle = box.querySelector(".cursor-nwse-resize")!
    fireEvent.pointerDown(handle, { button: 0, clientX: 0, clientY: 0 })
    fireEvent.pointerMove(window, { clientX: 1000, clientY: -1000 })
    fireEvent.pointerUp(window)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([202, 82, 1024, 98])
  })

  it("zooms with the buttons and the wheel, and fits again", () => {
    renderPage()
    const canvas = screen.getByTestId("json-builder-canvas")
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }))
    expect(canvas).toHaveTextContent("120%")
    fireEvent.wheel(canvas, { deltaY: 100 })
    fireEvent.wheel(canvas, { deltaY: 100 })
    expect(canvas).toHaveTextContent("83%")
    fireEvent.click(screen.getByRole("button", { name: "Fit to screen" }))
    expect(canvas).toHaveTextContent("100%")
  })

  it("opens the field editor on a canvas double click", async () => {
    renderPage()
    fireEvent.doubleClick(screen.getByLabelText("Box 1 caption"))
    const dialog = await screen.findByRole("dialog")
    const editor = within(dialog).getByLabelText("Edit the caption of item 1")
    fireEvent.change(editor, { target: { value: "a lamp" } })
    expect(screen.getByLabelText("Caption of item 1")).toHaveValue("a lamp")
  })

  it("moves the focused box with the arrow keys", () => {
    renderPage()
    const box = within(screen.getByTestId("json-builder-canvas")).getByRole("group")
    fireEvent.keyDown(box, { key: "ArrowRight" })
    fireEvent.keyDown(box, { key: "ArrowDown", shiftKey: true })
    expect(useBuilder.getState().items[0]!.bbox).toEqual([103, 118, 410, 425])
  })

  it.each(["txt2img", "img2img"] as const)("loads the prompt and size of %s, keeping its extra text", async (tab) => {
    const store = getFormStore(tab)
    store.getState().load(tabOwners(tabSchema(tab)))
    store.getState().setValue("core", "prompt", `${COMPILED} <lora:keep:1>`)
    store.getState().setValue("core", "width", 1024)
    store.getState().setValue("core", "height", 768)
    useBuilder.getState().set({ extra: "old" })
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: tab === "txt2img" ? "Load from txt2img" : "Load from img2img" }))
    await waitFor(() => expect(useBuilder.getState().items).toHaveLength(2))
    const s = useBuilder.getState()
    expect([s.width, s.height, s.committedWidth, s.committedHeight, s.extra]).toEqual([1024, 768, 1024, 768, "<lora:keep:1>"])
    expect(location.hash).toBe("#/ext/json-builder")
  })

  it("refuses a malformed form prompt and leaves the draft alone", async () => {
    const failure = vi.spyOn(toast, "error")
    const store = getFormStore("img2img")
    store.getState().load(tabOwners(tabSchema("img2img")))
    store.getState().setValue("core", "prompt", "not json")
    store.getState().setValue("core", "width", 512)
    store.getState().setValue("core", "height", 512)
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Load from img2img" }))
    await waitFor(() => expect(failure).toHaveBeenCalled())
    const s = useBuilder.getState()
    expect([s.width, s.height, s.items]).toEqual([1024, 1024, initial.items])
  })

  it("expands the item list, hiding the other sections until collapsed", () => {
    renderPage()
    const toggle = screen.getByRole("button", { name: "Expand the item list" })
    expect(toggle).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(toggle)
    expect(screen.getByRole("button", { name: "Collapse the item list" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.queryByRole("button", { name: "Set resolution" })).toBeNull()
    expect(screen.queryByLabelText("Caption")).toBeNull()
    // the list, canvas and send buttons stay
    expect(screen.getByRole("list", { name: "Items" })).toBeTruthy()
    expect(screen.getByTestId("json-builder-canvas")).toBeTruthy()
    expect(screen.getByRole("button", { name: "Send to txt2img" })).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Collapse the item list" }))
    expect(screen.getByRole("button", { name: "Set resolution" })).toBeTruthy()
  })

  it("un-expands the list when a failed send needs a hidden field", () => {
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Expand the item list" }))
    expect(screen.queryByLabelText("Caption")).toBeNull()
    // the summary caption is required but hidden: the failed send must reveal it
    fireEvent.click(screen.getByRole("button", { name: "Send to txt2img" }))
    expect(screen.getByLabelText("Caption")).toHaveAttribute("aria-invalid", "true")
  })

  it("puts a file or the selected result under the boxes, and revokes blob URLs", async () => {
    const created: string[] = []
    const revoked: string[] = []
    vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
      const url = `blob:mock-${created.length}`
      created.push(url)
      void blob
      return url
    })
    vi.spyOn(URL, "revokeObjectURL").mockImplementation((url) => void revoked.push(url))
    const view = renderPage()
    const canvas = screen.getByTestId("json-builder-canvas")
    // no session result: the gallery button is disabled
    expect(screen.getByRole("button", { name: "Use the selected result as background" })).toBeDisabled()
    fireEvent.change(screen.getByTestId("json-builder-background"), { target: { files: [new File(["x"], "bg.png", { type: "image/png" })] } })
    expect(canvas.querySelector("img")).toHaveAttribute("src", "blob:mock-0")
    // a second file replaces (and revokes) the first
    fireEvent.change(screen.getByTestId("json-builder-background"), { target: { files: [new File(["y"], "bg2.png", { type: "image/png" })] } })
    expect(canvas.querySelector("img")).toHaveAttribute("src", "blob:mock-1")
    expect(revoked).toEqual(["blob:mock-0"])
    // the selected session result is a file URL, not a blob
    act(() => useSession.setState({ items: [{ key: "j:0", fileId: "out/a.png", jobId: "j", tab: "txt2img", index: 0, infotext: null, createdAt: 1 } as ResultItem], selected: "j:0" }))
    fireEvent.click(screen.getByRole("button", { name: "Use the selected result as background" }))
    expect(canvas.querySelector("img")).toHaveAttribute("src", api.fileUrl("out/a.png"))
    expect(revoked).toEqual(["blob:mock-0", "blob:mock-1"])
    fireEvent.click(screen.getByRole("button", { name: "Clear the background" }))
    expect(canvas.querySelector("img")).toBeNull()
    // nothing of it is persisted or compiled
    expect(JSON.stringify(useBuilder.getState())).not.toContain("blob:mock")
    // one still loaded when the page unmounts is revoked by the cleanup
    fireEvent.change(screen.getByTestId("json-builder-background"), { target: { files: [new File(["z"], "bg3.png", { type: "image/png" })] } })
    view.unmount()
    expect(revoked).toEqual(["blob:mock-0", "blob:mock-1", "blob:mock-2"])
  })

  it("edits the summary caption and an item's fields through the field editor", async () => {
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Edit the summary caption in the field editor" }))
    let dialog = await screen.findByRole("dialog")
    fireEvent.change(within(dialog).getByLabelText("Edit the summary caption"), { target: { value: "a quiet room" } })
    expect(useBuilder.getState().caption).toBe("a quiet room")
    fireEvent.keyDown(dialog, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

    act(() => useBuilder.getState().updateItem(1, { label: "text", text: "OPEN" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit the text of item 1 in the field editor" }))
    dialog = await screen.findByRole("dialog")
    fireEvent.change(within(dialog).getByLabelText("Edit the text of item 1"), { target: { value: "CLOSED" } })
    expect(useBuilder.getState().items[0]!.text).toBe("CLOSED")
    fireEvent.keyDown(dialog, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

    // a deleted item's editor closes instead of resurrecting it
    fireEvent.click(screen.getByRole("button", { name: "Edit the caption of item 1 in the field editor" }))
    await screen.findByRole("dialog")
    act(() => useBuilder.getState().removeItem(1))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    expect(useBuilder.getState().items.map((item) => item.id)).toEqual([2])
    expect(useBuilder.getState().items[0]!.caption).toBe("")
  })

  it("applies prompt styles into the open field only (StrictMode)", async () => {
    const applied = vi.spyOn(api.styles, "apply").mockResolvedValue({ prompt: "styled caption", negative_prompt: "must not land" })
    useBuilder.getState().set({ caption: "a room" })
    render(
      <StrictMode>
        <QueryClientProvider client={new QueryClient()}>
          <JsonBuilderPage page={PAGE} />
        </QueryClientProvider>
      </StrictMode>,
    )
    fireEvent.click(screen.getByRole("button", { name: "Edit the summary caption in the field editor" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Prompt styles" }))
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply styles" }))
    await waitFor(() => expect(useBuilder.getState().caption).toBe("styled caption"))
    expect(applied).toHaveBeenCalledWith({ prompt: "a room", negative_prompt: "", styles: ["style-a"] })
    expect(within(dialog).getByRole("button", { name: "Prompt styles" })).toHaveTextContent("none")
  })

  it("leaves the generation forms' prompts and styles alone when applying styles", async () => {
    vi.spyOn(api.styles, "apply").mockResolvedValue({ prompt: "styled caption", negative_prompt: "must not land" })
    const before = (["txt2img", "img2img"] as const).map((tab) => {
      const store = getFormStore(tab)
      store.getState().load(tabOwners(tabSchema(tab)))
      store.getState().setValue("core", "prompt", `${tab} positive`)
      store.getState().setValue("core", "negative_prompt", `${tab} negative`)
      store.getState().setValue("core", "styles", ["chosen-style"])
      return { tab, core: { ...store.getState().values.core } }
    })
    useBuilder.getState().set({ caption: "a room" })
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Edit the summary caption in the field editor" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Prompt styles" }))
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply styles" }))
    await waitFor(() => expect(useBuilder.getState().caption).toBe("styled caption"))
    for (const { tab, core } of before) expect(getFormStore(tab).getState().values.core).toEqual(core)
  })

  it("drops the styles answer when the field changed while the dialog stays open", async () => {
    let resolve: (v: { prompt: string; negative_prompt: string }) => void = () => {}
    const pending = new Promise<{ prompt: string; negative_prompt: string }>((r) => (resolve = r))
    vi.spyOn(api.styles, "apply").mockReturnValue(pending)
    useBuilder.getState().set({ caption: "a room" })
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Edit the summary caption in the field editor" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Prompt styles" }))
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply styles" }))
    await waitFor(() => expect(api.styles.apply).toHaveBeenCalled())
    act(() => useBuilder.getState().set({ caption: "typed meanwhile" }))
    await act(async () => {
      resolve({ prompt: "styled caption", negative_prompt: "" })
      await pending
    })
    expect(useBuilder.getState().caption).toBe("typed meanwhile")
    expect(screen.getByRole("dialog")).toBeTruthy()
  })

  it("drops the styles answer when the dialog closed with the value unchanged", async () => {
    let resolve: (v: { prompt: string; negative_prompt: string }) => void = () => {}
    const pending = new Promise<{ prompt: string; negative_prompt: string }>((r) => (resolve = r))
    vi.spyOn(api.styles, "apply").mockReturnValue(pending)
    useBuilder.getState().set({ caption: "a room" })
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Edit the summary caption in the field editor" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Prompt styles" }))
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply styles" }))
    await waitFor(() => expect(api.styles.apply).toHaveBeenCalled())
    fireEvent.keyDown(dialog, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    await act(async () => {
      resolve({ prompt: "late answer", negative_prompt: "" })
      await pending
    })
    expect(useBuilder.getState().caption).toBe("a room")
  })

  it("drops the styles answer when a load replaced the draft, even with the same caption", async () => {
    let resolve: (v: { prompt: string; negative_prompt: string }) => void = () => {}
    const pending = new Promise<{ prompt: string; negative_prompt: string }>((r) => (resolve = r))
    vi.spyOn(api.styles, "apply").mockReturnValue(pending)
    useBuilder.getState().set({ caption: "a room" })
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Edit the summary caption in the field editor" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Prompt styles" }))
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply styles" }))
    await waitFor(() => expect(api.styles.apply).toHaveBeenCalled())
    // a load carrying the same caption still invalidates the pending answer
    const draft = decompilePrompt(COMPILED, 1024, 768)
    act(() => useBuilder.getState().load({ ...draft, summary: { ...draft.summary, caption: "a room" } }))
    await act(async () => {
      resolve({ prompt: "styled caption", negative_prompt: "" })
      await pending
    })
    expect(useBuilder.getState().caption).toBe("a room")
    expect(screen.queryByRole("dialog")).toBeNull()
  })

  it("keeps the value and the style selection when applying styles fails", async () => {
    const failure = vi.spyOn(toast, "error")
    vi.spyOn(api.styles, "apply").mockRejectedValue(new Error("boom"))
    useBuilder.getState().set({ caption: "a room" })
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Edit the summary caption in the field editor" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Prompt styles" }))
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply styles" }))
    await waitFor(() => expect(failure).toHaveBeenCalled())
    expect(useBuilder.getState().caption).toBe("a room")
    expect(within(dialog).getByRole("button", { name: "Prompt styles" })).toHaveTextContent("style-a")
  })

  it("switches to Fibo VGL mode, displays the CC BY-NC 4.0 badge, and compiles VGL 2.0 JSON", async () => {
    getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
    renderPage()
    fireEvent.click(screen.getByRole("radio", { name: "Fibo VGL" }))
    expect(screen.getByText("CC BY-NC 4.0")).toBeVisible()
    expect(screen.getByLabelText("Background setting")).toBeVisible()
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "A ceramic vase on stone" } })
    fireEvent.change(screen.getByLabelText("Caption of item 1"), { target: { value: "ceramic vase" } })
    fireEvent.click(screen.getByRole("button", { name: "Send to txt2img" }))
    await waitFor(() => expect(location.hash).toBe("#/txt2img"))
    const sent = JSON.parse(String(getFormStore("txt2img").getState().values.core?.prompt))
    expect(sent).toMatchObject({
      short_description: "A ceramic vase on stone",
      style_medium: "photograph",
      objects: [{ description: "ceramic vase" }],
    })
  })
})
