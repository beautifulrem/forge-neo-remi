import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { useHeaderSlotTarget } from "@/app/header-slot-store"
import type { ImageMetadata } from "@/api/types"
import { toast } from "sonner"
import { groupEntries } from "@/features/pnginfo/group"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import PngInfoPage from "./index"

const metadata: ImageMetadata = {
  name: "cat.png",
  format: "PNG",
  mode: "RGB",
  width: 512,
  height: 768,
  size: 1234,
  infotext: "a cat\nNegative prompt: ugly\nSteps: 20, Sampler: Euler, Seed: 42",
  params: { Prompt: "a cat", Steps: "20", Seed: "42" },
  entries: [
    { group: "Image", key: "Format", value: "PNG", kind: "text" },
    { group: "Metadata", key: "workflow", value: '{"nodes": []}', kind: "text" },
    { group: "EXIF", key: "Make", value: "Remi", kind: "text" },
  ],
  chunks: [
    { type: "IHDR", length: 13, count: 1 },
    { type: "tEXt", length: 80, count: 1, key: "parameters" },
    { type: "IDAT", length: 9000, count: 3 },
    { type: "IEND", length: 0, count: 1 },
  ],
}

function mockServer(answer: Partial<ImageMetadata> = {}) {
  const calls: { method: string; path: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const text = request.method === "GET" ? "" : await request.text()
    calls.push({ method: request.method, path, body: text && !text.startsWith("--") ? JSON.parse(text) : text })
    const data = path === "/pnginfo" ? { ...metadata, ...answer } : path === "/uploads" ? { id: "up1", name: "x.png", width: 1, height: 1, mime: "image/png", size: 1 } : {}
    return new Response(JSON.stringify(data), { status: path === "/uploads" ? 201 : 200, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

const renderPage = (query?: string) =>
  render(
    <QueryClientProvider client={queryClient}>
      <PngInfoPage route={{ name: "png-info", query: query ? new URLSearchParams(query) : undefined }} />
    </QueryClientProvider>,
  )

describe("PNG Info page", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("asks for an image and uploads a dropped one", async () => {
    const calls = mockServer()
    renderPage()
    const zone = screen.getByRole("button", { name: /^Choose an image to read/ })
    const file = new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], "cat.png", { type: "image/png" })
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    await waitFor(() => expect(calls.some((call) => call.path === "/uploads")).toBe(true))
    await waitFor(() => expect(location.hash).toBe("#/png-info?file=up1"))
  })

  it("refuses a file that is not an image without uploading it", async () => {
    const calls = mockServer()
    const error = vi.spyOn(toast, "error")
    renderPage()
    const zone = screen.getByRole("button", { name: /^Choose an image to read/ })
    // named and typed like a PNG, but text inside: the server would answer 415
    fireEvent.drop(zone, { dataTransfer: { files: [new File(["just some notes"], "notes.png", { type: "image/png" })] } })
    await waitFor(() => expect(error).toHaveBeenCalledWith("Not an image: notes.png"))
    expect(calls.some((call) => call.path === "/uploads")).toBe(false)
  })

  it("shows the parameters, every metadata entry and the PNG chunks", async () => {
    const calls = mockServer()
    renderPage("file=abc")
    expect(await screen.findByText("cat.png")).toBeInTheDocument()
    expect(calls.find((call) => call.path === "/pnginfo")?.body).toEqual({ file_id: "abc" })
    expect(screen.getByRole("button", { name: "To txt2img" })).toBeEnabled()
    const params = screen.getByRole("table", { name: "Generation parameters" })
    expect(within(params).getByRole("rowheader", { name: "Seed" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("tab", { name: /All metadata/ }))
    expect(await screen.findByRole("rowheader", { name: "workflow" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("tab", { name: /PNG chunks/ }))
    const chunks = await screen.findByRole("table", { name: "PNG chunks" })
    expect(within(chunks).getByText("× 3")).toBeInTheDocument()
  })

  // On a phone the Clear button showed its icon alone, its label display:none: a button with no
  // name, in a 36 px box beside the header's 28 px squares. A 28 px square there, its label read out.
  it("shows Clear in the header as a 28 px icon square on a phone, its label read out", async () => {
    mockServer()
    const header = document.createElement("div")
    document.body.append(header)
    act(() => useHeaderSlotTarget.getState().setElement(header))
    try {
      renderPage("file=abc")
      const clear = await within(header).findByRole("button", { name: "Clear" })
      expect(clear).toHaveClass("max-sm:size-7", "max-sm:px-0")
      const label = within(clear).getByText("Clear")
      expect(label).toHaveClass("max-sm:sr-only")
      expect(label).not.toHaveClass("hidden")
    } finally {
      act(() => useHeaderSlotTarget.getState().setElement(null))
      header.remove()
    }
  })

  // The section tabs kept their state across images: an image without PNG chunks opened on its
  // disabled "PNG chunks" tab.
  it("opens each image on its parameters", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const plain = (await (input as Request).text()).includes('"plain"')
      const answer = plain ? { ...metadata, name: "plain.jpg", format: "JPEG", chunks: [] } : metadata
      return new Response(JSON.stringify(answer), { headers: { "Content-Type": "application/json" } })
    })
    const page = (file: string) => (
      <QueryClientProvider client={queryClient}>
        <PngInfoPage route={{ name: "png-info", query: new URLSearchParams({ file }) }} />
      </QueryClientProvider>
    )
    const view = render(page("abc"))
    fireEvent.click(await screen.findByRole("tab", { name: /PNG chunks/ }))
    expect(await screen.findByRole("table", { name: "PNG chunks" })).toBeInTheDocument()

    view.rerender(page("plain"))
    expect(await screen.findByText("plain.jpg")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Parameters" })).toHaveAttribute("aria-selected", "true")
    expect(screen.getByRole("table", { name: "Generation parameters" })).toBeInTheDocument()
  })

  // On a phone the section tabs become a select, whose option read "All metadata (3)" glued
  // from two strings: Chinese got ASCII brackets ("全部元数据 (3)").
  it("counts the metadata entries in one translated label", async () => {
    mockServer()
    vi.spyOn(HTMLElement.prototype, "scrollWidth", "get").mockReturnValue(1000) // the tabs do not fit
    const user = userEvent.setup()
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    try {
      renderPage("file=abc")
      await user.click(await screen.findByRole("combobox", { name: "元数据" }))
      expect(await screen.findByRole("option", { name: "全部元数据（3）" })).toBeInTheDocument()
    } finally {
      await act(() => i18n.changeLanguage("en"))
    }
  })

  it("disables sending parameters when the image has none", async () => {
    mockServer({ infotext: null, params: {} })
    renderPage("file=plain")
    expect(await screen.findByText(/has no generation parameters/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "To txt2img" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Extras" })).toBeEnabled()
  })

  // The Parameters tab said "Nothing found in the image." under a note that the other metadata is there
  it("says where the other metadata is when there are no parameters, and when there is none", async () => {
    mockServer({ infotext: null, params: {} })
    renderPage("file=plain")
    const parameters = await screen.findByRole("tabpanel")
    expect(parameters).toHaveTextContent("No generation parameters. The image's other metadata is under All metadata.")
    expect(screen.queryByText("Nothing found in the image.")).not.toBeInTheDocument()
  })

  it("says the image holds nothing when it holds nothing", async () => {
    mockServer({ infotext: null, params: {}, entries: [] })
    renderPage("file=bare")
    expect(await screen.findByRole("tabpanel")).toHaveTextContent("Nothing found in the image.")
    fireEvent.click(screen.getByRole("tab", { name: /All metadata/ }))
    expect(await screen.findByRole("tabpanel")).toHaveTextContent("Nothing found in the image.")
  })

  // A WebSocket resync invalidates every query: a failed refetch replaced the metadata on screen
  it("keeps the metadata shown when a refetch fails", async () => {
    mockServer()
    renderPage("file=cat")
    expect(await screen.findByRole("heading", { name: "cat.png" })).toBeInTheDocument()
    vi.mocked(globalThis.fetch).mockImplementation(async () => new Response(JSON.stringify({ detail: "gone" }), { status: 404, headers: { "Content-Type": "application/json" } }))
    await act(() => queryClient.invalidateQueries())
    await waitFor(() => expect(queryClient.getQueryState(["pnginfo", "cat"])?.status).toBe("error"))
    await act(() => new Promise((resolve) => setTimeout(resolve, 20)))
    expect(screen.getByRole("heading", { name: "cat.png" })).toBeInTheDocument()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("offers a retry when the metadata cannot be read", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ detail: "boom" }), { status: 404, headers: { "Content-Type": "application/json" } }))
    renderPage("file=cat")
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("The image is no longer there.")
    vi.restoreAllMocks()
    mockServer()
    fireEvent.click(within(alert).getByRole("button", { name: "Retry" }))
    expect(await screen.findByRole("heading", { name: "cat.png" })).toBeInTheDocument()
  })

  it("listens for pasted images once, not again on every render", async () => {
    const calls = mockServer()
    const listen = vi.spyOn(window, "addEventListener")
    const { rerender } = renderPage()
    // the page's own listener and the drop zone's: neither is added again
    const pastes = () => listen.mock.calls.filter(([type]) => type === "paste").length
    const before = pastes()
    for (let i = 0; i < 3; i++) {
      rerender(
        <QueryClientProvider client={queryClient}>
          <PngInfoPage route={{ name: "png-info" }} />
        </QueryClientProvider>,
      )
    }
    expect(pastes()).toBe(before)
    // and the one listener still reads a pasted image
    const paste = new Event("paste", { bubbles: true, cancelable: true }) as ClipboardEvent
    Object.defineProperty(paste, "clipboardData", { value: { files: [new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], "p.png", { type: "image/png" })] } })
    act(() => void document.body.dispatchEvent(paste))
    await waitFor(() => expect(calls.some((call) => call.path === "/uploads")).toBe(true))
  })

  it("groups entries in order", () => {
    expect(groupEntries(metadata.entries).map(([group, rows]) => [group, rows.length])).toEqual([
      ["Image", 1],
      ["Metadata", 1],
      ["EXIF", 1],
    ])
  })
})
