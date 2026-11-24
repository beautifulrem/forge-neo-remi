import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
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
    const zone = screen.getByRole("button", { name: "Choose an image to read" })
    const file = new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], "cat.png", { type: "image/png" })
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    await waitFor(() => expect(calls.some((call) => call.path === "/uploads")).toBe(true))
    await waitFor(() => expect(location.hash).toBe("#/png-info?file=up1"))
  })

  it("refuses a file that is not an image without uploading it", async () => {
    const calls = mockServer()
    const error = vi.spyOn(toast, "error")
    renderPage()
    const zone = screen.getByRole("button", { name: "Choose an image to read" })
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

  it("groups entries in order", () => {
    expect(groupEntries(metadata.entries).map(([group, rows]) => [group, rows.length])).toEqual([
      ["Image", 1],
      ["Metadata", 1],
      ["EXIF", 1],
    ])
  })
})
