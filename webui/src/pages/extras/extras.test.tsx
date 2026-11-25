import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { extrasSchema } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { resetExtrasRuns, useExtrasRuns } from "@/features/extras/run"
import { getFormStore } from "@/ui/form-store"
import ExtrasPage from "./index"

function mockFetch() {
  const calls: { method: string; path: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const text = request.method === "GET" ? "" : await request.text()
    calls.push({ method: request.method, path, body: text ? JSON.parse(text) : undefined })
    const data = path === "/jobs" ? { job_id: "e1", job_ids: ["e1"], seq: 1 } : path === "/schema/extras" ? extrasSchema : {}
    return new Response(JSON.stringify(data), { status: path === "/jobs" ? 201 : 200, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

describe("extras page", () => {
  beforeEach(() => {
    resetExtrasRuns()
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
    queryClient.setQueryData(keys.schema("extras"), extrasSchema)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("renders the form from the schema and processes with Ctrl+Enter", async () => {
    const calls = mockFetch()
    render(
      <QueryClientProvider client={queryClient}>
        <ExtrasPage />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Scale by")).toBeInTheDocument()
    expect(screen.getByText("Results appear here, next to their source.")).toBeInTheDocument()
    act(() => getFormStore("extras").getState().setValue("core", "image", "src1")) // the source the schema requires
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true })
    await waitFor(() => expect(calls.some((call) => call.method === "POST" && call.path === "/jobs")).toBe(true))
    await waitFor(() => expect(screen.getByRole("button", { name: "Interrupt" })).toBeInTheDocument())
  })

  it("adds the form to the queue with Ctrl+Shift+Enter instead of processing it here", async () => {
    const calls = mockFetch()
    render(
      <QueryClientProvider client={queryClient}>
        <ExtrasPage />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Scale by")).toBeInTheDocument()
    act(() => getFormStore("extras").getState().setValue("core", "image", "src1"))
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true, shiftKey: true })
    await waitFor(() => expect(calls.filter((call) => call.method === "POST" && call.path === "/jobs")).toHaveLength(1))
    expect(calls.find((call) => call.path === "/jobs")?.body).toMatchObject({ tab: "extras" })
    expect(useExtrasRuns.getState().jobs).toEqual([])
    expect(screen.queryByRole("button", { name: "Interrupt" })).not.toBeInTheDocument()
  })

  it("compares a result with its source", async () => {
    mockFetch()
    useExtrasRuns.setState({ results: [{ jobId: "done", outputs: [{ id: "out1", kind: "image", width: 2048, height: 2048 }], source: "src1", info: "Postprocess upscale by: 4", at: 1 }] })
    render(
      <QueryClientProvider client={queryClient}>
        <ExtrasPage />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole("slider", { name: "Comparison divider" })).toBeInTheDocument()
    expect(screen.getByRole("img", { name: "Source" })).toHaveAttribute("src", expect.stringContaining("/files/src1"))
    fireEvent.click(screen.getByRole("radio", { name: "Result" }))
    expect(screen.getByRole("img", { name: "Processed image" })).toHaveAttribute("src", expect.stringContaining("/files/out1"))
    expect(screen.getByText("Postprocess upscale by: 4")).toBeInTheDocument()
  })

  it("says which values a run refused in the UI language", async () => {
    mockFetch()
    const { i18n } = await import("@/i18n")
    const choice = "{{value}} is not one of the choices ({{choices}})"
    i18n.addResource("zh-CN", "server", choice, "{{value}} 不在可选项（{{choices}}）中")
    i18n.addResourceBundle("zh-CN", "chrome", { Upscaler: "放大算法", "{{field}}: {{message}}": "{{field}}：{{message}}" }, true, true)
    const refused = { owner: "core", param: "upscaler_1", message: "'Gone' is not one of the choices ('None')", template: choice, params: { value: "'Gone'", choices: "'None'" } }
    useExtrasRuns.setState({ results: [{ jobId: "bad", outputs: [], source: null, info: null, error: "core.upscaler_1: 'Gone' is not one of the choices ('None')", errors: [refused], at: 1 }] })
    await act(() => i18n.changeLanguage("zh-CN"))
    try {
      render(
        <QueryClientProvider client={queryClient}>
          <ExtrasPage />
        </QueryClientProvider>,
      )
      // named by its field, as the toast of a refused submission names it
      expect(await screen.findByText("放大算法：'Gone' 不在可选项（'None'）中")).toBeInTheDocument()
    } finally {
      await act(() => i18n.changeLanguage("en"))
    }
  })

  // The placeholder was three tall blocks (224, 128 and 96 px), 172 px taller than the form on a phone,
  // and the result area under it jumped up when the schema arrived. It is the form's first blocks,
  // made of the same sizes (the source image's label over its empty drop zone, then two fields), and
  // on a phone the result area comes in with the form: no fixed height fits every form.
  describe("while its schema loads", () => {
    beforeEach(() => queryClient.removeQueries({ queryKey: keys.schema("extras") }))

    it("holds the form's shape, with nothing under it on a phone", () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise<Response>(() => {}))
      render(
        <QueryClientProvider client={queryClient}>
          <ExtrasPage />
        </QueryClientProvider>,
      )
      const skeleton = document.querySelector<HTMLElement>("[data-slot=form-skeleton]")!
      expect(skeleton).toHaveAttribute("aria-busy", "true")
      const classes = (element: Element) => element.className.split(/\s+/)
      const [image, ...fields] = [...skeleton.children]
      // the field's label line over the drop zone, with the upload hint's padding, icon and line of text
      expect(classes(image!.querySelector("[data-part=label]")!)).toContain("h-5")
      const zone = image!.querySelector("[data-part=drop-zone]")!
      expect(classes(zone)).toEqual(expect.arrayContaining(["rounded-lg", "border", "py-6"]))
      expect(zone.querySelector(".size-5")).not.toBeNull()
      expect(zone.querySelector(".text-sm")).not.toBeNull()
      // a label line over a control as tall as a select or a slider
      expect(fields).toHaveLength(2)
      for (const field of fields) {
        expect(classes(field.querySelector("[data-part=label]")!)).toContain("h-5")
        expect(classes(field.querySelector("[data-part=control]")!)).toContain("h-8")
      }
      expect(screen.queryByRole("region", { name: "Result" })).toBeNull()
    })

    it("brings the result area in with the form", async () => {
      mockFetch()
      render(
        <QueryClientProvider client={queryClient}>
          <ExtrasPage />
        </QueryClientProvider>,
      )
      expect(document.querySelector("[data-slot=form-skeleton]")).not.toBeNull()
      expect(await screen.findByText("Scale by")).toBeInTheDocument()
      expect(screen.getByRole("region", { name: "Result" })).toHaveTextContent("Results appear here, next to their source.")
      expect(document.querySelector("[data-slot=form-skeleton]")).toBeNull()
    })
  })
})
