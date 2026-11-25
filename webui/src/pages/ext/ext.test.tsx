import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { keys, pageSchemaQuery, queryClient } from "@/api/queries"
import { bootstrapWith } from "@/test/api-fixtures"
import type { PageSchema } from "@/ui/schema"
import { applyToPage, flushPending, loadedSchema, loadPage, PENDING_TIMEOUT_MS, pageStore, sendToPage } from "./deliver"
import ExtensionPage from "./index"
import { PageView } from "./PageView"
import { PageSendButtons } from "./send-targets"
import { sendImageToPage, sendTargetsOf } from "./targets"

const schema = (id: string, extra: Record<string, unknown> = {}): PageSchema =>
  ({
    version: 1,
    id,
    title: "Captioner",
    owner: `page:${id}`,
    layout: {
      type: "column",
      children: [
        { type: "image", name: "image", label: "Image", default: null, kind: "image", on_change: "image_changed" },
        { type: "text", name: "caption", label: "Caption", default: "", lines: 2, prompt: false },
        { type: "output", name: "status", kind: "text" },
      ],
    },
    defaults: { image: null, caption: "" },
    ...extra,
  }) as PageSchema

function answering(patch: Record<string, unknown> = {}) {
  const calls: { path: string; body: Record<string, unknown> }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    calls.push({ path: decodeURIComponent(new URL(request.url).pathname), body: (await request.json().catch(() => ({}))) as Record<string, unknown> })
    return new Response(JSON.stringify(patch), { headers: { "Content-Type": "application/json" } })
  })
  return calls
}

describe("page deliveries", () => {
  beforeEach(() => history.replaceState(null, "", "#/png-info"))
  afterEach(() => vi.restoreAllMocks())

  it("keeps values, uploads and outputs when the schema is reloaded", () => {
    const store = loadPage(schema("keep"))
    store.getState().setValues("page:keep", { image: "up-1", caption: "a cat" })
    store.getState().setOutputs("page:keep", { status: "ready" })
    const next = schema("keep")
    loadPage(next)
    expect(loadedSchema(store)).toBe(next)
    expect(store.getState().values["page:keep"]).toEqual({ image: "up-1", caption: "a cat" })
    expect(store.getState().outputs["page:keep"]).toEqual({ status: "ready" })
  })

  it("holds a send until the page has loaded, then applies it and runs on_change", async () => {
    const calls = answering({ caption: { value: "from the image" } })
    const sent = sendToPage("later", { image: "up-2", unknown: 1 })
    expect(location.hash).toBe("#/ext/later")
    loadPage(schema("later"))
    flushPending("later")
    await expect(sent).resolves.toEqual({ count: 1, locked: [] })
    expect(pageStore("later").getState().values["page:later"]?.image).toBe("up-2")
    await waitFor(() => expect(calls.map((call) => call.path)).toEqual(["/api/v2/actions/page:later/image_changed"]))
    expect(calls[0]!.body).toMatchObject({ tab: "page:later", values: { image: "up-2" } })
    await waitFor(() => expect(pageStore("later").getState().values["page:later"]?.caption).toBe("from the image"))
  })

  it("gives up a send when the page fails to load, is missing or never loads", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ detail: "no page" }), { status: 404, headers: { "Content-Type": "application/json" } }))
    const broken = sendToPage("broken", { image: "up-4" })
    await queryClient.prefetchQuery({ ...pageSchemaQuery("broken"), retry: false })
    await expect(broken).resolves.toEqual({ count: 0, locked: [] })

    queryClient.setQueryData(keys.app, { ...bootstrapWith({}), pages: [] })
    const gone = sendToPage("gone", { image: "up-5" })
    render(
      <QueryClientProvider client={queryClient}>
        <ExtensionPage route={{ name: "page", id: "gone" }} />
      </QueryClientProvider>,
    )
    await expect(gone).resolves.toEqual({ count: 0, locked: [] })
    expect(screen.getByText("Page not found")).toBeInTheDocument()
    queryClient.clear()

    vi.useFakeTimers()
    try {
      const slow = sendToPage("slow", { image: "up-6" })
      vi.advanceTimersByTime(PENDING_TIMEOUT_MS)
      await expect(slow).resolves.toEqual({ count: 0, locked: [] })
    } finally {
      vi.useRealTimers()
    }
  })

  it("applies at once to a loaded page and shows it", async () => {
    answering()
    loadPage(schema("now"))
    await expect(sendToPage("now", { image: "up-3" })).resolves.toEqual({ count: 1, locked: [] })
    expect(location.hash).toBe("#/ext/now")
    expect(applyToPage("now", { nope: 1 })).toEqual({ count: 0, locked: [] })
  })

  it("names the values a lock kept on the page, in the send's own toast, and does not count them", async () => {
    answering()
    const info = vi.spyOn(toast, "info")
    const failure = vi.spyOn(toast, "error")
    const locked = schema("locked")
    // a toggle of the page that locks its image (`locks: "image"`) is on
    const layout = locked.layout as { children: unknown[] }
    layout.children.push({ type: "toggle", name: "keep", label: "Keep the image", default: true, locks: "image" })
    loadPage({ ...locked, defaults: { ...locked.defaults, keep: true } })
    pageStore("locked").getState().setValue("page:locked", "image", "mine")
    await expect(sendToPage("locked", { image: "up-7", caption: "a cat" })).resolves.toEqual({ count: 1, locked: ["page:locked.image"] })
    await sendImageToPage({ pageId: "locked", target: { kind: "image", param: "image", label: "Captioner" } }, "up-8")
    expect(failure).toHaveBeenCalledWith("Could not send to Captioner", expect.objectContaining({ description: "Overridden by a lock: Image" }))
    expect(pageStore("locked").getState().values["page:locked"]?.image).toBe("mine")
    await Promise.resolve()
    expect(info).not.toHaveBeenCalled()
  })
})

describe("PageView", () => {
  afterEach(() => vi.restoreAllMocks())

  it("renders the page form and runs on_load each time it is shown", async () => {
    const calls = answering({ __outputs__: { status: "Model ready" } })
    const page = schema("view", { on_load: "status" })
    const { unmount } = render(<PageView schema={page} />)
    expect(screen.getByText("Caption")).toBeInTheDocument()
    expect(await screen.findByText("Model ready")).toBeInTheDocument()
    unmount()
    render(<PageView schema={page} />)
    await waitFor(() => expect(calls.filter((call) => call.path === "/api/v2/actions/page:view/status")).toHaveLength(2))
  })
})

describe("send targets", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("reads well-formed targets only", () => {
    expect(sendTargetsOf(schema("x", { send_to: [{ kind: "image", param: "image", label: "X" }, { kind: "audio", param: "a", label: "A" }, "bad"] }))).toEqual([
      { kind: "image", param: "image", label: "X" },
    ])
    expect(sendTargetsOf(schema("y"))).toEqual([])
  })

  it("offers a button per page accepting images and sends the file to it", async () => {
    queryClient.setQueryData(keys.app, {
      ...bootstrapWith({}),
      pages: [
        { id: "joycaption", title: "JoyCaption" },
        { id: "view-only", title: "View", widget: "custom-view" },
      ],
    })
    queryClient.setQueryData(keys.page("joycaption"), schema("joycaption", { send_to: [{ kind: "image", param: "image", label: "JoyCaption" }] }))
    answering()
    render(
      <QueryClientProvider client={queryClient}>
        <PageSendButtons fileId="out-9" />
      </QueryClientProvider>,
    )
    fireEvent.click(await screen.findByRole("button", { name: "JoyCaption" }))
    expect(location.hash).toBe("#/ext/joycaption")
    loadPage(schema("joycaption"))
    flushPending("joycaption")
    await waitFor(() => expect(pageStore("joycaption").getState().values["page:joycaption"]?.image).toBe("out-9"))
  })
})

// A failed bootstrap (a restart, a short 5xx) said "Page not found", dropped a pending send to the
// page, and nothing but a reload asked again
describe("extension page while the app cannot load", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.setQueryDefaults(keys.app, {})
    queryClient.removeQueries({ queryKey: keys.app })
  })

  it("says so, keeps a send waiting and asks again", async () => {
    let up = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      up ? Response.json({ ...bootstrapWith({}), pages: [{ id: "after-restart", title: "After restart", widget: "not-in-this-build" }] }) : Response.json({ detail: "boom" }, { status: 503 }),
    )
    queryClient.removeQueries({ queryKey: keys.app })
    queryClient.setQueryDefaults(keys.app, { retry: false })
    history.replaceState(null, "", "#/png-info")
    let settled: { count: number } | null = null
    void sendToPage("after-restart", { caption: "a cat" }).then((applied) => (settled = applied))
    render(
      <QueryClientProvider client={queryClient}>
        <ExtensionPage route={{ name: "page", id: "after-restart" }} />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Page unavailable")).toBeInTheDocument()
    expect(screen.getByText("The server is busy; try again shortly.")).toBeInTheDocument()
    expect(screen.queryByText("Page not found")).toBeNull()
    expect(settled).toBeNull()
    up = true
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("After restart")).toBeInTheDocument()
  })
})
