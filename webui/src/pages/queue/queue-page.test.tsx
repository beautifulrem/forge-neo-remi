import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import type { Job } from "@/api/types"
import { navigate, useRoute } from "@/app/router"
import { jobOf } from "@/test/api-fixtures"
import QueuePage from "./index"

const finished = (id: string, summary: string, status: Job["status"] = "completed"): Job => ({ ...jobOf(id, status), summary })

/** Serves the queue and the history; a history request whose status is in `held` waits for `release`. */
function mockServer(held: string[] = []) {
  let release: () => void = () => {}
  const gate = new Promise<void>((resolve) => (release = resolve))
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL((input as Request).url)
    const path = url.pathname.replace("/api/v2", "")
    const json = (data: unknown) => new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json" } })
    if (path === "/queue/items") return json([])
    if (path === "/queue") return json({ size: 0, paused: false, running: null })
    if (path === "/jobs") {
      const status = url.searchParams.get("status") ?? ""
      if (held.includes(status)) await gate
      const items = status === "failed" ? [finished("f1", "a failed fox", "failed")] : status === "cancelled,interrupted" ? [] : [finished("c1", "a calm cat"), finished("c2", "a sleepy dog")]
      return json({ items, next_cursor: null, total: items.length })
    }
    return json({})
  })
  return { release }
}

function Page() {
  return <QueuePage route={useRoute()} />
}

const renderPage = () =>
  render(
    <QueryClientProvider client={queryClient}>
      <Page />
    </QueryClientProvider>,
  )

describe("queue page on a phone", () => {
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(600)
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(390)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  // Back from a job opened in its form came back to the Queue tab, not the History it left
  it("keeps the History tab in the address", async () => {
    mockServer()
    navigate({ name: "queue" }, { replace: true })
    renderPage()
    fireEvent.click(screen.getByRole("tab", { name: "History" }))
    expect(new URLSearchParams(location.hash.split("?")[1]).get("section")).toBe("history")
    // a filter written later keeps the tab
    expect(await screen.findByText("a calm cat")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("radio", { name: "Failed" }))
    await waitFor(() => expect(new URLSearchParams(location.hash.split("?")[1]).get("status")).toBe("failed"))
    expect(new URLSearchParams(location.hash.split("?")[1]).get("section")).toBe("history")
    fireEvent.click(screen.getByRole("tab", { name: /^Queue\d*$/ }))
    expect(new URLSearchParams(location.hash.split("?")[1]).get("section")).toBeNull()
  })

  it("opens on the History tab the address names", async () => {
    mockServer()
    navigate({ name: "queue", query: new URLSearchParams("section=history") }, { replace: true })
    renderPage()
    expect(screen.getByRole("tab", { name: "History" })).toHaveAttribute("aria-selected", "true")
    expect(await screen.findByText("a calm cat")).toBeInTheDocument()
  })

  it("keeps the rows shown while another filter's history loads", async () => {
    const server = mockServer(["failed"])
    navigate({ name: "queue" }, { replace: true })
    renderPage()
    fireEvent.click(screen.getByRole("tab", { name: "History" }))
    const panel = screen.getByRole("tabpanel")
    expect(await within(panel).findByText("a calm cat")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("radio", { name: "Failed" }))
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)))
    // no skeleton in between: the old rows stay until the new ones arrive
    expect(within(panel).getByText("a calm cat")).toBeInTheDocument()
    expect(panel.querySelector("[data-slot=skeleton]")).toBeNull()
    await act(async () => server.release())
    expect(await within(panel).findByText("a failed fox")).toBeInTheDocument()
    expect(within(panel).queryByText("a calm cat")).toBeNull()
  })

  it("does not say an empty filter's 'no jobs' while another filter loads", async () => {
    const server = mockServer(["completed"])
    navigate({ name: "queue", query: new URLSearchParams("section=history&status=stopped") }, { replace: true })
    renderPage()
    const panel = screen.getByRole("tabpanel")
    expect(await within(panel).findByText("No jobs match.")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("radio", { name: "Completed" }))
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)))
    // the stopped jobs' empty page is not the completed ones' answer
    expect(within(panel).queryByText("No jobs match.")).toBeNull()
    expect(panel.querySelector("[data-slot=skeleton]")).not.toBeNull()
    await act(async () => server.release())
    expect(await within(panel).findByText("a calm cat")).toBeInTheDocument()
  })
})
