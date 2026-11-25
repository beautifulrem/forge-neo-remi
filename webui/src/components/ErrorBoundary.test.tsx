import { QueryClient, QueryClientProvider, useSuspenseQuery } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { Suspense } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ApiError } from "@/api/http"
import { ErrorBoundary } from "./ErrorBoundary"

function Schema({ load }: { load: () => Promise<string> }) {
  const { data } = useSuspenseQuery({ queryKey: ["schema"], queryFn: load, staleTime: Infinity, retry: false })
  return <p>{data}</p>
}

// A suspense query that has thrown is not fetched again on its next mount unless its error boundary is
// reset (TanStack Query): "Try again", and leaving the page and coming back, only showed the same error
// until a reload (a server restart or a short 5xx on the first visit of txt2img, img2img, Extras, a page).
describe("error boundary", () => {
  afterEach(() => vi.restoreAllMocks())

  it("asks again for the query that failed", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = vi.fn<() => Promise<string>>().mockRejectedValueOnce(new Error("The server is busy")).mockResolvedValue("the schema")
    const client = new QueryClient()
    const page = (key: string) => (
      <QueryClientProvider client={client}>
        <ErrorBoundary resetKey={key}>
          <Suspense fallback={<p>Loading</p>}>
            <Schema load={load} />
          </Suspense>
        </ErrorBoundary>
      </QueryClientProvider>
    )
    render(page("txt2img"))
    expect(await screen.findByText("Something went wrong")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    expect(await screen.findByText("the schema")).toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(2)
    client.clear()
  })

  it("asks again when the page is shown again", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = vi.fn<() => Promise<string>>().mockRejectedValueOnce(new Error("The server is busy")).mockResolvedValue("the schema")
    const client = new QueryClient()
    const page = (key: string, shown: boolean) => (
      <QueryClientProvider client={client}>
        <ErrorBoundary resetKey={key}>
          <Suspense fallback={<p>Loading</p>}>{shown ? <Schema load={load} /> : <p>Queue</p>}</Suspense>
        </ErrorBoundary>
      </QueryClientProvider>
    )
    const { rerender } = render(page("txt2img", true))
    expect(await screen.findByText("Something went wrong")).toBeInTheDocument()
    rerender(page("queue", false))
    expect(await screen.findByText("Queue")).toBeInTheDocument()
    rerender(page("txt2img", true))
    expect(await screen.findByText("the schema")).toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(2)
    client.clear()
  })

  // The page fallback showed the server's English detail ("Internal server error (id …); the details
  // are in the server log") in the Chinese UI: a failed request is explained by its status.
  it("explains a failed request in the UI language", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = vi.fn<() => Promise<string>>().mockRejectedValue(new ApiError(500, "Internal server error (id 0a1b2c); the details are in the server log"))
    const client = new QueryClient()
    render(
      <QueryClientProvider client={client}>
        <ErrorBoundary resetKey="txt2img">
          <Suspense fallback={<p>Loading</p>}>
            <Schema load={load} />
          </Suspense>
        </ErrorBoundary>
      </QueryClientProvider>,
    )
    expect(await screen.findByText("The server ran into an error (500).")).toBeInTheDocument()
    expect(screen.queryByText(/Internal server error/)).toBeNull()
    client.clear()
  })

  // A request that got no answer reached the page view as the browser's "Failed to fetch", and the
  // inline boundary printed any failed request's English detail after its own sentence.
  it("says a request that got no answer in the UI language", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"))
    const { api } = await import("@/api/endpoints")
    const client = new QueryClient()
    render(
      <QueryClientProvider client={client}>
        <ErrorBoundary resetKey="txt2img">
          <Suspense fallback={<p>Loading</p>}>
            <Schema load={() => api.jobs.list({ limit: 1 }).then(() => "listed")} />
          </Suspense>
        </ErrorBoundary>
      </QueryClientProvider>,
    )
    expect(await screen.findByText("The server cannot be reached.")).toBeInTheDocument()
    expect(screen.queryByText(/Failed to fetch/)).toBeNull()
    client.clear()
  })

  it("explains a failed request inside a form control in the UI language", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = vi.fn<() => Promise<string>>().mockRejectedValue(new ApiError(500, "Internal server error (id 9f9f); the details are in the server log"))
    const client = new QueryClient()
    render(
      <QueryClientProvider client={client}>
        <ErrorBoundary inline>
          <Suspense fallback={<p>Loading</p>}>
            <Schema load={load} />
          </Suspense>
        </ErrorBoundary>
      </QueryClientProvider>,
    )
    expect(await screen.findByRole("alert")).toHaveTextContent("This control failed to render: The server ran into an error (500).")
    expect(screen.queryByText(/Internal server error/)).toBeNull()
    client.clear()
  })
})
