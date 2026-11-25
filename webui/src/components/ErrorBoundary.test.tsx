import { QueryClient, QueryClientProvider, useSuspenseQuery } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { Suspense } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
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
})
