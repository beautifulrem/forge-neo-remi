import { render, screen } from "@testing-library/react"
import { lazy, Suspense, type ComponentType } from "react"
import { describe, expect, it, vi } from "vitest"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { retryableLazy } from "./retryable-lazy"

function loader() {
  return vi
    .fn<() => Promise<{ default: ComponentType }>>()
    .mockRejectedValueOnce(new Error("Failed to fetch dynamically imported module"))
    .mockResolvedValue({ default: () => <p>loaded page</p> })
}

function mount(Page: ComponentType) {
  render(
    <ErrorBoundary>
      <Suspense fallback={<p>loading</p>}>
        <Page />
      </Suspense>
    </ErrorBoundary>,
  )
}

describe("retryableLazy", () => {
  it("fetches the chunk again after a failed load", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = loader()
    mount(retryableLazy(load))
    expect(await screen.findByText("loaded page")).toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("is needed: React.lazy keeps the failure", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = loader()
    mount(lazy(load))
    expect(await screen.findByRole("alert")).toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(1)
  })
})
