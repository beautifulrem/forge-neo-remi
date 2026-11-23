import { act, fireEvent, render, screen } from "@testing-library/react"
import { lazy, Suspense, type ComponentType, type ReactNode } from "react"
import { describe, expect, it, vi } from "vitest"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { retryableLazy } from "./retryable-lazy"

const chunkError = () => new Error("Failed to fetch dynamically imported module")
const page = { default: () => <p>loaded page</p> }

/** A chunk that fails `failures` times, then loads (Infinity: never). */
function loader(failures: number) {
  let left = failures
  return vi.fn<() => Promise<{ default: ComponentType }>>(() => (left-- > 0 ? Promise.reject(chunkError()) : Promise.resolve(page)))
}

function mount(Page: ComponentType, fallback?: ReactNode) {
  render(
    <ErrorBoundary fallback={fallback}>
      <Suspense fallback={<p>loading</p>}>
        <Page />
      </Suspense>
    </ErrorBoundary>,
  )
}

const settle = () => act(() => new Promise((resolve) => setTimeout(resolve, 200)))

describe("retryableLazy", () => {
  it("shows a chunk that keeps failing as an error, without fetching it in a loop", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = loader(Infinity)
    mount(retryableLazy(load))
    expect(await screen.findByRole("alert")).toBeInTheDocument()
    await settle()
    expect(load).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    expect(await screen.findByRole("alert")).toBeInTheDocument()
    await settle()
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("fetches the chunk again on Try again", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = loader(1)
    mount(retryableLazy(load))
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }))
    expect(await screen.findByText("loaded page")).toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("brings back a shell control that hid itself once the browser is online", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = loader(1)
    mount(retryableLazy(load), null)
    await settle()
    expect(screen.queryByText("loaded page")).not.toBeInTheDocument()
    act(() => {
      window.dispatchEvent(new Event("online"))
    })
    expect(await screen.findByText("loaded page")).toBeInTheDocument()
  })

  it("fetches the chunk again when a boundary mounts on a load that failed before", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = loader(1)
    const Sheet = retryableLazy(load)
    const first = render(
      <ErrorBoundary fallback={null}>
        <Suspense fallback={<p>loading</p>}>
          <Sheet />
        </Suspense>
      </ErrorBoundary>,
    )
    await settle()
    first.unmount() // the More sheet closed with nothing in it
    mount(Sheet, null) // and opened again
    expect(await screen.findByText("loaded page")).toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("is needed: React.lazy keeps the failure", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const load = loader(1)
    mount(lazy(load))
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }))
    expect(await screen.findByRole("alert")).toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(1)
  })
})

describe("lazy components in the app", () => {
  it("are all retryable: a chunk that failed once loads again on Try again or back online", async () => {
    const { readFileSync, readdirSync } = await import("node:fs")
    const { join } = await import("node:path")
    const files = (readdirSync("src", { recursive: true }) as string[]).filter((file) => /\.tsx?$/.test(file) && !/\.test\.tsx?$/.test(file))
    const plain = files.filter((file) => file !== join("lib", "retryable-lazy.ts") && /(?<![\w.])lazy\(/.test(readFileSync(join("src", file), "utf8")))
    expect(plain).toEqual([])
  })
})
