import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { navigate } from "./router"
import { useRouteParams, useRouteSearch } from "./route-params"

const DEFAULTS = { status: "all", q: "" }

function Filters() {
  const [params, setParams] = useRouteParams(DEFAULTS)
  const [text, setText] = useRouteSearch(params.q, (q) => setParams({ q }))
  return (
    <>
      <input aria-label="Search" value={text} onChange={(event) => setText(event.target.value)} />
      <button type="button" onClick={() => setParams({ status: params.status === "all" ? "failed" : "all" })}>
        {params.status}
      </button>
    </>
  )
}

describe("route params", () => {
  afterEach(() => vi.useRealTimers())

  it("keeps the filters in the address, beside its other entries, without their defaults", () => {
    navigate({ name: "queue", query: new URLSearchParams({ view: "recipes" }) })
    const length = history.length
    render(<Filters />)
    fireEvent.click(screen.getByRole("button", { name: "all" }))
    expect(location.hash).toBe("#/queue?view=recipes&status=failed")
    expect(history.length).toBe(length) // replaced, not pushed
    fireEvent.click(screen.getByRole("button", { name: "failed" }))
    expect(location.hash).toBe("#/queue?view=recipes")
  })

  it("writes the search after a pause, and takes the address's value on Back", () => {
    vi.useFakeTimers()
    navigate({ name: "queue", query: new URLSearchParams({ q: "dog" }) })
    render(<Filters />)
    const field = screen.getByRole("textbox", { name: "Search" })
    expect(field).toHaveValue("dog")
    fireEvent.change(field, { target: { value: "cat " } })
    expect(location.hash).toBe("#/queue?q=dog")
    act(() => vi.advanceTimersByTime(300))
    expect(location.hash).toBe("#/queue?q=cat")
    expect(field).toHaveValue("cat ") // the space being typed stays

    // Back (or a link) to another search: the field shows it, and nothing is written over it
    fireEvent.change(field, { target: { value: "bird" } })
    act(() => navigate({ name: "queue", query: new URLSearchParams({ q: "dog" }) }, { replace: true }))
    expect(field).toHaveValue("dog")
    act(() => vi.advanceTimersByTime(1000))
    expect(location.hash).toBe("#/queue?q=dog")
  })
})
