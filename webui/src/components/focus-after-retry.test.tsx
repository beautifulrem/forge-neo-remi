import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { LoadFailed } from "./LoadFailed"

type State = "failed" | "retrying" | "loaded" | "text" | "spinner"

/** A panel whose list failed: `state` says what it shows. Buttons before and after it are not its content. */
function Panel({ state }: { state: State }) {
  return (
    <div data-testid="panel" tabIndex={-1}>
      <button type="button">Before</button>
      {state === "failed" || state === "retrying" ? (
        <LoadFailed error={null} message="Could not load the list" retrying={state === "retrying"} onRetry={() => {}} />
      ) : state === "loaded" ? (
        <ul>
          <li>
            {/* out of the Tab order (a roving focus's other stops): not where the focus goes */}
            <button type="button" tabIndex={-1}>
              Roving stop
            </button>
            <button type="button">First item</button>
          </li>
        </ul>
      ) : state === "text" ? (
        <p>Nothing to press here</p>
      ) : (
        <p role="status">Loading…</p>
      )}
      <button type="button">After</button>
    </div>
  )
}

/** Renders the failure with its Retry focused and pressed, then shows it as retrying. */
function failedThenRetrying() {
  const view = render(<Panel state="failed" />)
  const retry = screen.getByRole("button", { name: "Retry" })
  act(() => retry.focus())
  fireEvent.click(retry)
  view.rerender(<Panel state="retrying" />)
  return { ...view, retry }
}

/** Lets the hand-off's frames run. */
const frames = () => act(() => new Promise((resolve) => setTimeout(resolve, 250)))

describe("the focus after a retry", () => {
  it("goes to the first tabbable element of what the retry loaded, not to the controls around it", async () => {
    const { rerender } = failedThenRetrying()
    rerender(<Panel state="loaded" />)
    await waitFor(() => expect(screen.getByRole("button", { name: "First item" })).toHaveFocus())
  })

  it("goes to the focusable container when what loaded has nothing to press", async () => {
    const { rerender } = failedThenRetrying()
    rerender(<Panel state="text" />)
    await waitFor(() => expect(screen.getByTestId("panel")).toHaveFocus())
  })

  it("waits for content drawn a render later than the data (a spinner first)", async () => {
    const { rerender } = failedThenRetrying()
    rerender(<Panel state="spinner" />)
    await act(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
    rerender(<Panel state="loaded" />)
    await waitFor(() => expect(screen.getByRole("button", { name: "First item" })).toHaveFocus())
  })

  it("goes there from the dialog that took a focus lost inside it", async () => {
    const view = render(
      <div role="dialog" aria-label="Import" tabIndex={-1}>
        <Panel state="failed" />
      </div>,
    )
    const retry = screen.getByRole("button", { name: "Retry" })
    act(() => retry.focus())
    fireEvent.click(retry)
    const dialog = (state: State) => (
      <div role="dialog" aria-label="Import" tabIndex={-1}>
        <Panel state={state} />
      </div>
    )
    view.rerender(dialog("retrying"))
    view.rerender(dialog("loaded"))
    // what Base UI's dialog does with a focus lost inside it
    act(() => screen.getByRole("dialog").focus())
    await waitFor(() => expect(screen.getByRole("button", { name: "First item" })).toHaveFocus())
  })

  it("stays where the user put it", async () => {
    const { rerender } = failedThenRetrying()
    const before = screen.getByRole("button", { name: "Before" })
    act(() => before.focus())
    rerender(<Panel state="loaded" />)
    await frames()
    expect(before).toHaveFocus()
  })

  it("is left alone when the failure goes without a retry", async () => {
    const view = render(<Panel state="failed" />)
    act(() => screen.getByRole("button", { name: "Retry" }).focus())
    view.rerender(<Panel state="loaded" />)
    await frames()
    expect(screen.getByRole("button", { name: "First item" })).not.toHaveFocus()
  })
})
