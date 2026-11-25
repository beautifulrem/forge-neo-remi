import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { useState } from "react"
import { toast } from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { RequestError } from "@/api/http"
import { ConfirmDialog } from "./ConfirmDialog"

const view = vi.hoisted(() => ({ broken: true }))

vi.mock("./ConfirmDialogView", () => ({
  ConfirmDialogView: ({ open, title }: { open: boolean; title: string }) => {
    if (view.broken) throw new RequestError(new TypeError("Failed to fetch dynamically imported module"))
    return open ? <div role="alertdialog">{title}</div> : null
  },
}))

function Page() {
  const [open, setOpen] = useState(false)
  return (
    <>
      <p>the history</p>
      <button type="button" onClick={() => setOpen(true)}>
        Clear
      </button>
      <ConfirmDialog open={open} onOpenChange={setOpen} title="Clear the history?" description="" confirmLabel="Clear" onConfirm={() => {}} />
    </>
  )
}

describe("confirm dialog", () => {
  afterEach(() => {
    view.broken = true
  })

  // The dialog's code loads when it first opens: a chunk that could not be fetched reached the
  // page's boundary and replaced the whole view (the history, the queue) with an error.
  it("keeps the page when the dialog cannot be shown, and opens it next time", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const failed = vi.spyOn(toast, "error")
    render(<Page />)
    fireEvent.click(screen.getByRole("button", { name: "Clear" }))
    await waitFor(() => expect(failed).toHaveBeenCalledWith("The dialog could not be shown", { description: "The server cannot be reached." }))
    expect(screen.getByText("the history")).toBeInTheDocument()
    expect(screen.queryByRole("alertdialog")).toBeNull()

    view.broken = false
    fireEvent.click(screen.getByRole("button", { name: "Clear" }))
    expect(await screen.findByRole("alertdialog")).toHaveTextContent("Clear the history?")
  })
})
