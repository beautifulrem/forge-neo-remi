import { render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { DeleteDialog } from "./DeleteDialog"

describe("DeleteDialog", () => {
  it("opens with the focus on Cancel, not on deleting the files", async () => {
    render(<DeleteDialog ids={["file-1", "file-2"]} onCancel={vi.fn()} onConfirm={vi.fn()} />)
    const cancel = await screen.findByRole("button", { name: "Cancel" })
    await waitFor(() => expect(document.activeElement).toBe(cancel))
    expect(screen.getByRole("checkbox", { name: "Also delete the files from disk" })).not.toHaveFocus()
  })
})
