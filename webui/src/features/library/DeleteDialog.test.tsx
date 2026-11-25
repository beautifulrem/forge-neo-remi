import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { DeleteDialog } from "./DeleteDialog"

function renderDialog(ids: string[]) {
  const onConfirm = vi.fn()
  render(<DeleteDialog ids={ids} onCancel={() => {}} onConfirm={onConfirm} />)
  return { dialog: screen.getByRole("alertdialog"), onConfirm }
}

describe("DeleteDialog", () => {
  it("opens with the focus on Cancel, not on deleting the files", async () => {
    render(<DeleteDialog ids={["file-1", "file-2"]} onCancel={vi.fn()} onConfirm={vi.fn()} />)
    const cancel = await screen.findByRole("button", { name: "Cancel" })
    await waitFor(() => expect(document.activeElement).toBe(cancel))
    expect(screen.getByRole("checkbox", { name: "Also delete the files from disk" })).not.toHaveFocus()
  })

  // Removing one image said "The images leave the gallery; their files stay…".
  it("speaks of one image in the singular", () => {
    const { dialog } = renderDialog(["a"])
    expect(within(dialog).getByText("Remove this image?")).toBeInTheDocument()
    expect(within(dialog).getByText("The image leaves the gallery; its file stays in the output folder.")).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole("checkbox"))
    expect(within(dialog).getByText("The file is deleted from the output folder. This cannot be undone.")).toBeInTheDocument()
    expect(within(dialog).getByText("Also delete the file from disk")).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Delete file" })).toBeInTheDocument()
  })

  it("speaks of several images in the plural", () => {
    const { dialog, onConfirm } = renderDialog(["a", "b"])
    expect(within(dialog).getByText("Remove 2 images?")).toBeInTheDocument()
    expect(within(dialog).getByText("The images leave the gallery; their files stay in the output folder.")).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole("checkbox"))
    expect(within(dialog).getByText("The files are deleted from the output folder. This cannot be undone.")).toBeInTheDocument()
    expect(within(dialog).getByText("Also delete the files from disk")).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete files" }))
    expect(onConfirm).toHaveBeenCalledWith(["a", "b"], true)
  })
})
