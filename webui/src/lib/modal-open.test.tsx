import { cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it } from "vitest"
import { SearchSelect } from "@/components/SearchSelect"
import { AlertDialog, AlertDialogContent, AlertDialogTitle } from "@/components/ui/alert-dialog"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover"
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet"
import { modalOpen } from "./modal-open"

// the first open of a popover, a menu or a search select loads its implementation
const slow = { timeout: 15_000 }

describe("modalOpen", { timeout: 30_000 }, () => {
  afterEach(() => {
    cleanup()
    document.body.innerHTML = ""
  })

  it("is false on a bare page", () => {
    render(<button type="button">Generate</button>)
    expect(modalOpen()).toBe(false)
  })

  it("counts the kit's dialog, alert dialog and sheet", async () => {
    const { unmount } = render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Edit style</DialogTitle>
        </DialogContent>
      </Dialog>,
    )
    await screen.findByRole("dialog", { name: "Edit style" })
    expect(modalOpen()).toBe(true)
    unmount()
    expect(modalOpen()).toBe(false)

    const alert = render(
      <AlertDialog open>
        <AlertDialogContent>
          <AlertDialogTitle>Delete the recipe?</AlertDialogTitle>
        </AlertDialogContent>
      </AlertDialog>,
    )
    await screen.findByRole("alertdialog", { name: "Delete the recipe?" })
    expect(modalOpen()).toBe(true)
    alert.unmount()

    // a sheet's popup says which side it slides from (`data-side`) but sits in no positioner
    render(
      <Sheet open>
        <SheetContent>
          <SheetTitle>Sections</SheetTitle>
        </SheetContent>
      </Sheet>,
    )
    const sheet = await screen.findByRole("dialog", { name: "Sections" })
    expect(sheet).toHaveAttribute("data-side")
    expect(modalOpen()).toBe(true)
  })

  it("leaves out a popover, a search select's popup and a menu", async () => {
    const user = userEvent.setup()
    render(
      <>
        <Popover defaultOpen>
          <PopoverTrigger>Aspect ratio</PopoverTrigger>
          <PopoverContent aria-labelledby="ratio-title">
            <PopoverTitle id="ratio-title">Aspect ratio</PopoverTitle>
          </PopoverContent>
        </Popover>
        <SearchSelect options={[{ value: "Euler", label: "Euler" }]} value="Euler" onValueChange={() => {}} aria-label="Sampler" />
      </>,
    )
    await screen.findByRole("dialog", { name: "Aspect ratio" }, slow)
    expect(modalOpen()).toBe(false)
    await user.click(screen.getByRole("combobox", { name: "Sampler" }))
    await screen.findByRole("dialog", { name: "Sampler" }, slow)
    expect(modalOpen()).toBe(false)
  })

  it("leaves out an open menu", async () => {
    render(
      <DropdownMenu defaultOpen>
        <DropdownMenuTrigger>More actions</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Download</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    await screen.findByRole("menu", {}, slow)
    expect(modalOpen()).toBe(false)
  })

  it("counts a dialog opened from inside a popover", async () => {
    render(
      <Popover defaultOpen>
        <PopoverTrigger>Model settings</PopoverTrigger>
        <PopoverContent aria-label="Model settings">
          <Dialog open>
            <DialogContent>
              <DialogTitle>Edit metadata</DialogTitle>
            </DialogContent>
          </Dialog>
        </PopoverContent>
      </Popover>,
    )
    await screen.findByRole("dialog", { name: "Edit metadata" }, slow)
    await waitFor(() => expect(modalOpen()).toBe(true))
  })

  it("leaves out a panel that says it is not modal, a closing dialog and a hidden one", () => {
    const drawer = document.body.appendChild(document.createElement("aside"))
    drawer.setAttribute("role", "dialog")
    drawer.setAttribute("aria-modal", "false")
    expect(modalOpen()).toBe(false)

    const closing = document.body.appendChild(document.createElement("div"))
    closing.setAttribute("role", "dialog")
    closing.setAttribute("data-closed", "")
    expect(modalOpen()).toBe(false)

    const kept = document.body.appendChild(document.createElement("div"))
    kept.hidden = true
    kept.appendChild(document.createElement("div")).setAttribute("role", "alertdialog")
    expect(modalOpen()).toBe(false)
  })

  it("counts a hand-written modal (the sign-in screen)", () => {
    const signIn = document.body.appendChild(document.createElement("div"))
    signIn.setAttribute("role", "dialog")
    signIn.setAttribute("aria-modal", "true")
    expect(modalOpen()).toBe(true)
  })
})
