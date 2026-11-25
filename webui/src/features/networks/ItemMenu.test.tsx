import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import type { NetworkItem, NetworkKind } from "@/api/types"
import { ItemMenu } from "./ItemMenu"
import type { MenuAnchor } from "./items"

const KIND: NetworkKind = { id: "lora", title: "LoRA", action: "prompt", negative: true, fields: ["description"], builtin: true }
const item = (extra: Partial<NetworkItem> = {}): NetworkItem => ({ name: "ink", title: "ink", kind: "lora", path: "ink.safetensors", folder: "", tags: [], preferred_modules: [], search_terms: [], editable: true, hidden: false, ...extra })

function open(props: Partial<Parameters<typeof ItemMenu>[0]> = {}) {
  const anchor = document.createElement("div")
  document.body.append(anchor)
  const onCommand = vi.fn()
  render(<ItemMenu item={item()} kind={KIND} anchor={anchor} active={false} canOpenFolder onClose={() => {}} onCommand={onCommand} {...props} />)
  return { menu: screen.getByRole("menu"), onCommand }
}

const labels = (menu: HTMLElement) => within(menu).getAllByRole("menuitem").map((entry) => entry.textContent?.replace(/[IE]$/, ""))

describe("item menu", () => {
  it("offers every command of an item with a file", () => {
    const { menu, onCommand } = open()
    expect(labels(menu)).toEqual(["Add to the prompt", "Metadata", "Edit metadata", "Set preview", "Copy path", "Open folder"])
    fireEvent.click(within(menu).getByRole("menuitem", { name: /Metadata/ }))
    expect(onCommand).toHaveBeenCalledWith("metadata", expect.objectContaining({ name: "ink" }))
  })

  it("leaves out what needs a model file, and Open folder when folders are hidden", () => {
    expect(labels(open({ item: item({ editable: false }) }).menu)).toEqual(["Add to the prompt", "Metadata", "Copy path"])
  })

  it("hides Open folder when the server does not allow it, and says Remove for an active card", () => {
    expect(labels(open({ canOpenFolder: false, active: true }).menu)).toEqual(["Remove from the prompt", "Metadata", "Edit metadata", "Set preview", "Copy path"])
  })

  it("gives the focus back to the grid on Escape, wherever it was opened from", async () => {
    // The card's "…" button is a pointer shortcut (out of the Tab order, hidden from screen readers,
    // with no focus ring): a menu it opened sent the focus back there, out of sight.
    function Grid() {
      const [anchor, setAnchor] = useState<MenuAnchor | null>(null)
      return (
        <>
          <div role="listbox" aria-label="LoRA" tabIndex={0} onKeyDown={(event) => event.key === "F10" && setAnchor(document.getElementById("card"))}>
            <div id="card" role="option" aria-selected={false}>
              ink
              <div aria-hidden>
                <button type="button" tabIndex={-1} onClick={(event) => setAnchor(event.currentTarget)}>
                  More actions
                </button>
              </div>
            </div>
          </div>
          <ItemMenu item={anchor ? item() : null} kind={KIND} anchor={anchor} active={false} canOpenFolder onClose={() => setAnchor(null)} onCommand={() => {}} />
        </>
      )
    }
    render(<Grid />)
    const listbox = screen.getByRole("listbox", { name: "LoRA" })
    const close = async () => {
      const menu = await screen.findByRole("menu")
      await waitFor(() => expect(menu).toContainElement(document.activeElement as HTMLElement))
      fireEvent.keyDown(document.activeElement!, { key: "Escape" })
      await waitFor(() => expect(screen.queryByRole("menu")).toBeNull())
      await waitFor(() => expect(document.activeElement).toBe(listbox))
    }
    const button = screen.getByText("More actions")
    act(() => button.focus())
    fireEvent.click(button)
    await close()
    fireEvent.keyDown(listbox, { key: "F10", shiftKey: true })
    await close()
  })

  it("opens at the pointer for a right click", () => {
    const element = document.createElement("div")
    document.body.append(element)
    const { menu } = open({ anchor: { x: 120, y: 80, element } })
    expect(menu).toBeInTheDocument()
  })
})
