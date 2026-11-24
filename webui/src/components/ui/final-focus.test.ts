import { afterEach, describe, expect, it } from "vitest"
import { finalFocus } from "./final-focus"

function layout() {
  document.body.innerHTML = `
    <button id="trigger">Board actions</button>
    <div id="popup" role="menu"><div role="menuitem" tabindex="-1" id="item">Rename…</div></div>
    <div role="menu" id="submenu"><div role="menuitem" tabindex="-1" id="subitem">Copy</div></div>
    <div role="alertdialog"><input id="field" aria-label="Board name" /></div>`
  const get = (id: string) => document.getElementById(id) as HTMLElement
  return { trigger: get("trigger"), item: get("item"), subitem: get("subitem"), field: get("field") }
}

describe("finalFocus", () => {
  afterEach(() => {
    document.body.innerHTML = ""
  })

  it("returns the focus to the trigger while it is still the popup's", () => {
    const { trigger, item, subitem } = layout()
    const target = finalFocus(trigger, "popup")
    for (const element of [item, subitem, trigger]) {
      element.focus()
      expect(target()).toBe(trigger)
    }
    ;(document.activeElement as HTMLElement).blur()
    expect(target()).toBe(trigger)
    expect(finalFocus(null, "popup")()).toBe(true)
  })

  it("leaves it where a dialog an item opened has put it", () => {
    const { trigger, field } = layout()
    field.focus()
    expect(finalFocus(trigger, "popup")()).toBe(false)
  })
})
