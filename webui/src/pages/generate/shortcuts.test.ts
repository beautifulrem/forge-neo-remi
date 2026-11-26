import { afterEach, describe, expect, it } from "vitest"
import { escapeIsFree, POPUP_SLOTS } from "./shortcuts"

const sources = import.meta.glob<string>(["../../**/*.{ts,tsx}", "!../../**/*.test.*"], { query: "?raw", import: "default", eager: true })

/** Dispatches Escape on `target` and reports what `escapeIsFree` said at the window. */
function escapeFrom(target: Element, { prevent = false } = {}): boolean {
  let free: boolean | null = null
  const listen = (event: KeyboardEvent) => {
    free = escapeIsFree(event)
  }
  window.addEventListener("keydown", listen)
  if (prevent) target.addEventListener("keydown", (event) => event.preventDefault(), { once: true })
  target.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }))
  window.removeEventListener("keydown", listen)
  return free!
}

/** The DOM of a prompt editor: wrapper, CodeMirror root, editable content. */
function promptEditor(parent: Element = document.body) {
  parent.insertAdjacentHTML(
    "beforeend",
    '<div data-slot="prompt-editor"><input data-testid="decorator-field"><div class="cm-editor"><div class="cm-scroller"><div class="cm-content" contenteditable="true"></div></div></div></div>',
  )
  const root = parent.lastElementChild!
  return { content: root.querySelector(".cm-content")!, decoratorField: root.querySelector("input")! }
}

describe("escapeIsFree", () => {
  afterEach(() => {
    document.body.innerHTML = ""
  })

  it("interrupts from the page and from a prompt editor (the placeholder's 'Esc to Interrupt')", () => {
    expect(escapeFrom(document.body)).toBe(true)
    expect(escapeFrom(promptEditor().content)).toBe(true)
  })

  it("leaves Escape to the prompt editor when it used it (completion list, selection)", () => {
    expect(escapeFrom(promptEditor().content, { prevent: true })).toBe(false)
  })

  it("leaves Escape to ordinary fields, a decorator's fields, dialogs and open popups", () => {
    const input = document.createElement("input")
    document.body.append(input)
    expect(escapeFrom(input)).toBe(false)
    expect(escapeFrom(promptEditor().decoratorField)).toBe(false)
    const dialog = document.createElement("div")
    dialog.setAttribute("role", "dialog")
    document.body.append(dialog)
    expect(escapeFrom(promptEditor(dialog).content)).toBe(false)
    dialog.remove()
    const popup = document.createElement("div")
    popup.setAttribute("role", "listbox")
    popup.setAttribute("data-open", "")
    document.body.append(popup)
    expect(escapeFrom(promptEditor().content)).toBe(false)
  })

  // The form's schema tabs are a `tabs-content` slot: a switch or a button in a tab kept Escape
  // from interrupting, as if it sat in a popup.
  it("interrupts from a button in a form tab, an accordion, a card or a field", () => {
    for (const slot of ["tabs-content", "accordion-content", "collapsible-content", "card-content", "field-content"]) {
      document.body.innerHTML = `<div data-slot="${slot}"><button>Hires. fix</button></div>`
      expect(escapeFrom(document.querySelector("button")!), slot).toBe(true)
    }
  })

  // No role and no data-open here: only the slot says the focus is in a popup.
  it("leaves Escape to a popover's or a menu's content", () => {
    for (const slot of ["popover-content", "hover-card-content", "dropdown-menu-content", "context-menu-sub-content"]) {
      document.body.innerHTML = `<div data-slot="${slot}"><button>Item</button></div>`
      expect(escapeFrom(document.querySelector("button")!), slot).toBe(false)
    }
  })
})

/**
 * Every `*-content` slot in the sources is sorted here, so a new one gets a decision. A popup
 * (`POPUP_SLOTS`) floats over the page in a portal and closes on Escape itself: dialog, alert
 * dialog, sheet, popover, hover card, tooltip (no focus ever lands in one, but it is dismissed
 * the same way), menus and their sub-menus, select, combobox. The rest are in the page, where
 * Escape has no use and interrupts.
 */
const IN_PAGE_SLOTS = [
  "tabs-content", // a tab's panel: the generate form's schema tabs
  "accordion-content",
  "collapsible-content",
  "card-content",
  "empty-content", // an empty state's actions
  "field-content", // a field's label and description column
  "field-separator-content",
]

describe("the slots that keep Escape", () => {
  const defined = new Set(Object.values(sources).flatMap((source) => [...source.matchAll(/data-slot"?\s*[=:]\s*"([a-z][a-z-]*-content)"/g)].map((match) => match[1]!)))

  it("sort every content slot into a popup or a part of the page", () => {
    expect(Object.keys(sources).length).toBeGreaterThan(100) // the scan works
    expect(defined).toContain("tabs-content")
    const unsorted = [...defined].filter((slot) => !(POPUP_SLOTS as readonly string[]).includes(slot) && !IN_PAGE_SLOTS.includes(slot))
    expect(unsorted).toEqual([])
  })

  it("name only slots that exist", () => {
    expect([...POPUP_SLOTS, ...IN_PAGE_SLOTS].filter((slot) => !defined.has(slot))).toEqual([])
  })
})
