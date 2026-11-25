import { afterEach, describe, expect, it } from "vitest"
import { inertOutside } from "./modal-inert"

/** body > #root (the app, a toaster in it) and a modal's portal, as Base UI lays them out. */
function page() {
  document.body.innerHTML = `
    <div id="root"><div id="shell"><a id="skip" href="#main">Skip</a><div id="app"><main><button>Generate</button></main></div>
      <section id="toasts" aria-live="polite" data-react-aria-top-layer="true"></section></div></div>
    <script></script>`
  return (id: string) => document.getElementById(id)!
}

/** A modal's portal (its backdrop, focus guards and popup) added to the page as it opens. */
function open(index: number) {
  document.body.insertAdjacentHTML(
    "beforeend",
    `<div id="portal${index}" data-base-ui-portal><div id="backdrop${index}"></div><span data-base-ui-focus-guard tabindex="0"></span><div id="popup${index}" role="dialog"></div><span data-base-ui-focus-guard tabindex="0"></span></div>`,
  )
  return document.getElementById(`popup${index}`)!
}

const inert = () => [...document.querySelectorAll("[inert]")].map((element) => element.id || element.tagName.toLowerCase())

afterEach(() => {
  document.body.innerHTML = ""
})

describe("inertOutside", () => {
  it("makes the page inert and leaves the popup's portal and the toasts live", () => {
    page()
    const release = inertOutside(open(0))
    // the boxes around the toasts are walked into, their other children marked; the script is left alone
    expect(inert().sort()).toEqual(["app", "skip"])
    release()
    expect(inert()).toEqual([])
  })

  it("holds a lower modal under a modal over it, and lets each go in turn", () => {
    page()
    const lower = inertOutside(open(0))
    const upper = inertOutside(open(1))
    expect(inert().sort()).toEqual(["app", "portal0", "skip"])
    upper()
    expect(inert().sort()).toEqual(["app", "skip"])
    lower()
    expect(inert()).toEqual([])
  })

  it("leaves an inert it did not set", () => {
    const get = page()
    get("skip").setAttribute("inert", "")
    const release = inertOutside(open(0))
    release()
    expect(inert()).toEqual(["skip"])
  })
})
