/**
 * Where the focus goes when a retry succeeds. The Retry of a failed load keeps the focus while its
 * retry runs (`LoadFailed`, `RetryButton`); when the data arrives the failure view is replaced by
 * what it loaded, and the focus it held fell to `<body>` (or to the dialog around it). It now goes
 * to that content: its first tabbable element, or, while it has none, the nearest focusable
 * element around it (a dialog, `#main`).
 *
 * Only a failure view that held the focus as it went, and only once a retry ran from it (pressed,
 * or a refetch on reconnect shown as retrying): one that leaves for another reason (a filter
 * changed, another page) leaves the focus where it is.
 */
import { type RefCallback, useCallback, useLayoutEffect, useRef } from "react"

const TABBABLE = [
  "a[href]",
  "button:not([disabled])",
  'input:not([disabled]):not([type="hidden"])',
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[contenteditable=""]',
  '[contenteditable="true"]',
  '[tabindex]:not([tabindex="-1"])',
].join(", ")
const OUT_OF_REACH = '[hidden], [inert], [aria-hidden="true"]'
const DIALOGS = '[role="dialog"], [role="alertdialog"]'
/** Frames to wait for the content (a panel may draw a spinner for one more render first). */
const FRAMES = 10

type Place = { node: Element; previous: Node | null; next: Node | null }

/**
 * A ref for the failure view (or its Retry): when it leaves the page holding the focus after a
 * retry ran, the focus goes to the content that took its place. `retrying` is the view's
 * `LoadFailure.retrying`.
 */
export function useFocusAfterRetry<T extends HTMLElement = HTMLElement>(retrying: boolean): RefCallback<T> {
  const retried = useRef(false)
  useLayoutEffect(() => {
    if (retrying) retried.current = true
  })
  return useCallback((view: T | null) => {
    if (!view) return
    const press = () => {
      retried.current = true
    }
    view.addEventListener("click", press)
    return () => {
      view.removeEventListener("click", press)
      // still in the page as this runs: React detaches refs before it removes the nodes
      if (retried.current && view.contains(document.activeElement)) focusWhatReplaces(view)
    }
  }, [])
}

/** Moves the focus to what takes `view`'s place once it is gone, unless the focus has gone elsewhere since. */
function focusWhatReplaces(view: Element) {
  const places: Place[] = []
  for (let node: Element | null = view; node; node = node.parentElement) places.push({ node, previous: node.previousSibling, next: node.nextSibling })
  let frame = 0
  const attempt = () => {
    const at = places.findIndex((place) => place.node.isConnected)
    // the view stayed (it was not removed after all), or nothing of the page around it did
    if (at <= 0) return
    const container = places[at]!.node
    if (!focusLost(container)) return
    const { previous, next } = places[at - 1]!
    const children = [...container.childNodes]
    const start = previous?.parentNode === container ? children.indexOf(previous as ChildNode) + 1 : 0
    const end = next?.parentNode === container ? children.indexOf(next as ChildNode) : children.length
    for (const node of children.slice(start, end)) {
      if (node instanceof HTMLElement && focusFirstTabbable(node)) return
    }
    if (++frame < FRAMES) {
      requestAnimationFrame(attempt)
      return
    }
    const around = container.closest<HTMLElement>("[tabindex]")
    around?.focus()
  }
  // Two frames: Base UI's dialogs put a focus lost inside them back on the popup on the next
  // frame (`restoreFocus`); the content takes it after that.
  requestAnimationFrame(() => requestAnimationFrame(attempt))
}

/** The focus fell out with the view: to the page, or to the dialog the view was in. */
function focusLost(container: Element): boolean {
  const active = document.activeElement
  if (!active || active === document.body || !active.isConnected) return true
  return active.matches(DIALOGS) && active.contains(container)
}

function focusFirstTabbable(root: HTMLElement): boolean {
  const candidates = root.matches(TABBABLE) ? [root] : []
  candidates.push(...root.querySelectorAll<HTMLElement>(TABBABLE))
  for (const candidate of candidates) {
    if (candidate.closest(OUT_OF_REACH)) continue
    candidate.focus()
    if (document.activeElement === candidate) return true
  }
  return false
}
