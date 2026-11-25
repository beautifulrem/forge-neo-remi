/**
 * A bar pinned over an edge of a scroller (a sticky run bar, an apply bar, a
 * send bar) must not hide the control the keyboard focus moves to (WCAG
 * 2.4.11): Tab scrolled the next control just into the scroller's box, under
 * the bar. The scroller gets scroll padding on that edge as deep as the bar
 * reaches into it, so a control scrolled into view stops clear of the bar.
 */
import { useCallback } from "react"

export type PinnedEdge = "top" | "bottom"

/** Room between the bar and a control scrolled up to it: the control's focus ring and some air. */
const CLEARANCE = 8

/** The bars pinned in each scroller, and how deep each reaches in. */
const pinned = new WeakMap<HTMLElement, Map<HTMLElement, { edge: PinnedEdge; depth: number }>>()

/** The nearest ancestor that scrolls vertically. */
export function scrollerOf(element: HTMLElement): HTMLElement | null {
  for (let node = element.parentElement; node; node = node.parentElement) {
    const { overflowY } = getComputedStyle(node)
    if (overflowY === "auto" || overflowY === "scroll") return node
  }
  return null
}

function pad(scroller: HTMLElement) {
  let top = 0
  let bottom = 0
  for (const { edge, depth } of pinned.get(scroller)?.values() ?? []) {
    if (edge === "top") top = Math.max(top, depth)
    else bottom = Math.max(bottom, depth)
  }
  scroller.style.scrollPaddingTop = top > 0 ? `${top}px` : ""
  scroller.style.scrollPaddingBottom = bottom > 0 ? `${bottom}px` : ""
}

function release(scroller: HTMLElement, bar: HTMLElement) {
  pinned.get(scroller)?.delete(bar)
  pad(scroller)
}

/**
 * A callback ref for a bar pinned to `edge` of its scroller: the nearest
 * scrolling ancestor, or `findScroller(bar)` for a bar laid over a scroller
 * it is not inside (keep that function stable: a new one remounts the ref).
 * The depth is the bar's height plus its inset (`bottom-3`, a safe area),
 * re-measured when the bar or the window changes size.
 */
export function usePinnedBar(edge: PinnedEdge, findScroller?: (bar: HTMLElement) => HTMLElement | null) {
  return useCallback(
    (bar: HTMLElement | null) => {
      if (!bar) return
      let scroller: HTMLElement | null = null
      const measure = () => {
        const next = findScroller ? findScroller(bar) : scrollerOf(bar)
        if (scroller && scroller !== next) release(scroller, bar)
        scroller = next
        if (!scroller) return
        const style = getComputedStyle(bar)
        const inset = parseFloat(edge === "top" ? style.top : style.bottom) || 0
        let bars = pinned.get(scroller)
        if (!bars) pinned.set(scroller, (bars = new Map()))
        bars.set(bar, { edge, depth: Math.ceil(bar.offsetHeight + Math.max(0, inset) + CLEARANCE) })
        pad(scroller)
      }
      const observer = new ResizeObserver(measure)
      observer.observe(bar)
      window.addEventListener("resize", measure)
      measure()
      return () => {
        observer.disconnect()
        window.removeEventListener("resize", measure)
        if (scroller) release(scroller, bar)
      }
    },
    [edge, findScroller],
  )
}
