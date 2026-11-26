/**
 * A strip that scrolls when it does not fit its pane (tab lists, the
 * navigation rail held sideways): its edges fade where there is more to
 * scroll to, and the selected item (`[data-active]`) is kept in view when
 * `active` changes. `axis` is the way it scrolls: "x" sideways, "y" up and down.
 * An item scrolled into view (focus, `active`) stops clear of the fade.
 */
import { type CSSProperties, type RefObject, useEffect, useLayoutEffect, useState } from "react"

/** The fade's depth; the room an item scrolled into view keeps from the edge is that plus its focus ring. */
const FADE = "1.5rem"
const CLEAR = 28

export function useScrollFade(ref: RefObject<HTMLElement | null>, active: unknown, axis: "x" | "y" = "x"): CSSProperties | undefined {
  const [edges, setEdges] = useState({ start: false, end: false })
  // The element the ref holds after each render: a strip can mount after the component holding the hook (the
  // editor's image toolbar, shown once there is an image), and one read at mount only would be none.
  const [strip, setStrip] = useState<HTMLElement | null>(null)
  // oxlint-disable-next-line react-hooks/exhaustive-deps -- after every render, to see a new element; it sets the state only when the element changed
  useLayoutEffect(() => {
    if (ref.current !== strip) setStrip(ref.current)
  })
  useEffect(() => {
    if (!strip) return
    const update = () => {
      const [offset, room, length] = axis === "x" ? [strip.scrollLeft, strip.clientWidth, strip.scrollWidth] : [strip.scrollTop, strip.clientHeight, strip.scrollHeight]
      const start = offset > 1
      const end = offset + room < length - 1
      setEdges((old) => (old.start === start && old.end === end ? old : { start, end }))
    }
    // the strip's own size, and its items': items that arrive or grow after it mounts (boards,
    // extension pages) change how far it scrolls without changing its size
    const sizes = new ResizeObserver(update)
    const watch = () => {
      sizes.disconnect()
      sizes.observe(strip)
      for (const item of strip.children) sizes.observe(item)
    }
    const items = new MutationObserver(() => {
      watch()
      update()
    })
    update()
    watch()
    items.observe(strip, { childList: true, subtree: true, characterData: true })
    strip.addEventListener("scroll", update, { passive: true })
    return () => {
      sizes.disconnect()
      items.disconnect()
      strip.removeEventListener("scroll", update)
    }
  }, [strip, axis])
  useEffect(() => {
    const strip = ref.current
    const item = strip?.querySelector<HTMLElement>("[data-active]")
    if (!strip || !item) return
    if (axis === "x") {
      const left = item.offsetLeft - strip.offsetLeft
      if (left < strip.scrollLeft + CLEAR) strip.scrollLeft = left - CLEAR
      else if (left + item.offsetWidth > strip.scrollLeft + strip.clientWidth - CLEAR) strip.scrollLeft = left + item.offsetWidth - strip.clientWidth + CLEAR
    } else {
      const top = item.offsetTop - strip.offsetTop
      if (top < strip.scrollTop + CLEAR) strip.scrollTop = top - CLEAR
      else if (top + item.offsetHeight > strip.scrollTop + strip.clientHeight - CLEAR) strip.scrollTop = top + item.offsetHeight - strip.clientHeight + CLEAR
    }
  }, [ref, strip, active, axis])
  if (!edges.start && !edges.end) return undefined
  const mask = `linear-gradient(to ${axis === "x" ? "right" : "bottom"}, transparent, #000 ${edges.start ? FADE : "0rem"}, #000 calc(100% - ${edges.end ? FADE : "0rem"}), transparent)`
  // the browser scrolls a focused item into view this far from the edge: past the fade, its ring whole
  const clear = `${CLEAR}px`
  return { maskImage: mask, WebkitMaskImage: mask, ...(axis === "x" ? { scrollPaddingInline: clear } : { scrollPaddingBlock: clear }) }
}
