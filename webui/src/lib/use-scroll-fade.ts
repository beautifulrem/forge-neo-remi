/**
 * A strip that scrolls when it does not fit its pane (tab lists, the
 * navigation rail held sideways): its edges fade where there is more to
 * scroll to, and the selected item (`[data-active]`) is kept in view when
 * `active` changes. `axis` is the way it scrolls: "x" sideways, "y" up and down.
 */
import { type CSSProperties, type RefObject, useEffect, useState } from "react"

export function useScrollFade(ref: RefObject<HTMLElement | null>, active: unknown, axis: "x" | "y" = "x"): CSSProperties | undefined {
  const [edges, setEdges] = useState({ start: false, end: false })
  useEffect(() => {
    const strip = ref.current
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
  }, [ref, axis])
  useEffect(() => {
    const strip = ref.current
    const item = strip?.querySelector<HTMLElement>("[data-active]")
    if (!strip || !item) return
    if (axis === "x") {
      const left = item.offsetLeft - strip.offsetLeft
      if (left < strip.scrollLeft) strip.scrollLeft = left - 16
      else if (left + item.offsetWidth > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = left + item.offsetWidth - strip.clientWidth + 16
    } else {
      const top = item.offsetTop - strip.offsetTop
      if (top < strip.scrollTop) strip.scrollTop = top - 16
      else if (top + item.offsetHeight > strip.scrollTop + strip.clientHeight) strip.scrollTop = top + item.offsetHeight - strip.clientHeight + 16
    }
  }, [ref, active, axis])
  if (!edges.start && !edges.end) return undefined
  const mask = `linear-gradient(to ${axis === "x" ? "right" : "bottom"}, transparent, #000 ${edges.start ? "1.5rem" : "0rem"}, #000 calc(100% - ${edges.end ? "1.5rem" : "0rem"}), transparent)`
  return { maskImage: mask, WebkitMaskImage: mask }
}
