/**
 * Whether a control's label fits beside the rest of its content (an icon) in
 * the width the control has. A label that does not fit is shown as the icon
 * alone, the label kept as the accessible name, rather than cut to "Qu…". The
 * label is measured in its own font, so the answer holds for every language: a
 * fixed breakpoint is too wide for one translation and too narrow for another.
 *
 * The control's width must not depend on the label (a button that fills its
 * row): hiding the label then cannot change the answer.
 */
import { type RefObject, useLayoutEffect, useState } from "react"

export function useLabelFits(control: RefObject<HTMLElement | null>, label: RefObject<HTMLElement | null>, text: string, enabled: boolean): boolean {
  const [fits, setFits] = useState(true)
  useLayoutEffect(() => {
    const box = control.current
    const span = label.current
    if (!enabled || !box || !span) return
    let active = true
    const check = () => {
      if (!active || box.clientWidth === 0) return // not laid out (hidden): keep the last answer
      const style = getComputedStyle(box)
      const gap = parseFloat(style.columnGap) || 0
      let room = box.clientWidth - (parseFloat(style.paddingLeft) || 0) - (parseFloat(style.paddingRight) || 0)
      for (const child of box.children) {
        if (child === span) continue
        const width = child.getBoundingClientRect().width
        if (width > 0) room -= width + gap
      }
      // scrollWidth is the label's whole width, shown, cut or visually hidden
      setFits(span.scrollWidth <= room + 0.5)
    }
    check()
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(check)
    observer?.observe(box)
    void document.fonts?.ready.then(check)
    return () => {
      active = false
      observer?.disconnect()
    }
  }, [control, label, text, enabled])
  return !enabled || fits
}
