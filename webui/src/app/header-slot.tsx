/**
 * Pages put controls into the header with <HeaderSlot> (rendered through a
 * portal into the header's slot element, so pages keep ownership and state).
 */
import type { ReactNode } from "react"
import { createPortal } from "react-dom"
import { useHeaderSlotTarget } from "./header-slot-store"

export function HeaderSlot({ children }: { children: ReactNode }) {
  const element = useHeaderSlotTarget((s) => s.element)
  return element ? createPortal(children, element) : null
}

/**
 * A header button whose label hides below `sm`: there a 28 px square whose icon is the 16 px
 * of the header's icon buttons (search, appearance), not the 14 px of a small labelled button.
 */
export const ICON_SQUARE_BELOW_SM = "max-sm:size-7 max-sm:px-0 max-sm:[&_svg:not([class*='size-'])]:size-4"

/** The same below `lg`, for the networks button, whose label needs room the model bar takes there. */
export const ICON_SQUARE_BELOW_LG = "max-lg:size-7 max-lg:px-0 max-lg:[&_svg:not([class*='size-'])]:size-4"
