/**
 * Blocks of a form column (the generate parameters, an extension page) are an
 * inspector's panels, not cards: no box, no fill, no radius; a hairline above
 * each one and the space around it separate them. A panel inside a panel (an
 * extension's section in the Extensions list, a section in a section) is a
 * level down: a fainter hairline and a smaller title.
 */
import { createContext, use } from "react"

/** 0: a panel of the column; 1 and deeper: nested in one. */
export const PanelDepthContext = createContext(0)

export const usePanelDepth = () => use(PanelDepthContext)

export const panelClass = (depth: number) => (depth === 0 ? "border-t border-border pt-3" : "border-t border-border/60 pt-2.5")

export const panelTitleClass = (depth: number) => (depth === 0 ? "text-[0.9375rem] font-semibold tracking-[-0.01em]" : "text-sm font-semibold")
