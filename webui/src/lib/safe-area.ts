/**
 * The screen's safe area (a notch, rounded corners, the home indicator: `env(safe-area-inset-*)`)
 * as numbers, for what positions itself in script: Base UI's popups keep their `collisionPadding`
 * inside it, the toaster keeps its offset from the edges outside it. CSS reads `env()` directly;
 * this reads it through a hidden probe element whose padding is the insets.
 */
import { useMemo, useSyncExternalStore } from "react"

export type Insets = { top: number; right: number; bottom: number; left: number }

const NONE: Insets = { top: 0, right: 0, bottom: 0, left: 0 }

/** Base UI's own `collisionPadding` (px) on each side. */
export const COLLISION_PADDING = 5

let probe: HTMLElement | null = null
let current: Insets = NONE
const listeners = new Set<() => void>()

function read(): Insets {
  if (typeof document === "undefined" || !document.body) return NONE
  if (!probe?.isConnected) {
    probe = document.createElement("div")
    probe.setAttribute("aria-hidden", "true")
    probe.style.cssText =
      "position:fixed;top:0;left:0;visibility:hidden;pointer-events:none;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)"
    document.body.append(probe)
  }
  const style = getComputedStyle(probe)
  const px = (value: string) => Math.max(0, Math.round(parseFloat(value) || 0))
  return { top: px(style.paddingTop), right: px(style.paddingRight), bottom: px(style.paddingBottom), left: px(style.paddingLeft) }
}

function update() {
  const next = read()
  if (next.top === current.top && next.right === current.right && next.bottom === current.bottom && next.left === current.left) return
  current = next
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void): () => void {
  if (listeners.size === 0) window.addEventListener("resize", update)
  listeners.add(listener)
  update() // a popup that opens reads the insets of now (a phone turned while it was closed)
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) window.removeEventListener("resize", update)
  }
}

/** The safe-area insets in px (0 where the screen has none); follows a turned phone. */
export function useSafeAreaInsets(): Insets {
  return useSyncExternalStore(subscribe, () => current, () => NONE)
}

/** A popup's `collisionPadding`: Base UI's own 5 px from each edge of the safe area. */
export function useCollisionPadding(): Insets {
  const insets = useSafeAreaInsets()
  return useMemo(
    () => ({
      top: insets.top + COLLISION_PADDING,
      right: insets.right + COLLISION_PADDING,
      bottom: insets.bottom + COLLISION_PADDING,
      left: insets.left + COLLISION_PADDING,
    }),
    [insets],
  )
}
