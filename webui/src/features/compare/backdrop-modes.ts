/**
 * Pure RGBA 4-Backdrop definitions and style helpers shared by Lightbox,
 * CompareSlider and CompareDialog (Checkerboard, Pure Black, Pure White, Green Screen).
 */
import type { CSSProperties } from "react"

export type BackdropMode = "checker" | "black" | "white" | "green"

export const BACKDROP_MODES: readonly BackdropMode[] = ["checker", "black", "white", "green"]

export const BACKDROP_META: Record<BackdropMode, { label: string; color?: string }> = {
  checker: { label: "Checkerboard" },
  black: { label: "Pure black", color: "#000000" },
  white: { label: "Pure white", color: "#ffffff" },
  green: { label: "Green screen", color: "#00b140" },
}

export function nextBackdrop(current: BackdropMode): BackdropMode {
  const index = BACKDROP_MODES.indexOf(current)
  return BACKDROP_MODES[(index + 1) % BACKDROP_MODES.length]!
}

/** Class name and inline style to paint a surface with the chosen RGBA backdrop. */
export function backdropSurface(mode: BackdropMode): { className?: string; style?: CSSProperties } {
  if (mode === "checker") return { className: "bg-checker" }
  return { style: { backgroundColor: BACKDROP_META[mode].color } }
}
