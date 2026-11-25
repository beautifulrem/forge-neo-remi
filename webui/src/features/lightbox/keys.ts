/**
 * The image viewer's keys (no modifier) and what each does; Lightbox handles them, the shortcuts
 * dialog lists every one (its test holds the two together). Letters in both cases: Shift or Caps
 * Lock must not turn a key off.
 */
export const VIEWER_KEYS = {
  ArrowLeft: "previous",
  ArrowRight: "next",
  Home: "first",
  End: "last",
  s: "download",
  S: "download",
  "+": "zoom-in",
  "=": "zoom-in",
  "-": "zoom-out",
  "0": "fit",
  "1": "actual-size",
  t: "tile",
  T: "tile",
  b: "backdrop",
  B: "backdrop",
} as const

export type ViewerAction = (typeof VIEWER_KEYS)[keyof typeof VIEWER_KEYS]

/** The action of a key, or null. */
export function viewerAction(key: string): ViewerAction | null {
  return Object.hasOwn(VIEWER_KEYS, key) ? VIEWER_KEYS[key as keyof typeof VIEWER_KEYS] : null
}
