/**
 * Whether the keyboard is a Mac's (macOS, iPadOS, iOS): ⌘ is the shortcut modifier and key labels
 * use the Mac's symbols. Read at each call, so labels follow the platform they are shown on.
 */
export function isMac(): boolean {
  return typeof navigator !== "undefined" && /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent)
}
