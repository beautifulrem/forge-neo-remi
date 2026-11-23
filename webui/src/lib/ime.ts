import type { KeyboardEvent } from "react"

/**
 * Whether a key belongs to an IME composition (pinyin, kana): its Enter picks
 * a candidate and must not also commit the field. Safari sends the confirming
 * Enter after `compositionend`, flagged only by the legacy key code 229.
 */
export function isComposing(event: Pick<KeyboardEvent, "nativeEvent" | "keyCode">): boolean {
  return event.nativeEvent.isComposing || event.keyCode === 229
}
