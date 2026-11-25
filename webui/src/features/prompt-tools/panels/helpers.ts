/** Helpers and props of the prompt tools panels. */
import type { PromptKind } from "@/features/prompt-editor/facets"
import type { PromptToolsConfig, TagSettings } from "../api"
import { tagText } from "../completion/insert"

export type PanelProps = {
  value: string
  setValue: (value: string) => void
  kind: PromptKind
  label: string
  disabled: boolean
  config: PromptToolsConfig
  close: () => void
}

/**
 * "3 minutes ago" in the UI language, counted from `now` (ms): a render passes the time it read
 * (`useNow`), so the text moves on while the page stays open.
 */
export function timeAgo(seconds: number, locale: string, now = Date.now()): string {
  const diff = seconds - now / 1000
  const format = new Intl.RelativeTimeFormat(locale, { numeric: "auto" })
  // half away from zero (Math.round(-59.5) is -59: "59 minutes ago" for 59.5 minutes)
  const round = (value: number) => Math.sign(value) * Math.round(Math.abs(value))
  // under a day: the rounded count, moving up a unit when it reaches the next one (59.6 s is a minute)
  const units: [number, number, Intl.RelativeTimeFormatUnit][] = [
    [1, 60, "second"],
    [60, 60, "minute"],
    [3600, 24, "hour"],
  ]
  for (const [size, limit, unit] of units) {
    const value = round(diff / size)
    if (Math.abs(value) < limit) return format.format(value, unit)
  }
  // a day or more: calendar days, so "yesterday" is the day before today whatever the hour
  const midnight = (ms: number) => new Date(ms).setHours(0, 0, 0, 0)
  const days = round((midnight(seconds * 1000) - midnight(now)) / 86_400_000)
  if (Math.abs(days) < 7) return format.format(days, "day")
  if (Math.abs(days) < 30) return format.format(round(days / 7), "week")
  const months = round(days / 30.44)
  if (Math.abs(months) < 12) return format.format(months, "month")
  return format.format(round(days / 365.25), "year")
}

/** Appends text to a prompt with a comma between (no doubled separators). */
export function appendText(current: string, text: string): string {
  const head = current.replace(/[\s,]+$/, "")
  const added = text.trim()
  if (!head) return added
  if (!added) return current
  return `${head}, ${added}`
}

/** A quick-group tag as it is written into the prompt, like an accepted completion (`hanten_(clothes)` → `hanten \(clothes\)`). */
export function groupTagText(tag: string, settings: TagSettings): string {
  return tagText({ text: tag, type: "tag", category: null }, settings)
}
