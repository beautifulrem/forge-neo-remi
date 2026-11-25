import { expect } from "vitest"

/**
 * The text an invalid control names as the reason (its aria-errormessage and aria-describedby),
 * read from the elements shown on screen: a text only a screen reader gets (`sr-only`, hidden)
 * does not count. An invalid field says why in text (docs/remi-webui/frontend.md).
 */
export function shownReason(control: Element): string {
  expect(control).toHaveAttribute("aria-invalid", "true")
  const ids = `${control.getAttribute("aria-errormessage") ?? ""} ${control.getAttribute("aria-describedby") ?? ""}`.split(/\s+/).filter(Boolean)
  return ids
    .map((id) => document.getElementById(id))
    .filter((element): element is HTMLElement => element !== null && element.closest(".sr-only, [hidden], [aria-hidden='true']") === null)
    .map((element) => element.textContent?.trim() ?? "")
    .filter(Boolean)
    .join(" ")
}
