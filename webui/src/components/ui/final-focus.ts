/**
 * Where a closing menu or popover sends the focus (Base UI's `finalFocus`):
 * back to its trigger, unless something else has taken it since. Base UI moves
 * it once the popup has faded out, by which time a dialog one of its items
 * opened has already moved the focus inside: sent back to the trigger, typing
 * went nowhere and Ctrl+A selected the page behind the dialog.
 */
export function finalFocus(trigger: HTMLElement | null, popupId: string): () => HTMLElement | boolean {
  return () => {
    const active = document.activeElement
    // still the popup's (or nobody's): on a submenu, an item, the trigger or the page
    const ours = active === null || active === document.body || active === trigger || Boolean(document.getElementById(popupId)?.contains(active) || active.closest("[role=menu]"))
    return ours ? (trigger ?? true) : false
  }
}
