/**
 * Whether a modal dialog is open over the page: the check the page-wide keys make before they act
 * (the generate keys, Ctrl/Cmd+S on the settings page, "?" for the shortcuts list).
 *
 * Every `role="dialog"` or `role="alertdialog"` counts (the kit's dialogs, alert dialogs and sheets,
 * the sign-in screen) except:
 * - a popover or a search select's popup: Base UI renders them as `role="dialog"` too, but anchored,
 *   inside a positioner (`role="presentation"` carrying `data-side`), and they leave the page usable;
 * - a panel that says `aria-modal="false"` (the Networks drawer);
 * - a dialog closing (Base UI's `data-closed` while it fades out) or hidden.
 *
 * Menus and listboxes are not dialogs and never count.
 */
const DIALOGS = '[role="dialog"], [role="alertdialog"]'
/** Base UI's positioner (`usePositioner`): what anchors a popover, a menu or a select's popup. */
const POSITIONER = '[role="presentation"][data-side]'

export function modalOpen(root: ParentNode = document): boolean {
  for (const element of root.querySelectorAll<HTMLElement>(DIALOGS)) {
    if (element.getAttribute("aria-modal") === "false" || element.hasAttribute("data-closed")) continue
    if (element.closest(POSITIONER) || element.closest("[hidden]")) continue
    return true
  }
  return false
}
