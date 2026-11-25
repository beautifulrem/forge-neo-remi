/**
 * While a modal popup is open (a dialog, an alert dialog, a sheet, the image viewer), the page
 * behind it is `inert`: out of the Tab order, out of the pointer's reach and out of the
 * accessibility tree.
 *
 * Base UI traps the focus of a modal popup with a focus guard on each side of it, which sends the
 * focus back inside a frame after Tab lands on it (`enqueueFocus`). A Tab pressed sooner than that
 * (a held key with no repeat delay) went on from the guard to the page behind, and from there
 * through the whole page. Base UI hides that page from assistive technology (`aria-hidden`) but
 * leaves it focusable, and offers no option to make it inert.
 *
 * Left live: the popup's portal (its backdrop, which closes it on a press, its focus guards, and
 * the menus, selects and popovers opened from inside it, which Base UI portals into it), and what
 * marks itself a top layer for modals (`data-react-aria-top-layer`: sonner's toasts, still read out
 * and dismissable). The page's own live regions go inert with the page, as behind a native modal
 * `<dialog>`: kept live, their boxes stayed live too, and a scroller among them still took the
 * focus. A modal over another (an alert dialog asking about a dialog's action) makes the lower one
 * inert too: each element counts the modals holding it, and only an `inert` set here is taken off.
 */
import * as React from "react"

/** Whether the dialog root around is modal (`modal`, not `false` or `"trap-focus"`). */
export const ModalRootContext = React.createContext(true)

/** The dialog root's `modal` prop as the popup needs it: only a full modal makes the page inert. */
export function isFullModal(modal: boolean | "trap-focus" | undefined): boolean {
  return modal === undefined || modal === true
}

/** How many open modals hold an element inert, and the elements that were inert before any did. */
const holders = new WeakMap<Element, number>()
const inertBefore = new WeakSet<Element>()
/** Elements that draw nothing: no need to mark them. */
const SKIP = new Set(["SCRIPT", "STYLE", "LINK", "META", "TEMPLATE"])

/** Makes everything outside `popup`'s portal (and outside the toasts) inert; returns the undo. */
export function inertOutside(popup: Element): () => void {
  const body = popup.ownerDocument.body
  const keep = [popup.closest("[data-base-ui-portal]") ?? popup, ...body.querySelectorAll("[data-react-aria-top-layer]")]
  const kept = new Set<Element>(keep)
  // the boxes that hold something kept: walked into, not marked
  const around = new Set<Element>()
  for (const element of keep) {
    for (let node = element.parentElement; node && node !== body && !around.has(node); node = node.parentElement) around.add(node)
  }
  const outside: Element[] = []
  const walk = (parent: Element) => {
    for (const child of parent.children) {
      if (kept.has(child) || SKIP.has(child.tagName)) continue
      if (around.has(child)) walk(child)
      else outside.push(child)
    }
  }
  walk(body)
  for (const element of outside) {
    const count = holders.get(element) ?? 0
    if (count === 0) {
      if (element.hasAttribute("inert")) inertBefore.add(element)
      else element.setAttribute("inert", "")
    }
    holders.set(element, count + 1)
  }
  return () => {
    for (const element of outside) {
      const count = (holders.get(element) ?? 1) - 1
      if (count > 0) {
        holders.set(element, count)
        continue
      }
      holders.delete(element)
      if (inertBefore.has(element)) inertBefore.delete(element)
      else element.removeAttribute("inert")
    }
  }
}

type PopupProps = React.HTMLAttributes<HTMLDivElement> & { ref?: React.Ref<HTMLDivElement> }

/**
 * Base UI's `render` for a modal popup (`Dialog.Popup`, `AlertDialog.Popup`): the popup's `div`,
 * holding the page behind it inert while it is open.
 */
export function renderModalPopup(props: PopupProps, state: { open: boolean }) {
  return <ModalPopup {...props} open={state.open} />
}

function ModalPopup({ open, ref, ...props }: PopupProps & { open: boolean }) {
  const modal = React.useContext(ModalRootContext)
  const [element, setElement] = React.useState<HTMLDivElement | null>(null)
  const release = React.useRef<(() => void) | null>(null)
  const holding = modal && open && element !== null
  // After the commit that opens it: Base UI has noted where the focus was (it gives it back there),
  // and moves it inside a frame later.
  React.useEffect(() => {
    if (!holding || !element) return
    release.current = inertOutside(element)
    return () => {
      release.current?.()
      release.current = null
    }
  }, [holding, element])
  // Off in the commit that closes it (it fades out with `data-closed`), before anything gives the
  // focus back to the page: `focus()` on an inert element does nothing.
  React.useLayoutEffect(
    () => () => {
      release.current?.()
      release.current = null
    },
    [holding, element],
  )
  const setRef = React.useCallback(
    (node: HTMLDivElement | null) => {
      setElement(node)
      if (typeof ref !== "function") {
        if (ref) ref.current = node
        return () => {
          if (ref) ref.current = null
        }
      }
      const cleanup = ref(node)
      return () => {
        if (typeof cleanup === "function") cleanup()
        else ref(null)
      }
    },
    [ref],
  )
  return <div {...props} ref={setRef} />
}
