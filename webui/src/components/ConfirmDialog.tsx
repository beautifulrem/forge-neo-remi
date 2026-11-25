/**
 * A confirmation dialog ("Clear the history?") whose code (the alert dialog,
 * focus trap and scroll lock) loads the first time it opens, so the views
 * that offer one do not carry it in their first load
 * (docs/research/ui-performance.md, "Bundle"). It renders nothing until then
 * and stays mounted afterwards (close animation). `preloadConfirmDialog()`
 * fetches it ahead, e.g. when the pointer enters the button that opens it.
 */
// oxlint-disable react/only-export-components -- the component and its preload
import { Suspense, useState } from "react"
import { toast } from "sonner"
import type { ConfirmDialogProps } from "./ConfirmDialogView"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { renderFailureText } from "@/features/errors/error-text"
import { t } from "@/i18n"
import { retryableLazy, retryFailedLoads } from "@/lib/retryable-lazy"

const load = () => import("./ConfirmDialogView")
const View = retryableLazy(() => load().then((module) => ({ default: module.ConfirmDialogView })))

/** Fetches the dialog's code (button intent); module loads are cached. */
export function preloadConfirmDialog() {
  void load().catch(() => {})
}

export function ConfirmDialog(props: ConfirmDialogProps) {
  const [used, setUsed] = useState(props.open)
  if (props.open && !used) setUsed(true)
  if (!used) return null
  // A dialog that cannot be shown (its chunk not fetched, a render bug) closes and says so rather
  // than taking the page that offered it down; the next open starts afresh and loads it again.
  return (
    <ErrorBoundary
      fallback={null}
      onError={(error) => {
        setUsed(false)
        props.onOpenChange(false)
        toast.error(t("The dialog could not be shown"), { description: renderFailureText(error) })
        retryFailedLoads()
      }}
    >
      <Suspense fallback={null}>
        <View {...props} />
      </Suspense>
    </ErrorBoundary>
  )
}

export type { ConfirmDialogProps }
