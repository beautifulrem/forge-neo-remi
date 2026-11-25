import type { ReactNode, Ref } from "react"
import { useTranslation } from "react-i18next"
import { Loader2 } from "@/components/icons"
import { Button } from "@/components/ui/button"
import { useFocusAfterRetry } from "@/components/focus-after-retry"
import { errorText } from "@/features/errors/error-text"
import { cn } from "@/lib/utils"

/**
 * A list that could not load at all: what failed, and a retry (never an "empty" list). `message`
 * replaces the error's own text (e.g. an explanation of admin access); `children` follow the Retry
 * (another way out: the text on the web). While `retrying` (see `api/load-failure.ts`) the failure
 * stays and its Retry is busy; when the retry brings the data, the focus the failure held goes to
 * what replaces it (`focus-after-retry.ts`).
 */
export function LoadFailed({
  error,
  message,
  onRetry,
  retrying = false,
  className,
  children,
}: {
  error: unknown
  message?: string
  onRetry: () => void
  retrying?: boolean
  className?: string
  children?: ReactNode
}) {
  const { i18n } = useTranslation()
  const focusAfterRetry = useFocusAfterRetry<HTMLDivElement>(retrying)
  return (
    <div ref={focusAfterRetry} role="alert" aria-busy={retrying || undefined} className={cn("flex flex-wrap items-center gap-x-3 gap-y-1 py-3 text-sm", className)}>
      <span className="text-destructive">{message ?? errorText(error, {}, i18n.language)}</span>
      <RetryControl retrying={retrying} onRetry={onRetry} />
      {children}
    </div>
  )
}

/**
 * Retry for a load that failed (a page's, in its `PagePlaceholder`). While the retry runs it is
 * disabled but keeps the focus (`aria-disabled`), and a spinner takes its label's place at the
 * label's width; when the retry brings the data, the focus goes to what replaces it.
 */
export function RetryButton({ retrying = false, onRetry }: { retrying?: boolean; onRetry: () => void }) {
  const focusAfterRetry = useFocusAfterRetry<HTMLButtonElement>(retrying)
  return <RetryControl ref={focusAfterRetry} retrying={retrying} onRetry={onRetry} />
}

function RetryControl({ retrying, onRetry, ref }: { retrying: boolean; onRetry: () => void; ref?: Ref<HTMLButtonElement> }) {
  const { t } = useTranslation()
  return (
    <Button ref={ref} variant="outline" size="sm" className="relative" disabled={retrying} focusableWhenDisabled onClick={onRetry}>
      {/* transparent, not hidden: the label stays the button's name */}
      <span className={cn(retrying && "opacity-0")}>{t("Retry")}</span>
      {retrying ? <Loader2 className="absolute inset-0 m-auto animate-spin" aria-hidden /> : null}
    </Button>
  )
}
