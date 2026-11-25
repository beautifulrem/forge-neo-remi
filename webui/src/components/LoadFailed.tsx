import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { Loader2 } from "@/components/icons"
import { Button } from "@/components/ui/button"
import { errorText } from "@/features/errors/error-text"
import { cn } from "@/lib/utils"

/**
 * A list that could not load at all: what failed, and a retry (never an "empty" list). `message`
 * replaces the error's own text (e.g. an explanation of admin access); `children` follow the Retry
 * (another way out: the text on the web). While `retrying` (see `api/load-failure.ts`) the failure
 * stays and its Retry is busy.
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
  return (
    <div role="alert" aria-busy={retrying || undefined} className={cn("flex flex-wrap items-center gap-x-3 gap-y-1 py-3 text-sm", className)}>
      <span className="text-destructive">{message ?? errorText(error, {}, i18n.language)}</span>
      <RetryButton retrying={retrying} onRetry={onRetry} />
      {children}
    </div>
  )
}

/**
 * Retry for a load that failed. While the retry runs it is disabled but keeps the focus
 * (`aria-disabled`), and a spinner takes its label's place at the label's width.
 */
export function RetryButton({ retrying = false, onRetry }: { retrying?: boolean; onRetry: () => void }) {
  const { t } = useTranslation()
  return (
    <Button variant="outline" size="sm" className="relative" disabled={retrying} focusableWhenDisabled onClick={onRetry}>
      {/* transparent, not hidden: the label stays the button's name */}
      <span className={cn(retrying && "opacity-0")}>{t("Retry")}</span>
      {retrying ? <Loader2 className="absolute inset-0 m-auto animate-spin" aria-hidden /> : null}
    </Button>
  )
}
