import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { errorText } from "@/features/errors/error-text"
import { cn } from "@/lib/utils"

/** A list that could not load at all: what failed, and a retry (never an "empty" list). */
export function LoadFailed({ error, onRetry, className }: { error: unknown; onRetry: () => void; className?: string }) {
  const { t } = useTranslation()
  return (
    <div role="alert" className={cn("flex flex-wrap items-center gap-x-3 gap-y-1 py-3 text-sm", className)}>
      <span className="text-destructive">{errorText(error)}</span>
      <Button variant="outline" size="sm" onClick={onRetry}>
        {t("Retry")}
      </Button>
    </div>
  )
}
