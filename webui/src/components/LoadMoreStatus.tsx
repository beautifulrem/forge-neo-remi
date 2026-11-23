import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"

/**
 * The row under an endless list: "Loading…" while the next page loads, and
 * a retry when it failed (the pages already shown stay; the list does not
 * ask again on its own, so a failing server is not hammered).
 */
export function LoadMoreStatus({ loading, failed, onRetry }: { loading: boolean; failed: boolean; onRetry: () => void }) {
  const { t } = useTranslation()
  if (loading) return <p className="py-3 text-center text-xs text-muted-foreground">{t("Loading…")}</p>
  if (!failed) return null
  return (
    <div role="alert" className="flex items-center justify-center gap-2 py-2 text-xs text-muted-foreground">
      <span>{t("Could not load more.")}</span>
      <Button variant="ghost" size="xs" onClick={onRetry}>
        {t("Retry")}
      </Button>
    </div>
  )
}
