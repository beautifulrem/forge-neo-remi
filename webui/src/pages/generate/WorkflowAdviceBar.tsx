/**
 * Pre-submit actionable workflow advice chips rendered directly above the
 * Generate run controls. Each `AdviceItem` carries a severity (`warning`,
 * `error`, `info`), a clear message, and an optional one-click "Fix" button
 * that applies the server-supplied parameter / option `patch`.
 */
import { CircleAlert, Info, Loader2, TriangleAlert, Wrench, X } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { useStore } from "zustand"
import { Button } from "@/components/ui/button"
import type { GenerateTab } from "@/features/progress/run-controller"
import { cn } from "@/lib/utils"
import { applyAdvicePatch } from "@/ui/actions"
import { getFormStore } from "@/ui/form-store"
import type { AdviceItem } from "@/ui/schema"
import { failureText } from "@/features/errors/error-text"

const COLLAPSED_LIMIT = 2

function adviceKey(item: AdviceItem): string {
  return `${item.code}:${item.message}`
}

export function WorkflowAdviceBar({ tab }: { tab: GenerateTab }) {
  const { t } = useTranslation()
  const store = getFormStore(tab)
  const advice = useStore(store, (s) => s.advice)
  const dismissed = useStore(store, (s) => s.dismissedAdvice)
  const [expanded, setExpanded] = useState(false)
  const [fixing, setFixing] = useState<string | null>(null)

  const visible = advice.filter((item) => !dismissed.has(adviceKey(item)))
  if (visible.length === 0) return null

  const shown = expanded ? visible : visible.slice(0, COLLAPSED_LIMIT)
  const hiddenCount = visible.length - shown.length

  const fix = async (item: AdviceItem) => {
    const key = adviceKey(item)
    if (fixing) return
    setFixing(key)
    try {
      await applyAdvicePatch({ store, tab }, item)
      toast.success(item.action_label ? t("Applied: {{action}}", { action: t(item.action_label) }) : t("Parameter fix applied"))
    } catch (error) {
      toast.error(t("Could not apply fix"), { description: failureText(error) })
    } finally {
      setFixing(null)
    }
  }

  return (
    <div role="region" aria-label={t("Workflow advice")} data-slot="workflow-advice" className="-mx-1 flex flex-col gap-1.5 px-1">
      {shown.map((item) => {
        const key = adviceKey(item)
        const Icon = item.severity === "error" ? CircleAlert : item.severity === "warning" ? TriangleAlert : Info
        const isFixing = fixing === key
        return (
          <div
            key={key}
            data-advice-code={item.code}
            className={cn(
              "flex items-start gap-2 rounded-r-sm border-l-2 px-2.5 py-1.5 text-xs leading-snug",
              item.severity === "warning" && "border-l-advice-warn bg-advice-warn-bg text-advice-warn",
              item.severity === "error" && "border-l-destructive bg-destructive/10 text-destructive",
              item.severity === "info" && "border-l-advice-info bg-advice-info-bg text-advice-info",
            )}
          >
            <Icon className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            <p className="min-w-0 flex-1 break-words">{t(item.message)}</p>
            <div className="flex shrink-0 items-center gap-2">
              {item.patch ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="xs"
                  disabled={fixing !== null}
                  onClick={() => void fix(item)}
                  className="relative h-5 gap-1 px-1.5 text-[0.6875rem] font-medium after:absolute after:-inset-y-2.5 after:inset-x-0"
                >
                  {isFixing ? <Loader2 className="size-3 animate-spin" aria-hidden /> : <Wrench className="size-3" aria-hidden />}
                  <span>{item.action_label ? t(item.action_label) : t("Fix")}</span>
                </Button>
              ) : null}
              <button
                type="button"
                aria-label={t("Dismiss advice")}
                onClick={() => store.getState().dismissAdvice(key)}
                className="relative grid size-5 place-items-center rounded-xs opacity-70 after:absolute after:-inset-x-2 after:-inset-y-2.5 outline-none hover:opacity-100 focus-visible:ring-2 focus-visible:ring-ring"
              >
                <X className="size-3" aria-hidden />
              </button>
            </div>
          </div>
        )
      })}
      {hiddenCount > 0 ? (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="self-start rounded-xs px-1 text-xs font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
        >
          {t("Show {{count}} more advice", { count: hiddenCount })}
        </button>
      ) : expanded && visible.length > COLLAPSED_LIMIT ? (
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="self-start rounded-xs px-1 text-xs font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
        >
          {t("Show fewer")}
        </button>
      ) : null}
    </div>
  )
}
