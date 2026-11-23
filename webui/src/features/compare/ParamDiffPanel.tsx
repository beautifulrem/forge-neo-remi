/**
 * Collapsible structured Parameters Diff panel rendered beneath the visual
 * comparison in `CompareDialog`. Lazily loaded so it costs 0 KB until needed.
 */
import { useQuery } from "@tanstack/react-query"
import { ArrowRightLeft, ChevronDown, ClipboardPaste, Loader2 } from "@/components/icons"
import { useState } from "react"
import { toast } from "sonner"
import type { FormTab } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { errorText } from "@/features/errors/error-text"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { applyCompareDiffToForm, fetchCompareDiff, type TokenDiff, totalDiffCount } from "./param-diff"

export type CompareSide = {
  id: string
  name: string
  infotext?: string | null
  tab?: string | null
}

type ParamDiffPanelProps = {
  a: CompareSide
  b: CompareSide
  onApplied?: () => void
}

function TokenDiffRow({ title, diff }: { title: string; diff: TokenDiff }) {
  const { t } = useTranslation()
  const changed = diff.added.length > 0 || diff.removed.length > 0
  return (
    <div className="flex flex-col gap-1 border-t border-border/60 pt-2">
      <div className="flex items-center gap-2 text-xs font-medium">
        <span>{title}</span>
        {changed ? (
          <span className="flex items-center gap-1.5 text-[0.6875rem] tabular-nums">
            {diff.removed.length > 0 ? <span className="text-diff-del">−{diff.removed.length}</span> : null}
            {diff.added.length > 0 ? <span className="text-diff-add">+{diff.added.length}</span> : null}
          </span>
        ) : (
          <span className="text-[0.6875rem] font-normal text-muted-foreground">{t("Unchanged")}</span>
        )}
      </div>
      {changed ? (
        <div className="flex flex-wrap gap-1 text-xs leading-relaxed">
          {diff.removed.map((token, idx) => (
            <span
              key={`r-${idx}-${token}`}
              className="rounded-sm bg-diff-del-bg px-1.5 py-0.5 text-diff-del line-through decoration-current/60"
              title={`${t("Removed")} (A)`}
            >
              − {token}
            </span>
          ))}
          {diff.added.map((token, idx) => (
            <span
              key={`a-${idx}-${token}`}
              className="rounded-sm bg-diff-add-bg px-1.5 py-0.5 text-diff-add"
              title={`${t("Added")} (B)`}
            >
              + {token}
            </span>
          ))}
          {diff.unchanged.map((token, idx) => (
            <span key={`u-${idx}-${token}`} className="rounded-sm bg-muted/60 px-1.5 py-0.5 text-muted-foreground">
              {token}
            </span>
          ))}
        </div>
      ) : diff.a ? (
        <p className="line-clamp-2 text-xs text-muted-foreground">{diff.a}</p>
      ) : null}
    </div>
  )
}

export function ParamDiffPanel({ a, b, onApplied }: ParamDiffPanelProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(true)
  const [onlyDiffs, setOnlyDiffs] = useState(true)
  const [applying, setApplying] = useState<"a" | "b" | null>(null)

  const query = useQuery({
    queryKey: ["compare", "diff", a.id, b.id, a.infotext ?? "", b.infotext ?? ""],
    queryFn: ({ signal }) => fetchCompareDiff(a, b, signal),
    staleTime: 60_000,
  })

  const diff = query.data
  const count = diff ? totalDiffCount(diff) : 0
  const hasMetadata = Boolean((diff?.infotext_a ?? a.infotext) || (diff?.infotext_b ?? b.infotext))
  const targetTab: FormTab = b.tab === "img2img" || a.tab === "img2img" ? "img2img" : "txt2img"

  const applySide = async (side: "a" | "b") => {
    if (!diff) return
    setApplying(side)
    try {
      const applied = await applyCompareDiffToForm(diff, side, targetTab)
      if (applied === 0) {
        toast.info(t("No parameter differences to apply"))
      } else {
        toast.success(
          side === "a" ? t("Applied A differences ({{count}})", { count: applied }) : t("Applied B differences ({{count}})", { count: applied }),
        )
        onApplied?.()
      }
    } catch (error) {
      toast.error(t("Could not apply parameter differences"), { description: errorText(error) })
    } finally {
      setApplying(null)
    }
  }

  const coreRows = diff
    ? onlyDiffs
      ? diff.core_diffs.map((row) => ({ key: row.key, a: row.a, b: row.b, changed: true }))
      : (diff.all_core ?? diff.core_diffs.map((row) => ({ key: row.key, a: row.a, b: row.b, changed: true })))
    : []

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="border-t border-border pt-2">
      <div className="flex flex-wrap items-center gap-2">
        <CollapsibleTrigger className="flex h-7 items-center gap-1.5 rounded-sm px-1 text-sm font-semibold outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50">
          <ChevronDown className={cn("size-4 text-muted-foreground transition-transform duration-150", !open && "-rotate-90")} aria-hidden />
          <ArrowRightLeft className="size-3.5 text-muted-foreground" aria-hidden />
          <span>{t("Parameters diff")}</span>
          {query.isPending ? (
            <Loader2 className="size-3.5 animate-spin text-muted-foreground" aria-hidden />
          ) : hasMetadata ? (
            <Badge variant={count > 0 ? "secondary" : "outline"} className="text-[0.625rem] tabular-nums">
              {count === 0 ? t("Identical parameters") : t("{{count}} differences", { count })}
            </Badge>
          ) : null}
        </CollapsibleTrigger>

        {open && diff && hasMetadata ? (
          <div className="ml-auto flex flex-wrap items-center gap-1.5">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs"
              aria-pressed={onlyDiffs}
              onClick={() => setOnlyDiffs((v) => !v)}
            >
              {onlyDiffs ? t("Show all") : t("Only differences")}
            </Button>
            {count > 0 ? (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  disabled={applying !== null}
                  onClick={() => void applySide("a")}
                >
                  <ClipboardPaste aria-hidden />
                  {t("Apply A to form")}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  disabled={applying !== null}
                  onClick={() => void applySide("b")}
                >
                  <ClipboardPaste aria-hidden />
                  {t("Apply B to form")}
                </Button>
              </>
            ) : null}
          </div>
        ) : null}
      </div>

      <CollapsibleContent>
        <div className="-mx-1 mt-2 max-h-60 overflow-y-auto px-1">
          {!hasMetadata ? (
            <p className="py-3 text-xs text-muted-foreground">{t("No parameter metadata available for either image.")}</p>
          ) : diff ? (
            <div className="flex flex-col gap-2.5 pb-1">
              {(!onlyDiffs || diff.prompt_diff.added.length > 0 || diff.prompt_diff.removed.length > 0) ? (
                <TokenDiffRow title={t("Prompt diff")} diff={diff.prompt_diff} />
              ) : null}

              {(!onlyDiffs || diff.negative_diff.added.length > 0 || diff.negative_diff.removed.length > 0) &&
              (diff.negative_diff.a || diff.negative_diff.b) ? (
                <TokenDiffRow title={t("Negative prompt diff")} diff={diff.negative_diff} />
              ) : null}

              {coreRows.length > 0 ? (
                <div className="flex flex-col gap-1 border-t border-border/60 pt-2">
                  <h4 className="text-xs font-medium">{t("Core parameters")}</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-border/70 text-left text-[0.6875rem] text-muted-foreground">
                          <th className="py-1 pr-3 font-medium">{t("Parameter")}</th>
                          <th className="py-1 pr-3 font-medium">A</th>
                          <th className="py-1 font-medium">B</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/40">
                        {coreRows.map((row) => (
                          <tr key={row.key} className={cn(row.changed && "bg-primary/[0.04]")}>
                            <td className="py-1 pr-3 font-medium text-muted-foreground">{row.key}</td>
                            <td className={cn("py-1 pr-3 font-mono break-all", row.changed && row.a && "text-diff-del")}>
                              {row.a ?? "—"}
                            </td>
                            <td className={cn("py-1 font-mono break-all", row.changed && row.b && "font-medium text-diff-add")}>
                              {row.b ?? "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}

              {diff.lora_diffs.length > 0 ? (
                <div className="flex flex-col gap-1 border-t border-border/60 pt-2">
                  <h4 className="text-xs font-medium">{t("LoRAs")}</h4>
                  <ul className="flex flex-col divide-y divide-border/40 text-xs">
                    {diff.lora_diffs.map((lora) => (
                      <li key={lora.name} className="flex items-center gap-2 py-1">
                        <span className="min-w-0 flex-1 truncate font-medium" title={lora.name}>
                          {lora.name}
                        </span>
                        <span className="font-mono text-muted-foreground">{lora.a ?? "—"}</span>
                        <span className="text-muted-foreground" aria-hidden>
                          →
                        </span>
                        <span className="font-mono font-medium">{lora.b ?? "—"}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {diff.extension_diffs.length > 0 ? (
                <div className="flex flex-col gap-1 border-t border-border/60 pt-2">
                  <h4 className="text-xs font-medium">{t("Active extensions")}</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-xs">
                      <tbody className="divide-y divide-border/40">
                        {diff.extension_diffs.map((ext) => (
                          <tr key={`${ext.group}:${ext.key}`}>
                            <td className="py-1 pr-3 text-muted-foreground">
                              <Badge variant="outline" className="mr-1.5 text-[0.625rem]">
                                {ext.group}
                              </Badge>
                              {ext.key}
                            </td>
                            <td className="py-1 pr-3 font-mono text-destructive">{ext.a ?? "—"}</td>
                            <td className="py-1 font-mono font-medium">{ext.b ?? "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}

              {count === 0 && onlyDiffs ? (
                <p className="py-2 text-xs text-muted-foreground">{t("Identical parameters")}</p>
              ) : null}
            </div>
          ) : null}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
