/**
 * Collapsible structured Parameters Diff panel rendered beneath the visual
 * comparison in `CompareDialog`. Razor-sharp monospaced diff table without
 * pill clutter. Lazily loaded so it costs 0 KB until needed.
 */
import { useQuery } from "@tanstack/react-query"
import { ArrowRightLeft, ChevronDown, ClipboardPaste, Loader2 } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import type { FormTab } from "@/api/types"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { errorText } from "@/features/errors/error-text"
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
    <div className="flex flex-col gap-1.5 border-t border-border/60 pt-2">
      <div className="flex items-center gap-2 text-xs font-medium text-foreground">
        <span>{title}</span>
        {changed ? (
          <span className="flex items-center gap-1.5 font-mono text-[0.6875rem] tabular-nums">
            {diff.removed.length > 0 ? <span className="text-diff-del">−{diff.removed.length}</span> : null}
            {diff.added.length > 0 ? <span className="text-diff-add">+{diff.added.length}</span> : null}
          </span>
        ) : (
          <span className="font-mono text-[0.6875rem] font-normal text-muted-foreground">{t("Unchanged")}</span>
        )}
      </div>
      {changed ? (
        <div className="flex flex-wrap gap-1 font-mono text-[0.6875rem] leading-relaxed">
          {diff.removed.map((token, idx) => (
            <span
              key={`r-${idx}-${token}`}
              className="rounded-xs bg-diff-del-bg px-1.5 py-0.5 text-diff-del line-through decoration-current/55"
              title={`${t("Removed")} (A)`}
            >
              − {token}
            </span>
          ))}
          {diff.added.map((token, idx) => (
            <span
              key={`a-${idx}-${token}`}
              className="rounded-xs bg-diff-add-bg px-1.5 py-0.5 text-diff-add"
              title={`${t("Added")} (B)`}
            >
              + {token}
            </span>
          ))}
          {diff.unchanged.map((token, idx) => (
            <span key={`u-${idx}-${token}`} className="px-1 py-0.5 text-muted-foreground">
              {token}
            </span>
          ))}
        </div>
      ) : diff.a ? (
        <p title={diff.a} className="line-clamp-2 font-mono text-xs text-muted-foreground">{diff.a}</p>
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
      {/* planned rows: the actions follow the title or take a row of their own as one group; where even
          that row is too narrow for the three (a phone), the toggle sits above the two apply buttons */}
      <div className="@container/diffbar flex flex-wrap items-center gap-2">
        <CollapsibleTrigger className="flex h-7 items-center gap-1.5 rounded-sm px-1 text-xs font-semibold tracking-wide text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
          <ChevronDown className={cn("size-3.5 text-muted-foreground transition-transform duration-150", !open && "-rotate-90")} aria-hidden />
          <ArrowRightLeft className="size-3.5 text-muted-foreground" aria-hidden />
          <span>{t("Parameters diff")}</span>
          {query.isPending ? (
            <Loader2 className="size-3.5 animate-spin text-muted-foreground" aria-hidden />
          ) : hasMetadata ? (
            <>
              <span className="text-muted-foreground/50 select-none" aria-hidden>
                ·
              </span>
              <span className="font-mono text-[0.6875rem] font-normal text-muted-foreground tabular-nums">
                {count === 0 ? t("Identical parameters") : t("{{count}} differences", { count })}
              </span>
            </>
          ) : null}
        </CollapsibleTrigger>

        {open && diff && hasMetadata ? (
          <div className="ml-auto flex items-center gap-1.5 @max-[24rem]/diffbar:grid @max-[24rem]/diffbar:w-full @max-[24rem]/diffbar:grid-cols-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground @max-[24rem]/diffbar:col-span-2 @max-[24rem]/diffbar:justify-self-start"
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
                  className="h-7 gap-1 px-2 text-xs"
                  disabled={applying !== null}
                  onClick={() => void applySide("a")}
                >
                  <ClipboardPaste className="size-3" aria-hidden />
                  {t("Apply A to form")}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1 px-2 text-xs"
                  disabled={applying !== null}
                  onClick={() => void applySide("b")}
                >
                  <ClipboardPaste className="size-3" aria-hidden />
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
              {!onlyDiffs || diff.prompt_diff.added.length > 0 || diff.prompt_diff.removed.length > 0 ? (
                <TokenDiffRow title={t("Prompt diff")} diff={diff.prompt_diff} />
              ) : null}

              {(!onlyDiffs || diff.negative_diff.added.length > 0 || diff.negative_diff.removed.length > 0) &&
              (diff.negative_diff.a || diff.negative_diff.b) ? (
                <TokenDiffRow title={t("Negative prompt diff")} diff={diff.negative_diff} />
              ) : null}

              {coreRows.length > 0 ? (
                <div className="flex flex-col gap-1.5 border-t border-border/60 pt-2">
                  <h4 className="text-xs font-medium text-foreground">{t("Core parameters")}</h4>
                  <div className="overflow-x-auto rounded-md bg-background">
                    <table className="w-full border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-border/70 bg-muted/35 text-left text-[0.6875rem] text-muted-foreground">
                          <th className="w-1/3 px-2.5 py-1 font-medium">{t("Parameter")}</th>
                          <th className="w-1/3 px-2.5 py-1 font-mono font-medium">A</th>
                          <th className="w-1/3 px-2.5 py-1 font-mono font-medium">B</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/50 font-mono tabular-nums">
                        {coreRows.map((row) => (
                          <tr key={row.key}>
                            <td className="px-2.5 py-1 font-sans font-medium text-muted-foreground">{row.key}</td>
                            <td className={cn("px-2.5 py-1 break-all", row.changed && row.a && "bg-diff-del-bg/60 text-diff-del")}>
                              {row.a ?? "—"}
                            </td>
                            <td className={cn("px-2.5 py-1 break-all", row.changed && row.b && "bg-diff-add-bg/60 font-medium text-diff-add")}>
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
                <div className="flex flex-col gap-1.5 border-t border-border/60 pt-2">
                  <h4 className="text-xs font-medium text-foreground">{t("LoRAs")}</h4>
                  <div className="overflow-x-auto rounded-md bg-background">
                    <table className="w-full border-collapse text-xs font-mono tabular-nums">
                      <tbody className="divide-y divide-border/50">
                        {diff.lora_diffs.map((lora) => (
                          <tr key={lora.name}>
                            <td className="w-1/3 truncate px-2.5 py-1 font-sans font-medium text-foreground" title={lora.name}>
                              {lora.name}
                            </td>
                            <td className={cn("w-1/3 px-2.5 py-1", lora.a ? "bg-diff-del-bg/60 text-diff-del" : "text-muted-foreground")}>
                              {lora.a ?? "—"}
                            </td>
                            <td className={cn("w-1/3 px-2.5 py-1", lora.b ? "bg-diff-add-bg/60 font-medium text-diff-add" : "text-muted-foreground")}>
                              {lora.b ?? "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}

              {diff.extension_diffs.length > 0 ? (
                <div className="flex flex-col gap-1.5 border-t border-border/60 pt-2">
                  <h4 className="text-xs font-medium text-foreground">{t("Active extensions")}</h4>
                  <div className="overflow-x-auto rounded-md bg-background">
                    <table className="w-full border-collapse text-xs font-mono tabular-nums">
                      <tbody className="divide-y divide-border/50">
                        {diff.extension_diffs.map((ext) => (
                          <tr key={`${ext.group}:${ext.key}`}>
                            <td className="w-1/3 px-2.5 py-1 font-sans text-muted-foreground">
                              <span className="font-mono text-[0.6875rem] text-foreground">{ext.group}</span>
                              <span className="mx-1 text-muted-foreground/50" aria-hidden>
                                /
                              </span>
                              {ext.key}
                            </td>
                            <td className={cn("w-1/3 px-2.5 py-1", ext.a ? "bg-diff-del-bg/60 text-diff-del" : "text-muted-foreground")}>
                              {ext.a ?? "—"}
                            </td>
                            <td className={cn("w-1/3 px-2.5 py-1", ext.b ? "bg-diff-add-bg/60 font-medium text-diff-add" : "text-muted-foreground")}>
                              {ext.b ?? "—"}
                            </td>
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
