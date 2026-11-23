/**
 * Asset Lineage Strip inside `GalleryDetails`: visualizes the
 * parent → current → child derivative chain (`GET /api/v2/gallery/{id}/lineage`),
 * allowing one-click navigation to ancestor/descendant images or loading the
 * source job directly into its form.
 */
import { useQuery } from "@tanstack/react-query"
import { ArrowRight, GitBranch, History } from "@/components/icons"
import { useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import type { FormTab, GalleryItem } from "@/api/types"
import { Button } from "@/components/ui/button"
import { errorText } from "@/features/errors/error-text"
import { parsedEntries } from "@/features/infotext/paste"
import { useTranslation } from "react-i18next"
import { deliver } from "@/features/send-to/send-to"
import { cn } from "@/lib/utils"
import { type LineageNode, lineageQuery } from "./boards-api"
import { openInPngInfo } from "./queries"

type LineageStripProps = {
  item: GalleryItem
  onSelectImage?: (fileId: string) => void
  onClose: () => void
}

function ThumbChip({
  node,
  current = false,
  onClick,
}: {
  node: LineageNode
  current?: boolean
  onClick?: () => void
}) {
  const { t } = useTranslation()
  const label = node.name ?? node.file_id
  return (
    <button
      type="button"
      disabled={current}
      title={current ? `${t("Current")} · ${label}` : t("Jump to {{name}}", { name: label })}
      onClick={onClick}
      className={cn(
        "group relative flex w-14 shrink-0 flex-col items-center gap-1 rounded-md p-1 text-left transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
        current ? "bg-white/15 ring-1 ring-primary" : "bg-white/6 hover:bg-white/14",
      )}
    >
      <div className="relative aspect-square w-12 overflow-hidden rounded-sm bg-black/40">
        <img src={api.thumbUrl(node.file_id, 128)} alt={label} className="size-full object-cover" loading="lazy" />
      </div>
      <span className="w-full truncate text-center text-[0.625rem] text-white/75">{current ? t("Current") : node.relation || node.tab || "—"}</span>
    </button>
  )
}

export function LineageStrip({ item, onSelectImage, onClose }: LineageStripProps) {
  const { t } = useTranslation()
  const [loadingJob, setLoadingJob] = useState(false)
  const { data } = useQuery(lineageQuery(item.id))

  const ancestors = data?.ancestors ?? []
  const descendants = data?.descendants ?? []
  const siblings = data?.siblings ?? []
  const sourceJobId = data?.item?.job_id ?? item.job_id ?? null

  if (ancestors.length === 0 && descendants.length === 0 && siblings.length === 0 && !sourceJobId) {
    return null
  }

  const jumpTo = (fileId: string) => {
    if (onSelectImage) {
      onSelectImage(fileId)
    } else {
      onClose()
      openInPngInfo(fileId)
    }
  }

  const loadSourceJob = async () => {
    if (!sourceJobId || loadingJob) return
    setLoadingJob(true)
    try {
      const detail = await api.jobs.get(sourceJobId)
      const tab: FormTab = detail.tab === "img2img" || detail.tab === "extras" ? detail.tab : "txt2img"
      const entries = parsedEntries({
        core: (detail.core ?? {}) as Record<string, unknown>,
        scripts: (detail.scripts ?? {}) as Record<string, Record<string, unknown>>,
      })
      await deliver(tab, entries, {})
      onClose()
      toast.success(t("Source job loaded into {{tab}}", { tab }))
    } catch (error) {
      toast.error(t("Could not load source job"), { description: errorText(error) })
    } finally {
      setLoadingJob(false)
    }
  }

  const currentNode: LineageNode = data?.item ?? {
    file_id: item.id,
    relation: "self",
    depth: 0,
    name: item.name,
    width: item.width,
    height: item.height,
    tab: item.tab,
    model: item.model,
    created_at: item.created_at,
    exists: true,
  }

  return (
    <section aria-label={t("Asset lineage")} className="flex flex-col gap-2 border-t border-white/10 pt-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-xs font-semibold text-white/85">
          <GitBranch className="size-3.5 text-white/60" aria-hidden />
          {t("Asset lineage")}
        </span>
        {sourceJobId ? (
          <Button
            variant="ghost"
            size="xs"
            disabled={loadingJob}
            onClick={() => void loadSourceJob()}
            className="h-6 gap-1 px-1.5 text-[0.6875rem] text-white/80 hover:bg-white/12 hover:text-white"
          >
            <History className="size-3" aria-hidden />
            {t("Load source job")}
          </Button>
        ) : null}
      </div>

      {ancestors.length > 0 || descendants.length > 0 ? (
        <div className="-mx-1 flex items-center gap-1.5 overflow-x-auto px-1 pb-1">
          {ancestors.length > 0 ? (
            <>
              <div className="flex items-center gap-1" aria-label={t("Parents")}>
                {ancestors.map((node) => (
                  <ThumbChip key={`anc-${node.file_id}`} node={node} onClick={() => jumpTo(node.file_id)} />
                ))}
              </div>
              <ArrowRight className="size-3.5 shrink-0 text-white/40" aria-hidden />
            </>
          ) : null}

          <ThumbChip node={currentNode} current />

          {descendants.length > 0 ? (
            <>
              <ArrowRight className="size-3.5 shrink-0 text-white/40" aria-hidden />
              <div className="flex items-center gap-1" aria-label={t("Children")}>
                {descendants.map((node) => (
                  <ThumbChip key={`desc-${node.file_id}`} node={node} onClick={() => jumpTo(node.file_id)} />
                ))}
              </div>
            </>
          ) : null}
        </div>
      ) : null}

      {siblings.length > 0 ? (
        <div className="flex flex-col gap-1">
          <span className="text-[0.6875rem] text-white/55">{t("Siblings")}</span>
          <div className="-mx-1 flex items-center gap-1 overflow-x-auto px-1 pb-0.5">
            {siblings.map((node) => (
              <ThumbChip key={`sib-${node.file_id}`} node={node} onClick={() => jumpTo(node.file_id)} />
            ))}
          </div>
        </div>
      ) : null}
    </section>
  )
}
