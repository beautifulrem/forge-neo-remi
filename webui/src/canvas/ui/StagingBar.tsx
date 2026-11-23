/**
 * InvokeAI-inspired floating Staging Bar inside the Konva canvas editor:
 * flip through 1–4 inpaint/img2img candidates, adjust opacity or split/original
 * comparison, and Accept as new base layer or Discard without losing zoom/pan.
 */
import { Check, ChevronLeft, ChevronRight, Columns2, Eye, EyeOff, Trash2, X } from "@/components/icons"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import type { StagedCandidate, StagingSession } from "../state/staging"
import { Divider, IconButton, PANEL } from "./controls"

type StagingBarProps = {
  session: StagingSession
  busy?: boolean
  onPrev: () => void
  onNext: () => void
  onSelect: (index: number) => void
  onOpacityChange: (opacity: number) => void
  onCompareOriginalChange: (compare: boolean) => void
  onToggleSplit: () => void
  onAccept: (candidate: StagedCandidate) => void
  onDiscardCurrent: () => void
  onDiscardAll: () => void
}

export function StagingBar({
  session,
  busy = false,
  onPrev,
  onNext,
  onSelect,
  onOpacityChange,
  onCompareOriginalChange,
  onToggleSplit,
  onAccept,
  onDiscardCurrent,
  onDiscardAll,
}: StagingBarProps) {
  const { t } = useTranslation()
  const count = session.candidates.length
  const active = session.candidates[session.index]
  if (!active) return null

  return (
    <div
      role="toolbar"
      aria-label={t("Canvas staging")}
      data-toast-obstacle
      className={cn(
        PANEL,
        "pointer-events-auto flex max-w-[calc(100%-1rem)] flex-wrap items-center gap-1 px-1.5 py-1 shadow-lg",
      )}
    >
      <IconButton label={t("Previous candidate")} keys={["←"]} onClick={onPrev} disabled={count <= 1 || busy}>
        <ChevronLeft />
      </IconButton>

      <div className="flex items-center gap-1 px-1">
        <span className="text-xs font-medium tabular-nums" aria-live="polite">
          {t("Candidate {{index}} of {{count}}", { index: session.index + 1, count })}
        </span>
        {count > 1 ? (
          <div className="hidden items-center gap-1 sm:flex">
            {session.candidates.map((candidate, idx) => (
              <button
                key={`${candidate.fileId}-${idx}`}
                type="button"
                aria-label={t("Candidate {{index}} of {{count}}", { index: idx + 1, count })}
                aria-pressed={idx === session.index}
                onClick={() => onSelect(idx)}
                className={cn(
                  "size-2 rounded-full transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  idx === session.index ? "bg-primary" : "bg-muted-foreground/40 hover:bg-muted-foreground/70",
                )}
              />
            ))}
          </div>
        ) : null}
      </div>

      <IconButton label={t("Next candidate")} keys={["→"]} onClick={onNext} disabled={count <= 1 || busy}>
        <ChevronRight />
      </IconButton>

      <Divider />

      <div className="hidden items-center gap-1.5 px-1 @md:flex">
        <Slider
          value={[Math.round(session.opacity * 100)]}
          min={0}
          max={100}
          step={1}
          getAriaLabel={() => t("Overlay opacity")}
          onValueChange={(value) => {
            const next = Array.isArray(value) ? value[0] : value
            if (typeof next === "number") onOpacityChange(next / 100)
          }}
          className="w-18"
        />
        <span className="w-8 text-right text-[0.6875rem] text-muted-foreground tabular-nums">
          {Math.round(session.opacity * 100)}%
        </span>
      </div>

      <IconButton
        label={t("Hold to view original")}
        pressed={session.compareOriginal}
        onClick={() => onCompareOriginalChange(!session.compareOriginal)}
        disabled={busy}
      >
        {session.compareOriginal ? <EyeOff /> : <Eye />}
      </IconButton>

      <IconButton label={t("Split compare")} pressed={session.splitCompare} onClick={onToggleSplit} disabled={busy}>
        <Columns2 />
      </IconButton>

      <Divider />

      <Button size="sm" className="h-7 gap-1 px-2.5 text-xs" disabled={busy} onClick={() => onAccept(active)}>
        <Check className="size-3.5" aria-hidden />
        <span>{t("Accept as base layer")}</span>
      </Button>

      <IconButton label={t("Discard candidate")} keys={["Esc"]} onClick={onDiscardCurrent} disabled={busy}>
        <X />
      </IconButton>

      {count > 1 ? (
        <IconButton label={t("Discard all")} onClick={onDiscardAll} disabled={busy}>
          <Trash2 />
        </IconButton>
      ) : null}
    </div>
  )
}
