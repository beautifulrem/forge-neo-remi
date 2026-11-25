/**
 * Tactile darkroom film-strip Staging Bar inside the Konva canvas editor:
 * flip through 1–4 inpaint/img2img candidates, adjust opacity or split/original
 * comparison, and Accept as new base layer (`--glow` CTA) or Discard without
 * losing zoom/pan.
 */
import { Check, ChevronLeft, ChevronRight, Columns2, Eye, Trash2, X } from "@/components/icons"
import { useEffect, useRef } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"
import { type StagedCandidate, type StagingSession, takeStagingFocus } from "../state/staging"
import { Divider, IconButton, PANEL } from "./controls"

type StagingBarProps = {
  session: StagingSession
  /** The canvas document, for a focus request (`requestStagingFocus`). */
  docKey: string
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
  docKey,
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
  const bar = useRef<HTMLDivElement>(null)
  const count = session.candidates.length
  const active = session.candidates[session.index]
  // The bar itself takes it, not Accept or Discard: a held Enter must not decide.
  useEffect(() => {
    if (takeStagingFocus(docKey)) bar.current?.focus()
  }, [docKey])
  if (!active) return null

  return (
    <div
      ref={bar}
      role="toolbar"
      aria-label={t("Canvas staging")}
      tabIndex={-1}
      data-toast-obstacle
      className={cn(
        PANEL,
        // One row: as its `@container/staging` (the canvas width) narrows, the candidate dots go
        // first (the counter and arrows stay), then the fader, then Accept's label (its icon stays,
        // named). Measured in English, the longest: 657, 553, 431 and 302 px of bar.
        "flex max-w-[calc(100%-0.5rem)] items-center gap-1 px-2 py-1 outline-none focus-visible:ring-3 focus-visible:ring-focus-ring",
      )}
    >
      <IconButton label={t("Previous candidate")} keys={["←"]} onClick={onPrev} disabled={count <= 1 || busy}>
        <ChevronLeft />
      </IconButton>

      <div className="flex items-center gap-2 px-1">
        <span className="font-mono text-xs font-medium text-foreground tabular-nums" aria-live="polite">
          <span aria-hidden>
            {session.index + 1} / {count}
          </span>
          <span className="sr-only">{t("Candidate {{index}} of {{count}}", { index: session.index + 1, count })}</span>
        </span>
        {count > 1 ? (
          <div className="hidden items-center @[43rem]/staging:flex">
            {session.candidates.map((candidate, idx) => (
              <button
                key={`${candidate.fileId}-${idx}`}
                type="button"
                aria-label={t("Candidate {{index}} of {{count}}", { index: idx + 1, count })}
                aria-pressed={idx === session.index}
                onClick={() => onSelect(idx)}
                className="group grid size-6 place-items-center rounded-sm outline-none focus-visible:ring-3 focus-visible:ring-ring"
              >
                <span
                  aria-hidden
                  className={cn(
                    "block h-1.5 rounded-full transition-colors duration-150",
                    idx === session.index ? "w-4 bg-foreground" : "w-1.5 bg-muted-foreground group-hover:bg-foreground",
                  )}
                />
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <IconButton label={t("Next candidate")} keys={["→"]} onClick={onNext} disabled={count <= 1 || busy}>
        <ChevronRight />
      </IconButton>

      <Divider />

      <div className="hidden items-center gap-1.5 px-1 @[36.5rem]/staging:flex">
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
        <span className="w-8 text-right font-mono text-[0.6875rem] text-muted-foreground tabular-nums">
          {Math.round(session.opacity * 100)}%
        </span>
      </div>

      <IconButton
        label={t("Show the original")}
        pressed={session.compareOriginal}
        onClick={() => onCompareOriginalChange(!session.compareOriginal)}
        disabled={busy}
      >
        {/* the label says what it does and the pressed tint that it is on: EyeOff read as "hide" */}
        <Eye />
      </IconButton>

      <IconButton label={t("Split compare")} pressed={session.splitCompare} onClick={onToggleSplit} disabled={busy}>
        <Columns2 />
      </IconButton>

      <Divider />

      <Button
        size="sm"
        // the editor sits beside Generate (the screen's lamp): accepting is the bar's lead, not a second lamp
        variant="secondary"
        className="h-7 gap-1 px-2 text-xs font-medium @[29rem]/staging:px-2.5"
        aria-label={t("Accept as base layer")}
        title={t("Accept as base layer")}
        disabled={busy}
        onClick={() => onAccept(active)}
      >
        <Check className="size-3.5" aria-hidden />
        <span className="hidden @[29rem]/staging:inline">{t("Accept as base layer")}</span>
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
