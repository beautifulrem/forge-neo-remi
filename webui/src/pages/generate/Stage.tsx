/**
 * The stage: the selected result large, or the live preview while one of
 * this client's generation jobs runs (unless the user picked an older
 * result), with step / percent / ETA over the image, the result actions and
 * the generation parameters.
 */
import { ChevronDown, Expand, Radio } from "@/components/icons"
import { type ReactNode, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useStore } from "zustand"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { progressFraction, type RunningJob } from "@/api/progress"
import { useWs } from "@/api/ws"
import { Button } from "@/components/ui/button"
import { useKeyboardHints, useMediaQuery } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import { selectedItem, useSession } from "@/features/gallery/session-store"
import { InfotextPanel } from "@/features/infotext/InfotextPanel"
import { formatShortcut } from "@/app/commands"
import { ExposureSheet } from "@/features/progress/ExposureSheet"
import { formatEta, stepText } from "@/features/progress/effects"
import type { GenerateTab } from "@/features/progress/run-controller"
import { useShowsStageProgress } from "@/features/progress/stage-overlay"
import { ResultActions } from "@/features/send-to/ResultActions"
import { Img2imgStage } from "@/canvas"
import { getFormStore } from "@/ui/form-store"
import { useOwnRunningJob } from "./hooks"
import { LivePreview } from "./LivePreview"
import { viewfinderMetrics } from "./viewfinder"

/** A short window (a phone held sideways): the empty viewfinder gives up its frame's room, the progress its block. */
const SHORT_WINDOW = "(max-height: 32rem)"

function ProgressOverlay({ job, tab }: { job: RunningJob; tab: GenerateTab }) {
  const { t, i18n } = useTranslation()
  useShowsStageProgress(tab)
  // A short stage's picture is not much taller than the progress block over its lower part, which
  // then covered its middle: there the progress is one line along the stage's bottom edge.
  const edge = useMediaQuery(SHORT_WINDOW)
  const fraction = progressFraction(job) ?? 0
  const eta = formatEta(job.eta, i18n.language)
  const facts = (
    <>
      <span className="min-w-0 truncate">{stepText(job)}</span>
      {job.jobCount && job.jobCount > 1 ? (
        <span className="shrink-0 text-muted-foreground">{t("Batch {{index}} of {{count}}", { index: (job.jobNo ?? 0) + 1, count: job.jobCount })}</span>
      ) : null}
    </>
  )
  const percent = <span className={cn("shrink-0 font-medium", !edge && "ml-auto")}>{Math.round(fraction * 100)}%</span>
  const time = eta ? <span className="shrink-0 text-muted-foreground">{t("ETA {{eta}}", { eta })}</span> : null
  const sheet = <ExposureSheet step={job.step} steps={job.steps} fraction={fraction} aria-label={t("Generation progress")} className={edge ? "h-1.5 min-w-8 flex-1" : undefined} />
  // No backdrop-filter: the preview under it changes every step, and a blur would be redrawn with it.
  return edge ? (
    <div data-slot="stage-progress" data-placement="edge" className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-background/92 px-2.5 py-1 text-xs tabular-nums">
      {facts}
      {sheet}
      {percent}
      {time}
    </div>
  ) : (
    <div data-slot="stage-progress" data-placement="float" className="absolute inset-x-3 bottom-3 rounded-md bg-background/92 px-3 pt-2 pb-2.5">
      <div className="mb-1.5 flex items-baseline gap-3 text-xs tabular-nums">
        {facts}
        {percent}
        {time}
      </div>
      {sheet}
    </div>
  )
}

/**
 * Live darkroom viewfinder shown while the stage has no result yet: reads the
 * form's current width and height to draw an architectural proportional frame
 * with four L-shaped crop marks, an optical center registration crosshair, and
 * monospaced resolution telemetry.
 */
function EmptyViewfinder({ tab, waiting }: { tab: GenerateTab; waiting: boolean }) {
  const { t } = useTranslation()
  const store = getFormStore(tab)
  const rawWidth = useStore(store, (s) => s.values.core?.width)
  const rawHeight = useStore(store, (s) => s.values.core?.height)
  const metrics = viewfinderMetrics(rawWidth, rawHeight)
  const ratioValue = metrics.width / metrics.height
  // a touch screen without a keyboard has no shortcut to press
  const keyboard = useKeyboardHints()

  return (
    // a short stage (a phone held sideways) gives up the frame's room, never the next step's: the
    // frame shrinks, and goes when there is too little of it to read as one
    <div className="flex size-full flex-col items-center justify-center gap-4 p-6 sm:p-8 [@media(max-height:32rem)]:gap-2 [@media(max-height:32rem)]:py-3">
      <div
        className="flex min-h-0 w-full max-w-md flex-1 items-center justify-center [@container(max-height:3rem)]:*:hidden"
        style={{ containerType: "size" }}
      >
        <div
          data-slot="stage-viewfinder"
          style={{
            aspectRatio: `${metrics.width} / ${metrics.height}`,
            width: `min(100cqw, 22rem, calc(min(100cqh, 22rem) * ${ratioValue}))`,
          }}
          className="relative flex max-h-full max-w-full items-center justify-center"
        >
          {/* Faint framing hairline */}
          <span aria-hidden className="pointer-events-none absolute inset-0 border border-border/75" />
          {/* Four crisp L-shaped architectural darkroom crop marks */}
          <span aria-hidden className="pointer-events-none absolute -top-px -left-px size-3 border-t-[1.5px] border-l-[1.5px] border-foreground/45" />
          <span aria-hidden className="pointer-events-none absolute -top-px -right-px size-3 border-t-[1.5px] border-r-[1.5px] border-foreground/45" />
          <span aria-hidden className="pointer-events-none absolute -bottom-px -left-px size-3 border-b-[1.5px] border-l-[1.5px] border-foreground/45" />
          <span aria-hidden className="pointer-events-none absolute -bottom-px -right-px size-3 border-b-[1.5px] border-r-[1.5px] border-foreground/45" />
          {/* Optical center registration crosshair */}
          <span aria-hidden className="pointer-events-none relative size-3.5 opacity-30">
            <span className="absolute top-1/2 left-0 h-px w-full -translate-y-1/2 bg-foreground" />
            <span className="absolute top-0 left-1/2 h-full w-px -translate-x-1/2 bg-foreground" />
          </span>
        </div>
      </div>
      <div className="flex max-w-[34ch] flex-col items-center gap-1 text-center">
        <p data-slot="viewfinder-spec" className="font-mono text-xs text-muted-foreground tabular-nums">
          {metrics.width} × {metrics.height} · {metrics.ratio} · {metrics.megapixels}
        </p>
        <p className="text-xs text-balance text-muted-foreground">
          {waiting
            ? t("Waiting for the first preview…")
            : keyboard
              ? t("Write a prompt, then press {{shortcut}} to generate", { shortcut: formatShortcut("Mod+Enter") })
              : t("Write a prompt, then tap Generate")}
        </p>
      </div>
    </div>
  )
}

/** Announces start and end of generation to screen readers (not every step). */
function LiveAnnouncer({ running }: { running: boolean }) {
  const { t } = useTranslation()
  const [message, setMessage] = useState("")
  const [was, setWas] = useState(running)
  if (was !== running) {
    setWas(running)
    setMessage(running ? t("Generation started") : t("Generation finished"))
  }
  return (
    <span role="status" className="sr-only">
      {message}
    </span>
  )
}

type StageProps = {
  tab: GenerateTab
  onOpenViewer: () => void
  className?: string
  /** Header items at the start / end (the pane toggles). */
  leading?: ReactNode
  trailing?: ReactNode
  /** A row above the header (the run controls while the parameters pane is collapsed). */
  toolbar?: ReactNode
}

/** Below this height the parameters panel starts collapsed: the image keeps the room. */
const COMPACT_STAGE_PX = 520

/**
 * In the img2img Source view the editor's floor (16rem) holds the stage open: a column with less
 * room (a phone held sideways) scrolls, and the history strip follows the editor instead of lying
 * under it. The result view has no floor: its picture scales down (the viewer shows it whole).
 */
const SOURCE_FLOOR = "has-[#img2img-stage-source]:min-h-fit"

export function Stage({ tab, onOpenViewer, className, leading, trailing, toolbar }: StageProps) {
  const { t } = useTranslation()
  const job = useOwnRunningJob()
  const follow = useSession((s) => s.follow)
  const item = useSession(selectedItem)
  const showProgress = useClientOption("show_progressbar")
  const livePreviews = useClientOption("live_previews_enable")
  const videoAutoplay = useClientOption("video_player_auto")
  const videoLoop = useClientOption("video_player_loop")
  const preview = useWs((s) => (job && s.preview && (job.seq === undefined || s.preview.jobSeq === job.seq) ? s.preview.url : null))
  const live = job && follow && livePreviews ? preview : null
  const src = live ?? (item ? api.fileUrl(item.fileId) : null)
  const video = !live && item?.videoId ? item.videoId : null
  const [infoOpen, setInfoOpen] = useState(true)
  const section = useRef<HTMLElement>(null)

  // A new job takes the stage again.
  const jobId = job?.jobId ?? null
  useEffect(() => {
    if (jobId) useSession.getState().setFollow(true)
  }, [jobId])

  // A short stage (a small window, or one resized smaller) shows the image first; the parameters are
  // a click away. Only becoming short closes them: the user's choice stands until the next time.
  useEffect(() => {
    const element = section.current
    if (!element) return
    let short = false
    const check = () => {
      const height = element.clientHeight
      const now = height > 0 && height < COMPACT_STAGE_PX
      if (now && !short) setInfoOpen(false)
      short = now
    }
    check()
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(check)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const image = (
    <div data-lit={job ? "" : undefined} className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-md bg-canvas data-lit:shadow-[inset_0_0_0_1px_color-mix(in_oklch,var(--glow)_30%,transparent),inset_0_0_80px_-20px_var(--glow-dim)]">
      {video && item ? (
        // A generated video (e.g. Wan): played here; its frames stay in the history and the viewer.
        <video
          key={video}
          src={api.fileUrl(video)}
          poster={api.fileUrl(item.fileId)}
          controls
          playsInline
          autoPlay={videoAutoplay}
          loop={videoLoop}
          aria-label={t("Generated video")}
          className={cn("max-h-full max-w-full object-contain", job && "opacity-90")}
        />
      ) : src ? (
        <button
          type="button"
          onClick={onOpenViewer}
          aria-label={t("Open image viewer")}
          className="group/stage flex size-full items-center justify-center outline-none focus-visible:ring-3 focus-visible:ring-focus-ring"
        >
          {live ? (
            <LivePreview src={live} alt={t("Live preview")} />
          ) : (
            // a finished picture is laid on the table: one short settle, only when the picture changes
            // (`key`). A CSS animation, not motion's: the image must show even when motion's feature
            // chunk never loads (an element motion starts at opacity 0 stays there without it).
            <img
              key={src}
              src={src}
              alt={t("Generated image")}
              className={cn(
                "max-h-full max-w-full object-contain transition-opacity duration-200 animate-in fade-in zoom-in-[0.985] ease-[cubic-bezier(0.2,0,0,1)]",
                job && "opacity-90",
              )}
            />
          )}
        </button>
      ) : (
        <EmptyViewfinder tab={tab} waiting={job !== null} />
      )}
      {job && !follow ? (
        <Button
          variant="secondary"
          size="sm"
          onClick={() => useSession.getState().setFollow(true)}
          className="absolute top-3 left-1/2 -translate-x-1/2 gap-1.5 shadow-sm"
        >
          <Radio className="text-primary" />
          {t("Show live preview")}
        </Button>
      ) : null}
      {job && showProgress ? <ProgressOverlay job={job} tab={tab} /> : null}
      <LiveAnnouncer running={job !== null} />
    </div>
  )

  const actions = (
    <>
      <ResultActions item={live ? null : item} tab={tab} upscalable collapsible />
      <Button variant="ghost" size="icon-sm" className="ml-auto" aria-label={t("Open image viewer")} onClick={onOpenViewer} disabled={!src}>
        <Expand />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        aria-expanded={infoOpen}
        onClick={() => setInfoOpen((open) => !open)}
        className="gap-1 text-muted-foreground"
      >
        {t("Parameters")}
        <ChevronDown className={cn("transition-transform", !infoOpen && "-rotate-90")} />
      </Button>
    </>
  )

  // Never more than about a quarter of the stage: the image keeps the room. It scrolls only up and
  // down: the copy button's hit box reaches past the panel's 4 px of room.
  const info = infoOpen && item && !live ? <InfotextPanel infotext={item.infotext} className="max-h-[max(4.5rem,min(12rem,25%))] shrink-0 overflow-x-hidden overflow-y-auto rounded-sm px-1 py-1.5" /> : null
  const toolbarRow = toolbar ? (
    <div className="flex items-center gap-1">
      {leading}
      {toolbar}
      {trailing}
    </div>
  ) : null

  if (tab === "img2img") {
    return (
      <section ref={section} aria-label={t("Image area")} className={cn("@container/actions flex min-h-0 flex-col gap-2", className, SOURCE_FLOOR)}>
        {toolbarRow}
        <Img2imgStage
          className={cn("min-h-0 flex-1", SOURCE_FLOOR)}
          leading={toolbar ? null : leading}
          trailing={toolbar ? null : trailing}
          actions={actions}
          result={
            <>
              {image}
              {info}
            </>
          }
        />
      </section>
    )
  }

  return (
    <section ref={section} aria-label={t("Result")} className={cn("@container/actions flex min-h-0 flex-col gap-2", className)}>
      {toolbarRow}
      <div className="flex min-h-7 flex-wrap items-center gap-1">
        {toolbar ? null : leading}
        {actions}
        {toolbar ? null : trailing}
      </div>
      {image}
      {info}
    </section>
  )
}
