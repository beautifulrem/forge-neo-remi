/**
 * Before / after comparison: two images stacked at the same size, the
 * "after" image clipped at a divider the user drags (pointer anywhere on
 * the frame, or the handle with ←/→, Shift for big steps, Home / End).
 * Images of different sizes are both fitted to the frame (the "before"
 * image's aspect ratio), so an upscale compares feature for feature.
 */
import { ChevronsLeftRight } from "@/components/icons"
import { useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { type BackdropMode, backdropSurface } from "./backdrop-modes"
import { clampPosition, keyPosition } from "./position"

type CompareSliderProps = {
  before: string
  after: string
  beforeLabel?: string
  afterLabel?: string
  /** Width / height of the frame (defaults to the "before" image's once loaded). */
  aspect?: number
  className?: string
  /** Initial divider position, 0..1. */
  initial?: number
  /** The frame is as wide as fits this height (a CSS length, e.g. "70dvh") and its container. */
  fitHeight?: string
  /** RGBA inspection backdrop behind transparent areas (default: checkerboard). */
  backdrop?: BackdropMode
}

export function CompareSlider({ before, after, beforeLabel, afterLabel, aspect, className, initial = 0.5, fitHeight, backdrop = "checker" }: CompareSliderProps) {
  const { t } = useTranslation()
  const [position, setPosition] = useState(clampPosition(initial))
  const [natural, setNatural] = useState<number | null>(null)
  const frame = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const ratio = aspect ?? natural ?? 1

  const fromPointer = (clientX: number) => {
    const rect = frame.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    setPosition(clampPosition((clientX - rect.left) / rect.width))
  }
  const percent = `${(position * 100).toFixed(1)}%`

  const surface = backdropSurface(backdrop)

  return (
    <div
      ref={frame}
      data-backdrop={backdrop}
      className={cn("relative isolate max-h-full max-w-full touch-none overflow-hidden rounded-lg select-none", surface.className, className)}
      style={{
        ...surface.style,
        aspectRatio: String(ratio),
        width: fitHeight ? `min(100%, calc(${fitHeight} * ${ratio}))` : undefined,
      }}
      onPointerDown={(event) => {
        if (event.button !== 0) return
        dragging.current = true
        event.currentTarget.setPointerCapture?.(event.pointerId)
        fromPointer(event.clientX)
      }}
      onPointerMove={(event) => dragging.current && fromPointer(event.clientX)}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
    >
      <img
        src={before}
        alt={beforeLabel ?? t("Before")}
        draggable={false}
        onLoad={(event) => {
          const { naturalWidth, naturalHeight } = event.currentTarget
          if (naturalWidth && naturalHeight) setNatural(naturalWidth / naturalHeight)
        }}
        className="absolute inset-0 size-full object-contain"
      />
      <img
        src={after}
        alt={afterLabel ?? t("After")}
        draggable={false}
        className="absolute inset-0 size-full object-contain"
        style={{ clipPath: `inset(0 0 0 ${percent})` }}
      />
      {/* each label keeps to its half of the frame (a narrow one, a portrait image held sideways, ran them
          together), its full text a title; it takes the pointer for that, and a press on it still drags (the frame's) */}
      <span title={beforeLabel ?? t("Before")} className="absolute top-2 left-2 max-w-[calc(50%-0.75rem)] truncate rounded-md bg-black/55 px-1.5 py-0.5 text-[0.6875rem] font-medium text-white">
        {beforeLabel ?? t("Before")}
      </span>
      <span title={afterLabel ?? t("After")} className="absolute top-2 right-2 max-w-[calc(50%-0.75rem)] truncate rounded-md bg-black/55 px-1.5 py-0.5 text-[0.6875rem] font-medium text-white">
        {afterLabel ?? t("After")}
      </span>
      <div aria-hidden className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.25)]" style={{ left: percent }} />
      <div
        role="slider"
        tabIndex={0}
        aria-label={t("Comparison divider")}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(position * 100)}
        aria-valuetext={t("{{percent}} % of the after image shown", { percent: Math.round((1 - position) * 100) })}
        aria-orientation="horizontal"
        onKeyDown={(event) => {
          const next = keyPosition(position, event.key, event.shiftKey)
          if (next === null) return
          event.preventDefault()
          setPosition(next)
        }}
        className="absolute top-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full bg-white text-neutral-900 shadow-md ring-1 ring-black/10 outline-none focus-visible:ring-4 focus-visible:ring-focus-ring"
        style={{ left: percent }}
      >
        <ChevronsLeftRight className="size-4" aria-hidden />
      </div>
    </div>
  )
}
