/**
 * The img2img stage: a Source / Result switch over the image editor bound
 * to the img2img form, or the results. Batch modes have no canvas: the
 * stage shows the results only.
 */
import { Images, Layers, PencilRuler } from "@/components/icons"
import { type ReactNode, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Toggle } from "@/components/ui/toggle"
import { useSession } from "@/features/gallery/session-store"
import { cn } from "@/lib/utils"
import { ImageEditor } from "../ImageEditor"
import { preloadImageEditor } from "../preload"
import { MAX_STAGED_CANDIDATES, useCanvasStaging, wireImg2imgStaging } from "../state/staging"
import { useImg2imgStageView } from "./stage-view"
import { IMG2IMG, setCoreValues, setSizeFromImage, setSourceImage, useImg2imgBinding, useImg2imgEditorProps, useModeLayers } from "./use-img2img"

function ViewSwitch({ view, onChange }: { view: "source" | "result"; onChange: (view: "source" | "result") => void }) {
  const { t } = useTranslation()
  const options = [
    { value: "source" as const, label: t("Source"), icon: PencilRuler },
    { value: "result" as const, label: t("Result"), icon: Images },
  ]
  return (
    <div role="tablist" aria-label={t("Image view")} className="inline-flex shrink-0 items-center gap-0.5 rounded-lg bg-muted p-0.5">
      {options.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          role="tab"
          aria-selected={view === value}
          // Only the shown view's panel is mounted.
          aria-controls={view === value ? `img2img-stage-${value}` : undefined}
          id={`img2img-stage-tab-${value}`}
          onClick={() => onChange(value)}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
              event.preventDefault()
              const next = value === "source" ? "result" : "source"
              onChange(next)
              document.getElementById(`img2img-stage-tab-${next}`)?.focus()
            }
          }}
          tabIndex={view === value ? 0 : -1}
          className={cn(
            "flex h-6 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium whitespace-nowrap text-muted-foreground transition-colors outline-none select-none",
            "hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/60",
            "aria-selected:bg-card aria-selected:text-foreground dark:aria-selected:bg-accent",
          )}
        >
          <Icon className="size-3.5" aria-hidden />
          {label}
        </button>
      ))}
    </div>
  )
}

type Img2imgStageProps = {
  /** The result view (image and parameters). */
  result: ReactNode
  /** The result's actions, shown in the header with the result view. */
  actions?: ReactNode
  /** Header items before / after the switch (pane toggles, the run bar). */
  leading?: ReactNode
  trailing?: ReactNode
  className?: string
}

export function Img2imgStage({ result, actions, leading, trailing, className }: Img2imgStageProps) {
  const { t } = useTranslation()
  const binding = useImg2imgBinding()
  const layers = useModeLayers(binding)
  const editor = useImg2imgEditorProps(binding, layers)
  const [view, setView] = useImg2imgStageView(binding ? [binding.background, ...binding.layers.map((layer) => layer.param)] : undefined)
  const stayOnCanvas = useCanvasStaging((s) => s.stayOnCanvas)
  const setStayOnCanvas = useCanvasStaging((s) => s.setStayOnCanvas)
  const sessionItems = useSession((s) => s.items)
  const img2imgItems = sessionItems.filter((item) => item.tab === "img2img" && !item.role)
  const canvas = Boolean(binding && layers.canvas)
  const shown = canvas ? view : "result"

  // The editor chunk is fetched with the page, so switching to Source is instant.
  useEffect(() => {
    void preloadImageEditor()
    wireImg2imgStaging()
  }, [])

  const stageLatest = () => {
    const latestJobId = img2imgItems[0]?.jobId
    if (!latestJobId) return
    const batch = img2imgItems.filter((item) => item.jobId === latestJobId).slice(0, MAX_STAGED_CANDIDATES)
    if (batch.length === 0) return
    useCanvasStaging.getState().stageCandidates(
      IMG2IMG,
      batch.map((item) => ({ fileId: item.fileId, width: item.width, height: item.height, infotext: item.infotext })),
    )
    setView("source")
    toast.success(t("Staged {{count}} candidate on canvas", { count: batch.length }))
  }

  return (
    <div className={cn("flex min-h-0 flex-col gap-2", className)}>
      <div className="flex min-h-7 flex-wrap items-center gap-1">
        {leading}
        {canvas ? <ViewSwitch view={shown} onChange={setView} /> : null}
        {canvas && editor.image ? (
          <>
            <Toggle
              size="sm"
              pressed={stayOnCanvas}
              onPressedChange={setStayOnCanvas}
              aria-label={t("Stage results in canvas")}
              title={t("Stage results in canvas")}
              className="h-7 gap-1 px-2 text-xs"
            >
              <Layers className="size-3.5" aria-hidden />
              <span className="hidden @[32rem]/actions:inline">{t("Stage in canvas")}</span>
            </Toggle>
            {shown === "result" && img2imgItems.length > 0 ? (
              <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs" aria-label={t("Stage latest result")} title={t("Stage latest result")} onClick={stageLatest}>
                <Layers className="size-3.5" aria-hidden />
                <span className="hidden @[32rem]/actions:inline">{t("Stage latest result")}</span>
              </Button>
            ) : null}
          </>
        ) : null}
        {/* The labels above go by the bar's width (the stage's `@container/actions`). A wide bar is
            one row; a narrow one gives the result actions a row of their own under the canvas
            controls (a planned second row, not a wrap), where they fold into "More actions" by the
            room they have. */}
        {shown === "result" && actions ? (
          <div className="@container/actions order-last flex min-w-0 grow basis-full items-center gap-1 @[46rem]/actions:order-none @[46rem]/actions:basis-0">{actions}</div>
        ) : (
          <div className="min-w-0 flex-1" />
        )}
        {trailing ? <div className="ml-auto flex items-center">{trailing}</div> : null}
      </div>
      {shown === "source" && binding ? (
        <div role="tabpanel" id="img2img-stage-source" aria-labelledby="img2img-stage-tab-source" className="flex min-h-0 flex-1 flex-col">
          <ImageEditor
            docKey={IMG2IMG}
            pendingKey={IMG2IMG}
            label={t("Source image")}
            className="min-h-64 flex-1"
            fill
            {...editor}
            onImageChange={(image) => {
              void setSourceImage(binding, image?.id ?? null, image)
            }}
            onLayerChange={(key, id) => setCoreValues({ [key]: id })}
            onUseImageSize={setSizeFromImage}
          />
        </div>
      ) : (
        <div role={canvas ? "tabpanel" : undefined} id="img2img-stage-result" aria-labelledby={canvas ? "img2img-stage-tab-result" : undefined} className="flex min-h-0 flex-1 flex-col">
          {result}
        </div>
      )}
    </div>
  )
}
