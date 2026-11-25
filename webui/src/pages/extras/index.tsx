/**
 * Extras: upscale and post-process one image, a batch of uploads or a folder
 * on the server (the form and every postprocessing operation are rendered
 * from the server's schema). The result shows with a before / after divider
 * against its source, with download, send-to and the viewer; this session's
 * runs stay in a strip below.
 */
import { Ban, ImageIcon, ImageUpscale, Loader2, Maximize2 } from "@/components/icons"
import { useQuery } from "@tanstack/react-query"
import { Suspense, useEffect, useLayoutEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { progressFraction, runningFor, useProgress } from "@/api/progress"
import { schemaQuery, useSchema } from "@/api/queries"
import { useWs } from "@/api/ws"
import { formatShortcut } from "@/app/commands"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { ExposureSheet } from "@/features/progress/ExposureSheet"
import { Skeleton } from "@/components/ui/skeleton"
import { useIsDesktop } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import { CompareSlider } from "@/features/compare/CompareSlider"
import { jobErrorText } from "@/features/errors/error-text"
import { type ExtrasResult, interruptExtras, runExtras, useExtrasRuns, wireExtrasRuns } from "@/features/extras/run"
import type { ResultItem } from "@/features/gallery/session-store"
import { formatEta } from "@/features/progress/effects"
import { enqueueForm } from "@/features/queue/edit-session"
import { EnqueueButton, QueueEditBanner } from "@/features/queue/FormQueueControls"
import { ResultActions } from "@/features/send-to/ResultActions"
import { useFormLoader } from "@/pages/generate/use-form-loader"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { DEFAULT_SLOT } from "@/ui/form-index"
import { getFormStore } from "@/ui/form-store"
import { ScriptsProvider, Slot } from "@/ui/layout/Slot"
import { ImageFieldSkeleton, LabelSkeleton } from "@/ui/lazy-views"
import { RenderLayout } from "@/ui/Renderer"
import { schemaIndex } from "@/ui/tab"
import { retryableLazy } from "@/lib/retryable-lazy"
import { usePinnedBar } from "@/lib/use-pinned-bar"

const Lightbox = retryableLazy(() => import("@/features/lightbox/Lightbox").then((m) => ({ default: m.Lightbox })))

wireExtrasRuns()

function ExtrasForm({ onShown }: { onShown: () => void }) {
  const { t } = useTranslation()
  const { data: latest } = useSchema("extras")
  const store = getFormStore("extras")
  const schema = useFormLoader(store, latest)
  const ready = schema !== null
  // before paint, so the result area comes in the same frame as the form
  useLayoutEffect(() => {
    if (ready) onShown()
  }, [ready, onShown])
  if (!schema) return <FormSkeleton />
  const hasDefaultSlot = store.getState().index.slots.has(DEFAULT_SLOT)
  return (
    <FormProvider formId="extras" store={store}>
      <ScriptsProvider scripts={schema.scripts}>
        <div role="form" aria-label={t("Parameters")} className="flex flex-col gap-3">
          <OwnerProvider owner="core">
            <RenderLayout layout={schema.core.layout} />
            {hasDefaultSlot ? null : <Slot section={DEFAULT_SLOT} />}
          </OwnerProvider>
        </div>
      </ScriptsProvider>
    </FormProvider>
  )
}

/**
 * Stands in while the form's schema loads: the opening of a form of one image and two fields (the
 * source image's label over its empty drop zone, then two fields of a label over an h-8 control),
 * made of the same sizes so a form like that does not change height when it arrives. A form with the
 * source tabs and every postprocessing section is taller: on a phone the result area comes in with
 * the form (`onShown`), so nothing under the skeleton moves whatever form arrives.
 */
function FormSkeleton() {
  return (
    <div data-slot="form-skeleton" className="flex flex-col gap-3" aria-busy>
      <ImageFieldSkeleton label />
      {["w-20", "w-16"].map((width) => (
        <div key={width} className="flex flex-col gap-1.5">
          <LabelSkeleton className={width} />
          <Skeleton data-part="control" className="h-8 rounded-md" />
        </div>
      ))}
    </div>
  )
}

function RunBar() {
  const { t, i18n } = useTranslation()
  const submitting = useExtrasRuns((s) => s.submitting)
  const waiting = useExtrasRuns((s) => s.jobs.length)
  const job = useProgress((s) => runningFor(s, "extras"))
  const fraction = progressFraction(job)
  const eta = formatEta(job?.eta, i18n.language)
  const busy = waiting > 0
  return (
    <div className="flex flex-col gap-2">
      <QueueEditBanner tab="extras" />
      <div className="flex items-center gap-1.5">
        <Button variant="glow" size="lg" className="h-11 min-w-0 flex-1 gap-2 text-[0.9375rem] font-semibold" onClick={() => void runExtras()} disabled={submitting} aria-keyshortcuts="Control+Enter Meta+Enter">
          {submitting ? <Loader2 className="animate-spin" aria-hidden /> : null}
          <span className="truncate">{busy ? t("Queue another") : t("Process")}</span>
          <Kbd className="ml-1 hidden border border-glow-foreground/30 bg-transparent font-normal text-glow-foreground/75 sm:inline-flex touch-only:hidden!">{formatShortcut("Mod+Enter")}</Kbd>
        </Button>
        <EnqueueButton tab="extras" />
        {busy ? (
          <Button variant="destructive" size="icon-lg" className="size-11" onClick={interruptExtras} aria-label={t("Interrupt")} title={`${t("Interrupt")} (Esc)`} aria-keyshortcuts="Escape">
            <Ban />
          </Button>
        ) : null}
      </div>
      {busy ? (
        <div className="flex flex-col gap-1.5" role="status">
          {job && fraction !== null ? <ExposureSheet step={job.step} steps={job.steps} fraction={fraction} aria-label={t("Processing progress")} /> : null}
          <p className="flex flex-wrap gap-x-3 text-xs text-muted-foreground tabular-nums">
            {(job
              ? [job.text ? t(job.text) : t("Processing"), job.jobCount && job.jobCount > 1 ? t("Image {{index}} of {{count}}", { index: (job.jobNo ?? 0) + 1, count: job.jobCount }) : null, eta ? t("ETA {{eta}}", { eta }) : null]
              : [t("Queued")]
            )
              .filter(Boolean)
              .map((fact, index) => (
                <span key={index}>{fact}</span>
              ))}
          </p>
        </div>
      ) : null}
    </div>
  )
}

function asItems(result: ExtrasResult): ResultItem[] {
  return result.outputs
    .filter((output) => (output.kind ?? "image") === "image")
    .map((output, index) => ({
      key: `${result.jobId}:${index}`,
      fileId: output.id,
      jobId: result.jobId,
      tab: "extras",
      index,
      infotext: output.infotext ?? result.info,
      width: output.width ?? undefined,
      height: output.height ?? undefined,
      createdAt: result.at,
    }))
}

function ResultStage() {
  const { t, i18n } = useTranslation()
  const results = useExtrasRuns((s) => s.results)
  const selected = useExtrasRuns((s) => s.selected)
  const result = (selected ? results.find((item) => item.jobId === selected) : undefined) ?? results[0] ?? null
  const items = result ? asItems(result) : []
  const [current, setCurrent] = useState<string | null>(null)
  const item = items.find((entry) => entry.key === current) ?? items[0] ?? null
  const [mode, setMode] = useState<"compare" | "result">("compare")
  const [viewer, setViewer] = useState(false)
  const job = useProgress((s) => runningFor(s, "extras"))
  const preview = useWs((s) => (job && s.preview && (job.seq === undefined || s.preview.jobSeq === job.seq) ? s.preview.url : null))
  const comparable = Boolean(result?.source && item && items.length === 1)
  // the form names the values a run refused (the page has its schema)
  const { data: schema } = useQuery({ ...schemaQuery("extras"), enabled: Boolean(result?.errors?.length) })
  const index = schema ? schemaIndex(schema) : undefined

  let view: React.ReactNode
  if (job && preview && !item) {
    view = <img src={preview} alt={t("Live preview")} className="max-h-full max-w-full rounded-lg object-contain" />
  } else if (result?.error) {
    // the "Generation failed" toast announces it; the result area keeps it in view, as a plain line
    view = <p className="max-w-md text-sm text-destructive">{jobErrorText(result.error, result.errors, i18n.language, index)}</p>
  } else if (!item) {
    view = (
      <div className="flex flex-col items-center gap-2 text-center text-sm text-muted-foreground">
        <ImageUpscale className="size-7 opacity-50" aria-hidden />
        <p>{result ? t("This run made no image (a folder run saves to its output folder).") : t("Results appear here, next to their source.")}</p>
      </div>
    )
  } else if (comparable && mode === "compare") {
    view = (
      <CompareSlider
        key={item.key}
        before={api.fileUrl(result!.source!)}
        after={api.fileUrl(item.fileId)}
        beforeLabel={t("Source")}
        afterLabel={t("Result")}
        aspect={item.width && item.height ? item.width / item.height : undefined}
        fitHeight="calc(100dvh - 15rem)"
      />
    )
  } else {
    view = (
      <button type="button" className="contents" onClick={() => setViewer(true)} aria-label={t("Open in the viewer")}>
        <img src={api.fileUrl(item.fileId)} alt={t("Processed image")} className="max-h-full max-w-full cursor-zoom-in rounded-lg bg-checker object-contain" />
      </button>
    )
  }

  return (
    // the actions fold by the column's width (`ResultActions` collapsible), as on the generation stage
    <div className="@container/actions flex h-full min-h-0 flex-col gap-2">
      {/* Planned rows, never a wrap (sizes up to five digits a side): one row where the view
          switch, the size and the folded actions fit (27rem; the size and the actions alone,
          17rem); narrower, the actions take a row of their own, and below 16rem the size gives
          way to the view switch. The actions unfold at 34rem (`ResultActions`) and need more
          room than the switch and the size leave them until 36rem: there the size gives way too.
          The viewer shows the size. */}
      <div className="flex min-h-9 flex-wrap items-center gap-2">
        {comparable ? (
          <Segmented
            size="sm"
            aria-label={t("View")}
            value={mode}
            onValueChange={setMode}
            options={[
              { value: "compare", label: t("Compare") },
              { value: "result", label: t("Result") },
            ]}
          />
        ) : null}
        {item?.width && item.height ? (
          <span className={cn("text-xs text-muted-foreground tabular-nums", comparable && "hidden @[16rem]/actions:inline @[34rem]/actions:hidden @[36rem]/actions:inline")}>
            {item.width}×{item.height}
          </span>
        ) : null}
        <div className={cn("ml-auto flex min-w-0 basis-full items-center justify-end gap-0.5", comparable ? "@[27rem]/actions:basis-auto" : "@[17rem]/actions:basis-auto")}>
          {item ? (
            <Button variant="ghost" size="icon-sm" aria-label={t("Open in the viewer")} onClick={() => setViewer(true)}>
              <Maximize2 />
            </Button>
          ) : null}
          <ResultActions item={item} tab="txt2img" removable={false} collapsible />
        </div>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center rounded-md bg-canvas p-2">{view}</div>
      {items.length > 1 ? (
        // 4 px of room around the thumbnails: the scroll box would cut their focus ring
        <div role="listbox" aria-label={t("Images of this run")} className="-mx-1 flex gap-1.5 overflow-x-auto p-1">
          {items.map((entry) => (
            <button
              key={entry.key}
              type="button"
              role="option"
              aria-selected={entry.key === item?.key}
              onClick={() => setCurrent(entry.key)}
              className={cn(
                // the focus ring outside the thumbnail, the selection a layer inside it over the image:
                // the selected one shows the focus too (a ring drawn inside sat under that layer)
                "relative size-16 shrink-0 overflow-hidden rounded-md bg-checker outline-none focus-visible:ring-3 focus-visible:ring-focus-ring",
                "after:pointer-events-none after:absolute after:inset-0 after:rounded-[inherit]",
                entry.key === item?.key && "after:shadow-[inset_0_0_0_2px_var(--primary),inset_0_0_0_3px_var(--background)]",
              )}
            >
              <img src={api.thumbUrl(entry.fileId, 128)} alt="" className="size-full object-cover" />
            </button>
          ))}
        </div>
      ) : null}
      {result?.info ? <p title={result.info} className="line-clamp-3 text-xs break-words text-muted-foreground select-text">{result.info}</p> : null}
      <RecentRuns />
      {viewer && item ? (
        <Suspense fallback={null}>
          <Lightbox open onOpenChange={setViewer} items={items} current={item.key} onNavigate={setCurrent} tab="extras" />
        </Suspense>
      ) : null}
    </div>
  )
}

function RecentRuns() {
  const { t } = useTranslation()
  const results = useExtrasRuns((s) => s.results)
  const selected = useExtrasRuns((s) => s.selected) ?? results[0]?.jobId
  if (results.length < 2) return null
  return (
    <div className="flex flex-col gap-1">
      <h2 className="text-[0.8rem] font-medium text-muted-foreground">{t("This session")}</h2>
      {/* 4 px of room around the thumbnails: the scroll box would cut their rings */}
      <div role="listbox" aria-label={t("Recent runs")} className="-mx-1 flex gap-1.5 overflow-x-auto p-1">
        {results.map((result) => {
          const first = result.outputs.find((output) => (output.kind ?? "image") === "image")
          return (
            <button
              key={result.jobId}
              type="button"
              role="option"
              aria-selected={result.jobId === selected}
              aria-label={first ? t("Images of this run: {{count}}", { count: result.outputs.length }) : t("Run without images")}
              onClick={() => useExtrasRuns.setState({ selected: result.jobId })}
              className={cn("grid size-12 shrink-0 place-items-center overflow-hidden rounded-md bg-muted outline-none focus-visible:ring-3 focus-visible:ring-focus-ring", result.jobId === selected && "ring-2 ring-primary")}
            >
              {first ? <img src={api.thumbUrl(first.id, 96)} alt="" className="size-full object-cover" /> : result.error ? <Ban className="size-4 text-destructive" aria-hidden /> : <ImageIcon className="size-4 text-muted-foreground" aria-hidden />}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function useExtrasShortcuts() {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.isComposing || event.repeat) return
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        if (document.querySelector("[role=dialog][data-open], [role=alertdialog]")) return
        event.preventDefault()
        // Ctrl/Cmd+Shift+Enter: "Add to queue", as on the Generate page
        if (event.shiftKey) void enqueueForm("extras")
        else void runExtras()
      } else if (event.key === "Escape" && useExtrasRuns.getState().jobs.length > 0) {
        const target = event.target instanceof Element ? event.target : null
        if (target?.closest("input, textarea, [contenteditable], [role=dialog], [role=menu], [role=listbox]")) return
        interruptExtras()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
}

export default function ExtrasPage() {
  const { t } = useTranslation()
  const desktop = useIsDesktop()
  // nothing run yet: on a phone the result area stays a short box under the form instead of a screen of empty canvas
  const noResults = useExtrasRuns((s) => s.results.length === 0)
  const running = useProgress((s) => runningFor(s, "extras") !== null)
  const idle = noResults && !running
  // a phone scrolls the page under the run bar; the desktop columns scroll below it
  const runBarRef = usePinnedBar("top")
  // on a phone the result area is under the form: it comes in with the form, never under the skeleton
  const [formShown, setFormShown] = useState(false)
  useExtrasShortcuts()
  return (
    <div className={cn("h-full", desktop ? "grid grid-cols-[minmax(22rem,30rem)_minmax(0,1fr)]" : "overflow-y-auto")}>
      <div className={cn("flex flex-col", desktop && "min-h-0 border-r")}>
        <div ref={runBarRef} className="sticky top-0 z-10 border-b bg-background p-3">
          <RunBar />
        </div>
        <div className={cn("p-3", desktop && "min-h-0 flex-1 overflow-y-auto")}>
          <Suspense fallback={<FormSkeleton />}>
            <ExtrasForm onShown={() => setFormShown(true)} />
          </Suspense>
        </div>
      </div>
      {desktop || formShown ? (
        <section aria-label={t("Result")} className={cn("p-3", desktop ? "min-h-0" : idle ? "h-64" : "h-[80dvh]")}>
          <ResultStage />
        </section>
      ) : null}
    </div>
  )
}
