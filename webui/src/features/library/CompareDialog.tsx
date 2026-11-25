/** Compare two images: before / after divider or side by side, with RGBA 4-backdrop and collapsible structured Parameters Diff. */
import { Suspense, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { Segmented } from "@/components/Segmented"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { type BackdropMode, backdropSurface } from "@/features/compare/backdrop-modes"
import { BackdropPicker } from "@/features/compare/BackdropPicker"
import { CompareSlider } from "@/features/compare/CompareSlider"
import { cn } from "@/lib/utils"
import { retryableLazy } from "@/lib/retryable-lazy"

const ParamDiffPanel = retryableLazy(() => import("@/features/compare/ParamDiffPanel").then((m) => ({ default: m.ParamDiffPanel })))

export type CompareImage = {
  id: string
  name: string
  width?: number | null
  height?: number | null
  infotext?: string | null
  model?: string | null
  tab?: string | null
}

type CompareDialogProps = {
  images: readonly [CompareImage, CompareImage] | null
  onOpenChange: (open: boolean) => void
}

export function CompareDialog({ images, onOpenChange }: CompareDialogProps) {
  const { t } = useTranslation()
  const [mode, setMode] = useState<"slider" | "side">("slider")
  const [backdrop, setBackdrop] = useState<BackdropMode>("checker")
  const [a, b] = images ?? []
  const sameSize = a && b && a.width === b.width && a.height === b.height
  const surface = backdropSurface(backdrop)
  return (
    <Dialog open={images !== null} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[calc(94dvh-env(safe-area-inset-top)-env(safe-area-inset-bottom))] w-[min(calc(96vw-env(safe-area-inset-left)-env(safe-area-inset-right)),1400px)] max-w-none flex-col gap-3 sm:max-w-none">
        <DialogHeader className="flex-row flex-wrap items-center justify-between gap-2 pr-8">
          <div className="min-w-0">
            <DialogTitle>{t("Compare images")}</DialogTitle>
            <DialogDescription className="truncate" title={a && b ? `${a.name} · ${b.name}` : undefined}>
              {a && b ? `${a.name} · ${b.name}` : null}
              {a && b && !sameSize ? ` · ${t("Different sizes: both are fitted to the frame")}` : null}
            </DialogDescription>
          </div>
          <div className="flex items-center gap-1.5">
            <BackdropPicker value={backdrop} onChange={setBackdrop} />
            <Segmented
              size="sm"
              aria-label={t("Comparison view")}
              value={mode}
              onValueChange={setMode}
              options={[
                { value: "slider", label: t("Divider") },
                { value: "side", label: t("Side by side") },
              ]}
            />
          </div>
        </DialogHeader>
        {a && b ? (
          <>
            {mode === "slider" ? (
              <div className="flex min-h-0 flex-1 items-center justify-center">
                <CompareSlider
                  before={api.fileUrl(a.id)}
                  after={api.fileUrl(b.id)}
                  beforeLabel={t("A")}
                  afterLabel={t("B")}
                  aspect={a.width && a.height ? a.width / a.height : undefined}
                  fitHeight="60dvh"
                  backdrop={backdrop}
                />
              </div>
            ) : (
              <div className="grid min-h-0 flex-1 grid-cols-2 gap-2">
                {[a, b].map((image, index) => (
                  <figure key={image.id} className="flex min-h-0 flex-col gap-1">
                    <img
                      src={api.fileUrl(image.id)}
                      alt={image.name}
                      data-backdrop={backdrop}
                      className={cn("min-h-0 flex-1 rounded-lg object-contain", surface.className)}
                      style={surface.style}
                    />
                    <figcaption className="truncate text-center text-xs text-muted-foreground" title={image.name}>
                      {index === 0 ? t("A") : t("B")} · {image.name}
                      {image.width && image.height ? ` · ${image.width}×${image.height}` : ""}
                    </figcaption>
                  </figure>
                ))}
              </div>
            )}
            <Suspense fallback={null}>
              <ParamDiffPanel a={a} b={b} onApplied={() => onOpenChange(false)} />
            </Suspense>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
