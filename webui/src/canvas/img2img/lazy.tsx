/**
 * The img2img surfaces of the Generate page, loaded with the img2img tab:
 * txt2img (the default view) never renders them, so their code stays out of
 * its first load (docs/research/ui-performance.md, "Bundle").
 */
import { type ComponentProps, Suspense, useEffect } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { lazyModule, prefetch } from "@/lib/lazy-module"
import { cn } from "@/lib/utils"
import { retryableLazy } from "@/lib/retryable-lazy"

const SourcePanel = retryableLazy(() => import("./SourcePanel").then((m) => ({ default: m.Img2imgSourcePanel })))
const Stage = retryableLazy(() => import("./Img2imgStage").then((m) => ({ default: m.Img2imgStage })))

/**
 * What stands in for the source card's panel while it loads: the panel's frame without its controls
 * (`SourceFrame.tsx`), as tall as the panel at every width and in every mode. A chunk of its own,
 * like the panel: the parameter form that renders the card is txt2img's too. It is fetched as the
 * img2img stage renders, with the page and before the form's code and schema are in; a block
 * stands in until it is.
 */
const frame = lazyModule(() => import("./SourceFrame"))

type SourcePanelProps = ComponentProps<typeof import("./SourcePanel").Img2imgSourcePanel>
type StageProps = ComponentProps<typeof import("./Img2imgStage").Img2imgStage>

/** The source card's content (`SourcePanel`), fetched the first time it renders. */
export function Img2imgSourcePanel(props: SourcePanelProps) {
  const loaded = frame.useModule()
  useEffect(() => prefetch(frame), [])
  return (
    <Suspense fallback={loaded ? <loaded.SourcePanelStandIn {...props} /> : <Skeleton className="h-40 rounded-lg" />}>
      <SourcePanel {...props} />
    </Suspense>
  )
}

/** The img2img stage (`Img2imgStage`), fetched the first time it renders; the source card's stand-in with it. */
export function Img2imgStage(props: StageProps) {
  useEffect(() => prefetch(frame), [])
  return (
    <Suspense fallback={<Skeleton className={cn("rounded-md", props.className)} />}>
      <Stage {...props} />
    </Suspense>
  )
}
