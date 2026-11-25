/**
 * Node views that the default views' first screen does not show: image,
 * file, color and JSON inputs, tabs and output views. txt2img has them only
 * in collapsed sections, settings not at all, so their code (with the image
 * editor's form glue and the tabs primitives) is fetched the first time one
 * renders instead of with every form (docs/research/ui-performance.md,
 * "Bundle"). A skeleton as tall as what it stands for fills in while one
 * loads (the content below must not jump when it arrives); `preloadViews()`
 * fetches them all (tests, or a view that is about to show them).
 */
// oxlint-disable react/only-export-components -- the exports are components made by deferred(), plus their preload
import { type ComponentType, type ReactNode, Suspense } from "react"
import { useImageRole } from "@/canvas/form/image-role"
import { Skeleton } from "@/components/ui/skeleton"
import { retryableLazy } from "@/lib/retryable-lazy"
import { cn } from "@/lib/utils"
import { EnabledProvider } from "./context"
import { useCondition } from "./hooks"
import { useSelectedTab } from "./layout/tab-selection"
import { RenderChildren } from "./Renderer"
import type { ImageNode, TabNode, TabsNode } from "./schema"

// oxlint-disable-next-line typescript/no-explicit-any -- each view receives its own node variant
type View = ComponentType<{ node: any }>

const loaders: (() => Promise<unknown>)[] = []

/** What stands in while a view loads: a skeleton's classes, or one made for the node. */
// oxlint-disable-next-line typescript/no-explicit-any -- as View: each view receives its own node variant
type Fallback = string | ((node: any) => ReactNode)

function deferred(load: () => Promise<View>, fallback: Fallback): View {
  let loaded: View | null = null
  const fetch = () =>
    load().then((component) => {
      loaded = component
      return component
    })
  loaders.push(fetch)
  // Once fetched (by an earlier render or preloadViews), React.lazy resolves synchronously: a
  // thenable that calls back at once renders without suspending.
  // (a thenable with `catch`, which retryableLazy attaches its retry to; it never rejects)
  const ready = (view: View): Promise<{ default: View }> => {
    const thenable = { then: (resolve: (module: { default: View }) => void) => resolve({ default: view }), catch: () => thenable }
    return thenable as unknown as Promise<{ default: View }>
  }
  const Lazy = retryableLazy(() => (loaded ? ready(loaded) : fetch().then((component) => ({ default: component }))))
  function DeferredView(props: { node: unknown }) {
    return (
      <Suspense fallback={typeof fallback === "string" ? <Skeleton className={fallback} /> : fallback(props.node)}>
        <Lazy {...props} />
      </Suspense>
    )
  }
  return DeferredView
}

/** A field's label line while it loads (`Field`'s header: as tall as the label's leading). */
export function LabelSkeleton({ className }: { className?: string }) {
  return (
    <div data-part="label" className="flex h-5 items-center">
      <Skeleton className={cn("h-3 w-16 rounded-sm", className)} />
    </div>
  )
}

/**
 * An empty image upload field while it loads: its label line over the drop zone, which has the upload
 * hint's padding, icon and line of text. Spacing utilities scale with the density, so it is as tall
 * as the field in either.
 */
export function ImageFieldSkeleton({ label }: { label: boolean }) {
  return (
    <div className="flex flex-col gap-1.5" aria-hidden>
      {label ? <LabelSkeleton className="w-14" /> : null}
      <Skeleton data-part="drop-zone" className="flex flex-col items-center gap-1.5 rounded-lg border border-transparent px-3 py-6">
        <span className="size-5" />
        <span className="text-sm">{"\u00a0"}</span>
      </Skeleton>
    </div>
  )
}

/** An upload field starts empty; the image editor keeps its block; a layer painted on it renders nothing. */
function ImagePlaceholder({ node }: { node: ImageNode }) {
  const role = useImageRole(node)
  if (role === "layer") return null
  return role === "upload" ? <ImageFieldSkeleton label={Boolean(node.label)} /> : <Skeleton className="h-40 w-full rounded-lg" />
}

export const ImageInput = deferred(
  () => import("./inputs/Image").then((m) => m.ImageInput),
  (node: ImageNode) => <ImagePlaceholder node={node} />,
)
export const ImageListInput = deferred(() => import("./inputs/Image").then((m) => m.ImageListInput), "h-24 w-full rounded-lg")
export const Color = deferred(() => import("./inputs/Color").then((m) => m.Color), "h-9 w-full rounded-md")
export const FileInput = deferred(() => import("./inputs/File").then((m) => m.FileInput), "h-9 w-full rounded-md")
export const JsonInput = deferred(() => import("./inputs/Json").then((m) => m.JsonInput), "h-20 w-full rounded-md")
/** A tab's content as `TabPanel` lays it out (`TabsContent`'s text size, a column, `enabled_if`). */
function TabContent({ tab }: { tab: TabNode }) {
  const enabled = useCondition(tab.enabled_if)
  return (
    <div className="flex flex-col gap-3 text-sm">
      <EnabledProvider enabled={enabled}>
        <RenderChildren nodes={tab.children} />
      </EnabledProvider>
    </div>
  )
}

/**
 * Tabs while the strip's code loads: a stand-in as tall as the strip (an h-8 tab list, or the select it
 * becomes) over the selected tab's content, rendered as it will be; nothing when no tab shows.
 */
function TabsPlaceholder({ node }: { node: TabsNode }) {
  const tab = useSelectedTab(node)
  if (!tab) return null
  return (
    <div className="flex flex-col gap-3">
      <Skeleton data-part="tab-strip" className="h-8 w-full max-w-72 rounded-lg" aria-hidden />
      <TabContent tab={tab} />
    </div>
  )
}

export const TabsView = deferred(
  () => import("./layout/Tabs").then((m) => m.TabsView),
  (node: TabsNode) => <TabsPlaceholder node={node} />,
)
export const GalleryView = deferred(() => import("./layout/Outputs").then((m) => m.GalleryView), "h-40 w-full rounded-lg")
// An output that has nothing yet: its label over the "No output yet" line, each a line of the same
// text size (as tall in either density), or nothing at all without a label
export const OutputView = deferred(
  () => import("./layout/Outputs").then((m) => m.OutputView),
  (node: { label?: string | null }) =>
    node.label ? (
      <div className="flex flex-col gap-1.5" aria-hidden>
        <Skeleton className="w-24 rounded-sm text-[0.8rem]">{"\u00a0"}</Skeleton>
        <Skeleton className="py-2 text-sm">{"\u00a0"}</Skeleton>
      </div>
    ) : null,
)

/** Fetches every deferred view; afterwards they render without a placeholder. */
export function preloadViews(): Promise<unknown> {
  return Promise.all(loaders.map((load) => load()))
}
