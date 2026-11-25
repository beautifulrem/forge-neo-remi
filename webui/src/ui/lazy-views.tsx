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
import { Skeleton } from "@/components/ui/skeleton"
import { retryableLazy } from "@/lib/retryable-lazy"

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

export const ImageInput = deferred(() => import("./inputs/Image").then((m) => m.ImageInput), "h-40 w-full rounded-lg")
export const ImageListInput = deferred(() => import("./inputs/Image").then((m) => m.ImageListInput), "h-24 w-full rounded-lg")
export const Color = deferred(() => import("./inputs/Color").then((m) => m.Color), "h-9 w-full rounded-md")
export const FileInput = deferred(() => import("./inputs/File").then((m) => m.FileInput), "h-9 w-full rounded-md")
export const JsonInput = deferred(() => import("./inputs/Json").then((m) => m.JsonInput), "h-20 w-full rounded-md")
export const TabsView = deferred(() => import("./layout/Tabs").then((m) => m.TabsView), "h-24 w-full rounded-lg")
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
