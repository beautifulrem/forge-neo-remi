import { type ComponentType, createElement, lazy, type LazyExoticComponent } from "react"

/** Loads that failed, each with the swap that lets its component load again. */
const failed = new Set<() => void>()

/**
 * Lets every component whose chunk failed to load fetch it again on its next
 * render. Called by an error boundary's "Try again" and when the browser is
 * back online; never by the failure itself, or a chunk that keeps failing
 * would suspend and fetch in a loop instead of reaching the boundary.
 */
export function retryFailedLoads(): void {
  const swaps = [...failed]
  failed.clear()
  for (const swap of swaps) swap()
}

/**
 * `React.lazy` that can load again: lazy() keeps a failed import for good, so
 * after a chunk failed (offline, a replaced build) an error boundary's "Try
 * again" rendered the same failure. A failed load reaches the boundary as
 * usual; {@link retryFailedLoads} then swaps in a fresh lazy component, and
 * the next render fetches the chunk again.
 */
export function retryableLazy<P extends object>(load: () => Promise<{ default: ComponentType<P> }>): ComponentType<P> {
  const make = (): LazyExoticComponent<ComponentType<P>> =>
    lazy(() =>
      load().catch((error: unknown) => {
        failed.add(() => {
          current = make()
        })
        throw error
      }),
    )
  let current = make()
  function Retryable(props: P) {
    return createElement(current, props)
  }
  return Retryable
}
