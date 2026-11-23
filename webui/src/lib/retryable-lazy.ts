import { type ComponentType, createElement, lazy, type LazyExoticComponent } from "react"

/**
 * `React.lazy` that can load again: lazy() keeps a failed import for good, so
 * after a chunk failed (offline, a replaced build) an error boundary's "Try
 * again" rendered the same failure. A failed load swaps in a fresh lazy
 * component before it throws; the next render fetches the chunk again.
 */
export function retryableLazy<P extends object>(load: () => Promise<{ default: ComponentType<P> }>): ComponentType<P> {
  const make = (): LazyExoticComponent<ComponentType<P>> =>
    lazy(() =>
      load().catch((error: unknown) => {
        current = make()
        throw error
      }),
    )
  let current = make()
  function Retryable(props: P) {
    return createElement(current, props)
  }
  return Retryable
}
