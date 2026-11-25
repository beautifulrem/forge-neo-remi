/**
 * The time a relative time ("3 minutes ago") is counted from, moving on while the page stays
 * open: a render that shows one reads `useNow()` and passes it on (`timeAgo(…, now)`), so the
 * React Compiler, which memoizes a render on what it reads, recomputes the text when it moves.
 * One timer serves every component that reads it and runs only while one is mounted.
 */
import { useSyncExternalStore } from "react"

/** How often the time moves on: relative times change by the second only under a minute. */
export const NOW_TICK_MS = 30_000

const listeners = new Set<() => void>()
let timer: ReturnType<typeof setInterval> | undefined
let now = 0
/** Nobody read the time since the timer stopped: the next read takes the clock's. */
let stale = true

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  timer ??= setInterval(() => {
    now = Date.now()
    for (const notify of listeners) notify()
  }, NOW_TICK_MS)
  return () => {
    listeners.delete(listener)
    if (listeners.size > 0 || timer === undefined) return
    clearInterval(timer)
    timer = undefined
    stale = true
  }
}

function snapshot(): number {
  if (stale) {
    now = Date.now()
    stale = false
  }
  return now
}

/** The current time in ms, updated every {@link NOW_TICK_MS} while the caller is mounted. */
export function useNow(): number {
  return useSyncExternalStore(subscribe, snapshot, snapshot)
}
