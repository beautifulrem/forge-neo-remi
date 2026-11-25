/**
 * Values a form is still producing: a canvas layer being exported and uploaded after the last brush
 * stroke (a registered flush), or a file on its way with the value it sets (`pendingWrite`). A submit
 * (Generate, Process, an action button, saving a queued job) awaits them first, so it carries what the
 * user sees. Writers register per form id.
 */

type Flush = () => Promise<void>

const writers = new Map<string, Set<Flush>>()
/** Values on their way, per form: each settles when its write has ended, failed or not. */
const writes = new Map<string, Set<Promise<void>>>()

function setOf<T>(map: Map<string, Set<T>>, formId: string): Set<T> {
  let set = map.get(formId)
  if (!set) map.set(formId, (set = new Set()))
  return set
}

/** Registers `flush` for form `formId`; returns the unregister function. */
export function registerPendingWrite(formId: string, flush: Flush): () => void {
  const set = setOf(writers, formId)
  set.add(flush)
  return () => void set.delete(flush)
}

/**
 * Runs `write` (an upload and the value it sets) as a pending value of form `formId`, so a submit
 * started meanwhile sends what it writes. A failed write does not fail the submit (it says so itself).
 */
export async function pendingWrite(formId: string, write: () => Promise<void>): Promise<void> {
  const done = write()
  const settled = done.then(
    () => {},
    () => {},
  )
  const set = setOf(writes, formId)
  set.add(settled)
  try {
    await done
  } finally {
    set.delete(settled)
  }
}

/** Whether form `formId` has anything to flush (a registered flush or a value on its way). */
export function hasPendingWrites(formId: string): boolean {
  return (writers.get(formId)?.size ?? 0) > 0 || (writes.get(formId)?.size ?? 0) > 0
}

/**
 * Resolves when every pending value of the form is written (rejects if a registered flush fails).
 * `values: false` runs the registered flushes only: what a write itself needs first (replacing an
 * image flushes the layers painted on it), which would otherwise wait for that write to end.
 */
export async function flushPendingWrites(formId: string, { values = true }: { values?: boolean } = {}): Promise<void> {
  const flushes = [...(writers.get(formId) ?? [])]
  const onTheirWay = values ? [...(writes.get(formId) ?? [])] : []
  if (flushes.length === 0 && onTheirWay.length === 0) return
  await Promise.all([...flushes.map((flush) => flush()), ...onTheirWay])
}
