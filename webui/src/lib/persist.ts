/**
 * Hydration guard for zustand `persist` stores. What localStorage holds was
 * written by an older build, by hand, or by another app on this origin; the
 * default shallow merge lets a field of the wrong type through and the first
 * render that reads it throws. `mergeTyped` keeps a stored field only when its
 * type matches the store's own value (a `null` default stands for "a string
 * or nothing"), and drops empty entries from stored lists.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value)

function sameKind(value: unknown, fallback: unknown): boolean {
  if (fallback === null) return value === null || typeof value === "string"
  if (Array.isArray(fallback)) return Array.isArray(value)
  if (isRecord(fallback)) return isRecord(value)
  return typeof value === typeof fallback
}

export function mergeTyped<T extends object>(persisted: unknown, current: T): T {
  if (!isRecord(persisted)) return current
  const out = { ...current } as Record<string, unknown>
  for (const [key, value] of Object.entries(persisted)) {
    if (!(key in current)) continue
    const fallback = (current as Record<string, unknown>)[key]
    if (typeof fallback === "function" || !sameKind(value, fallback)) continue
    out[key] = Array.isArray(value) ? value.filter((item) => item !== null && item !== undefined) : value
  }
  return out as T
}
