/**
 * Hydration guard for zustand `persist` stores. What localStorage holds was
 * written by an older build, by hand, or by another app on this origin; the
 * default shallow merge lets a field of the wrong type through and the first
 * render that reads it throws. `mergeTyped` keeps a stored field only when its
 * type matches the store's own value (a `null` default stands for "a string
 * or nothing", a number must be finite), and drops empty entries from stored
 * lists. A field that takes only some values of its type (an enum, a colour)
 * names them in `rules`: a value of the right type it does not take (an older
 * build's choice, a hand edit) crashed the pages that look it up. Numbers with
 * a range are clamped by the store's own `merge`. `mergeTyped` does not look
 * inside lists or maps: a store whose lists or maps hold objects checks each
 * entry (`keepEntries`, a type guard on its items).
 */
import { createJSONStorage } from "zustand/middleware"

export const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value)

/** A string, or nothing stored (an optional or nullable text field). */
export const isText = (value: unknown): value is string | null | undefined => value === null || value === undefined || typeof value === "string"

/** The entries of a stored map whose values `keep` accepts. */
export function keepEntries<T>(value: Record<string, unknown>, keep: (item: unknown) => item is T): Record<string, T> {
  return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, T] => keep(entry[1])))
}

/** The values a field takes (an enum's, with `null` where it may be unset), or a test of them. */
export type FieldRules<T> = { [K in keyof T]?: readonly T[K][] | ((value: unknown) => boolean) }

function sameKind(value: unknown, fallback: unknown): boolean {
  if (fallback === null) return value === null || typeof value === "string"
  if (Array.isArray(fallback)) return Array.isArray(value)
  if (isRecord(fallback)) return isRecord(value)
  // JSON reads 1e999 as Infinity
  if (typeof fallback === "number") return typeof value === "number" && Number.isFinite(value)
  return typeof value === typeof fallback
}

function takes(rule: readonly unknown[] | ((value: unknown) => boolean) | undefined, value: unknown): boolean {
  if (rule === undefined) return true
  return typeof rule === "function" ? rule(value) : rule.includes(value)
}

export function mergeTyped<T extends object>(persisted: unknown, current: T, rules: FieldRules<T> = {}): T {
  if (!isRecord(persisted)) return current
  const out = { ...current } as Record<string, unknown>
  for (const [key, value] of Object.entries(persisted)) {
    if (!(key in current)) continue
    const fallback = (current as Record<string, unknown>)[key]
    if (typeof fallback === "function" || !sameKind(value, fallback) || !takes(rules[key as keyof T], value)) continue
    out[key] = Array.isArray(value) ? value.filter((item) => item !== null && item !== undefined) : value
  }
  return out as T
}

/**
 * `createJSONStorage` over a storage whose writes may fail (a full quota, storage turned off):
 * persist writes after every `set`, so a write that throws escaped from the action that set the
 * state, and from the store's subscribers, cutting short the event that caused it (a finished
 * job). A failed write keeps the state in memory only.
 */
export function bestEffortStorage<S>(getStorage: () => Storage) {
  return createJSONStorage<S>(() => {
    const storage = getStorage()
    return {
      getItem: (name) => storage.getItem(name),
      setItem: (name, value) => {
        try {
          storage.setItem(name, value)
        } catch {
          // kept in memory for this page's life
        }
      },
      removeItem: (name) => storage.removeItem(name),
    }
  })
}
