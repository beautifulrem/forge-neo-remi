/**
 * Loads a tab schema into its form store before the first paint. On a
 * reload (options changed, UI preset switched) it keeps what `load` alone
 * would lose: uploaded images (never kept in drafts) stay, and params the
 * user never touched follow their new defaults.
 */
import { useLayoutEffect } from "react"
import { useStore } from "zustand"
import { CONFLICTS_ACTION, runAction } from "@/ui/actions"
import { jsonEqual, splitRef } from "@/ui/conditions"
import type { FormIndex } from "@/ui/form-index"
import { type FormState, type FormStore, isModified, type OwnerValues } from "@/ui/form-store"
import type { TabSchema } from "@/ui/schema"
import { tabOwners } from "@/ui/tab"
import { isPersisted } from "@/ui/values"

type Snapshot = Pick<FormState, "values" | "defaults"> & { index: FormIndex }

/** Values to set after a reload, per owner (see the module comment). */
export function carryOver(before: Snapshot, after: Snapshot): Record<string, OwnerValues> {
  const out: Record<string, OwnerValues> = {}
  for (const [owner, nodes] of after.index.inputs) {
    for (const [name, node] of nodes) {
      const previous = before.index.inputs.get(owner)?.get(name)
      if (!previous) continue
      const old = before.values[owner]?.[name]
      const oldDefault = before.defaults[owner]?.[name]
      const now = after.values[owner]?.[name]
      const newDefault = after.defaults[owner]?.[name]
      let value: unknown
      if (!isPersisted(node)) {
        if (previous.type !== node.type || old === null || old === undefined || !jsonEqual(now, newDefault)) continue
        value = old
      } else {
        if (!jsonEqual(old, oldDefault) || jsonEqual(oldDefault, newDefault) || !jsonEqual(now, old)) continue
        value = newDefault
      }
      ;(out[owner] ??= {})[name] = value
    }
  }
  return out
}

export function loadSchema(store: FormStore, schema: TabSchema) {
  const before = store.getState()
  const snapshot: Snapshot | null = before.revision > 0 ? { values: before.values, defaults: before.defaults, index: before.index } : null
  before.load(tabOwners(schema), {
    conflictParams: schema.conflict_params,
    capabilities: schema.capabilities ?? null,
    advice: schema.advice ?? [],
  })
  if (snapshot) {
    for (const [owner, values] of Object.entries(carryOver(snapshot, store.getState()))) store.getState().setValues(owner, values)
  }
  // Patch locks are not persisted: re-derive them when the form holds a non-default conflict param.
  const state = store.getState()
  if ((schema.conflict_params ?? []).some((ref) => isModified(state, ...splitRef(ref, "core")))) {
    void runAction({ store, tab: schema.tab }, "core", CONFLICTS_ACTION)
  }
}

/** The schema each store was last loaded with. */
const loadedWith = new WeakMap<FormStore, TabSchema>()

/**
 * Loads `schema` into `store` unless it already holds it: the form page does this before its first
 * paint, and code that reads a form away from its page (capturing a recipe on the Queue page) does
 * it first too, since until then the store holds only the drafts, with no defaults to compare them to.
 */
export function ensureSchemaLoaded(store: FormStore, schema: TabSchema) {
  if (loadedWith.get(store) === schema) return
  loadedWith.set(store, schema)
  loadSchema(store, schema)
}

/** True once `schema` is loaded into `store` (loaded before paint; `load` bumps the revision that re-renders). */
export function useFormLoader(store: FormStore, schema: TabSchema): boolean {
  // Read the revision into the result so a memoising compiler re-evaluates after `load`.
  const revision = useStore(store, (s) => s.revision)
  useLayoutEffect(() => ensureSchemaLoaded(store, schema), [store, schema])
  return revision > 0 && loadedWith.get(store) === schema
}
