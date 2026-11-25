/**
 * Param locks: a toggle with `locks: "owner.param"` protects that param while
 * it is on. The form store drops programmatic writes to it (`setValues`:
 * pasted parameters, "Send to", reused jobs, carried-over defaults;
 * `applyPatches`: presets and action patches); user edits (`setValue`) and
 * resets still apply. Display parts of a patch (choices, enabled, …) apply.
 * Both writes return what a lock kept from changing, and every path that
 * reports a write names it with `lockNote` ("Overridden by a lock: …").
 */
import { i18n, t } from "@/i18n"
import { splitRef } from "./conditions"
import type { FormIndex } from "./form-index"
import { messageText } from "./messages"

export type ParamLock = {
  /** The lock toggle. */
  owner: string
  name: string
  /** The protected param. */
  target: readonly [owner: string, name: string]
}

export function collectLocks(index: FormIndex): ParamLock[] {
  const locks: ParamLock[] = []
  for (const [owner, inputs] of index.inputs) {
    for (const [name, node] of inputs) {
      if (node.type === "toggle" && typeof node.locks === "string" && node.locks) {
        locks.push({ owner, name, target: splitRef(node.locks, owner) })
      }
    }
  }
  return locks
}

/** `"owner.param"` refs as the form labels them, in `language`, joined its way (a ref without a label as it is). */
export function lockedLabels(index: FormIndex, refs: readonly string[], language = i18n.language): string {
  const labels = refs.map((ref) => {
    const dot = ref.indexOf(".")
    const label = index.inputs.get(ref.slice(0, dot))?.get(ref.slice(dot + 1))?.label
    return label ? messageText(i18n, label, language) : ref
  })
  return new Intl.ListFormat(language, { type: "conjunction" }).format(labels)
}

/** "Overridden by a lock: …" for the values a lock kept (`refs`, "owner.param"); undefined when none was. */
export function lockNote(index: FormIndex, refs: readonly string[], language = i18n.language): string | undefined {
  return refs.length > 0 ? t("Overridden by a lock: {{names}}", { names: lockedLabels(index, refs, language), lng: language }) : undefined
}

/** Names of `owner` that are locked now (their lock toggle is on). */
export function lockedNames(locks: readonly ParamLock[], values: Record<string, Record<string, unknown> | undefined>, owner: string): Set<string> {
  const names = new Set<string>()
  for (const lock of locks) {
    if (lock.target[0] === owner && values[lock.owner]?.[lock.name] === true) names.add(lock.target[1])
  }
  return names
}
