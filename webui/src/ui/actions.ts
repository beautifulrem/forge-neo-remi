/**
 * Action client: POST /actions/{owner}/{action} with the owner's values and
 * apply the returned patch (schema-format.md §5).
 */
import { api } from "@/api/endpoints"
import { sentenceParams } from "@/api/http"
import { keys, queryClient } from "@/api/queries"
import type { ActionExtra, ActionRequest, ActionResponse, ActionToast, Bootstrap, ParamPatch, ParamValues } from "@/api/types"
import { errorText } from "@/features/errors/error-text"
import { i18n, t } from "@/i18n"
import { showToast } from "@/lib/toast"
import { splitRef } from "./conditions"
import type { FormStore } from "./form-store"
import { lockNote } from "./locks"
import { isMessage, messageText } from "./messages"
import type { AdviceItem, AdviceSeverity, ModelCapabilitySummary } from "./schema"

const TOAST_LEVELS = new Set(["info", "success", "warning", "error"])
const ADVICE_SEVERITIES = new Set<AdviceSeverity>(["info", "warning", "error"])
const PATCH_KEYS = new Set(["value", "choices", "visible", "enabled", "label", "info", "hint", "minimum", "maximum", "step"])

export type SplitResponse = {
  params: [owner: string, name: string, patch: ParamPatch][]
  toast?: ActionToast
  outputs?: Record<string, unknown>
  /** Settings the action already stored on the server (`__settings__`). */
  settings?: Record<string, unknown>
  /** Pre-submit workflow advice items (`__advice__`). */
  advice?: AdviceItem[]
  /** Updated active model capability summary (`__capabilities__`). */
  capabilities?: ModelCapabilitySummary
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function parseAdviceList(raw: unknown): AdviceItem[] | undefined {
  if (!Array.isArray(raw)) return undefined
  const items: AdviceItem[] = []
  for (const entry of raw) {
    if (!isRecord(entry) || typeof entry.code !== "string" || typeof entry.message !== "string") continue
    const severity: AdviceSeverity = ADVICE_SEVERITIES.has(entry.severity as AdviceSeverity) ? (entry.severity as AdviceSeverity) : "info"
    const patch = isRecord(entry.patch) ? (entry.patch as AdviceItem["patch"]) : null
    const action_label = typeof entry.action_label === "string" ? entry.action_label : null
    const params = sentenceParams(entry.params)
    const sentence = typeof entry.template === "string" && params ? { template: entry.template, params } : {}
    items.push({ code: entry.code, severity, message: entry.message, patch, action_label, ...sentence })
  }
  return items
}

function parseCapabilities(raw: unknown): ModelCapabilitySummary | undefined {
  if (!isRecord(raw) || typeof raw.preset !== "string" || !Array.isArray(raw.capabilities)) return undefined
  return raw as unknown as ModelCapabilitySummary
}

/** Separates reserved keys from param patches and resolves `"owner.param"` keys. */
export function splitResponse(owner: string, response: ActionResponse | unknown): SplitResponse {
  const out: SplitResponse = { params: [] }
  if (!isRecord(response)) return out
  for (const [key, entry] of Object.entries(response)) {
    if (key === "__toast__") {
      if (isRecord(entry) && (typeof entry.message === "string" || isMessage(entry.message))) {
        const level = TOAST_LEVELS.has(entry.level as string) ? (entry.level as ActionToast["level"]) : undefined
        out.toast = { level, message: entry.message }
      }
    } else if (key === "__outputs__") {
      if (isRecord(entry)) out.outputs = entry
    } else if (key === "__settings__") {
      if (isRecord(entry)) out.settings = entry
    } else if (key === "__advice__") {
      const advice = parseAdviceList(entry)
      if (advice !== undefined) out.advice = advice
    } else if (key === "__capabilities__") {
      const capabilities = parseCapabilities(entry)
      if (capabilities !== undefined) out.capabilities = capabilities
    } else if (key.startsWith("__")) {
      // Unknown reserved key: ignore (forward compatible).
    } else if (isRecord(entry)) {
      const patch: ParamPatch = {}
      for (const [field, value] of Object.entries(entry)) {
        if (PATCH_KEYS.has(field)) (patch as Record<string, unknown>)[field] = value
      }
      out.params.push([...splitRef(key, owner), patch])
    }
  }
  return out
}

export type ActionContext = {
  store: FormStore
  /** Form id sent as `tab`: txt2img / img2img / extras or `page:<id>`. */
  tab: string
}

/**
 * The request body: the owner's values, the core form's (for scripts) and
 * every other script's (`scripts`), so cross-owner actions such as
 * `core.__conflicts__` see the form as it is, not its defaults.
 */
export function actionRequest(ctx: ActionContext, owner: string, extra: ActionExtra = {}): ActionRequest {
  const { values } = ctx.store.getState()
  const scripts: Record<string, ParamValues> = {}
  for (const [id, ownerValues] of Object.entries(values)) {
    if (id !== "core" && id !== owner) scripts[id] = ownerValues
  }
  return {
    ...extra,
    tab: ctx.tab,
    values: values[owner] ?? {},
    ...(values.core && owner !== "core" ? { core: values.core } : {}),
    ...(Object.keys(scripts).length > 0 ? { scripts } : {}),
  }
}

const generations = new Map<string, number>()

/**
 * Runs `owner.action` and applies its patch. Responses that were overtaken by
 * a newer call of the same action are dropped (last write wins).
 */
export async function runAction(
  ctx: ActionContext,
  owner: string,
  action: string,
  extra: ActionExtra = {},
): Promise<SplitResponse | null> {
  const key = `${ctx.tab}/${owner}/${action}`
  const generation = (generations.get(key) ?? 0) + 1
  generations.set(key, generation)

  let response: ActionResponse
  try {
    response = await api.action(owner, action, actionRequest(ctx, owner, extra))
  } catch (error) {
    if (generations.get(key) === generation) {
      // the server's details are English, for its log: the status says what failed
      showToast("error", t("Action failed"), errorText(error))
    }
    return null
  }
  if (generations.get(key) !== generation) return null

  const split = splitResponse(owner, response)
  // the conflicts pass (on load, after a paste, on conflict param edits) leaves locks unnamed
  applyResponse(ctx.store, owner, split, { quietLocks: action === CONFLICTS_ACTION })
  return split
}

/** Applies an action's answer; the values a lock kept are named unless `quietLocks`. */
export function applyResponse(store: FormStore, owner: string, split: SplitResponse, { quietLocks = false } = {}) {
  const state = store.getState()
  const kept = split.params.length > 0 ? state.applyPatches(split.params) : []
  if (split.outputs) state.setOutputs(owner, split.outputs)
  if (split.advice !== undefined) state.setAdvice(split.advice)
  if (split.capabilities !== undefined) state.setCapabilities(split.capabilities)
  // the values a lock kept: under the action's own toast, or in one of their own
  const note = quietLocks ? undefined : lockNote(state.index, kept)
  if (split.toast) showToast(split.toast.level, messageText(i18n, split.toast.message), note)
  else if (note) showToast("info", note)
  if (split.settings) {
    // Stored by the server already: update the cached values, never write them back.
    const settings = split.settings
    queryClient.setQueryData<Bootstrap>(keys.app, (app) => (app ? { ...app, options: { ...app.options, ...settings } } : app))
    void queryClient.invalidateQueries({ queryKey: keys.options })
  }
}

/** The core action resolving Anima conflict locks (schema-format.md §4, §5). */
export const CONFLICTS_ACTION = "__conflicts__"
/** The core action returning pre-submit structured workflow advice (`__advice__`). */
export const ADVICE_ACTION = "__advice__"

/** Applies an `AdviceItem`'s one-click patch (form params and/or server `__options__`). */
export async function applyAdvicePatch(ctx: ActionContext, item: AdviceItem): Promise<boolean> {
  if (!item.patch) return false
  const { __options__: optionUpdates, ...rest } = item.patch
  const split = splitResponse("core", rest)
  const changedByOwner = new Map<string, string[]>()
  if (split.params.length > 0) {
    const note = lockNote(ctx.store.getState().index, ctx.store.getState().applyPatches(split.params))
    if (note) showToast("info", note)
    for (const [owner, name] of split.params) {
      const list = changedByOwner.get(owner) ?? []
      if (!list.includes(name)) list.push(name)
      changedByOwner.set(owner, list)
    }
  }
  if (isRecord(optionUpdates) && Object.keys(optionUpdates).length > 0) {
    await api.patchOptions(optionUpdates)
    queryClient.setQueryData<Bootstrap>(keys.app, (app) => (app ? { ...app, options: { ...app.options, ...optionUpdates } } : app))
    void queryClient.invalidateQueries({ queryKey: keys.options })
  }
  if (changedByOwner.size > 0) {
    runBulkChangeEffects(
      ctx,
      [...changedByOwner.entries()].map(([owner, names]) => ({ owner, names })),
    )
  }
  // Remove the resolved chip immediately, then refresh live advice when the server supports capabilities.
  const current = ctx.store.getState().advice
  ctx.store.getState().setAdvice(current.filter((entry) => entry.code !== item.code || entry.message !== item.message))
  if (ctx.store.getState().capabilities !== null) {
    void runAction(ctx, "core", ADVICE_ACTION)
  }
  return true
}

/** Params of one owner a user edit or a reset changed. */
export type ParamChange = { owner: string; names: readonly string[] }

/**
 * Runs what follows a user edit of `names` of `owner` (one field, or a bulk
 * reset): each distinct `on_change` action of those params, and the core
 * `__conflicts__` action when any of them is a conflict param. `selected` is
 * that reference when exactly one changed (the user's last action wins); a
 * bulk reset of several sends none.
 */
export function runChangeEffects(ctx: ActionContext, owner: string, names: readonly string[]) {
  runBulkChangeEffects(ctx, [{ owner, names }])
}

/** `runChangeEffects` over several owners at once: one call per distinct action, one conflicts call. */
export function runBulkChangeEffects(ctx: ActionContext, changes: readonly ParamChange[]) {
  const state = ctx.store.getState()
  const actions = new Map<string, [owner: string, action: string]>()
  const conflicts: string[] = []
  for (const { owner, names } of changes) {
    const nodes = state.index.inputs.get(owner)
    for (const name of names) {
      const action = nodes?.get(name)?.on_change
      if (action) actions.set(`${owner}\0${action}`, [owner, action])
      if (state.conflictParams.has(`${owner}.${name}`)) conflicts.push(`${owner}.${name}`)
    }
  }
  for (const [owner, action] of actions.values()) void runAction(ctx, owner, action)
  if (conflicts.length > 0) {
    void runAction(ctx, "core", CONFLICTS_ACTION, conflicts.length === 1 ? { selected: conflicts[0] } : {})
  }
}

/** Resets `names` of `owner` (all when omitted); returns the names whose value changed. */
function resetOwner(ctx: ActionContext, owner: string, names?: readonly string[]): string[] {
  const before = ctx.store.getState().values[owner] ?? {}
  ctx.store.getState().reset(owner, names)
  const after = ctx.store.getState().values[owner] ?? {}
  return Object.keys(after).filter((name) => (names ? names.includes(name) : true) && after[name] !== before[name])
}

/** Resets `names` of `owner` (all when omitted) to their defaults and runs the change effects. */
export function resetParams(ctx: ActionContext, owner: string, names?: readonly string[]) {
  resetOwners(ctx, [owner], names)
}

/** Resets every param of `owners` (or `names` of each) and runs the change effects once for all of them. */
export function resetOwners(ctx: ActionContext, owners: readonly string[], names?: readonly string[]) {
  const changes = owners.map((owner) => ({ owner, names: resetOwner(ctx, owner, names) })).filter((change) => change.names.length > 0)
  if (changes.length > 0) runBulkChangeEffects(ctx, changes)
}
