/**
 * "Send to" and "use these params": move a result (its image and/or its
 * generation parameters) into a form, switching to that form's page when
 * needed. Values are applied once the target form has loaded its schema, so
 * a page that mounts later still receives them (upload ids are not kept in
 * drafts, which a load would otherwise drop).
 */
import { clientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import type { FormTab, ParamValues } from "@/api/types"
import { currentRoute, navigate, type Route, routeKey } from "@/app/router"
import { applyPaste, type PasteEntry, type PasteFilter, type PasteResult, parsedEntries, planEntries } from "@/features/infotext/paste"
import { runChangeEffects } from "@/ui/actions"
import type { FormIndex } from "@/ui/form-index"
import { getFormStore } from "@/ui/form-store"
import { describeFile } from "@/ui/uploads"

export type SendTarget = "txt2img" | "img2img" | "inpaint" | "extras"

/** The form a target fills. */
export const formOf = (target: SendTarget): FormTab => (target === "inpaint" ? "img2img" : target)

export function routeOf(tab: FormTab): Route {
  return tab === "extras" ? { name: "extras" } : { name: "generate", tab }
}

/** The core values that put an uploaded image into a target form (schema-format.md core names). */
export function imageEntry(target: SendTarget, fileId: string): PasteEntry | null {
  switch (target) {
    case "img2img":
      return { owner: "core", values: { init_img: fileId, mode: "img2img" } }
    case "inpaint":
      return { owner: "core", values: { init_img: fileId, mode: "Inpaint" } }
    case "extras":
      return { owner: "core", values: { image: fileId, mode: "Single Image" } }
    case "txt2img":
      return null
  }
}

/**
 * Image layers drawn over a source image (sketch, mask: image params whose
 * `source` names it) belong to that image: entries that set a new source
 * also clear its layers, unless they set them too.
 */
export function withLayersCleared(index: FormIndex, entries: readonly PasteEntry[]): PasteEntry[] {
  return entries.map((entry) => {
    const cleared: ParamValues = {}
    for (const [name, node] of index.inputs.get(entry.owner) ?? []) {
      if (node.type === "image" && node.source && node.source in entry.values && !(name in entry.values)) cleared[name] = null
    }
    return Object.keys(cleared).length > 0 ? { owner: entry.owner, values: { ...entry.values, ...cleared } } : entry
  })
}

type Delivery = { entries: PasteEntry[]; filter: PasteFilter; unsubscribe: () => void; resolve: (result: PasteResult) => void }
const pending = new Map<FormTab, Delivery>()

function apply(tab: FormTab, entries: readonly PasteEntry[], filter: PasteFilter): PasteResult {
  const store = getFormStore(tab)
  const { index } = store.getState()
  const plan = planEntries(index, withLayersCleared(index, entries), filter)
  return applyPaste({ store, tab }, plan.entries)
}

/**
 * Forms whose page keeps a loaded store across visits (the generate view
 * and Extras load a schema once and carry uploads over reloads). Other pages load
 * their schema again on every mount, which drops upload ids.
 */
const KEEPS_VALUES: ReadonlySet<FormTab> = new Set(["txt2img", "img2img", "extras"])

/**
 * Applies `entries` to form `tab` and shows its page: right away when the
 * form is loaded (and on screen, or a form that keeps its values),
 * otherwise after its next schema load. Resolves with the number of params set.
 */
export function deliver(tab: FormTab, entries: readonly PasteEntry[], filter: PasteFilter = {}): Promise<number> {
  return deliverPaste(tab, entries, filter).then((result) => result.count)
}

/** `deliver`, resolving with the params set and the ones a lock kept (`applyPaste`). */
export function deliverPaste(tab: FormTab, entries: readonly PasteEntry[], filter: PasteFilter = {}): Promise<PasteResult> {
  const store = getFormStore(tab)
  const onScreen = routeKey(currentRoute()) === routeKey(routeOf(tab))
  if (store.getState().revision > 0 && (onScreen || KEEPS_VALUES.has(tab))) {
    pending.get(tab)?.unsubscribe()
    pending.delete(tab)
    const result = apply(tab, entries, filter)
    if (!onScreen) navigate(routeOf(tab))
    return Promise.resolve(result)
  }

  const previous = pending.get(tab)
  previous?.unsubscribe()
  previous?.resolve({ count: 0, locked: [] })
  return new Promise<PasteResult>((resolve) => {
    const unsubscribe = store.subscribe((s, prev) => {
      if (s.revision === prev.revision) return
      unsubscribe()
      pending.delete(tab)
      resolve(apply(tab, entries, filter))
    })
    pending.set(tab, { entries: [...entries], filter, unsubscribe, resolve })
    navigate(routeOf(tab))
  })
}

/** A result to send: its file, its parameters, and its pixel size when the output metadata has it. */
export type SendItem = { fileId?: string | null; infotext?: string | null; width?: number | null; height?: number | null }

/** The `send_seed` / `send_size` / `send_cfg` settings as a paste filter. */
export function sendFilter(): PasteFilter {
  return { seed: clientOption("send_seed"), size: clientOption("send_size"), cfg: clientOption("send_cfg") }
}

/**
 * Sends a result to a form: its image (output file ids are valid image
 * inputs) and, for the generation forms, its parameters (`send_seed` /
 * `send_size` / `send_cfg` respected).
 */
export async function sendTo(target: SendTarget, item: SendItem, filter: PasteFilter = sendFilter()): Promise<number> {
  const tab = formOf(target)
  const parsed = item.infotext && tab !== "extras" ? await api.parseInfotext(item.infotext, tab).catch(() => null) : null
  const entries = parsed ? parsedEntries(parsed) : []
  const image = item.fileId ? imageEntry(target, item.fileId) : null
  if (image && item.fileId) {
    entries.push(image)
    describeFile(item.fileId, item)
  }
  return deliver(tab, entries, filter)
}

export type ReferenceTarget = {
  owner: string
  param: string
  enableParam?: string
}

/**
 * Finds the ImageStitch reference list input on a loaded form: prefers an owner
 * whose id matches `imagestitch` / `image_stitch`, else any script owner with
 * an `image_list` input named `references`.
 */
export function findReferenceTarget(index: FormIndex): ReferenceTarget | null {
  let fallback: ReferenceTarget | null = null
  for (const [owner, nodes] of index.inputs) {
    if (owner === "core") continue
    const isStitch = /image_?stitch/i.test(owner)
    const refNode = nodes.get("references")
    const listEntry =
      refNode?.type === "image_list"
        ? (["references", refNode] as const)
        : isStitch
          ? [...nodes.entries()].find(([, node]) => node.type === "image_list")
          : undefined
    if (!listEntry) continue
    const enableParam = nodes.get("enable")?.type === "toggle" ? "enable" : undefined
    const target: ReferenceTarget = { owner, param: listEntry[0], enableParam }
    if (isStitch) return target
    fallback ??= target
  }
  return fallback
}

export type AppendReferenceResult = {
  owner: string | null
  added: number
  total: number
}

function applyReferences(tab: "txt2img" | "img2img", items: readonly SendItem[]): AppendReferenceResult {
  const store = getFormStore(tab)
  const state = store.getState()
  const target = findReferenceTarget(state.index)
  if (!target) return { owner: null, added: 0, total: 0 }

  const rawExisting = state.values[target.owner]?.[target.param]
  const existing = Array.isArray(rawExisting) ? rawExisting.filter((entry): entry is string => typeof entry === "string" && entry.length > 0) : []
  const seen = new Set(existing)
  const incoming: string[] = []
  for (const item of items) {
    if (!item.fileId || seen.has(item.fileId)) continue
    seen.add(item.fileId)
    incoming.push(item.fileId)
    describeFile(item.fileId, item)
  }

  const nextList = [...existing, ...incoming]
  const values: ParamValues = { [target.param]: nextList }
  const changed: string[] = []
  if (incoming.length > 0) changed.push(target.param)
  if (target.enableParam && !state.values[target.owner]?.[target.enableParam]) {
    values[target.enableParam] = true
    changed.push(target.enableParam)
  }

  if (changed.length > 0) {
    state.setValues(target.owner, values)
    const reveal = state.index.reveal.get(target.owner)?.get(target.param)
    for (const key of reveal?.open ?? []) state.setOpen(key, true)
    for (const [key, value] of reveal?.tabs ?? []) state.setTab(key, value)
    runChangeEffects({ store, tab }, target.owner, changed)
  }

  return { owner: target.owner, added: incoming.length, total: nextList.length }
}

const pendingRefs = new Map<"txt2img" | "img2img", { unsubscribe: () => void; resolve: (res: AppendReferenceResult) => void }>()

/**
 * Appends one or more images to the active generation form's ImageStitch
 * reference list without clobbering existing references. Waits for the form's
 * schema to load if it has not been visited yet.
 */
export function sendToReference(items: readonly SendItem[], preferredTab?: "txt2img" | "img2img"): Promise<AppendReferenceResult> {
  const route = currentRoute()
  const tab: "txt2img" | "img2img" = preferredTab ?? (route.name === "generate" ? route.tab : "txt2img")
  const valid = items.filter((item) => Boolean(item.fileId))
  if (valid.length === 0) return Promise.resolve({ owner: null, added: 0, total: 0 })

  const store = getFormStore(tab)
  const onScreen = routeKey(route) === routeKey(routeOf(tab))
  if (store.getState().revision > 0) {
    pendingRefs.get(tab)?.unsubscribe()
    pendingRefs.delete(tab)
    const result = applyReferences(tab, valid)
    if (!onScreen) navigate(routeOf(tab))
    return Promise.resolve(result)
  }

  const previous = pendingRefs.get(tab)
  previous?.unsubscribe()
  previous?.resolve({ owner: null, added: 0, total: 0 })
  return new Promise<AppendReferenceResult>((resolve) => {
    const unsubscribe = store.subscribe((s, prev) => {
      if (s.revision === prev.revision) return
      unsubscribe()
      pendingRefs.delete(tab)
      resolve(applyReferences(tab, valid))
    })
    pendingRefs.set(tab, { unsubscribe, resolve })
    navigate(routeOf(tab))
  })
}
