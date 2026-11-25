/**
 * One run controller per generation form: holds the `run-machine` state,
 * performs its effects (submit the form, interrupt, skip, cancel) and feeds
 * it the WebSocket job events. Jobs in flight are remembered in session storage,
 * with "Generate forever" and the runs left of "Generate several times", so
 * a reload picks them up again and goes on (`restoreRuns`).
 */
import { toast } from "sonner"
import { create } from "zustand"
import { clientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { idempotencyKey, validationErrors } from "@/api/http"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import type { FormTab, TabSchema } from "@/api/types"
import { onWsEvent } from "@/api/ws"
import { t } from "@/i18n"
import { getFormStore } from "@/ui/form-store"
import { flushPendingWrites } from "@/ui/pending-writes"
import { jobRequest, missingInputs } from "@/ui/tab"
import { errorText, toastInvalid } from "@/features/errors/error-text"
import { notifyResults } from "./effects"
import { answerLoopQueries } from "./loop-owner"
import { noteUpscale } from "@/features/gallery/session-store"
import { IDLE, type JobOutcome, reduce, type RunEffect, type RunEvent, type RunState, type UpscaleSource } from "./run-machine"

export type GenerateTab = Extract<FormTab, "txt2img" | "img2img">
export const GENERATE_FORMS: readonly GenerateTab[] = ["txt2img", "img2img"]

type RunStore = { runs: Record<GenerateTab, RunState> }

export const useRuns = create<RunStore>()(() => ({ runs: { txt2img: IDLE, img2img: IDLE } }))

export const persistKey = (tab: GenerateTab) => `remi-run:${tab}`

/** The most runs "Generate several times" takes. */
export const MAX_RUNS = 999

/**
 * What a page load leaves for the next one in the same tab: the jobs in flight and the loops they
 * belong to. Kept in session storage, which a reload keeps and other tabs do not share: a tab
 * opened while this one loops "Generate forever" must not take the loop over and queue it twice.
 */
export type PersistedRun = { jobs: string[]; forever: boolean; repeatLeft: number }

function persist(tab: GenerateTab, state: RunState) {
  try {
    if (state.jobs.length > 0 || state.forever || state.repeatLeft > 0) {
      const saved: PersistedRun = { jobs: [...state.jobs], forever: state.forever, repeatLeft: state.repeatLeft }
      sessionStorage.setItem(persistKey(tab), JSON.stringify(saved))
    } else sessionStorage.removeItem(persistKey(tab))
  } catch {
    // Storage disabled: restoring after a reload is best-effort.
  }
}

/** The run a previous page load left (also the bare job list older versions wrote). */
export function persistedRun(tab: GenerateTab): PersistedRun {
  const ids = (value: unknown) => (Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : [])
  try {
    const raw = JSON.parse(sessionStorage.getItem(persistKey(tab)) ?? "[]") as unknown
    if (Array.isArray(raw)) return { jobs: ids(raw), forever: false, repeatLeft: 0 }
    const saved = (raw ?? {}) as Partial<Record<keyof PersistedRun, unknown>>
    // at most what the Repeat field gives (a hand-edited 1e9 generated for good)
    const repeatLeft = typeof saved.repeatLeft === "number" && Number.isFinite(saved.repeatLeft) ? Math.min(MAX_RUNS - 1, Math.max(0, Math.floor(saved.repeatLeft))) : 0
    return { jobs: ids(saved.jobs), forever: saved.forever === true, repeatLeft }
  } catch {
    return { jobs: [], forever: false, repeatLeft: 0 }
  }
}

/**
 * Drops the run a previous page load stored for `tab` (another tab's, copied into a duplicated
 * tab), keeping what this page has queued since.
 */
export function forgetStoredRun(tab: GenerateTab) {
  persist(tab, useRuns.getState().runs[tab])
}

/** Job ids a previous page load left in flight. */
export const persistedJobs = (tab: GenerateTab): string[] => persistedRun(tab).jobs

export function dispatchRun(tab: GenerateTab, event: RunEvent) {
  // a press gets its key and time here: the machine tells a double click from a second press by them
  if (event.type === "generate" && event.key === undefined) event = { ...event, key: idempotencyKey(), at: Date.now() }
  const before = useRuns.getState().runs[tab]
  const [after, effects] = reduce(before, event, { interruptOnly: clientOption("ctrl_enter_interrupt") })
  if (after !== before) {
    useRuns.setState((s) => ({ runs: { ...s.runs, [tab]: after } }))
    if (after.jobs !== before.jobs || after.forever !== before.forever || after.repeatLeft !== before.repeatLeft) persist(tab, after)
  }
  for (const effect of effects) perform(tab, effect)
}

function perform(tab: GenerateTab, effect: RunEffect) {
  switch (effect.type) {
    case "submit":
      void submit(tab, effect.key ?? idempotencyKey(), effect.upscale)
      return
    case "interrupt":
      void api.jobs.interrupt(effect.jobId).catch(() => {})
      return
    case "skip":
      void api.jobs.skip(effect.jobId).catch(() => {})
      return
    case "cancel":
      void api.jobs.remove(effect.jobId).catch(() => {})
      return
  }
}

/** Queues the form's job; `key` is its `Idempotency-Key` (a double click sends one key twice: one job). */
async function submit(tab: GenerateTab, key: string, upscale?: UpscaleSource) {
  const store = getFormStore(tab)
  const schema = queryClient.getQueryData<TabSchema>(keys.schema(tab))
  if (!schema) {
    dispatchRun(tab, { type: "rejected", key })
    return
  }
  try {
    // Canvas layers still uploading after the last stroke go out first.
    await flushPendingWrites(tab)
    const request = jobRequest(store, schema)
    const missing = missingInputs(schema, request)
    if (missing.length > 0) {
      // the server would reject it: mark the fields without sending a request that must fail
      store.getState().setErrors(missing)
      toastInvalid(missing, store.getState().index)
      dispatchRun(tab, { type: "rejected", key })
      return
    }
    const created = await api.jobs.create(upscale ? { ...request, upscale_from: upscale.fileId } : request, key)
    store.getState().clearErrors()
    const jobIds = created.job_ids?.length ? created.job_ids : [created.job_id]
    if (upscale) {
      const placement = { source: upscale.key, insert: clientOption("hires_button_gallery_insert"), selectNew: clientOption("hires_insert_index") }
      for (const jobId of jobIds) noteUpscale(jobId, placement)
    }
    for (const jobId of jobIds) useProgress.getState().track(jobId, tab)
    dispatchRun(tab, { type: "submitted", jobIds, key })
    for (const jobId of jobIds) replayEarly(tab, jobId)
  } catch (error) {
    const errors = validationErrors(error)
    if (errors) {
      store.getState().setErrors(errors)
      toastInvalid(errors, store.getState().index)
    } else {
      toast.error(t("Could not queue the job"), { description: errorText(error) })
    }
    dispatchRun(tab, { type: "rejected", key })
  }
}

/**
 * Job events that arrived before the `POST /jobs` answer named their job
 * (a short job can start, even finish, first). Replayed once it is known.
 */
const early = new Map<string, { started?: boolean; outcome?: JobOutcome }>()
const EARLY_LIMIT = 50

function remember(jobId: string, event: { started?: boolean; outcome?: JobOutcome }) {
  early.set(jobId, { ...early.get(jobId), ...event })
  while (early.size > EARLY_LIMIT) early.delete(early.keys().next().value!)
}

function replayEarly(tab: GenerateTab, jobId: string) {
  const seen = early.get(jobId)
  if (!seen) return
  early.delete(jobId)
  if (seen.started) dispatchRun(tab, { type: "started", jobId })
  if (seen.outcome) dispatchRun(tab, { type: "finished", jobId, outcome: seen.outcome })
}

/** The form whose controller owns `jobId`, if any. */
function ownerOf(jobId: string): GenerateTab | null {
  const { runs } = useRuns.getState()
  return GENERATE_FORMS.find((tab) => runs[tab].jobs.includes(jobId)) ?? null
}

/** Whether one of the forms is waiting for a `POST /jobs` answer. */
const submitting = () => GENERATE_FORMS.some((tab) => useRuns.getState().runs[tab].pending > 0)

/**
 * The last jobs each form's loop finished. A tab duplicated just as the loop moved on asks about
 * the job its copy of the storage names, which this tab may have finished a moment before: it is
 * still this tab's loop.
 */
const loopFinished = new Map<GenerateTab, string[]>()
const LOOP_FINISHED_KEPT = 8

function finished(jobId: string, outcome: (tab: GenerateTab | null) => JobOutcome) {
  const tab = ownerOf(jobId)
  if (tab) {
    const run = useRuns.getState().runs[tab]
    if (run.forever || run.repeatLeft > 0) loopFinished.set(tab, [...(loopFinished.get(tab) ?? []), jobId].slice(-LOOP_FINISHED_KEPT))
    dispatchRun(tab, { type: "finished", jobId, outcome: outcome(tab) })
  } else if (submitting()) remember(jobId, { outcome: outcome(null) })
}

let wired = false

/** Subscribes the controllers to job events (once per app). */
export function wireRunControllers() {
  if (wired) return
  wired = true
  // a duplicated tab asks before it resumes a loop this tab runs (restoreRuns)
  answerLoopQueries((tab, jobs) => {
    const run = useRuns.getState().runs[tab]
    const recent = loopFinished.get(tab) ?? []
    return (run.forever || run.repeatLeft > 0) && [...run.jobs, ...recent].some((id) => jobs.includes(id))
  })
  const started = (jobId: string) => {
    const tab = ownerOf(jobId)
    if (tab && useRuns.getState().runs[tab].running !== jobId) dispatchRun(tab, { type: "started", jobId })
    else if (!tab && submitting()) remember(jobId, { started: true })
  }
  onWsEvent("job.started", (e) => started(e.job_id))
  // A reload loses `job.started`: the first progress event marks the job running.
  onWsEvent("job.progress", (e) => started(e.job_id))
  onWsEvent("job.completed", (e) => {
    const stopped = e.status === "interrupted" || e.status === "cancelled"
    if (ownerOf(e.job_id) && !stopped) notifyResults(e.outputs.filter((o) => (o.kind ?? "image") === "image").length)
    finished(e.job_id, () => (stopped ? "interrupted" : "completed"))
  })
  onWsEvent("job.failed", (e) =>
    finished(e.job_id, (tab) =>
      e.status === "interrupted" || e.status === "cancelled" || (tab && useRuns.getState().runs[tab].interrupting) ? "interrupted" : "failed",
    ),
  )
}

/** Hires. fix `source` (a txt2img result) with the txt2img form's values: A1111's ✨ button. */
export function upscaleResult(source: UpscaleSource) {
  dispatchRun("txt2img", { type: "upscale", source })
}

/** Tests: back to idle, nothing persisted. */
export function resetRunControllers() {
  early.clear()
  loopFinished.clear()
  useRuns.setState({ runs: { txt2img: IDLE, img2img: IDLE } })
  for (const tab of GENERATE_FORMS) sessionStorage.removeItem(persistKey(tab))
}
