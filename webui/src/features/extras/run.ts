/**
 * Extras runs: submit the form as an `extras` job, follow it through the
 * progress store (its results arrive by WebSocket), interrupt it, and keep
 * this session's results with the source image each one came from (for the
 * before / after comparison).
 */
import { toast } from "sonner"
import { create } from "zustand"
import { persist } from "zustand/middleware"
import { api } from "@/api/endpoints"
import { ApiError, fieldMessages, validationErrors } from "@/api/http"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import type { Job, JobOutput, TabSchema } from "@/api/types"
import { onWsEvent } from "@/api/ws"
import { errorText, toastInvalid } from "@/features/errors/error-text"
import { t } from "@/i18n"
import { getFormStore } from "@/ui/form-store"
import { flushPendingWrites } from "@/ui/pending-writes"
import type { FieldMessage } from "@/ui/schema"
import { jobRequest, missingInputs } from "@/ui/tab"
import { bestEffortStorage, isRecord, isText, keepEntries, mergeTyped } from "@/lib/persist"

export type ExtrasResult = {
  jobId: string
  outputs: JobOutput[]
  /** The single source image (upload / output id), when the run had one. */
  source: string | null
  info: string | null
  error?: string
  /** The values a failed run refused when it ran (shown translated). */
  errors?: FieldMessage[]
  at: number
}

type ExtrasRunState = {
  submitting: boolean
  /** Jobs this page queued that have not finished, oldest first. */
  jobs: string[]
  sources: Record<string, string | null>
  /** Finished runs, newest first. */
  results: ExtrasResult[]
  /** The result shown (job id); null = the newest. */
  selected: string | null
}

const KEEP = 30

export const useExtrasRuns = create<ExtrasRunState>()(
  persist(() => ({ submitting: false, jobs: [], sources: {}, results: [], selected: null }) as ExtrasRunState, {
    name: "remi-extras-runs",
    version: 1,
    storage: bestEffortStorage(() => sessionStorage),
    partialize: ({ results, jobs, sources }) => ({ results, jobs, sources }),
    merge: (persisted, current) => {
      const merged = mergeTyped(persisted, current)
      return {
        ...merged,
        jobs: merged.jobs.filter((job) => typeof job === "string"),
        sources: keepEntries(merged.sources, (source): source is string | null => source === null || typeof source === "string"),
        results: merged.results.filter(isResult).map(({ errors, ...result }) => (errors === undefined ? result : { ...result, errors: fieldMessages(errors) })),
      }
    },
  }),
)

/** A stored run the page can show (sessionStorage edited by hand or written by another app). */
function isResult(value: unknown): value is ExtrasResult {
  return (
    isRecord(value) &&
    typeof value.jobId === "string" &&
    typeof value.at === "number" &&
    isText(value.source) &&
    isText(value.info) &&
    isText(value.error) &&
    (value.errors === undefined || Array.isArray(value.errors)) &&
    Array.isArray(value.outputs) &&
    value.outputs.every(isOutput)
  )
}

/** A size in pixels, or none stored. A function declaration: `persist` rehydrates inside `create()` above, before a const below it exists. */
function isPixels(value: unknown): boolean {
  return value === null || value === undefined || (typeof value === "number" && Number.isInteger(value) && value >= 0)
}

/** A stored output: every field the page renders or passes on is of its kind (an object as a React child crashed the page). */
function isOutput(value: unknown): value is JobOutput {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.kind === "string" &&
    isPixels(value.width) &&
    isPixels(value.height) &&
    isText(value.infotext) &&
    isText(value.role) &&
    (value.missing === null || value.missing === undefined || typeof value.missing === "boolean")
  )
}

/** The source image of the form when it processes one image ("Single Image"). */
export function singleSource(core: Record<string, unknown>): string | null {
  const mode = core.mode ?? "Single Image"
  return mode === "Single Image" && typeof core.image === "string" ? core.image : null
}

export async function runExtras(): Promise<void> {
  const schema = queryClient.getQueryData<TabSchema>(keys.schema("extras"))
  const state = useExtrasRuns.getState()
  if (!schema || state.submitting) return
  const store = getFormStore("extras")
  // submitting from the press: a second one while an upload lands queues nothing more
  useExtrasRuns.setState({ submitting: true })
  try {
    // a source image still uploading goes into the job, instead of "an image is required"
    await flushPendingWrites("extras")
    const body = jobRequest(store, schema)
    const missing = missingInputs(schema, body)
    if (missing.length > 0) {
      store.getState().setErrors(missing)
      toastInvalid(missing)
      return
    }
    const created = await api.jobs.create(body)
    store.getState().clearErrors()
    const ids = created.job_ids?.length ? created.job_ids : [created.job_id]
    const source = singleSource(body.core)
    useExtrasRuns.setState((s) => ({ jobs: [...s.jobs, ...ids], sources: { ...s.sources, ...Object.fromEntries(ids.map((id) => [id, source])) } }))
    for (const id of ids) useProgress.getState().track(id, "extras")
  } catch (error) {
    const errors = validationErrors(error)
    if (errors) {
      store.getState().setErrors(errors)
      toastInvalid(errors)
    } else {
      toast.error(t("Could not queue the job"), { description: errorText(error) })
    }
  } finally {
    useExtrasRuns.setState({ submitting: false })
  }
}

/** Interrupts the running Extras job and cancels the queued ones of this page. */
export function interruptExtras() {
  const { jobs } = useExtrasRuns.getState()
  const running = useProgress.getState().running?.jobId
  for (const id of jobs) void (id === running ? api.jobs.interrupt(id) : api.jobs.remove(id)).catch(() => {})
}

/** Records a finished job of this page (from the progress store). */
export function finishExtras(result: { jobId: string; outputs: JobOutput[]; infotext?: string | null; error?: string; errors?: FieldMessage[] }) {
  useExtrasRuns.setState((s) => {
    if (!s.jobs.includes(result.jobId) && !(result.jobId in s.sources)) return s
    const finished: ExtrasResult = {
      jobId: result.jobId,
      outputs: result.outputs,
      source: s.sources[result.jobId] ?? null,
      info: result.infotext ?? null,
      error: result.error,
      ...(result.errors?.length ? { errors: result.errors } : {}),
      at: Date.now(),
    }
    const sources = { ...s.sources }
    delete sources[result.jobId]
    return {
      jobs: s.jobs.filter((id) => id !== result.jobId),
      sources,
      results: [finished, ...s.results.filter((item) => item.jobId !== result.jobId)].slice(0, KEEP),
      selected: null,
    }
  })
}

let wired = false

/** Feeds finished Extras jobs into the run store (once per app). */
export function wireExtrasRuns() {
  if (wired) return
  wired = true
  useProgress.subscribe((s, prev) => {
    const job = s.last.extras
    if (job && job !== prev.last.extras) finishExtras(job)
  })
  // Jobs still in flight when the page was reloaded: follow them again.
  for (const id of useExtrasRuns.getState().jobs) useProgress.getState().track(id, "extras")
  void resyncExtras()
  // a reconnect: a job that ended while the socket was down sent its event to nobody
  onWsEvent("hello", () => void resyncExtras())
}

/**
 * Asks the server about this page's jobs in flight: an ended one records its
 * result, one the server no longer knows (404) its error. Any other failure
 * (the server restarting) leaves the job followed until the next `hello`.
 */
export async function resyncExtras(): Promise<void> {
  await Promise.all(
    useExtrasRuns.getState().jobs.map(async (id) => {
      let job: Job
      try {
        job = await api.jobs.get(id)
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) finishExtras({ jobId: id, outputs: [], error: t("The job is gone") })
        return
      }
      if (job.status === "queued" || job.status === "running") return
      finishExtras({ jobId: id, outputs: job.outputs, infotext: job.infotext, error: job.status === "failed" ? (job.error ?? undefined) : undefined })
    }),
  )
}

/** Tests: forget everything. */
export function resetExtrasRuns() {
  useExtrasRuns.setState({ submitting: false, jobs: [], sources: {}, results: [], selected: null })
}
