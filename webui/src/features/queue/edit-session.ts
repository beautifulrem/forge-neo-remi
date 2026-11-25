/**
 * Opening a job's parameters in its form (txt2img, img2img or Extras). For a
 * queued job this starts an *edit*: the form shows a banner, and "Update job"
 * replaces the queued job's parameters with the form's (`PUT
 * /jobs/{id}/params`) instead of queueing a new one. Finished jobs (and
 * imported ones that cannot be replayed) simply load their parameters.
 */
import { toast } from "sonner"
import { create } from "zustand"
import { createJSONStorage, persist } from "zustand/middleware"
import { api } from "@/api/endpoints"
import { ApiError, validationErrors } from "@/api/http"
import { keys, queryClient } from "@/api/queries"
import type { FormTab, JobDetail, ParamValues, TabSchema } from "@/api/types"
import { navigate } from "@/app/router"
import type { PasteEntry } from "@/features/infotext/paste"
import { deliver } from "@/features/send-to/send-to"
import { errorText, toastInvalid } from "@/features/errors/error-text"
import { DOUBLE_PRESS_MS } from "@/features/progress/run-machine"
import { t } from "@/i18n"
import { isRecord } from "@/lib/persist"
import { getFormStore } from "@/ui/form-store"
import { lockNote } from "@/ui/locks"
import { flushPendingWrites } from "@/ui/pending-writes"
import { jobRequest, missingInputs } from "@/ui/tab"

export type EditSession = { jobId: string; seq: number; tab: FormTab; title: string }

type EditState = {
  sessions: Partial<Record<FormTab, EditSession>>
  saving: Partial<Record<FormTab, boolean>>
}

const FORM_TABS: readonly FormTab[] = ["txt2img", "img2img", "extras"]

/** A stored edit of the shape written, under its own form (session storage edited by hand or by another app). */
function isSession(tab: string, value: unknown): value is EditSession {
  return (
    isRecord(value) &&
    value.tab === tab &&
    FORM_TABS.includes(tab as FormTab) &&
    typeof value.jobId === "string" &&
    typeof value.seq === "number" &&
    Number.isFinite(value.seq) &&
    typeof value.title === "string"
  )
}

/**
 * The edits in progress, kept in session storage: a reload keeps the form's draft (the job's
 * values), so it keeps the edit too, or Generate would queue the job a second time instead of
 * updating it. A job that started meanwhile ends the edit when it is saved (409).
 */
export const useQueueEdit = create<EditState>()(
  persist(() => ({ sessions: {}, saving: {} }) as EditState, {
    name: "remi-queue-edit",
    version: 1,
    storage: createJSONStorage(() => sessionStorage),
    partialize: ({ sessions }) => ({ sessions }),
    merge: (persisted, current) => {
      const sessions = isRecord(persisted) && isRecord(persisted.sessions) ? persisted.sessions : {}
      return { ...current, sessions: Object.fromEntries(Object.entries(sessions).filter(([tab, session]) => isSession(tab, session))) }
    },
  }),
)

/** One paste entry per owner: the job's core values, then each script's. */
export function jobEntries(job: Pick<JobDetail, "core" | "scripts">): PasteEntry[] {
  const scripts = Object.entries(job.scripts ?? {}).filter((entry): entry is [string, ParamValues] => entry[1] !== null && typeof entry[1] === "object" && !Array.isArray(entry[1]))
  return [{ owner: "core", values: job.core ?? {} }, ...scripts.map(([owner, values]) => ({ owner, values }))]
}

/** The failure as text in the UI language; rejected values are marked on the form itself. */
const message = (error: unknown) => errorText(error, { 404: t("The job was removed meanwhile."), 422: t("Check the marked fields.") })

/** Loads a job's parameters into its form (and, while it is queued, edits it). */
export async function openJob(jobId: string): Promise<void> {
  const id = toast.loading(t("Opening the job…"))
  try {
    const job = await api.jobs.get(jobId)
    const editing = job.status === "queued"
    useQueueEdit.setState((s) => {
      const sessions = { ...s.sessions }
      if (editing) sessions[job.tab] = { jobId: job.id, seq: job.seq, tab: job.tab, title: job.name || job.summary || "" }
      else delete sessions[job.tab]
      return { sessions }
    })
    const { count, locked } = await deliver(job.tab, jobEntries(job))
    const note = lockNote(getFormStore(job.tab).getState().index, locked)
    toast.success(editing ? t("Editing queued job #{{seq}}", { seq: job.seq }) : t("Parameters applied: {{count}}", { count }), { id, ...(note ? { description: note } : {}) })
  } catch (error) {
    toast.error(t("Could not open the job"), { id, description: message(error) })
  }
}

export function cancelEdit(tab: FormTab) {
  useQueueEdit.setState((s) => {
    const sessions = { ...s.sessions }
    delete sessions[tab]
    return { sessions }
  })
}

/** Writes the form's values into the edited queued job; true when it was updated. */
export async function saveEdit(tab: FormTab): Promise<boolean> {
  const session = useQueueEdit.getState().sessions[tab]
  const schema = queryClient.getQueryData<TabSchema>(keys.schema(tab))
  if (!session || !schema) return false
  const store = getFormStore(tab)
  // canvas layers still uploading after the last stroke go out first, as for Generate
  try {
    await flushPendingWrites(tab)
  } catch (error) {
    toast.error(t("Could not update the job"), { description: errorText(error) })
    return false
  }
  const { core, scripts } = jobRequest(store, schema)
  const missing = missingInputs(schema, { core })
  if (missing.length > 0) {
    store.getState().setErrors(missing)
    toastInvalid(missing, store.getState().index)
    return false
  }
  useQueueEdit.setState((s) => ({ saving: { ...s.saving, [tab]: true } }))
  try {
    await api.jobs.updateParams(session.jobId, { core, scripts })
    store.getState().clearErrors()
    cancelEdit(tab)
    void queryClient.invalidateQueries({ queryKey: keys.jobs })
    void queryClient.invalidateQueries({ queryKey: keys.queueItems })
    toast.success(t("Queued job #{{seq}} updated", { seq: session.seq }), { action: { label: t("Open queue"), onClick: () => navigate({ name: "queue" }) } })
    return true
  } catch (error) {
    const errors = validationErrors(error)
    if (errors) {
      store.getState().setErrors(errors)
      toastInvalid(errors, store.getState().index)
    } else if (error instanceof ApiError && (error.status === 409 || error.status === 404)) {
      cancelEdit(tab)
      toast.error(t("Job #{{seq}} already started or was removed", { seq: session.seq }), {
        description: t("Your changes are still in the form: generate or add them to the queue as a new job."),
      })
    } else {
      toast.error(t("Could not update the job"), { description: message(error) })
    }
    return false
  } finally {
    useQueueEdit.setState((s) => ({ saving: { ...s.saving, [tab]: false } }))
  }
}

/** The newest "Add to queue" press of each form still waiting for its answer. */
const pressed = new Map<FormTab, { at: number; job: Promise<string | null> }>()

/**
 * Queues the form as it is, without tracking it in the form's run (the "Add to queue" button,
 * Ctrl/Cmd+Shift+Enter). A press within `DOUBLE_PRESS_MS` of one still waiting for its answer is
 * a double click, as for Generate: it joins that press (its request, its `Idempotency-Key`, its
 * toast) and resolves with the same job.
 */
export function enqueueForm(tab: FormTab): Promise<string | null> {
  const at = Date.now()
  const last = pressed.get(tab)
  if (last && at - last.at >= 0 && at - last.at < DOUBLE_PRESS_MS) return last.job
  const job = enqueue(tab)
  const press = { at, job }
  pressed.set(tab, press)
  const answered = () => {
    if (pressed.get(tab) === press) pressed.delete(tab)
  }
  void job.then(answered, answered)
  return job
}

async function enqueue(tab: FormTab): Promise<string | null> {
  const schema = queryClient.getQueryData<TabSchema>(keys.schema(tab))
  if (!schema) return null
  const store = getFormStore(tab)
  try {
    await flushPendingWrites(tab)
  } catch (error) {
    toast.error(t("Could not queue the job"), { description: errorText(error) })
    return null
  }
  const request = jobRequest(store, schema)
  const missing = missingInputs(schema, request)
  if (missing.length > 0) {
    store.getState().setErrors(missing)
    toastInvalid(missing, store.getState().index)
    return null
  }
  try {
    const created = await api.jobs.create(request)
    store.getState().clearErrors()
    const queued = queryClient.getQueryData<unknown[]>(keys.queueItems)?.length
    toast.success(t("Added to the queue as #{{seq}}", { seq: created.seq }), {
      description: queued ? t("{{count}} jobs were waiting", { count: queued }) : undefined,
      action: { label: t("Open queue"), onClick: () => navigate({ name: "queue" }) },
    })
    void queryClient.invalidateQueries({ queryKey: keys.queueItems })
    return created.job_id
  } catch (error) {
    const errors = validationErrors(error)
    if (errors) {
      store.getState().setErrors(errors)
      toastInvalid(errors, store.getState().index)
    } else {
      toast.error(t("Could not queue the job"), { description: message(error) })
    }
    return null
  }
}
