/**
 * Following a page task (`remi.pages.Page.start_task`): poll the owner's
 * `__task__` action until the task ends; every answer is applied like an
 * action's patch (the status lands in the outputs as `task:<name>`, the
 * final answer carries the task's result). One follower per form, owner and
 * task; it outlives the widget, so a result still lands when the user has
 * left the page meanwhile.
 */
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import type { ActionExtra } from "@/api/types"
import { type ActionContext, actionRequest, applyResponse, splitResponse } from "@/ui/actions"
import type { Message } from "@/ui/messages"

export type TaskStatus = {
  id: string
  name: string
  state: "queued" | "running" | "completed" | "failed" | "cancelled"
  progress: number | null
  text: string | Message | null
  error: string | null
  done: boolean
  cancelling?: boolean
}

export const outputKey = (task: string) => `task:${task}`

export function isTaskStatus(value: unknown): value is TaskStatus {
  if (typeof value !== "object" || value === null) return false
  const status = value as Record<string, unknown>
  return typeof status.id === "string" && typeof status.state === "string" && typeof status.done === "boolean"
}

const followers = new Map<string, Promise<void>>()
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export function isFollowing(ctx: ActionContext, owner: string, task: string): boolean {
  return followers.has(`${ctx.tab}/${owner}/${task}`)
}

/** Asks for the current status only (a view that mounts again); applies it. */
export async function resumeTask(ctx: ActionContext, owner: string, task: string): Promise<TaskStatus | null> {
  try {
    return await poll(ctx, owner, task, { resume: true })
  } catch {
    return null
  }
}

/** Consecutive failed polls after which a follower gives up (about a minute at the 5 s cap). */
export const MAX_POLL_FAILURES = 15

/** Whether the server refused the poll itself (an unknown task, a handler that raises): asking again will not help. */
function refused(error: unknown): boolean {
  return error instanceof ApiError && error.status >= 400 && error.status < 500 && error.status !== 401 && error.status !== 408 && error.status !== 429
}

/** The session ended: the sign-in form is up and the task still runs, so the follower waits for it. */
const signedOut = (error: unknown) => error instanceof ApiError && error.status === 401

/** Marks the followed task failed in the form, so the widget stops showing it as running. */
function giveUp(ctx: ActionContext, owner: string, task: string, id: string, error: unknown) {
  const current = ctx.store.getState().outputs[owner]?.[outputKey(task)]
  if (isTaskStatus(current) && current.id !== id) return // a newer task is shown: not this follower's to fail
  const base = isTaskStatus(current) ? current : { id, name: task, progress: null, text: null }
  const message = error instanceof Error ? error.message : String(error)
  ctx.store.getState().setOutputs(owner, { [outputKey(task)]: { ...base, state: "failed", done: true, error: message } satisfies TaskStatus })
}

/**
 * Polls until task `id` ends (`interval` ms apart; network errors back off up
 * to 5 s and never toast). Resolves when it has ended or was replaced, or
 * when the server refuses the poll or keeps failing: the task then shows as
 * failed instead of running forever.
 */
export function followTask(ctx: ActionContext, owner: string, task: string, id: string, interval = 750): Promise<void> {
  const key = `${ctx.tab}/${owner}/${task}`
  const running = followers.get(key)
  if (running) return running
  const loop = (async () => {
    let delay = interval
    let failures = 0
    for (;;) {
      await sleep(delay)
      try {
        const status = await poll(ctx, owner, task, { id })
        delay = interval
        failures = 0
        if (!status || status.done || status.id !== id) return
      } catch (error) {
        if (signedOut(error)) {
          delay = Math.min(delay * 2, 5000) // not a failure: asked again until the user is back
          continue
        }
        failures += 1
        if (refused(error) || failures >= MAX_POLL_FAILURES) {
          giveUp(ctx, owner, task, id, error)
          return
        }
        delay = Math.min(delay * 2, 5000)
      }
    }
  })().finally(() => followers.delete(key))
  followers.set(key, loop)
  return loop
}

async function poll(ctx: ActionContext, owner: string, task: string, extra: Record<string, unknown>): Promise<TaskStatus | null> {
  // SAFETY: the task's own arguments travel in the action body as the server's task runner reads
  // them; ActionExtra types only the core actions' extras, so the body is widened to it.
  const body = actionRequest(ctx, owner, { task, ...extra } as unknown as ActionExtra)
  const split = splitResponse(owner, await api.action(owner, "__task__", body))
  applyResponse(ctx.store, owner, split)
  const status = split.outputs?.[outputKey(task)]
  return isTaskStatus(status) ? status : null
}
