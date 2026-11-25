/**
 * Generate / Interrupt / Skip state machine of one form, as a pure reducer:
 * `(state, event) → [state, effects]`. The controller (`run-controller.ts`)
 * feeds it user input and WebSocket job events and performs the effects.
 *
 * - Generate queues a job; pressing it again while one runs queues another.
 *   A second press within `DOUBLE_PRESS_MS` of one whose submit has not
 *   answered yet is a double click: it is sent with the first press's
 *   `Idempotency-Key`, so the server queues one job and answers both. The
 *   press is one submit: its first answer settles it, the other is ignored.
 * - Ctrl/Cmd+Enter while this form's job runs interrupts it and, unless the
 *   setting `ctrl_enter_interrupt` is on, generates again once it has ended.
 * - Interrupt stops the running job, cancels this form's queued jobs and
 *   ends "generate forever" and repeats. Skip only skips the current image.
 * - "Generate forever" and "repeat N" queue the next job when the previous
 *   one has finished (so each run picks up edits); a failure stops them. A
 *   reload keeps them going (`resume`, after the jobs are restored).
 */

export type RunPhase = "idle" | "submitting" | "queued" | "running" | "interrupting"

export type RunState = {
  /** Jobs this form queued that have not finished, in queue order. */
  jobs: readonly string[]
  /** The job of `jobs` the worker is running. */
  running: string | null
  /** Submits in flight (Generate can be pressed again meanwhile); a double click is one. */
  pending: number
  interrupting: boolean
  forever: boolean
  /** Runs still to queue, one after another (repeat N). */
  repeatLeft: number
  /** Generate again when the interrupted job has ended (Ctrl/Cmd+Enter). */
  restart: boolean
  /** In-flight submits to cancel when they answer (interrupted while they were pending). */
  dropPending: number
  /** Generate presses with requests in flight, the newest last: a press soon after the newest reuses its key. */
  presses: readonly Press[]
}

/**
 * A Generate press: its submit's `Idempotency-Key`, when it was pressed, its requests in flight
 * (2 for a double click) and whether one of them answered with its jobs.
 */
export type Press = { key: string; at: number; requests: number; answered: boolean }

/** A second Generate press this soon (ms) after a pending one is a double click, not another job. */
export const DOUBLE_PRESS_MS = 400

export type JobOutcome = "completed" | "failed" | "interrupted"

export type RunEvent =
  /**
   * `runs` > 1: generate that many times in a row. `key` (a fresh `Idempotency-Key`) and `at`
   * (`Date.now()`) let a double click be told from a second press (the controller sets them).
   */
  | { type: "generate"; runs?: number; key?: string; at?: number }
  | { type: "shortcut" }
  /** A `POST /jobs` answer: the jobs it queued (several with `repeat`); `key` is the one it was sent with. */
  | { type: "submitted"; jobIds: readonly string[]; key?: string }
  | { type: "rejected"; key?: string }
  | { type: "started"; jobId: string }
  | { type: "finished"; jobId: string; outcome: JobOutcome }
  | { type: "interrupt" }
  | { type: "skip" }
  | { type: "forever"; on: boolean }
  /** Drop the remaining repeats (the current run goes on). */
  | { type: "stop-repeats" }
  /** Jobs found again after a page reload. */
  | { type: "restore"; jobs: readonly string[]; running: string | null }
  /**
   * The loops ("Generate forever", runs left of "Generate several times") a
   * previous page load had going: they go on after the restored jobs, if one
   * of them is still in flight.
   */
  | { type: "resume"; forever: boolean; repeatLeft: number }
  /** Hires. fix an existing result with the form's values (txt2img's ✨ button): one job. */
  | { type: "upscale"; source: UpscaleSource }

/** The result a Hires. fix-only run starts from. */
export type UpscaleSource = { fileId: string; key: string }

export type RunEffect =
  /** `key`: the `Idempotency-Key` to send (a fresh one when absent). */
  | { type: "submit"; upscale?: UpscaleSource; key?: string }
  | { type: "interrupt"; jobId: string }
  | { type: "skip"; jobId: string }
  /** Remove a queued (not yet started) job. */
  | { type: "cancel"; jobId: string }

export type RunOptions = {
  /** `ctrl_enter_interrupt`: Ctrl/Cmd+Enter only interrupts. */
  interruptOnly: boolean
}

export const IDLE: RunState = {
  jobs: [],
  running: null,
  pending: 0,
  interrupting: false,
  forever: false,
  repeatLeft: 0,
  restart: false,
  dropPending: 0,
  presses: [],
}

export function phaseOf(s: RunState): RunPhase {
  if (s.interrupting) return "interrupting"
  if (s.running) return "running"
  if (s.jobs.length > 0) return "queued"
  if (s.pending > 0) return "submitting"
  return "idle"
}

/** Busy = something of this form is queued, running or being submitted. */
export const isBusy = (s: RunState) => s.pending > 0 || s.jobs.length > 0

type Result = [RunState, RunEffect[]]

function submit(s: RunState): Result {
  return [{ ...s, pending: s.pending + 1 }, [{ type: "submit" }]]
}

/** A Generate press: a double click sends the pending press again (its key), another press is a submit of its own. */
function press(s: RunState, key: string | undefined, at: number | undefined): Result {
  if (key === undefined || at === undefined) return submit(s)
  const last = s.presses.at(-1)
  if (last && !last.answered && at - last.at >= 0 && at - last.at < DOUBLE_PRESS_MS) {
    const presses = [...s.presses.slice(0, -1), { ...last, requests: last.requests + 1 }]
    return [{ ...s, presses }, [{ type: "submit", key: last.key }]]
  }
  return [{ ...s, pending: s.pending + 1, presses: [...s.presses, { key, at, requests: 1, answered: false }] }, [{ type: "submit", key }]]
}

/**
 * One request of a press answered (`ok`: with its jobs): the press's other requests are left in
 * flight. `first`: the answer settles the press (its first with jobs, or the last of a refused
 * press); any other answer only counts down its requests.
 */
function answer(s: RunState, key: string | undefined, ok: boolean): { presses: readonly Press[]; first: boolean } {
  const index = key === undefined ? -1 : s.presses.findIndex((p) => p.key === key)
  if (index < 0) return { presses: s.presses, first: true } // a submit of its own (loops, upscale) or no key
  const p = s.presses[index]!
  const requests = p.requests - 1
  const first = !p.answered && (ok || requests === 0)
  const rest = s.presses.filter((_, i) => i !== index)
  const presses = requests > 0 ? [...rest.slice(0, index), { ...p, requests, answered: p.answered || ok }, ...rest.slice(index)] : rest
  return { presses, first }
}

function interrupt(s: RunState): Result {
  const effects: RunEffect[] = []
  if (s.running) effects.push({ type: "interrupt", jobId: s.running })
  const queued = s.jobs.filter((id) => id !== s.running)
  for (const jobId of queued) effects.push({ type: "cancel", jobId })
  return [
    {
      ...s,
      jobs: s.running ? [s.running] : [],
      interrupting: Boolean(s.running),
      forever: false,
      repeatLeft: 0,
      restart: false,
      dropPending: s.pending,
    },
    effects,
  ]
}

/** What follows when this form has nothing queued any more. */
function settle(s: RunState, outcome: JobOutcome, effects: RunEffect[] = []): Result {
  if (s.jobs.length > 0 || s.pending > 0) return [s, effects]
  const next = { ...s, interrupting: false }
  const then = ([state, more]: Result): Result => [state, [...effects, ...more]]
  if (next.restart) return then(submit({ ...next, restart: false }))
  if (outcome === "failed") return [{ ...next, forever: false, repeatLeft: 0 }, effects]
  if (next.forever) return then(submit(next))
  if (next.repeatLeft > 0) return then(submit({ ...next, repeatLeft: next.repeatLeft - 1 }))
  return [next, effects]
}

export function reduce(s: RunState, event: RunEvent, options: RunOptions): Result {
  switch (event.type) {
    case "generate": {
      const runs = Math.max(1, Math.floor(event.runs ?? 1))
      return press(runs > 1 ? { ...s, repeatLeft: runs - 1 } : s, event.key, event.at)
    }
    case "shortcut": {
      // A second press while the first submit is in flight must not queue a duplicate.
      if (!s.running) return s.pending > 0 ? [s, []] : submit(s)
      const [next, effects] = interrupt(s)
      return [{ ...next, restart: !options.interruptOnly }, effects]
    }
    case "submitted": {
      const { presses, first } = answer(s, event.key, true)
      // the other request of a double click: the same jobs, already taken (and maybe ended) with the first answer
      if (!first) return [{ ...s, presses }, []]
      const pending = Math.max(0, s.pending - 1)
      if (s.dropPending > 0) {
        const effects = event.jobIds.map<RunEffect>((jobId) => ({ type: "cancel", jobId }))
        return settle({ ...s, pending, presses, dropPending: s.dropPending - 1 }, "interrupted", effects)
      }
      const added = event.jobIds.filter((id) => !s.jobs.includes(id))
      return [{ ...s, pending, presses, jobs: added.length > 0 ? [...s.jobs, ...added] : s.jobs }, []]
    }
    case "rejected": {
      const { presses, first } = answer(s, event.key, false)
      // the press's other request may still queue its job
      if (!first) return [{ ...s, presses }, []]
      const next = { ...s, presses, pending: Math.max(0, s.pending - 1), dropPending: Math.max(0, s.dropPending - 1) }
      return [{ ...next, forever: false, repeatLeft: 0, restart: false }, []]
    }
    case "started":
      return s.jobs.includes(event.jobId) ? [{ ...s, running: event.jobId }, []] : [s, []]
    case "finished": {
      if (!s.jobs.includes(event.jobId)) return [s, []]
      const next = { ...s, jobs: s.jobs.filter((id) => id !== event.jobId), running: s.running === event.jobId ? null : s.running }
      return settle(next, event.outcome)
    }
    case "interrupt":
      return interrupt(s)
    case "skip":
      return [s, s.running ? [{ type: "skip", jobId: s.running }] : []]
    case "forever":
      if (!event.on) return [{ ...s, forever: false }, []]
      return isBusy(s) ? [{ ...s, forever: true }, []] : submit({ ...s, forever: true })
    case "stop-repeats":
      return [{ ...s, repeatLeft: 0 }, []]
    case "upscale":
      return [{ ...s, pending: s.pending + 1 }, [{ type: "submit", upscale: event.source }]]
    case "resume":
      // Nothing in flight any more: the loop ended while the page was away.
      if (!isBusy(s)) return [s, []]
      return [{ ...s, forever: s.forever || event.forever, repeatLeft: Math.max(s.repeatLeft, event.repeatLeft) }, []]
    case "restore": {
      const jobs = [...new Set([...s.jobs, ...event.jobs])]
      return [{ ...s, jobs, running: event.running && jobs.includes(event.running) ? event.running : s.running }, []]
    }
  }
}
