/**
 * Which tab runs a form's "Generate forever" / "Generate several times" loop. A reload resumes the
 * loop its session storage remembers (`restoreRuns`); a duplicated tab starts with a copy of that
 * storage, so before resuming it asks the other tabs of this browser whether one still runs that
 * loop (owns one of its jobs in flight), and leaves it to that tab. Another tab's own loop of the
 * same form does not answer: its jobs are its own. A BroadcastChannel, not Web Locks: those exist
 * only in secure contexts, and a server reached with `--listen` over plain HTTP is not one.
 */
import type { GenerateTab } from "./run-controller"

const CHANNEL = "remi-run-loops"
/** How long a restoring tab waits for a looping tab to answer (a live tab answers at once). */
const ANSWER_WAIT_MS = 250

type Message = { type: "who-loops"; tab: GenerateTab; id: string; jobs: string[] } | { type: "loops"; tab: GenerateTab; id: string }

let answering: BroadcastChannel | null = null

/** Answers other tabs' "who loops this form?" when `loops(tab, jobs)` says this tab runs that loop. */
export function answerLoopQueries(loops: (tab: GenerateTab, jobs: readonly string[]) => boolean): void {
  if (answering || typeof BroadcastChannel === "undefined") return
  const channel = new BroadcastChannel(CHANNEL)
  channel.onmessage = (event: MessageEvent<Message>) => {
    const message = event.data
    if (message?.type === "who-loops" && Array.isArray(message.jobs) && loops(message.tab, message.jobs)) channel.postMessage({ type: "loops", tab: message.tab, id: message.id } satisfies Message)
  }
  answering = channel
}

/** True when another tab of this browser runs the loop of `tab` whose jobs in flight are `jobs`. */
export function loopRunsElsewhere(tab: GenerateTab, jobs: readonly string[], wait = ANSWER_WAIT_MS): Promise<boolean> {
  if (typeof BroadcastChannel === "undefined") return Promise.resolve(false)
  const channel = new BroadcastChannel(CHANNEL)
  const id = Math.random().toString(36).slice(2)
  return new Promise<boolean>((resolve) => {
    const done = (answer: boolean) => {
      clearTimeout(timer)
      channel.close()
      resolve(answer)
    }
    const timer = setTimeout(() => done(false), wait)
    channel.onmessage = (event: MessageEvent<Message>) => {
      if (event.data?.type === "loops" && event.data.tab === tab && event.data.id === id) done(true)
    }
    channel.postMessage({ type: "who-loops", tab, id, jobs: [...jobs] } satisfies Message)
  })
}
