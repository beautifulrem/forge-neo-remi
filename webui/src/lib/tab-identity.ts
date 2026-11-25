/**
 * Which browser tab the session storage belongs to. A reload keeps a tab's session storage, and
 * the runs it remembers are picked up again; but duplicating a tab copies that storage too, and
 * the copy must not take over the jobs its original still follows. Each tab keeps an id in its
 * session storage (`remi-tab-id`); at start a tab that has one asks the browser's other tabs on
 * the `remi-tabs` BroadcastChannel whether one of them is open with it. One that is answers: the
 * asking tab is its copy, takes a new id and leaves the copied runs alone. No answer within the
 * wait: a reload (or the original was closed), and the storage is its own. A tab without an id is
 * new (or its storage predates the ids) and owns what it holds at once.
 *
 * A BroadcastChannel, not Web Locks: those exist only in secure contexts, and a server reached with
 * `--listen` over plain HTTP is not one. Started at boot (`main.tsx`), so a tab answers its copies
 * from its first moment.
 */
export const TAB_ID_KEY = "remi-tab-id"
export const TAB_CHANNEL = "remi-tabs"
/** How long a starting tab waits for an open tab with its id to answer (a live tab answers at once). */
const ANSWER_WAIT_MS = 250

type Channel = Pick<BroadcastChannel, "postMessage" | "close"> & { onmessage: ((event: MessageEvent) => void) | null }

/** `from` / `to`: a page load's own random name, never stored (a tab tells its own question by it). */
type Message = { type: "who-has"; id: string; from: string } | { type: "has"; id: string; to: string }

export type ClaimOptions = {
  storage: Pick<Storage, "getItem" | "setItem"> | null
  channel: Channel | null
  wait?: number
}

export type TabClaim = {
  /** True when the session storage is this tab's own; false in the copy of a tab that is still open. */
  owned: Promise<boolean>
  close: () => void
}

const randomName = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36)

function readId(storage: ClaimOptions["storage"]): string | null {
  try {
    return storage?.getItem(TAB_ID_KEY) ?? null
  } catch {
    return null
  }
}

function writeId(storage: ClaimOptions["storage"], id: string) {
  try {
    storage?.setItem(TAB_ID_KEY, id)
  } catch {
    // storage disabled: every page load is a new tab, with nothing stored to restore either
  }
}

/** Settles this page load's tab id (see the module comment) and answers the tabs that ask for it. */
export function claimTabId({ storage, channel, wait = ANSWER_WAIT_MS }: ClaimOptions): TabClaim {
  const load = randomName()
  const stored = readId(storage)
  let id = stored ?? randomName()
  if (stored === null) writeId(storage, id)
  let settled = stored === null || channel === null
  let resolve!: (owned: boolean) => void
  const owned = new Promise<boolean>((done) => (resolve = done))
  let timer: ReturnType<typeof setTimeout> | undefined
  const isCopy = () => {
    settled = true
    clearTimeout(timer)
    id = randomName()
    writeId(storage, id)
    resolve(false)
  }
  if (channel) {
    channel.onmessage = (event: MessageEvent<Message>) => {
      const message = event.data
      if (message?.type === "who-has" && message.id === id && typeof message.from === "string" && message.from !== load) {
        // A tab that owns the id answers. Two tabs asking at once (a tab duplicated while its
        // original reloads; the one that asked first may not have heard the other): the one whose
        // page load sorts first keeps the id, the other is the copy.
        if (settled || load < message.from) channel.postMessage({ type: "has", id, to: message.from } satisfies Message)
        else isCopy()
      } else if (message?.type === "has" && message.to === load && !settled) {
        // an open tab has this id: this tab is its copy
        isCopy()
      }
    }
  }
  if (settled) resolve(true)
  else {
    timer = setTimeout(() => {
      settled = true
      resolve(true)
    }, wait)
    channel!.postMessage({ type: "who-has", id, from: load } satisfies Message)
  }
  return {
    owned,
    close: () => {
      clearTimeout(timer)
      channel?.close()
    },
  }
}

function sessionStore(): Storage | null {
  try {
    return typeof sessionStorage === "undefined" ? null : sessionStorage
  } catch {
    return null
  }
}

let claim: TabClaim | null = null
let nextWait = ANSWER_WAIT_MS

/**
 * Whether what this tab's session storage holds from a previous page load (jobs in flight) is its
 * own: true after a reload or in a new tab, false in a tab duplicated from one still open.
 */
export function sessionIsOwn(): Promise<boolean> {
  claim ??= claimTabId({
    storage: sessionStore(),
    channel: typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel(TAB_CHANNEL),
    wait: nextWait,
  })
  return claim.owned
}

/** Tests: the next `sessionIsOwn` checks afresh (waiting `wait` for an answer). */
export function resetTabIdentity({ wait = ANSWER_WAIT_MS }: { wait?: number } = {}) {
  claim?.close()
  claim = null
  nextWait = wait
}
