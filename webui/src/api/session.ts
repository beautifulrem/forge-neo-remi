/**
 * Whether this browser is signed in (`--auth`). The server renders no login
 * page: a page load without a login gets the SPA shell (status 401), the app
 * asks `GET /auth/status` and shows its own sign-in form (`features/auth`).
 *
 *  - `signed-out`: no session at start; only the sign-in form is rendered.
 *  - `expired`: a request was refused with 401 while the app was in use
 *    (the session ended, or the server now requires a login); the form is
 *    shown over the app, which keeps its state.
 */
import { create } from "zustand"
import type { AuthStatus } from "./types"

export type SessionState = "signed-in" | "signed-out" | "expired"

type Session = {
  state: SessionState
  /** The logged-in user, when logins are on. */
  user: string | null
  /**
   * Where the source code of the server is (GNU AGPL v3 section 13; `--source-url`): the
   * running version's once signed in, the repository's before. Null until the server said.
   */
  source: string | null
}

export const useSession = create<Session>()(() => ({ state: "signed-in", user: null, source: null }))

/** The answer of `GET /auth/status`, before the first render (`main.tsx`). */
export function applyAuthStatus({ required, user, source_url }: AuthStatus) {
  useSession.setState({ state: required && !user ? "signed-out" : "signed-in", user: user ?? null, source: source_url })
}

/** A request was refused with 401 (not the sign-in requests themselves). */
export function sessionRefused() {
  const { state } = useSession.getState()
  if (state === "signed-in") useSession.setState({ state: "expired" })
}
