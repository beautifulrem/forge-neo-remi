/**
 * What `GET /auth/status` tells the app before its first render (`main.tsx`):
 * whether it is signed in, and where the source code of the server is
 * (GNU AGPL v3 section 13; the sign-in screen links it too).
 */
import { afterEach, describe, expect, it } from "vitest"
import { applyAuthStatus, useSession } from "./session"

afterEach(() => useSession.setState({ state: "signed-in", user: null, source: null }))

describe("applyAuthStatus", () => {
  it("is signed out when a login is needed and none was given, and keeps the source link", () => {
    applyAuthStatus({ required: true, user: null, source_url: "https://example.org/remi" })
    expect(useSession.getState()).toMatchObject({ state: "signed-out", user: null, source: "https://example.org/remi" })
  })

  it("is signed in with a login, or when none is needed", () => {
    applyAuthStatus({ required: true, user: "ann", source_url: "https://example.org/remi/tree/0123abc" })
    expect(useSession.getState()).toMatchObject({ state: "signed-in", user: "ann", source: "https://example.org/remi/tree/0123abc" })
    applyAuthStatus({ required: false, user: null, source_url: "https://example.org/remi/tree/0123abc" })
    expect(useSession.getState()).toMatchObject({ state: "signed-in", user: null })
  })
})
