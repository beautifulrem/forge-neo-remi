/**
 * What the page waits for before its first render: the sign-in state and the
 * chosen locale's catalogs. A server that accepts the connection but does not
 * answer (a busy start, a stalled proxy) must not keep the page blank, so the
 * wait ends after `ms` either way; whatever arrives later still applies (the
 * sign-in form, the translated shell).
 */
export function settleWithin(work: Promise<unknown>, ms: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms)
    work
      .catch(() => {})
      .finally(() => {
        clearTimeout(timer)
        resolve()
      })
  })
}

/** How long the first render waits for the server (ms). */
export const BOOT_WAIT = 4000
