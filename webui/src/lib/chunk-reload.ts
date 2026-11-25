/**
 * A tab opened before the WebUI was updated asks for chunk names the new build no longer has: only
 * loading the new build helps. A chunk also fails while the server is down or restarting, and then
 * a reload would only land on the browser's error page; the page's own error (with "Try again")
 * stays instead. So when a page or control could not load its chunk (`retryableLazy`), the app asks
 * the server for its shell, and reloads only if that shell starts other scripts than this page's: a
 * new build. A prefetch that failed (a nav item on hover) never reloads: nobody asked for the page
 * yet, and a reload there would come without a click (with a leave-page prompt over unsaved work).
 * At most one reload a minute, so a server that really lacks a chunk shows the error instead of
 * looping.
 */
import { onLoadFailed } from "./retryable-lazy"

const LAST_RELOAD = "remi-chunk-reload"
const RELOAD_EVERY_MS = 60_000

/** The `src` of the module scripts a shell starts, as written (the server prefixes them under `--subpath`). */
function moduleScripts(doc: Document): string[] {
  return [...doc.querySelectorAll('script[type="module"][src]')].map((script) => script.getAttribute("src") ?? "")
}

/** Whether the server now serves a shell that starts other scripts than this page's. */
async function serverHasNewBuild(): Promise<boolean> {
  let html: string
  try {
    const response = await fetch(location.pathname, { cache: "no-store", credentials: "same-origin" })
    html = await response.text()
  } catch {
    return false // no answer: the server is down, the page's error stays
  }
  const served = moduleScripts(new DOMParser().parseFromString(html, "text/html"))
  // a proxy's error page starts no scripts: not a build
  return served.length > 0 && served.join(" ") !== moduleScripts(document).join(" ")
}

/** Reloads once a minute at most (no session storage to count with: never, to rule out a loop). */
function reloadOnce(reload: () => void): void {
  try {
    if (Date.now() - Number(sessionStorage.getItem(LAST_RELOAD) ?? 0) < RELOAD_EVERY_MS) return
    sessionStorage.setItem(LAST_RELOAD, String(Date.now()))
  } catch {
    return
  }
  reload()
}

/** Starts watching for chunks that failed to load; returns the function that stops it. */
export function installChunkReload({ reload = () => location.reload() }: { reload?: () => void } = {}): () => void {
  let checking = false
  return onLoadFailed(() => {
    if (checking) return // several chunks failed at once: one question to the server
    checking = true
    void serverHasNewBuild()
      .then((rebuilt) => {
        if (rebuilt) reloadOnce(reload)
      })
      .finally(() => {
        checking = false
      })
  })
}
