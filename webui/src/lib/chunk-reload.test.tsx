import { act, render, screen } from "@testing-library/react"
import { Suspense } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { installChunkReload } from "./chunk-reload"
import { retryableLazy } from "./retryable-lazy"

const SCRIPTS = ["/assets/react-a1.js", "/assets/shell-b2.js"]
const shell = (scripts: string[]) => `<!doctype html><html><head>${scripts.map((src) => `<script type="module" crossorigin src="${src}"></script>`).join("")}</head><body><div id="root"></div></body></html>`
const settle = () => act(() => new Promise((resolve) => setTimeout(resolve, 50)))

/** A lazy page whose chunk cannot be fetched (as a stale tab's, or any while the server is down). */
function mountFailingPage() {
  const Page = retryableLazy(() => Promise.reject(new TypeError("Failed to fetch dynamically imported module: /assets/Gone-c3.js")))
  render(
    <ErrorBoundary>
      <Suspense fallback={<p>loading</p>}>
        <Page />
      </Suspense>
    </ErrorBoundary>,
  )
}

// A tab opened before the WebUI was updated asks for chunks the new build lacks; only loading the
// new build helps. The page reloaded on any failed chunk while the browser said it was online: on
// localhost that is also the server being down, so a hover on a nav item (its chunk is prefetched)
// threw the tab onto the browser's error page, with no click and a leave-page prompt over unsaved
// settings.
describe("chunk reload", () => {
  let reload: ReturnType<typeof vi.fn<() => void>>
  let uninstall: () => void
  beforeEach(() => {
    sessionStorage.clear()
    for (const src of SCRIPTS) {
      const script = document.createElement("script")
      script.type = "module"
      script.setAttribute("src", src)
      document.head.append(script)
    }
    reload = vi.fn<() => void>()
    uninstall = installChunkReload({ reload })
    vi.spyOn(console, "error").mockImplementation(() => {})
  })
  afterEach(() => {
    uninstall()
    document.head.querySelectorAll("script").forEach((script) => script.remove())
    vi.restoreAllMocks()
  })

  it("never reloads for a prefetch", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(shell(["/assets/react-a1.js", "/assets/shell-NEW.js"])))
    const event = new Event("vite:preloadError", { cancelable: true })
    window.dispatchEvent(event)
    await settle()
    expect(reload).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(false) // the import fails as usual: a prefetch ignores it, a page shows it
  })

  it("does not reload while the server does not answer: the page shows its error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"))
    mountFailingPage()
    expect(await screen.findByText("The server cannot be reached.")).toBeInTheDocument()
    await settle()
    expect(reload).not.toHaveBeenCalled()
  })

  it("does not reload when the server still serves this build, or a page that is not the app", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(shell(SCRIPTS)))
    mountFailingPage()
    await settle()
    expect(fetch).toHaveBeenCalled()
    fetch.mockResolvedValue(new Response("<html><body>502 Bad Gateway</body></html>", { status: 502 }))
    mountFailingPage()
    await settle()
    expect(reload).not.toHaveBeenCalled()
  })

  it("loads the new build once the server serves one, at most once a minute", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(() => Promise.resolve(new Response(shell(["/assets/react-a1.js", "/assets/shell-NEW.js"]))))
    mountFailingPage()
    await settle()
    expect(fetch.mock.calls[0]![1]).toMatchObject({ cache: "no-store" })
    expect(reload).toHaveBeenCalledTimes(1)
    mountFailingPage()
    await settle()
    expect(reload).toHaveBeenCalledTimes(1)
  })
})
