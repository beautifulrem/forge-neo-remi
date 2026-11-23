import { QueryClientProvider } from "@tanstack/react-query"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { I18nextProvider } from "react-i18next"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import { useSession } from "@/api/session"
import { App } from "@/app/App"
import { BOOT_WAIT, settleWithin } from "@/app/boot"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { AuthGate } from "@/features/auth/SignIn"
import { activateLocale, i18n, initialLocale } from "@/i18n"
import "@/features/prompt-tools/register"
import "./index.css"

// A tab opened before the WebUI was updated asks for chunk names the new
// build no longer has: load the new build (at most once a minute, so a
// server that really lacks a chunk shows the error instead of looping).
// Offline, a reload would only land on the browser's error page: the
// page's own error (with "Try again") stays instead.
window.addEventListener("vite:preloadError", (event) => {
  if (!navigator.onLine) return
  try {
    const last = Number(sessionStorage.getItem("remi-chunk-reload") ?? 0)
    if (Date.now() - last < 60_000) return
    sessionStorage.setItem("remi-chunk-reload", String(Date.now()))
  } catch {
    return // no session storage to throttle with: never risk a reload loop
  }
  event.preventDefault()
  location.reload()
})

const root = createRoot(document.getElementById("root")!)

// Whether a login is needed (`--auth`): signed out, only the sign-in form
// renders and the app requests nothing. If the server does not answer, the
// app starts and shows its own connection errors.
const session = api.auth
  .status()
  .then(({ required, user }) => useSession.setState({ state: required && !user ? "signed-out" : "signed-in", user: user ?? null }))
  .catch(() => {})

// Load the stored/browser locale first so the shell never flashes English;
// a failure only means English, so render either way. A server that never
// answers holds the first render for BOOT_WAIT at most.
await settleWithin(Promise.all([session, activateLocale(initialLocale()).catch(() => {})]), BOOT_WAIT)

root.render(
  <StrictMode>
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={queryClient}>
        {/* Last line of defence: a shell error shows a message, never a white page. */}
        <ErrorBoundary>
          <AuthGate>
            <App />
          </AuthGate>
        </ErrorBoundary>
      </QueryClientProvider>
    </I18nextProvider>
  </StrictMode>,
)
