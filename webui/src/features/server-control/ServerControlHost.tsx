/**
 * Mounted once by the shell: the unsaved-changes unload guard, the palette
 * commands for "Reload UI" / "Restart server", and (lazily, only while a
 * command is being confirmed or followed) the confirm dialog and the
 * reconnecting overlay.
 */
import { RefreshCw, Power } from "@/components/icons"
import { Suspense, useEffect, useMemo } from "react"
import { type Command, useRegisterCommands } from "@/app/commands"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { requestServerCommand, useServerControl } from "./server-control"
import { installBeforeUnloadGuard } from "./unsaved"
import { retryableLazy } from "@/lib/retryable-lazy"

const ServerOverlay = retryableLazy(() => import("./ServerOverlay"))

export function ServerControlHost() {
  const active = useServerControl((s) => s.pending !== null || s.phase !== "idle")
  useEffect(installBeforeUnloadGuard, [])
  const commands = useMemo<Command[]>(
    () => [
      { id: "server:reload", label: "Reload UI", group: "System", icon: RefreshCw, keywords: ["restart", "reload", "refresh"], run: () => requestServerCommand("reload") },
      { id: "server:restart", label: "Restart server", group: "System", icon: Power, keywords: ["restart", "reboot"], run: () => requestServerCommand("restart") },
    ],
    [],
  )
  useRegisterCommands("server-control", commands)
  // its own boundary: a chunk that cannot load (the server is restarting) must not take the app down
  return active ? (
    <ErrorBoundary fallback={null}>
      <Suspense fallback={null}>
        <ServerOverlay />
      </Suspense>
    </ErrorBoundary>
  ) : null
}
