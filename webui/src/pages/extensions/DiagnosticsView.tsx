/**
 * Extensions Health Diagnostics & Worker / Multi-GPU / MLX Monitor view: a
 * telemetry strip, then two inspector panels (`ui/layout/panel.ts`) — the
 * worker monitor and the tabular extension health matrix, rows with a
 * hairline between them. Lazily loaded on `#/extensions?tab=diagnostics`.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { RefreshCw, RotateCw, Search } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { ApiError } from "@/api/http"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { errorText } from "@/features/errors/error-text"
import { cn } from "@/lib/utils"
import { panelClass, panelTitleClass } from "@/ui/layout/panel"
import {
  type ExtensionDiagnosticItem,
  extensionDiagnosticsQuery,
  refreshDiagnostics,
  restartWorkerApi,
  type WorkersStatusResponse,
  workersStatusQuery,
} from "./diagnostics-api"

type DiagFilter = "all" | "issues" | "manifest" | "disabled"

function formatMb(mb: number | null | undefined): string {
  if (mb == null || !Number.isFinite(mb)) return "—"
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`
}

function modeLabel(mode: WorkersStatusResponse["mode"], t: (key: string) => string): string {
  switch (mode) {
    case "pool":
      return t("Multi-GPU pool")
    case "subprocess":
      return t("Subprocess worker")
    default:
      return t("In-process")
  }
}

function workerDotTone(state: string): string {
  const s = state.toLowerCase()
  if (s === "idle" || s === "ready" || s === "running") return "bg-success"
  if (s === "error" || s === "crashed" || s === "dead" || s === "stopped") return "bg-destructive"
  return "bg-warning"
}

function HealthIndicator({ health }: { health: ExtensionDiagnosticItem["health"] }) {
  const { t } = useTranslation()
  const config = {
    healthy: { dot: "bg-success", text: "text-foreground", label: t("Healthy") },
    degraded: { dot: "bg-warning", text: "text-advice-warn", label: t("Degraded") },
    error: { dot: "bg-destructive", text: "text-destructive", label: t("Error") },
    disabled: { dot: "bg-muted-foreground/50", text: "text-muted-foreground", label: t("Disabled") },
  }[health] ?? { dot: "bg-muted-foreground/50", text: "text-muted-foreground", label: t("Disabled") }

  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs font-medium", config.text)}>
      <span className={cn("size-1.5 shrink-0 rounded-full", config.dot)} aria-hidden />
      <span>{config.label}</span>
    </span>
  )
}

export function DiagnosticsView() {
  const { t } = useTranslation()
  const [search, setSearch] = useState("")
  const [filter, setFilter] = useState<DiagFilter>("all")

  const diag = useQuery(extensionDiagnosticsQuery)
  const workers = useQuery(workersStatusQuery)

  const restart = useMutation({
    mutationFn: (workerId: string) => restartWorkerApi(workerId),
    onSuccess: () => {
      refreshDiagnostics()
      toast.success(t("Worker restarted"))
    },
    onError: (error) => toast.error(t("Could not restart worker"), { description: errorText(error) }),
  })

  const terms = search.trim().toLowerCase().split(/\s+/).filter(Boolean)
  const items = (diag.data?.items ?? []).filter((item) => {
    if (
      filter === "issues" &&
      item.health !== "degraded" &&
      item.health !== "error" &&
      item.hook_warnings === 0 &&
      item.hook_errors === 0 &&
      item.action_errors === 0
    ) {
      return false
    }
    if (filter === "manifest" && item.manifest_status === "missing") return false
    if (filter === "disabled" && item.enabled && item.health !== "disabled") return false
    if (terms.length === 0) return true
    const haystack = `${item.name} ${item.last_error ?? ""} ${item.disabled_reason ?? ""} ${item.manifest?.requires_capabilities?.join(" ") ?? ""}`.toLowerCase()
    return terms.every((term) => haystack.includes(term))
  })

  const filterButtons: { id: DiagFilter; text: string }[] = [
    { id: "all", text: t("All") },
    { id: "issues", text: t("Issues only") },
    { id: "manifest", text: t("Manifest only") },
    { id: "disabled", text: t("Disabled") },
  ]

  const sep = (
    <span className="text-muted-foreground/45 select-none" aria-hidden>
      ·
    </span>
  )

  return (
    <div className="flex flex-col gap-5 pb-12">
      {/* Telemetry strip: the panel below draws the hairline under it */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground tabular-nums">
          {diag.data ? (
            <span className="font-mono text-foreground">
              {t("Remi API version")}: {diag.data.api_version}
            </span>
          ) : null}
          {diag.data && workers.data ? sep : null}
          {workers.data ? (
            <span>
              {t("Worker mode")}: <span className="font-medium text-foreground">{modeLabel(workers.data.mode, t)}</span>
            </span>
          ) : null}
          {diag.data?.summary ? (
            <>
              {sep}
              <span className="inline-flex items-center gap-1.5">
                <span className="size-1.5 rounded-full bg-success" aria-hidden />
                {diag.data.summary.healthy} {t("Healthy")}
              </span>
              {diag.data.summary.degraded > 0 ? (
                <>
                  {sep}
                  <span className="inline-flex items-center gap-1.5 text-advice-warn">
                    <span className="size-1.5 rounded-full bg-warning" aria-hidden />
                    {diag.data.summary.degraded} {t("Degraded")}
                  </span>
                </>
              ) : null}
              {diag.data.summary.error > 0 ? (
                <>
                  {sep}
                  <span className="inline-flex items-center gap-1.5 text-destructive">
                    <span className="size-1.5 rounded-full bg-destructive" aria-hidden />
                    {diag.data.summary.error} {t("Error")}
                  </span>
                </>
              ) : null}
              {sep}
              <span className="font-mono">
                {diag.data.summary.with_manifest}/{diag.data.summary.total} {t("With manifest")}
              </span>
            </>
          ) : null}
        </div>
        <Button variant="ghost" size="sm" className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground" onClick={refreshDiagnostics}>
          <RefreshCw className="size-3.5" aria-hidden />
          {t("Refresh diagnostics")}
        </Button>
      </div>

      {/* Worker / Multi-GPU / MLX monitor: an inspector panel, not a card */}
      <section className={cn("flex flex-col gap-1", panelClass(0))} aria-label={t("Runtime & workers")}>
        <div className="flex min-h-8 items-center justify-between">
          <h2 className={panelTitleClass(0)}>{t("Runtime & workers")}</h2>
        </div>

        {workers.isPending ? (
          <div className="py-2">
            <Skeleton className="h-20 w-full" />
          </div>
        ) : workers.isError ? (
          <p className="py-2 text-sm text-muted-foreground">{t("Workers status unavailable on this server.")}</p>
        ) : (
          <div className="divide-y divide-border/70">
            {workers.data.workers.length === 0 ? (
              <p className="py-2 text-sm text-muted-foreground">{t("No workers reported")}</p>
            ) : (
              workers.data.workers.map((worker) => {
                const usedMb =
                  worker.vram_total_mb != null && worker.vram_free_mb != null
                    ? Math.max(0, worker.vram_total_mb - worker.vram_free_mb)
                    : null
                const usedPct = usedMb != null && worker.vram_total_mb ? Math.min(100, (usedMb / worker.vram_total_mb) * 100) : null
                return (
                  <div key={worker.worker_id} className="flex flex-col gap-2.5 py-3">
                    {/* Row 1: identity, backend/device, live status dot, restart action */}
                    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
                      <span className="font-mono text-sm font-semibold text-foreground">{worker.worker_id}</span>
                      {sep}
                      <span className="font-mono text-xs text-muted-foreground">
                        {worker.backend} · {worker.device}
                      </span>
                      {sep}
                      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-foreground">
                        <span className={cn("size-1.5 shrink-0 rounded-full", workerDotTone(worker.state))} aria-hidden />
                        <span className="capitalize">{worker.state}</span>
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="ml-auto h-7 gap-1.5 px-2.5 text-xs"
                        disabled={restart.isPending}
                        onClick={() => restart.mutate(worker.worker_id)}
                      >
                        <RotateCw className="size-3" aria-hidden />
                        {t("Restart worker")}
                      </Button>
                    </div>

                    {/* Row 2: clean tabular key-value telemetry + razor-thin 2px memory track */}
                    <div className="flex flex-col gap-1.5">
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground tabular-nums">
                        {worker.gpu_name ? <span className="text-foreground/90">{worker.gpu_name}</span> : null}
                        {worker.pid ? <span className="font-mono">PID {worker.pid}</span> : null}
                        {worker.loaded_model ? (
                          <span className="min-w-0 truncate" title={worker.loaded_model}>
                            {t("Loaded model")}: <span className="font-mono text-foreground">{worker.loaded_model}</span>
                          </span>
                        ) : null}
                        <span>
                          {t("Jobs completed")}: <span className="font-mono text-foreground">{worker.jobs_completed}</span>
                        </span>
                        <span>
                          {t("Restarts")}: <span className="font-mono text-foreground">{worker.restarts}</span>
                        </span>
                        {usedMb != null && worker.vram_total_mb != null ? (
                          <span className="font-mono text-foreground sm:ml-auto">
                            {t("VRAM")}: {formatMb(usedMb)} / {formatMb(worker.vram_total_mb)}
                          </span>
                        ) : null}
                      </div>

                      {usedPct != null ? (
                        <div
                          role="progressbar"
                          aria-label={t("VRAM")}
                          aria-valuenow={Math.round(usedPct)}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          className="h-0.5 w-full overflow-hidden rounded-full bg-[var(--slider-groove)]"
                        >
                          <div className="h-full w-full origin-left rounded-full bg-foreground/70 transition-transform duration-200" style={{ transform: `scaleX(${usedPct / 100})` }} />
                        </div>
                      ) : null}
                    </div>

                    {/* Row 3: quant formats as quiet inline mono */}
                    {worker.quant_formats && worker.quant_formats.length > 0 ? (
                      <p className="font-mono text-[0.6875rem] text-muted-foreground">{worker.quant_formats.join(" · ")}</p>
                    ) : null}
                  </div>
                )
              })
            )}

            {/* MLX sidecar sub-row */}
            {workers.data.mlx ? (
              <div className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-xs">
                <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                  <span className="font-medium text-foreground">{t("Apple Silicon MLX sidecar")}</span>
                  {sep}
                  <span className={cn("inline-flex items-center gap-1.5 font-medium", workers.data.mlx.available ? "text-success" : "text-muted-foreground")}>
                    <span
                      className={cn("size-1.5 shrink-0 rounded-full", workers.data.mlx.available ? "bg-success" : "bg-muted-foreground/50")}
                      aria-hidden
                    />
                    {workers.data.mlx.available ? t("MLX available") : t("MLX unavailable")}
                  </span>
                  {workers.data.mlx.unified_memory_mb ? (
                    <>
                      {sep}
                      <span className="font-mono text-muted-foreground tabular-nums">
                        {t("Unified memory")}: {formatMb(workers.data.mlx.unified_memory_mb)}
                      </span>
                    </>
                  ) : null}
                  {workers.data.mlx.mlx_version ? (
                    <>
                      {sep}
                      <span className="font-mono text-muted-foreground">mlx {workers.data.mlx.mlx_version}</span>
                    </>
                  ) : null}
                </div>
                {workers.data.mlx.reason ? <span className="text-muted-foreground">{workers.data.mlx.reason}</span> : null}
              </div>
            ) : null}
          </div>
        )}
      </section>

      {/* Extension health matrix: an inspector panel, not a card */}
      <section className={cn("flex flex-col gap-1", panelClass(0))} aria-label={t("Extension health & API")}>
        <div className="flex min-h-8 flex-wrap items-center gap-2">
          <h2 className={cn("mr-2", panelTitleClass(0))}>{t("Extension health & API")}</h2>

          <div className="flex flex-wrap items-center gap-0.5" role="group" aria-label={t("Filter diagnostics…")}>
            {filterButtons.map((btn) => (
              <button
                key={btn.id}
                type="button"
                aria-pressed={filter === btn.id}
                onClick={() => setFilter(btn.id)}
                className={cn(
                  "relative h-7 rounded-sm px-2 text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                  filter === btn.id
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                )}
              >
                {btn.text}
              </button>
            ))}
          </div>

          <div className="relative ml-auto min-w-44 flex-1 sm:max-w-56 sm:flex-initial">
            <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
            <Input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder={t("Filter diagnostics…")}
              aria-label={t("Filter diagnostics…")}
              className="h-7 pl-7 text-xs"
            />
          </div>
        </div>

        {diag.isPending ? (
          <div className="flex flex-col gap-2 py-2" aria-busy="true">
            {Array.from({ length: 4 }, (_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        ) : diag.isError && diag.error instanceof ApiError && diag.error.status === 404 ? (
          <p className="py-2 text-sm text-muted-foreground">{t("Diagnostics unavailable on this server.")}</p>
        ) : diag.isError && !diag.data ? (
          <div role="alert" className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2 text-sm">
            <span className="text-destructive">{errorText(diag.error)}</span>
            <Button variant="outline" size="sm" onClick={() => void diag.refetch()}>
              {t("Retry")}
            </Button>
          </div>
        ) : items.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">{t("No extension diagnostics match.")}</p>
        ) : (
          <ul className="divide-y divide-border/70">
            {items.map((item) => {
              const totalErrors = (item.hook_errors ?? 0) + (item.action_errors ?? 0) + (item.action_timeouts ?? 0)
              const disabledReason =
                item.disabled_reason ||
                item.disabled_scripts
                  ?.map((s) => (s.name && s.reason ? `${s.name}: ${s.reason}` : s.reason || s.name))
                  .filter(Boolean)
                  .join("; ")
              const manifestText =
                item.manifest_status === "valid"
                  ? `v${item.manifest?.api_version ?? "1.0"}`
                  : item.manifest_status === "incompatible" || item.manifest_status === "invalid"
                    ? t("Incompatible")
                    : "—"
              const caps = item.manifest?.requires_capabilities ?? []

              return (
                <li key={item.name} className="py-2.5">
                  <div className="grid grid-cols-1 items-center gap-x-3 gap-y-1 md:grid-cols-[minmax(12rem,1.4fr)_6.25rem_5.25rem_5.5rem_minmax(0,1fr)_11rem]">
                    <span className="truncate font-mono text-[0.8rem] font-medium text-foreground" title={item.name}>
                      {item.name}
                    </span>

                    <HealthIndicator health={item.health} />

                    <span
                      className={cn(
                        "font-mono text-xs tabular-nums",
                        item.manifest_status === "incompatible" || item.manifest_status === "invalid"
                          ? "font-medium text-destructive"
                          : "text-muted-foreground",
                      )}
                      title={t("Manifest")}
                    >
                      {manifestText}
                    </span>

                    <span className="truncate text-xs text-muted-foreground" title={item.manifest?.license ?? undefined}>
                      {item.manifest?.license ?? "—"}
                    </span>

                    <span
                      className="truncate font-mono text-xs text-muted-foreground"
                      title={caps.length > 0 ? `${t("Required capabilities")}: ${caps.join(", ")}` : undefined}
                    >
                      {caps.length > 0 ? caps.join(" · ") : "—"}
                    </span>

                    <div className="flex items-center gap-2.5 font-mono text-xs text-muted-foreground tabular-nums md:justify-end">
                      {item.hook_warnings > 0 ? (
                        <span className="text-advice-warn">
                          {t("Warnings")}: {item.hook_warnings}
                        </span>
                      ) : null}
                      {totalErrors > 0 ? (
                        <span className="text-destructive">
                          {t("Errors")}: {totalErrors}
                        </span>
                      ) : null}
                      {item.load_time_ms != null ? (
                        <span className="min-w-14 text-right" title={t("Load time")}>
                          {item.load_time_ms.toFixed(1)} ms
                        </span>
                      ) : null}
                    </div>
                  </div>

                  {disabledReason ? (
                    <p className="mt-1.5 border-l-2 border-border pl-2.5 text-xs text-muted-foreground">
                      <span className="font-medium text-foreground/85">{t("Disabled reason")}:</span> {disabledReason}
                    </p>
                  ) : null}

                  {item.last_error ? (
                    <p className="mt-1.5 overflow-x-auto border-l-2 border-destructive/70 pl-2.5 font-mono text-xs text-destructive">
                      {item.last_error}
                    </p>
                  ) : null}
                </li>
              )
            })}
          </ul>
        )}
      </section>
    </div>
  )
}
