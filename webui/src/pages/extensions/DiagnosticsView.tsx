/**
 * Extensions Health Diagnostics & Worker / Multi-GPU / MLX Monitor view:
 * surfaces `REMI_API_VERSION`, manifest compatibility, per-extension load time,
 * warning/error counts, disabled reasons, and the supervised worker pool card.
 * Lazily loaded on `#/extensions?tab=diagnostics`.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { CheckCircle2, CircleAlert, Cpu, Gauge, RefreshCw, RotateCw, Search, Server, ShieldAlert, ShieldCheck, TriangleAlert } from "@/components/icons"
import { useState } from "react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { errorText } from "@/features/errors/error-text"
import { useTranslation } from "react-i18next"
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

function HealthBadge({ health }: { health: ExtensionDiagnosticItem["health"] }) {
  const { t } = useTranslation()
  switch (health) {
    case "healthy":
      return (
        <Badge variant="secondary" className="gap-1 bg-diff-add-bg text-[0.625rem] text-diff-add">
          <CheckCircle2 className="size-3" aria-hidden />
          {t("Healthy")}
        </Badge>
      )
    case "degraded":
      return (
        <Badge variant="secondary" className="gap-1 bg-advice-warn-bg text-[0.625rem] text-advice-warn">
          <TriangleAlert className="size-3" aria-hidden />
          {t("Degraded")}
        </Badge>
      )
    case "error":
      return (
        <Badge variant="destructive" className="gap-1 text-[0.625rem]">
          <CircleAlert className="size-3" aria-hidden />
          {t("Error")}
        </Badge>
      )
    default:
      return (
        <Badge variant="outline" className="text-[0.625rem] text-muted-foreground">
          {t("Disabled")}
        </Badge>
      )
  }
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
    if (filter === "issues" && item.health !== "degraded" && item.health !== "error" && item.hook_warnings === 0 && item.hook_errors === 0 && item.action_errors === 0) {
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

  return (
    <div className="flex flex-col gap-5 pb-12">
      {/* Top refresh bar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          {diag.data ? (
            <Badge variant="outline" className="gap-1 font-mono text-[0.6875rem]">
              <ShieldCheck className="size-3.5 text-primary" aria-hidden />
              {t("Remi API version")}: {diag.data.api_version}
            </Badge>
          ) : null}
          {workers.data ? (
            <Badge variant="secondary" className="gap-1 text-[0.6875rem]">
              <Server className="size-3" aria-hidden />
              {t("Worker mode")}: {modeLabel(workers.data.mode, t)}
            </Badge>
          ) : null}
        </div>
        <Button variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs" onClick={refreshDiagnostics}>
          <RefreshCw className="size-3.5" aria-hidden />
          {t("Refresh diagnostics")}
        </Button>
      </div>

      {/* Worker / Multi-GPU / MLX monitor panel */}
      <section className={panelClass(0)} aria-label={t("Runtime & workers")}>
        <h2 className={cn(panelTitleClass(0), "mb-3 flex items-center gap-2")}>
          <Cpu className="size-4 text-muted-foreground" aria-hidden />
          {t("Runtime & workers")}
        </h2>

        {workers.isPending ? (
          <Skeleton className="h-24 w-full" />
        ) : workers.isError ? (
          <p className="text-sm text-muted-foreground">{t("Workers status unavailable on this server.")}</p>
        ) : (
          <div className="flex flex-col gap-3">
            {workers.data.workers.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("No workers reported")}</p>
            ) : (
              <ul className="divide-y divide-border/70">
                {workers.data.workers.map((worker) => {
                  const usedMb =
                    worker.vram_total_mb != null && worker.vram_free_mb != null
                      ? Math.max(0, worker.vram_total_mb - worker.vram_free_mb)
                      : null
                  const usedPct = usedMb != null && worker.vram_total_mb ? Math.min(100, (usedMb / worker.vram_total_mb) * 100) : null
                  return (
                    <li key={worker.worker_id} className="flex flex-col gap-2 py-2.5">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-sm font-medium">{worker.worker_id}</span>
                        <Badge variant="secondary" className="text-[0.625rem]">
                          {worker.backend} · {worker.device}
                        </Badge>
                        <Badge variant="outline" className="text-[0.625rem]">
                          {worker.state}
                        </Badge>
                        {worker.gpu_name ? <span className="text-xs text-muted-foreground">{worker.gpu_name}</span> : null}
                        {worker.pid ? <span className="font-mono text-xs text-muted-foreground">PID {worker.pid}</span> : null}
                        <Button
                          variant="outline"
                          size="sm"
                          className="ml-auto h-7 gap-1 px-2 text-xs"
                          disabled={restart.isPending}
                          onClick={() => restart.mutate(worker.worker_id)}
                        >
                          <RotateCw className="size-3" aria-hidden />
                          {t("Restart worker")}
                        </Button>
                      </div>

                      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground tabular-nums">
                        {worker.loaded_model ? (
                          <span>
                            {t("Loaded model")}: <strong className="font-medium text-foreground">{worker.loaded_model}</strong>
                          </span>
                        ) : null}
                        <span>
                          {t("Jobs completed")}: {worker.jobs_completed}
                        </span>
                        <span>
                          {t("Restarts")}: {worker.restarts}
                        </span>
                        {usedMb != null && worker.vram_total_mb != null ? (
                          <span>
                            {t("VRAM")}: {formatMb(usedMb)} / {formatMb(worker.vram_total_mb)}
                          </span>
                        ) : null}
                      </div>

                      {usedPct != null ? <Progress value={usedPct} aria-label={t("VRAM")} className="h-1.5" /> : null}

                      {worker.quant_formats && worker.quant_formats.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {worker.quant_formats.map((fmt) => (
                            <span key={fmt} className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.625rem] text-muted-foreground">
                              {fmt}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </li>
                  )
                })}
              </ul>
            )}

            {workers.data.mlx ? (
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-2.5 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{t("Apple Silicon MLX sidecar")}</span>
                  <Badge variant={workers.data.mlx.available ? "secondary" : "outline"} className="text-[0.625rem]">
                    {workers.data.mlx.available ? t("MLX available") : t("MLX unavailable")}
                  </Badge>
                  {workers.data.mlx.unified_memory_mb ? (
                    <span className="text-muted-foreground tabular-nums">
                      {t("Unified memory")}: {formatMb(workers.data.mlx.unified_memory_mb)}
                    </span>
                  ) : null}
                  {workers.data.mlx.mlx_version ? <span className="font-mono text-muted-foreground">mlx {workers.data.mlx.mlx_version}</span> : null}
                </div>
                {workers.data.mlx.reason ? <span className="text-muted-foreground">{workers.data.mlx.reason}</span> : null}
              </div>
            ) : null}
          </div>
        )}
      </section>

      {/* Extension Health & Manifest Diagnostics */}
      <section className={panelClass(0)} aria-label={t("Extension health & API")}>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className={cn(panelTitleClass(0), "flex items-center gap-2")}>
            <Gauge className="size-4 text-muted-foreground" aria-hidden />
            {t("Extension health & API")}
          </h2>
          {diag.data?.summary ? (
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground tabular-nums">
              <span>
                {t("Healthy")}: {diag.data.summary.healthy}
              </span>
              <span>·</span>
              <span>
                {t("Degraded")}: {diag.data.summary.degraded}
              </span>
              <span>·</span>
              <span>
                {t("Error")}: {diag.data.summary.error}
              </span>
              <span>·</span>
              <span>
                {t("With manifest")}: {diag.data.summary.with_manifest}/{diag.data.summary.total}
              </span>
            </div>
          ) : null}
        </div>

        <div className="mb-3 flex flex-wrap items-center gap-2">
          <div className="relative min-w-44 flex-1 sm:max-w-xs">
            <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
            <Input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder={t("Filter diagnostics…")}
              aria-label={t("Filter diagnostics…")}
              className="h-7 pl-7 text-[0.8rem]"
            />
          </div>
          <div className="flex flex-wrap items-center gap-1" role="group" aria-label={t("Filter diagnostics…")}>
            {filterButtons.map((btn) => (
              <button
                key={btn.id}
                type="button"
                aria-pressed={filter === btn.id}
                onClick={() => setFilter(btn.id)}
                className={cn(
                  "h-7 rounded-md px-2 text-xs font-medium transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
                  filter === btn.id ? "bg-primary/12 text-primary" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                {btn.text}
              </button>
            ))}
          </div>
        </div>

        {diag.isPending ? (
          <div className="flex flex-col gap-2" aria-busy="true">
            {Array.from({ length: 4 }, (_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : diag.isError ? (
          <p className="text-sm text-muted-foreground">{t("Diagnostics unavailable on this server.")}</p>
        ) : items.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">{t("No extension diagnostics match.")}</p>
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
              return (
                <li key={item.name} className="flex flex-col gap-1.5 py-2.5">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="truncate text-sm font-medium" title={item.name}>
                      {item.name}
                    </span>
                    <HealthBadge health={item.health} />
                    {item.manifest_status === "valid" ? (
                      <Badge variant="outline" className="gap-1 font-mono text-[0.625rem]">
                        {t("Manifest")} v{item.manifest?.api_version ?? "1.0"}
                      </Badge>
                    ) : item.manifest_status === "incompatible" || item.manifest_status === "invalid" ? (
                      <Badge variant="destructive" className="gap-1 text-[0.625rem]">
                        <ShieldAlert className="size-3" aria-hidden />
                        {t("Incompatible")}
                      </Badge>
                    ) : null}
                    {item.manifest?.license ? (
                      <span className="text-[0.6875rem] text-muted-foreground">{item.manifest.license}</span>
                    ) : null}

                    <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground tabular-nums">
                      {item.load_time_ms != null ? (
                        <span title={t("Load time")}>{item.load_time_ms.toFixed(1)} ms</span>
                      ) : null}
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
                    </div>
                  </div>

                  {item.manifest?.requires_capabilities && item.manifest.requires_capabilities.length > 0 ? (
                    <div className="flex flex-wrap items-center gap-1 text-[0.6875rem] text-muted-foreground">
                      <span>{t("Required capabilities")}:</span>
                      {item.manifest.requires_capabilities.map((cap) => (
                        <span key={cap} className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.625rem] text-foreground">
                          {cap}
                        </span>
                      ))}
                    </div>
                  ) : null}

                  {disabledReason ? (
                    <p className="text-xs text-muted-foreground">
                      <span className="font-medium">{t("Disabled reason")}:</span> {disabledReason}
                    </p>
                  ) : null}

                  {item.last_error ? (
                    <pre className="overflow-x-auto rounded bg-muted px-2 py-1 font-mono text-[0.6875rem] text-destructive">
                      {item.last_error}
                    </pre>
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
