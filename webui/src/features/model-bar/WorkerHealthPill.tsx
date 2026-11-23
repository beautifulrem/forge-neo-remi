/**
 * Compact Worker / Multi-GPU / MLX health pill in the Model Bar.
 * Talks to `GET /api/v2/workers` cleanly (gracefully hides when unavailable
 * on older servers or unit tests without worker endpoints).
 */
import { useQuery } from "@tanstack/react-query"
import { Cpu } from "@/components/icons"
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { apiUrl, MOCK } from "@/api/http"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { FIELD_BUTTON_CLASS } from "@/components/ui/select-styles"
import { cn } from "@/lib/utils"
import { type WorkersStatus, workersSummary } from "./model-badges"

const MOCK_WORKERS: WorkersStatus = {
  mode: "inprocess",
  workers: [
    {
      worker_id: "worker-0",
      state: "ready",
      backend: "pytorch",
      device: "mps",
      gpu_name: "Apple Silicon Unified",
      vram_total_mb: 24576,
      vram_free_mb: 16384,
      loaded_model: "animagine-xl-4.0.safetensors",
      jobs_completed: 12,
      jobs_failed: 0,
      restarts: 0,
    },
  ],
  mlx: {
    available: false,
    platform_supported: true,
    unified_memory_mb: 24576,
    supported_families: ["flux3", "qwen21", "zimage"],
    reason: "Optional sidecar idle",
  },
}

async function fetchWorkers(signal?: AbortSignal): Promise<WorkersStatus | null> {
  if (MOCK) return MOCK_WORKERS
  try {
    const response = await fetch(apiUrl("/workers"), { headers: { Accept: "application/json" }, signal })
    if (!response.ok) return null
    const data = (await response.json()) as unknown
    if (!data || typeof data !== "object" || !Array.isArray((data as WorkersStatus).workers)) return null
    return data as WorkersStatus
  } catch {
    return null
  }
}

function formatMemGb(mb: number | null | undefined): string | null {
  if (typeof mb !== "number" || !Number.isFinite(mb) || mb <= 0) return null
  return `${(mb / 1024).toFixed(1)} GB`
}

const MODE_LABELS: Record<string, string> = {
  inprocess: "In-process",
  subprocess: "Isolated worker",
  pool: "Multi-GPU pool",
}

export function WorkerHealthPill({ className }: { className?: string }) {
  const { t } = useTranslation()
  const id = useId()
  const { data } = useQuery({
    queryKey: ["workers"],
    queryFn: ({ signal }) => fetchWorkers(signal),
    staleTime: 30_000,
    retry: false,
  })

  if (!data || data.workers.length === 0) return null
  const summary = workersSummary(data)

  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            variant="ghost"
            size="sm"
            aria-label={t("Worker status")}
            className={cn(FIELD_BUTTON_CLASS, "gap-1.5 px-2 font-normal text-foreground", className)}
          />
        }
      >
        <span
          aria-hidden
          className={cn(
            "size-1.5 shrink-0 rounded-full",
            summary.tone === "ok" && "bg-success",
            summary.tone === "warn" && "bg-warning",
            summary.tone === "error" && "bg-destructive",
          )}
        />
        <Cpu className="size-3.5 text-muted-foreground" aria-hidden />
        <span className="text-xs font-medium tabular-nums">{summary.label}</span>
      </PopoverTrigger>
      <PopoverContent aria-labelledby={`${id}-title`} align="end" className="w-[min(22rem,calc(100vw-2rem))] gap-2.5 p-3">
        <div className="flex items-center justify-between gap-2">
          <p id={`${id}-title`} className="text-sm font-semibold">
            {t("Inference workers")}
          </p>
          <Badge variant="outline" className="border-capability-badge-border bg-capability-badge-bg text-capability-badge">
            {t(MODE_LABELS[data.mode] ?? data.mode)}
          </Badge>
        </div>
        <ul className="flex flex-col gap-2">
          {data.workers.map((worker) => {
            const totalGb = formatMemGb(worker.vram_total_mb)
            const freeGb = formatMemGb(worker.vram_free_mb)
            const usedRatio =
              worker.vram_total_mb && worker.vram_free_mb != null
                ? Math.min(1, Math.max(0, (worker.vram_total_mb - worker.vram_free_mb) / worker.vram_total_mb))
                : null
            return (
              <li key={worker.worker_id} className="flex flex-col gap-1 border-t border-border/60 pt-2 first:border-t-0 first:pt-0">
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="truncate font-medium text-foreground">{worker.gpu_name || worker.device || worker.worker_id}</span>
                  <Badge variant="secondary" className="ml-auto h-4 px-1.5 text-[0.625rem]">
                    {worker.state}
                  </Badge>
                </div>
                {totalGb ? (
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[0.6875rem] text-muted-foreground tabular-nums">
                      <span>{freeGb ? t("{{free}} free of {{total}}", { free: freeGb, total: totalGb }) : totalGb}</span>
                      {typeof worker.jobs_completed === "number" ? <span>{t("{{count}} done", { count: worker.jobs_completed })}</span> : null}
                    </div>
                    {usedRatio !== null ? (
                      <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full bg-primary" style={{ width: `${Math.round(usedRatio * 100)}%` }} />
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </li>
            )
          })}
        </ul>
        {data.mlx?.platform_supported ? (
          <div className="flex items-center justify-between gap-2 border-t border-border/60 pt-2 text-xs text-muted-foreground">
            <span>{t("MLX sidecar")}</span>
            <span className="font-medium text-foreground">{data.mlx.available ? t("Available") : t("Standby")}</span>
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}
