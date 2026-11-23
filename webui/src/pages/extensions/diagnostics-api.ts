/**
 * Thin typed API client for Extension Health Diagnostics (`/api/v2/extensions/diagnostics`)
 * and Worker / Multi-GPU / MLX Status (`/api/v2/workers`), backed by `workbenchRequest`.
 */
import { queryOptions } from "@tanstack/react-query"
import { queryClient } from "@/api/queries"
import { workbenchRequest } from "@/features/pipelines/workbench-http"

export type ExtensionManifest = {
  api_version?: string | null
  min_remi_version?: string | null
  requires_capabilities?: string[]
  requires_widgets?: string[]
  license?: string | null
  homepage?: string | null
  author?: string | null
  version?: string | null
}

export type ExtensionDiagnosticItem = {
  name: string
  enabled: boolean
  builtin: boolean
  health: "healthy" | "degraded" | "error" | "disabled"
  load_time_ms: number | null
  hook_warnings: number
  hook_errors: number
  action_errors: number
  action_timeouts: number
  last_error?: string | null
  last_error_at?: string | null
  manifest_status: "valid" | "missing" | "incompatible" | "invalid"
  manifest?: ExtensionManifest | null
  manifest_errors?: string[]
  manifest_warnings?: string[]
  disabled_scripts?: { name?: string; reason?: string }[]
  disabled_reason?: string | null
}

export type DiagnosticsSummary = {
  total: number
  healthy: number
  degraded: number
  error: number
  disabled: number
  with_manifest: number
}

export type ExtensionDiagnosticsResponse = {
  api_version: string
  schema_version: number
  summary: DiagnosticsSummary
  items: ExtensionDiagnosticItem[]
}

export type WorkerInfo = {
  worker_id: string
  pid?: number | null
  state: string
  backend: string
  device: string
  gpu_name?: string | null
  vram_total_mb?: number | null
  vram_free_mb?: number | null
  loaded_model?: string | null
  supported_families?: string[]
  quant_formats?: string[]
  current_job_id?: string | null
  jobs_completed: number
  jobs_failed: number
  restarts: number
  uptime_s?: number | null
  last_heartbeat?: string | null
}

export type MlxStatus = {
  available: boolean
  platform_supported: boolean
  mlx_version?: string | null
  mflux_version?: string | null
  unified_memory_mb?: number | null
  supported_families?: string[]
  reason?: string | null
}

export type WorkersStatusResponse = {
  mode: "inprocess" | "subprocess" | "pool"
  workers: WorkerInfo[]
  mlx?: MlxStatus | null
}

export const diagnosticsKeys = {
  diagnostics: ["extensions", "diagnostics"] as const,
  workers: ["extensions", "workers"] as const,
}

export async function fetchExtensionDiagnostics(signal?: AbortSignal): Promise<ExtensionDiagnosticsResponse> {
  return workbenchRequest<ExtensionDiagnosticsResponse>("/extensions/diagnostics", { signal })
}

export async function fetchWorkersStatus(signal?: AbortSignal): Promise<WorkersStatusResponse> {
  return workbenchRequest<WorkersStatusResponse>("/workers", { signal })
}

export async function restartWorkerApi(workerId: string): Promise<{ ok: boolean; worker?: WorkerInfo }> {
  return workbenchRequest<{ ok: boolean; worker?: WorkerInfo }>(`/workers/${encodeURIComponent(workerId)}/restart`, { method: "POST" })
}

export const extensionDiagnosticsQuery = queryOptions({
  queryKey: diagnosticsKeys.diagnostics,
  queryFn: ({ signal }) => fetchExtensionDiagnostics(signal),
  staleTime: 15_000,
  retry: false,
})

export const workersStatusQuery = queryOptions({
  queryKey: diagnosticsKeys.workers,
  queryFn: ({ signal }) => fetchWorkersStatus(signal),
  staleTime: 10_000,
  retry: false,
})

export function refreshDiagnostics() {
  void queryClient.invalidateQueries({ queryKey: diagnosticsKeys.diagnostics })
  void queryClient.invalidateQueries({ queryKey: diagnosticsKeys.workers })
}
