/**
 * Typed API client for Extension Health Diagnostics (`/api/v2/extensions/diagnostics`)
 * and Worker / Multi-GPU / MLX Status (`/api/v2/workers`), backed by `api` (`schema.d.ts`).
 */
import { queryOptions } from "@tanstack/react-query"
import { api } from "@/api/endpoints"
import { keys, queryClient, workersQuery } from "@/api/queries"
import type {
  DiagnosticsSummary,
  DisabledScriptInfo,
  ExtensionDiagnosticItem,
  ExtensionDiagnosticsResponse,
  ExtensionErrorEntry,
  ExtensionManifest,
  ManifestErrorDetail,
  MlxStatus,
  WorkerInfo,
  WorkerRestartResponse,
  WorkersResponse,
} from "@/api/types"

export type {
  DiagnosticsSummary,
  DisabledScriptInfo,
  ExtensionDiagnosticItem,
  ExtensionDiagnosticsResponse,
  ExtensionErrorEntry,
  ExtensionManifest,
  ManifestErrorDetail,
  MlxStatus,
  WorkerInfo,
  WorkerRestartResponse,
  WorkersResponse as WorkersStatusResponse,
}

export const diagnosticsKeys = {
  diagnostics: ["extensions", "diagnostics"] as const,
  workers: keys.workers,
}

export function fetchExtensionDiagnostics(signal?: AbortSignal): Promise<ExtensionDiagnosticsResponse> {
  return api.extensions.diagnostics(signal)
}

export function fetchWorkersStatus(signal?: AbortSignal): Promise<WorkersResponse> {
  return api.workers.status(signal)
}

export function getWorkerApi(workerId: string, signal?: AbortSignal): Promise<WorkerInfo> {
  return api.workers.get(workerId, signal)
}

export function restartWorkerApi(workerId: string): Promise<WorkerRestartResponse> {
  return api.workers.restart(workerId)
}

export const extensionDiagnosticsQuery = queryOptions({
  queryKey: diagnosticsKeys.diagnostics,
  queryFn: ({ signal }) => fetchExtensionDiagnostics(signal),
  staleTime: 15_000,
  retry: false,
})

export const workersStatusQuery = workersQuery

export function refreshDiagnostics() {
  void queryClient.invalidateQueries({ queryKey: diagnosticsKeys.diagnostics })
  void queryClient.invalidateQueries({ queryKey: keys.workers })
}
