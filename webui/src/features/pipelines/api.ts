/**
 * Thin typed API client for Multi-Stage Linear Pipelines (`/api/v2/chains`),
 * backed by `workbenchRequest` (which routes `VITE_MOCK=1` through `@/api/mock/workbench`).
 */
import { queryOptions } from "@tanstack/react-query"
import { queryClient } from "@/api/queries"
import type { ChainListResponse, ChainPreviewResponse, ChainTemplate, JobChain, StageSpec } from "./types"
import { workbenchRequest } from "./workbench-http"

export const chainKeys = {
  all: ["chains"] as const,
  list: ["chains", "list"] as const,
  templates: ["chains", "templates"] as const,
}

export function refreshChains() {
  void queryClient.invalidateQueries({ queryKey: chainKeys.all })
}

export async function listChains(signal?: AbortSignal): Promise<ChainListResponse> {
  try {
    const data = await workbenchRequest<ChainListResponse | JobChain[]>("/chains", { query: { limit: 25 }, signal })
    if (Array.isArray(data)) return { items: data, next_cursor: null, total: data.length }
    return {
      items: Array.isArray(data?.items) ? data.items : [],
      next_cursor: data?.next_cursor ?? null,
      total: typeof data?.total === "number" ? data.total : 0,
    }
  } catch {
    return { items: [], next_cursor: null, total: 0 }
  }
}

export async function listChainTemplates(signal?: AbortSignal): Promise<ChainTemplate[]> {
  try {
    const data = await workbenchRequest<ChainTemplate[] | { items?: ChainTemplate[] }>("/chains/templates", { signal })
    if (Array.isArray(data)) return data
    if (data && typeof data === "object" && Array.isArray(data.items)) return data.items
    return []
  } catch {
    return []
  }
}

export const chainsQuery = queryOptions({
  queryKey: chainKeys.list,
  queryFn: ({ signal }) => listChains(signal),
  staleTime: 10_000,
})

export const chainTemplatesQuery = queryOptions({
  queryKey: chainKeys.templates,
  queryFn: ({ signal }) => listChainTemplates(signal),
  staleTime: 30_000,
})

export async function previewChainApi(stages: readonly StageSpec[], modelFingerprint?: string, signal?: AbortSignal): Promise<ChainPreviewResponse | null> {
  try {
    return await workbenchRequest<ChainPreviewResponse>("/chains/preview", {
      method: "POST",
      body: { stages, ...(modelFingerprint ? { model_fingerprint: modelFingerprint } : {}) },
      signal,
    })
  } catch {
    return null
  }
}

export async function createChainApi(body: {
  name?: string
  stages: readonly StageSpec[]
  run_now?: boolean
  priority?: number
  model_fingerprint?: string
}): Promise<JobChain> {
  return workbenchRequest<JobChain>("/chains", { method: "POST", body: { run_now: true, ...body } })
}

export async function rerunChainApi(
  id: string,
  body: { stages?: readonly StageSpec[]; from_stage?: number; force_stages?: number[]; model_fingerprint?: string },
): Promise<JobChain> {
  return workbenchRequest<JobChain>(`/chains/${encodeURIComponent(id)}/rerun`, { method: "POST", body })
}

export async function cancelChainApi(id: string): Promise<JobChain> {
  return workbenchRequest<JobChain>(`/chains/${encodeURIComponent(id)}/cancel`, { method: "POST" })
}

export async function saveChainTemplateApi(body: {
  name: string
  description?: string
  model_family?: string
  stages: readonly StageSpec[]
  tags?: string[]
}): Promise<ChainTemplate> {
  return workbenchRequest<ChainTemplate>("/chains/templates", { method: "POST", body })
}

export async function deleteChainTemplateApi(id: string): Promise<void> {
  await workbenchRequest<void>(`/chains/templates/${encodeURIComponent(id)}`, { method: "DELETE" })
}
