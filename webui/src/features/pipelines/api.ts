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
  run: (id: string) => ["chains", "run", id] as const,
}

/** Refetches the run list and templates (a run in view is kept current by `chain.updated`). */
export function refreshChains() {
  void queryClient.invalidateQueries({ queryKey: chainKeys.list })
  void queryClient.invalidateQueries({ queryKey: chainKeys.templates })
}

export async function getChain(id: string, signal?: AbortSignal): Promise<JobChain> {
  return workbenchRequest<JobChain>(`/chains/${encodeURIComponent(id)}`, { signal })
}

/** One run, kept current by `chain.updated` events (see {@link storeChainRun}). */
export const chainRunQuery = (id: string) =>
  queryOptions({
    queryKey: chainKeys.run(id),
    queryFn: ({ signal }) => getChain(id, signal),
    staleTime: 5_000,
  })

/** Puts a chain the server returned (a mutation's answer or a `chain.updated` event) in the cache. */
export function storeChainRun(chain: JobChain) {
  // an HTTP answer can arrive after the events it raced (a stage that failed at once): keep the newer one
  const known = queryClient.getQueryData<JobChain>(chainKeys.run(chain.id))
  if (known && known.updated_at > chain.updated_at) return
  queryClient.setQueryData(chainKeys.run(chain.id), chain)
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

/**
 * The server's cache plan for `stages`. `chainId`: plan a re-run of that run
 * (stages with a random seed can reuse its outputs; in a new run they cannot).
 */
export async function previewChainApi(stages: readonly StageSpec[], chainId?: string, signal?: AbortSignal): Promise<ChainPreviewResponse | null> {
  try {
    return await workbenchRequest<ChainPreviewResponse>("/chains/preview", {
      method: "POST",
      body: { stages, ...(chainId ? { chain_id: chainId } : {}) },
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
