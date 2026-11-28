/**
 * Thin typed API client for Multi-Stage Linear Pipelines (`/api/v2/chains`),
 * backed by `@/api/endpoints`.
 */
import { queryOptions } from "@tanstack/react-query"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import type { ChainListResponse, ChainPreviewResponse, ChainTemplate, JobChain, StageSpec } from "./types"

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
  return api.chains.get(id, signal)
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
  const runKey = chainKeys.run(chain.id)
  if (typeof chain.updated_at !== "string" || chain.updated_at.length === 0) {
    void queryClient.invalidateQueries({ queryKey: runKey })
    return
  }
  const known = queryClient.getQueryData<JobChain>(runKey)
  if (known?.updated_at && known.updated_at > chain.updated_at) return
  queryClient.setQueryData(runKey, chain)
}

export async function listChains(signal?: AbortSignal): Promise<ChainListResponse> {
  try {
    const data = (await api.chains.list({ limit: 25 }, signal)) as ChainListResponse | JobChain[] | undefined
    if (Array.isArray(data)) return { items: data, next_cursor: null, total: data.length }
    return {
      items: Array.isArray(data?.items) ? data.items : [],
      next_cursor: data?.next_cursor ?? null,
      total: typeof data?.total === "number" ? data.total : 0,
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return { items: [], next_cursor: null, total: 0 }
    throw error // shown with a retry, not cached as "no runs yet"
  }
}

export async function listChainTemplates(signal?: AbortSignal): Promise<ChainTemplate[]> {
  try {
    const data = (await api.chains.templates(signal)) as ChainTemplate[] | { items?: ChainTemplate[] } | undefined
    if (Array.isArray(data)) return data
    if (data && typeof data === "object" && Array.isArray(data.items)) return data.items
    return []
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return []
    throw error
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
export async function previewChainApi(stages: readonly StageSpec[], chainId?: string, signal?: AbortSignal): Promise<ChainPreviewResponse> {
  return api.chains.preview({ stages, ...(chainId ? { chain_id: chainId } : {}) }, signal)
}

export async function createChainApi(body: {
  name?: string
  stages: readonly StageSpec[]
  run_now?: boolean
  priority?: number
  model_fingerprint?: string
}): Promise<JobChain> {
  return api.chains.create({ run_now: true, ...body })
}

export async function rerunChainApi(
  id: string,
  body: { stages?: readonly StageSpec[]; from_stage?: number; force_stages?: number[]; model_fingerprint?: string },
): Promise<JobChain> {
  return api.chains.rerun(id, body)
}

export async function pauseChainApi(id: string): Promise<JobChain> {
  return api.chains.pause(id)
}

export async function resumeChainApi(id: string): Promise<JobChain> {
  return api.chains.resume(id)
}

export async function cancelChainApi(id: string): Promise<JobChain> {
  return api.chains.cancel(id)
}

export async function saveChainTemplateApi(body: {
  name: string
  description?: string
  model_family?: string
  stages: readonly StageSpec[]
  tags?: string[]
}): Promise<ChainTemplate> {
  return api.chains.createTemplate(body)
}

export async function deleteChainTemplateApi(id: string): Promise<void> {
  await api.chains.deleteTemplate(id)
}
