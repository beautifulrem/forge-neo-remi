/** Non-node multi-stage linear pipeline types (matching backend contract #12). */
import type { FormTab, ParamValues } from "@/api/types"

export type StageKind = "prompt_upsample" | "generate" | "refine" | "finish"

export type StageStatus = "pending" | "running" | "cached" | "completed" | "failed" | "skipped" | "cancelled"

export type ChainStatus = "queued" | "running" | "paused" | "completed" | "failed" | "cancelled"

export type StageSpec = {
  index?: number
  kind: StageKind
  name?: string
  enabled?: boolean
  tab?: FormTab
  core?: ParamValues
  scripts?: Record<string, ParamValues>
  config?: Record<string, unknown>
}

export type StageRun = StageSpec & {
  index: number
  status: StageStatus
  stage_hash: string
  cached: boolean
  job_id?: string | null
  output_file_ids: string[]
  output_prompt?: string | null
  infotext?: string | null
  error?: string | null
  started_at?: string | null
  finished_at?: string | null
}

export type JobChain = {
  id: string
  name: string
  status: ChainStatus
  current_stage: number
  total_stages: number
  stages: StageRun[]
  created_at: string
  updated_at: string
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
}

export type ChainListResponse = {
  items: JobChain[]
  next_cursor?: string | null
  total: number
}

export type ChainPreviewStage = {
  index: number
  kind: StageKind
  name: string
  enabled: boolean
  stage_hash: string
  cache_hit: boolean
  cached_outputs: string[]
  cached_prompt?: string | null
}

export type ChainPreviewResponse = {
  stages: ChainPreviewStage[]
}

export type ChainTemplate = {
  id: string
  name: string
  description: string
  model_family: string
  stages: StageSpec[]
  tags: string[]
  created_at: string
  updated_at: string
}
