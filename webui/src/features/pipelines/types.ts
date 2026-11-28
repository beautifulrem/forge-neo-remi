/** Non-node multi-stage linear pipeline types aligned with `@/api/types`. */
import type {
  ChainListResponse as ApiChainListResponse,
  ChainPreviewResponse,
  ChainPreviewStage,
  ChainStatus,
  ChainTemplate,
  JobChain as ApiJobChain,
  StageKind,
  StageRun as ApiStageRun,
  StageSpec,
  StageStatus,
} from "@/api/types"

export type {
  ChainPreviewResponse,
  ChainPreviewStage,
  ChainStatus,
  ChainTemplate,
  StageKind,
  StageSpec,
  StageStatus,
}

export type StageRun = Omit<ApiStageRun, "name" | "enabled" | "tab" | "core" | "scripts" | "config"> & StageSpec

export type JobChain = Omit<ApiJobChain, "stages"> & {
  stages: StageRun[]
}

export type ChainListResponse = Omit<ApiChainListResponse, "items"> & {
  items: JobChain[]
}
