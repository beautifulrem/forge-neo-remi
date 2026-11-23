/**
 * The pipeline being built and the run it follows. Kept outside the builder
 * so switching to another Queue tab and back does not reset them.
 */
import { create } from "zustand"
import { STARTER_PIPELINES } from "./cache-hash"
import type { StageSpec } from "./types"

type Update<T> = T | ((prev: T) => T)

type PipelineDraft = {
  name: string
  stages: StageSpec[]
  /** The run shown in the builder; its data comes from `chainRunQuery`. */
  activeRunId: string | null
  setName: (name: string) => void
  setStages: (update: Update<StageSpec[]>) => void
  setActiveRunId: (id: string | null) => void
}

const starter = STARTER_PIPELINES[0]!

export const usePipelineDraft = create<PipelineDraft>()((set) => ({
  name: starter.name,
  stages: starter.stages.map((s) => structuredClone(s)),
  activeRunId: null,
  setName: (name) => set({ name }),
  setStages: (update) => set((state) => ({ stages: typeof update === "function" ? update(state.stages) : update })),
  setActiveRunId: (activeRunId) => set({ activeRunId }),
}))
