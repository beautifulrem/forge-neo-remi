/**
 * The pipeline being built and the run it follows. Kept outside the builder
 * so switching to another Queue tab and back does not reset them.
 */
import { create } from "zustand"
import { STARTER_PIPELINES } from "./cache-hash"
import type { StageSpec } from "./types"

type Update<T> = T | ((prev: T) => T)

type PipelineDraft = {
  name: string | null
  stages: StageSpec[]
  /** The run shown in the builder; its data comes from `chainRunQuery`. */
  activeRunId: string | null
  setName: (name: string) => void
  setStages: (update: Update<StageSpec[]>) => void
  setActiveRunId: (id: string | null) => void
}

const starter = STARTER_PIPELINES[0]!

export const usePipelineDraft = create<PipelineDraft>()((set) => ({
  // null: the starter's name, shown in the current language until the user names the pipeline
  name: null,
  stages: starter.stages.map((s) => structuredClone(s)),
  activeRunId: null,
  setName: (name) => set({ name }),
  setStages: (update) => set((state) => ({ stages: typeof update === "function" ? update(state.stages) : update })),
  setActiveRunId: (activeRunId) => set({ activeRunId }),
}))

/**
 * A key for each stage of the draft that stays with it through moves (a stage is the plain object
 * the API takes, with no id of its own): the builder keys its cards by it, so a moved card keeps
 * its DOM and its focus.
 */
const stageKeys = new WeakMap<StageSpec, number>()
let lastKey = 0

export function stageKey(stage: StageSpec): number {
  let key = stageKeys.get(stage)
  if (key === undefined) {
    key = ++lastKey
    stageKeys.set(stage, key)
  }
  return key
}

/** `next` replaces `stage` in the draft (an edit): it keeps the stage's key. */
export function keepStageKey(stage: StageSpec, next: StageSpec): StageSpec {
  stageKeys.set(next, stageKey(stage))
  return next
}
