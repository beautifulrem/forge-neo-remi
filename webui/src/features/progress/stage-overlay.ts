/**
 * Which generate forms show their running job over the stage right now
 * (`pages/generate/Stage.tsx` ProgressOverlay). The run bar's status line
 * leaves the step and the progress sheet to that overlay instead of showing
 * them a second time and pushing the form down for the length of the job; it
 * shows them itself when the stage does not (img2img on its Source view, the
 * progress bar turned off in the settings).
 */
import { useEffect } from "react"
import { create } from "zustand"
import type { GenerateTab } from "./run-controller"

export const useStageOverlay = create<Partial<Record<GenerateTab, number>>>(() => ({}))

/** Marks `tab`'s stage as showing its progress while the calling component is mounted. */
export function useShowsStageProgress(tab: GenerateTab) {
  useEffect(() => {
    useStageOverlay.setState((s) => ({ [tab]: (s[tab] ?? 0) + 1 }))
    return () => useStageOverlay.setState((s) => ({ [tab]: Math.max(0, (s[tab] ?? 0) - 1) }))
  }, [tab])
}
