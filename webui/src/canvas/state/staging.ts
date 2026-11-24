/**
 * Canvas In-Place Staging store (InvokeAI-inspired): holds 1–4 inpaint/img2img
 * candidate outputs overlaid directly on the active Konva document without
 * leaving the canvas or resetting zoom/pan.
 */
import { create } from "zustand"
import { useProgress } from "@/api/progress"

export const MAX_STAGED_CANDIDATES = 4

export type StagedCandidate = {
  fileId: string
  width?: number
  height?: number
  infotext?: string | null
  label?: string
}

export type StagingSession = {
  candidates: StagedCandidate[]
  index: number
  /** Candidate overlay opacity, 0..1. */
  opacity: number
  /** Temporarily hides the candidate to inspect the original base image. */
  compareOriginal: boolean
  /** Split-wipe before/after overlay mode. */
  splitCompare: boolean
  /** Split divider position, 0..1. */
  splitPosition: number
}

type StagingState = {
  sessions: Record<string, StagingSession>
  /** Keep the img2img stage on the canvas and auto-stage completed runs into it. */
  stayOnCanvas: boolean
  stageCandidates: (docKey: string, candidates: readonly StagedCandidate[]) => void
  next: (docKey: string) => void
  prev: (docKey: string) => void
  select: (docKey: string, index: number) => void
  setOpacity: (docKey: string, opacity: number) => void
  setCompareOriginal: (docKey: string, value: boolean) => void
  toggleSplitCompare: (docKey: string) => void
  setSplitPosition: (docKey: string, position: number) => void
  discardCurrent: (docKey: string) => void
  discardAll: (docKey: string) => void
  setStayOnCanvas: (enabled: boolean) => void
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

export const useCanvasStaging = create<StagingState>()((set) => ({
  sessions: {},
  stayOnCanvas: false,

  stageCandidates: (docKey, raw) => {
    const candidates = raw.filter((c) => Boolean(c.fileId)).slice(0, MAX_STAGED_CANDIDATES)
    if (candidates.length === 0) return
    set((s) => ({
      sessions: {
        ...s.sessions,
        [docKey]: {
          candidates,
          index: 0,
          opacity: 1,
          compareOriginal: false,
          splitCompare: false,
          splitPosition: 0.5,
        },
      },
    }))
  },

  next: (docKey) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session || session.candidates.length <= 1) return s
      return {
        sessions: {
          ...s.sessions,
          [docKey]: { ...session, index: (session.index + 1) % session.candidates.length, compareOriginal: false },
        },
      }
    }),

  prev: (docKey) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session || session.candidates.length <= 1) return s
      return {
        sessions: {
          ...s.sessions,
          [docKey]: {
            ...session,
            index: (session.index - 1 + session.candidates.length) % session.candidates.length,
            compareOriginal: false,
          },
        },
      }
    }),

  select: (docKey, index) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session) return s
      const clamped = Math.min(session.candidates.length - 1, Math.max(0, index))
      return {
        sessions: {
          ...s.sessions,
          [docKey]: { ...session, index: clamped, compareOriginal: false },
        },
      }
    }),

  setOpacity: (docKey, opacity) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session) return s
      return { sessions: { ...s.sessions, [docKey]: { ...session, opacity: clamp01(opacity) } } }
    }),

  setCompareOriginal: (docKey, value) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session) return s
      return { sessions: { ...s.sessions, [docKey]: { ...session, compareOriginal: value } } }
    }),

  toggleSplitCompare: (docKey) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session) return s
      return {
        sessions: {
          ...s.sessions,
          [docKey]: { ...session, splitCompare: !session.splitCompare, compareOriginal: false },
        },
      }
    }),

  setSplitPosition: (docKey, position) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session) return s
      return { sessions: { ...s.sessions, [docKey]: { ...session, splitPosition: clamp01(position) } } }
    }),

  discardCurrent: (docKey) =>
    set((s) => {
      const session = s.sessions[docKey]
      if (!session) return s
      const remaining = session.candidates.filter((_, i) => i !== session.index)
      if (remaining.length === 0) {
        const rest = { ...s.sessions }
        delete rest[docKey]
        return { sessions: rest }
      }
      return {
        sessions: {
          ...s.sessions,
          [docKey]: {
            ...session,
            candidates: remaining,
            index: Math.min(session.index, remaining.length - 1),
          },
        },
      }
    }),

  discardAll: (docKey) =>
    set((s) => {
      if (!(docKey in s.sessions)) return s
      const rest = { ...s.sessions }
      delete rest[docKey]
      return { sessions: rest }
    }),

  setStayOnCanvas: (enabled) => set({ stayOnCanvas: enabled }),
}))

/** Docs whose staging bar takes the focus when it shows next, with when that was asked. */
const focusRequests = new Map<string, number>()

/**
 * Asks the doc's staging bar to take the focus when it shows: staged by a button that goes away
 * with the view it sits in (the focus would fall to the page). Results staged on their own (a
 * finished run) never ask: the user may be typing a prompt.
 */
export function requestStagingFocus(docKey: string) {
  focusRequests.set(docKey, Date.now())
}

/** Whether the doc's staging bar should take the focus now (once, and only shortly after asked). */
export function takeStagingFocus(docKey: string): boolean {
  const asked = focusRequests.get(docKey)
  focusRequests.delete(docKey)
  return asked !== undefined && Date.now() - asked < 5_000
}

let wired = false

/**
 * Automatically captures completed `img2img` job outputs (1–4 candidates) into
 * the `"img2img"` canvas staging area whenever `stayOnCanvas` is enabled.
 */
export function wireImg2imgStaging() {
  if (wired) return
  wired = true
  useProgress.subscribe((s, prev) => {
    const job = s.last.img2img
    if (!job || job === prev.last.img2img || job.error) return
    if (!useCanvasStaging.getState().stayOnCanvas) return
    const candidates: StagedCandidate[] = job.outputs
      .filter((output) => (output.kind ?? "image") === "image" && !output.role)
      .slice(0, MAX_STAGED_CANDIDATES)
      .map((output) => ({
        fileId: output.id,
        width: output.width ?? undefined,
        height: output.height ?? undefined,
        infotext: output.infotext ?? job.infotext ?? null,
      }))
    if (candidates.length > 0) useCanvasStaging.getState().stageCandidates("img2img", candidates)
  })
}
