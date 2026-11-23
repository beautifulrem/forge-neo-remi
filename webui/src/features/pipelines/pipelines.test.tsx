import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { computeStageHashes, defaultStage, firstExecuteStage, planPipelineCache } from "./cache-hash"
import { PipelineBuilder } from "./PipelineBuilder"
import type { JobChain, StageSpec } from "./types"

function completedRun(stages: StageSpec[]): JobChain {
  const hashes = computeStageHashes(stages)
  const now = "2026-11-22T12:00:00Z"
  return {
    id: "chain-prev",
    name: "Previous run",
    status: "completed",
    current_stage: stages.length - 1,
    total_stages: stages.length,
    stages: stages.map((spec, index) => ({
      ...spec,
      index,
      status: spec.enabled === false ? "skipped" : "completed",
      stage_hash: hashes[index]!,
      cached: false,
      output_file_ids: [`file-stage-${index}`],
      output_prompt: spec.kind === "prompt_upsample" ? "upsampled" : null,
    })),
    created_at: now,
    updated_at: now,
  }
}

describe("pipeline stage hashing & cache-hit planner", () => {
  it("reuses cached upstream stages when only a downstream stage is modified", () => {
    const initial: StageSpec[] = [
      defaultStage("prompt_upsample"),
      defaultStage("generate"),
      defaultStage("refine"),
      defaultStage("finish"),
    ]
    const prev = completedRun(initial)

    // Unchanged: all 4 stages hit cache.
    const allHit = planPipelineCache(initial, { previousRun: prev })
    expect(allHit.map((s) => s.decision)).toEqual(["cached", "cached", "cached", "cached"])
    expect(firstExecuteStage(allHit)).toBeNull()

    // Modify Stage 3 (Refine denoising strength): Stages 1 & 2 stay cached, Stages 3 & 4 execute.
    const tweaked = initial.map((s, i) => (i === 2 ? { ...s, core: { ...s.core, denoising_strength: 0.55 } } : s))
    const partial = planPipelineCache(tweaked, { previousRun: prev })
    expect(partial.map((s) => s.decision)).toEqual(["cached", "cached", "execute", "execute"])
    expect(partial[1]!.cached_outputs).toEqual(["file-stage-1"])
    expect(firstExecuteStage(partial)).toBe(2)

    // Modify Stage 1: invalidates all downstream stages 1..4.
    const rootChange = initial.map((s, i) => (i === 0 ? { ...s, core: { prompt: "new scene" } } : s))
    expect(planPipelineCache(rootChange, { previousRun: prev }).map((s) => s.decision)).toEqual(["execute", "execute", "execute", "execute"])
  })

  it("marks disabled stages as skipped", () => {
    const stages = [defaultStage("generate"), { ...defaultStage("refine"), enabled: false }, defaultStage("finish")]
    const plan = planPipelineCache(stages)
    expect(plan.map((s) => s.decision)).toEqual(["execute", "skipped", "execute"])
  })
})

describe("PipelineBuilder UI", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("shows cache-hit vs will-execute badges and re-runs from modified stage", async () => {
    const runPayload: unknown[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      const method = init?.method ?? (typeof input === "string" ? "GET" : (input as Request).method)
      if (url.includes("/chains/templates")) return new Response(JSON.stringify([]), { status: 200 })
      if (url.includes("/chains/preview")) return new Response(JSON.stringify({ stages: [] }), { status: 200 })
      if (url.endsWith("/chains") && method === "GET") return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 })
      if (url.endsWith("/chains") && method === "POST") {
        const body = JSON.parse(String(init?.body ?? "{}")) as { name: string; stages: StageSpec[] }
        runPayload.push(body)
        return new Response(JSON.stringify(completedRun(body.stages)), { status: 201 })
      }
      return new Response("{}", { status: 200 })
    })

    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )

    // Initially all 3 starter stages show "Will execute".
    expect(screen.getAllByText("Will execute")).toHaveLength(3)

    // Run pipeline -> all 3 stages complete and now show "Reused from cached output".
    fireEvent.click(screen.getByRole("button", { name: "Run pipeline" }))
    await waitFor(() => expect(screen.getAllByText("Reused from cached output")).toHaveLength(3))
    expect(runPayload).toHaveLength(1)

    // Tweak Stage 3 (Upscale factor): Stages 1 & 2 stay cached, Stage 3 switches to "Will execute".
    const factorInput = screen.getByLabelText("Upscale factor")
    fireEvent.change(factorInput, { target: { value: "4" } })
    expect(screen.getAllByText("Reused from cached output")).toHaveLength(2)
    expect(screen.getAllByText("Will execute")).toHaveLength(1)
    expect(screen.getByRole("button", { name: "Re-run modified" })).toBeInTheDocument()
  })
})
