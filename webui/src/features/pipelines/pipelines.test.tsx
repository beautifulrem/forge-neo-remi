import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { computeStageHashes, defaultStage, firstExecuteStage, planPipelineCache } from "./cache-hash"
import { usePipelineDraft } from "./draft-store"
import { PipelineBuilder } from "./PipelineBuilder"
import type { ChainStatus, JobChain, StageSpec } from "./types"

const ws = vi.hoisted(() => ({ handlers: new Map<string, Set<(event: unknown) => void>>() }))
vi.mock("@/api/ws", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/ws")>()),
  onWsEvent: (type: string, handler: (event: unknown) => void) => {
    const set = ws.handlers.get(type) ?? new Set()
    ws.handlers.set(type, set)
    set.add(handler)
    return () => set.delete(handler)
  },
}))

function emit(event: { type: string } & Record<string, unknown>) {
  for (const handler of ws.handlers.get(event.type) ?? []) handler(event)
}

const initialDraft = usePipelineDraft.getState()

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
    usePipelineDraft.setState(initialDraft, true)
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
      if (url.includes("/chains/chain-prev")) return new Response(JSON.stringify(completedRun(runPayload.length ? (runPayload[0] as { stages: StageSpec[] }).stages : [])), { status: 200 })
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

  it("follows its run over chain.updated and lets the server's plan decide", async () => {
    const previews: Record<string, unknown>[] = []
    const withStatus = (chain: JobChain, status: ChainStatus, stage: "running" | "completed"): JobChain => ({
      ...chain,
      status,
      updated_at: `2026-11-23T00:00:0${previews.length}Z`,
      stages: chain.stages.map((st, i) => ({ ...st, status: i === 0 ? stage : stage === "completed" ? "completed" : "pending" })),
    })
    let created: JobChain | null = null
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      const method = init?.method ?? "GET"
      if (url.includes("/chains/templates")) return new Response(JSON.stringify([]), { status: 200 })
      if (url.includes("/chains/preview")) {
        const body = JSON.parse(String(init?.body ?? "{}")) as { stages: StageSpec[]; chain_id?: string }
        previews.push(body)
        // the server says nothing is cached (say the seed is random and this is a new run)
        const stages = body.stages.map((st, index) => ({ index, kind: st.kind, name: st.name ?? "", enabled: true, stage_hash: `h${index}`, cache_hit: false, cached_outputs: [] }))
        return new Response(JSON.stringify({ stages }), { status: 200 })
      }
      if (url.endsWith("/chains") && method === "GET") return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 })
      if (url.endsWith("/chains") && method === "POST") {
        const body = JSON.parse(String(init?.body ?? "{}")) as { stages: StageSpec[] }
        created = withStatus(completedRun(body.stages), "running", "running")
        return new Response(JSON.stringify(created), { status: 201 })
      }
      if (created && url.includes(`/chains/${created.id}`)) return new Response(JSON.stringify(created), { status: 200 })
      return new Response("{}", { status: 200 })
    })

    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(previews.length).toBeGreaterThan(0))
    expect(previews[0]!.chain_id).toBeUndefined()

    fireEvent.click(screen.getByRole("button", { name: "Run pipeline" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel pipeline" })).toBeInTheDocument())

    // the run finishes on the server: the builder hears it and offers to run again
    emit({ type: "chain.updated", chain_id: created!.id, ...withStatus(created!, "completed", "completed") })
    await waitFor(() => expect(screen.queryByRole("button", { name: "Cancel pipeline" })).not.toBeInTheDocument())
    await waitFor(() => expect(previews.at(-1)!.chain_id).toBe(created!.id))

    // the local estimate would call every stage cached; the server's plan says otherwise
    await waitFor(() => expect(screen.getAllByText("Will execute")).toHaveLength(3))
    expect(screen.queryByText("Reused from cached output")).not.toBeInTheDocument()
  })

  it("keeps the pipeline being built across a remount (switching Queue tabs)", () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify([]), { status: 200 }))
    const view = (
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>
    )
    const first = render(view)
    fireEvent.change(screen.getByLabelText("Pipeline name"), { target: { value: "My night shots" } })
    first.unmount()
    render(view)
    expect(screen.getByLabelText("Pipeline name")).toHaveValue("My night shots")
  })
})
