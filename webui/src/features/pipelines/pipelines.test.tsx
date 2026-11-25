import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { queryClient } from "@/api/queries"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
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

  it("lets a number field be emptied while typing and restores it when left empty", () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify([]), { status: 200 }))
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    const factor = screen.getByLabelText("Upscale factor")
    fireEvent.change(factor, { target: { value: "" } })
    expect(factor).toHaveValue(null) // not snapped back to the default while the user retypes
    const finish = () => usePipelineDraft.getState().stages.find((stage) => stage.kind === "finish")!
    expect(finish().core?.upscaling_resize).toBe(2)
    fireEvent.change(factor, { target: { value: "3" } })
    expect(finish().core?.upscaling_resize).toBe(3)
    fireEvent.change(factor, { target: { value: "" } })
    fireEvent.blur(factor)
    expect(factor).toHaveValue(3)
  })

  it("says the run list is on its way while it loads, not that there are no runs", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      const method = init?.method ?? (typeof input === "string" ? "GET" : (input as Request).method)
      if (new URL(url, "http://x").pathname.endsWith("/chains") && method === "GET") return new Promise<Response>(() => {})
      return new Response(JSON.stringify([]), { status: 200 })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    const runs = screen.getByRole("region", { name: "Recent pipeline runs" })
    expect(await within(runs).findByRole("status")).toHaveTextContent("Loading…")
    expect(within(runs).queryByText("No pipeline runs yet")).not.toBeInTheDocument()
    expect(within(runs).getByRole("heading").textContent).toBe("Recent pipeline runs") // no count of 0
  })

  it("keeps a failed run list apart from an empty one and retries it", async () => {
    let fail = true
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      const method = init?.method ?? (typeof input === "string" ? "GET" : (input as Request).method)
      if (new URL(url, "http://x").pathname.endsWith("/chains") && method === "GET") {
        // a client error: the query client does not retry it, so the test needs no backoff
        if (fail) return new Response(JSON.stringify({ detail: "database is locked" }), { status: 400 })
        return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 })
      }
      return new Response(JSON.stringify([]), { status: 200 })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    const retry = await screen.findByRole("button", { name: "Retry" })
    expect(screen.queryByText("No pipeline runs yet")).not.toBeInTheDocument()
    fail = false
    fireEvent.click(retry)
    expect(await screen.findByText("No pipeline runs yet")).toBeInTheDocument()
  })
  // The cards were keyed by kind and place, so a moved stage was a new card: the focus fell to the
  // page, and the arrows said "left" / "right" on a layout that is a column below xl.
  it("keeps the focus on the moved stage's button and names the stage it moves", () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify([]), { status: 200 }))
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    const kinds = () => usePipelineDraft.getState().stages.map((stage) => stage.kind)
    const later = screen.getByRole("button", { name: "Move Base Generate later" })
    later.focus()
    fireEvent.click(later)
    expect(kinds()).toEqual(["refine", "generate", "finish"])
    expect(screen.getByRole("button", { name: "Move Base Generate later" })).toHaveFocus()

    fireEvent.click(screen.getByRole("button", { name: "Move Base Generate later" }))
    expect(kinds()).toEqual(["refine", "finish", "generate"])
    // last now: its "later" button is disabled, so the focus goes to the other one
    expect(screen.getByRole("button", { name: "Move Base Generate later" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Move Base Generate earlier" })).toHaveFocus()
  })
  // The form is read after an await (its schema, its conflicts): the result was written to the
  // stage at the index the button had, which is another stage once the stages were moved meanwhile.
  it("gives a pulled form to the stage that asked for it when the stages moved meanwhile", async () => {
    const schema = tabSchema("txt2img")
    const defaults = tabOwners(schema).find((spec) => spec.owner === "core")!.defaults
    getFormStore("txt2img").setState({ revision: 0, values: { core: { ...defaults, cfg_scale: 3.5 } } })
    const pending: { send?: () => void } = {}
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (!url.includes("/schema")) return new Response(JSON.stringify([]), { status: 200 })
      return new Promise<Response>((resolve) => {
        pending.send = () => resolve(new Response(JSON.stringify(schema), { status: 200, headers: { "Content-Type": "application/json" } }))
      })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    const generate = screen.getAllByRole("listitem").find((item) => item.dataset.stageIndex === "0")!
    fireEvent.click(within(generate).getByRole("button", { name: "Pull from form" }))
    fireEvent.click(screen.getByRole("button", { name: "Move Base Generate later" }))
    await waitFor(() => expect(pending.send).toBeDefined())
    act(() => pending.send!())

    const stage = (kind: string) => usePipelineDraft.getState().stages.find((s) => s.kind === kind)!
    await waitFor(() => expect(stage("generate").core?.cfg_scale).toBe(3.5))
    expect(stage("refine").tab).toBe("img2img")
    expect(stage("refine").core?.cfg_scale).toBeUndefined()
  })
  it("disables Cancel pipeline while the cancel is on its way", async () => {
    const running: JobChain = { ...completedRun(usePipelineDraft.getState().stages), id: "chain-run", status: "running" }
    let cancels = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/chains/chain-run/cancel")) {
        cancels++
        return new Promise<Response>(() => {})
      }
      if (url.includes("/chains/chain-run")) return new Response(JSON.stringify(running), { status: 200 })
      return new Response(JSON.stringify(init?.method === "POST" ? {} : []), { status: 200 })
    })
    usePipelineDraft.setState({ activeRunId: "chain-run" })
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    const cancel = await screen.findByRole("button", { name: "Cancel pipeline" })
    fireEvent.click(cancel)
    await waitFor(() => expect(cancel).toBeDisabled())
    fireEvent.click(cancel)
    expect(cancels).toBe(1)
  })
})

describe("starter pipelines", () => {
  afterEach(async () => {
    vi.restoreAllMocks()
    queryClient.clear()
    usePipelineDraft.setState(initialDraft, true)
    await act(() => i18n.changeLanguage("en"))
  })

  it("names the starter draft and the built-in templates in the current language", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/chains/templates")) return new Response(JSON.stringify([]), { status: 200 })
      if (url.includes("/chains/preview")) return new Response(JSON.stringify({ stages: [] }), { status: 200 })
      if (url.includes("/chains")) return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 })
      return new Response("{}", { status: 200 })
    })
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("textbox", { name: "管线名称" })).toHaveValue("生成 → 精修 → 放大")
    fireEvent.click(screen.getByRole("button", { name: "管线模板" }))
    expect(await screen.findByRole("menuitem", { name: "生成 → 放大" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("menuitem", { name: "提示词扩写 → 生成 → 精修 → 收尾" }))
    expect(screen.getByRole("textbox", { name: "管线名称" })).toHaveValue("提示词扩写 → 生成 → 精修 → 收尾")
  })

  it("names each stage's switch with one sentence in the current language", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify([]), { status: 200 }))
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("switch", { name: `阶段 2：${zhCN["Refine / Inpaint"]}` })).toBeChecked()
  })

  it("shows recent runs' status and cache reuse in the current language", async () => {
    const stages = [defaultStage("generate"), defaultStage("finish")]
    const run = completedRun(stages)
    run.stages[0]!.cached = true
    const failed = { ...completedRun(stages), id: "chain-failed", status: "failed" as ChainStatus }
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/chains/templates")) return new Response(JSON.stringify([]), { status: 200 })
      if (url.includes("/chains/preview")) return new Response(JSON.stringify({ stages: [] }), { status: 200 })
      if (url.includes("/chains")) return new Response(JSON.stringify({ items: [run, failed], total: 2 }), { status: 200 })
      return new Response("{}", { status: 200 })
    })
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("已完成")).toBeInTheDocument()
    expect(screen.getByText("失败")).toBeInTheDocument()
    expect(screen.queryByText("completed")).toBeNull()
    expect(screen.getByText(/复用缓存 1\/2/)).toBeInTheDocument()
  })

  it("keeps the built-in starters in the menu next to saved templates", async () => {
    const saved = { id: "t1", name: "My chain", description: "", model_family: "universal", stages: [], tags: [], created_at: "2026-11-20T10:00:00Z", updated_at: "2026-11-20T10:00:00Z" }
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/chains/templates")) return new Response(JSON.stringify({ items: [saved] }), { status: 200 })
      if (url.includes("/chains/preview")) return new Response(JSON.stringify({ stages: [] }), { status: 200 })
      if (url.includes("/chains")) return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 })
      return new Response("{}", { status: 200 })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineBuilder />
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole("button", { name: "Templates" }))
    expect(await screen.findByRole("menuitem", { name: "My chain" })).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: "Generate → Refine → Upscale" })).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: "Generate → Upscale" })).toBeInTheDocument()
    // the items of one menu are all text or all iconed: Delete template carried the only icon
    const menu = screen.getByRole("menu")
    expect(within(menu).getByRole("menuitem", { name: "Delete template" })).toBeInTheDocument()
    // (a submenu's own chevron aside)
    expect(menu.querySelectorAll("[role=menuitem] svg:not(.cn-rtl-flip)")).toHaveLength(0)
  })
})
