/**
 * Tests for the Wave 2 mock API handlers (`src/api/mock/workbench.ts`, `library.ts`,
 * `fixtures.ts`, `server.ts`) exercising every Workbench endpoint via `handleMockRequest`.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { bootstrap, tabSchema } from "./fixtures"
import { handleMockRequest } from "./server"
import { mockAdviceFor, mockCapabilitiesForPreset, resetMockWorkbench } from "./workbench"

async function call(method: string, path: string, body?: unknown) {
  const init: RequestInit = { method, headers: { Accept: "application/json" } }
  if (body !== undefined) {
    init.headers = { Accept: "application/json", "Content-Type": "application/json" }
    init.body = JSON.stringify(body)
  }
  const response = await handleMockRequest(new Request(`http://localhost/api/v2${path}`, init))
  const data = response.status === 204 ? undefined : await response.json()
  return { status: response.status, data }
}

describe("mock workbench routes", () => {
  const initialPreset = bootstrap.options.forge_preset
  const initialCheckpoint = bootstrap.options.sd_model_checkpoint

  beforeEach(() => {
    resetMockWorkbench()
    bootstrap.options.forge_preset = initialPreset
    bootstrap.options.sd_model_checkpoint = initialCheckpoint
  })

  afterEach(() => {
    bootstrap.options.forge_preset = initialPreset
    bootstrap.options.sd_model_checkpoint = initialCheckpoint
  })

  it("serves capabilities, advice, /rules/derive, core/__advice__ and core/__forge_preset__", async () => {
    const xlCaps = mockCapabilitiesForPreset("xl", "animagine-xl-4.0.safetensors", "Automatic")
    expect(xlCaps.preset).toBe("xl")
    expect(xlCaps.engine_label).toBe("SDXL")

    const qwenCaps = mockCapabilitiesForPreset("qwen-2.1", "qwen_image_2.1_diffusers_fp8", "float8-e4m3fn")
    expect(qwenCaps.preset).toBe("qwen21")
    expect(qwenCaps.supports_rgba).toBe(true)
    expect(qwenCaps.reference_slot_format).toBe("<image_{i}>")
    expect(qwenCaps.badges).toEqual(expect.arrayContaining(["Diffusers", "FP8", "Turbo", "RGBA", "Control"]))

    expect(mockCapabilitiesForPreset("flux-3", "model.gguf").badges).toContain("GGUF")
    expect(mockCapabilitiesForPreset("anima", "anima-convrot.safetensors", "int8").badges).toContain("INT8-ConvRot")
    expect(mockCapabilitiesForPreset("flux", "flux-nvfp4.safetensors").badges).toContain("NVFP4")
    expect(mockCapabilitiesForPreset("flux", "flux-svdq.safetensors").badges).toContain("SVDQuant")

    // Default xl schema has capabilities + all scripts applicable
    const xlSchema = tabSchema("txt2img")
    expect(xlSchema.capabilities?.preset).toBe("xl")
    expect(xlSchema.advice).toEqual([])
    expect(xlSchema.scripts.every((s) => s.applicable !== false)).toBe(true)

    // Switching preset via core/__forge_preset__ updates capabilities, advice, and script gating
    const presetRes = await call("POST", "/actions/core/__forge_preset__", { tab: "txt2img", preset: "qwen21", values: { cfg_scale: 5 } })
    expect(presetRes.status).toBe(200)
    expect(presetRes.data.__capabilities__.preset).toBe("qwen21")
    expect(presetRes.data.__settings__.forge_preset).toBe("qwen21")

    const qwenSchema = tabSchema("txt2img")
    expect(qwenSchema.capabilities?.preset).toBe("qwen21")
    const animaScript = qwenSchema.scripts.find((s) => s.id === "anima")
    expect(animaScript?.applicable).toBe(false)
    expect(animaScript?.inapplicable_reason).toContain("Anima engine hooks")

    // core/__advice__ & /rules/derive emit actionable advice patches
    const adviceRes = await call("POST", "/actions/core/__advice__", {
      tab: "txt2img",
      preset: "qwen21",
      values: { cfg_scale: 5 },
      scripts: { sampler: { steps: 28, sampler_name: "Euler" } },
    })
    expect(adviceRes.status).toBe(200)
    expect(adviceRes.data.__advice__.some((a: { code: string }) => a.code === "qwen21.turbo_mismatch")).toBe(true)

    const rulesRes = await call("POST", "/rules/derive", {
      tab: "txt2img",
      core: { enable_hr: true, cfg_scale: 5 },
      scripts: { refiner: { enable: true }, sampler: { sampler_name: "LCM", steps: 20 } },
      selected: "sampler.sampler_name",
    })
    expect(rulesRes.status).toBe(200)
    expect(rulesRes.data["sampler.scheduler"]).toEqual({ value: "SGM Uniform", enabled: false })
    expect(rulesRes.data.__advice__.some((a: { code: string }) => a.code === "sampler.lcm_cfg")).toBe(true)

    // Extra advice branches (JPEG + RGBA, FLUX, Anima, MOSS)
    expect(mockAdviceFor("qwen21", { cfg_scale: 1 }, { sampler: { steps: 8 } }, { samples_format: "jpg" }).some((a) => a.code === "qwen21.rgba_jpeg")).toBe(true)
    expect(mockAdviceFor("flux3", { cfg_scale: 4 }).some((a) => a.code === "flux3.distilled_cfg")).toBe(true)
    expect(mockAdviceFor("anima", { cfg_scale: 7 }, { sampler: { steps: 12 } }).some((a) => a.code === "anima.sampling_range")).toBe(true)
    expect(mockAdviceFor("moss", { cfg_scale: 6 }).some((a) => a.code === "moss.cfg_hint")).toBe(true)
  })

  it("supports full Recipes CRUD, filters, import/export, and apply", async () => {
    const listed = await call("GET", "/recipes?model_family=qwen21&tag=rgba&capability=rgba_output&q=turbo")
    expect(listed.status).toBe(200)
    expect(listed.data.total).toBe(1)

    const bad = await call("POST", "/recipes", { name: "   " })
    expect(bad.status).toBe(422)

    const created = await call("POST", "/recipes", {
      name: "Studio Portrait",
      description: "Clean lighting",
      model_family: "anima",
      capabilities: ["anima_hooks"],
      tab: "txt2img",
      core_patch: { cfg_scale: 4.5, width: 832, height: 1216 },
      scripts_patch: { sampler: { steps: 32, sampler_name: "ER SDE" } },
      tags: ["portrait"],
    })
    expect(created.status).toBe(200)
    const id = created.data.id as string

    const fetched = await call("GET", `/recipes/${id}`)
    expect(fetched.status).toBe(200)
    expect(fetched.data.name).toBe("Studio Portrait")

    const patched = await call("PATCH", `/recipes/${id}`, { name: "Studio Portrait v2", tags: ["portrait", "v2"] })
    expect(patched.status).toBe(200)
    expect(patched.data.name).toBe("Studio Portrait v2")

    const applied = await call("POST", `/recipes/${id}/apply`, { tab: "txt2img", core: { prompt: "hi" }, scripts: {} })
    expect(applied.status).toBe(200)
    expect(applied.data.core.cfg_scale).toBe(4.5)
    expect(applied.data.patch["sampler.steps"]).toEqual({ value: 32 })
    expect(applied.data.warnings.length).toBeGreaterThan(0) // xl lacks anima_hooks

    const exported = await call("GET", "/recipes/export")
    expect(exported.status).toBe(200)
    expect(exported.data.version).toBe(1)

    const imported = await call("POST", "/recipes/import", {
      recipes: [
        { name: "" }, // skipped
        { id, name: "Studio Portrait Overwritten", core_patch: { cfg_scale: 4 } },
        { name: "Brand New Imported", model_family: "xl", core_patch: { cfg_scale: 5 } },
      ],
      overwrite: true,
    })
    expect(imported.status).toBe(200)
    expect(imported.data).toMatchObject({ added: 1, updated: 1, skipped: 1 })

    const skipDup = await call("POST", "/recipes/import", { recipes: [{ id, name: "Skip Me" }], overwrite: false })
    expect(skipDup.data.skipped).toBe(1)

    expect((await call("DELETE", `/recipes/${id}`)).status).toBe(204)
    expect((await call("GET", `/recipes/${id}`)).status).toBe(404)
    expect((await call("DELETE", `/recipes/${id}`)).status).toBe(404)
  })

  it("supports multi-stage Chains with deterministic stage caching, rerun, lifecycle, and templates", async () => {
    const tpls = await call("GET", "/chains/templates")
    expect(tpls.status).toBe(200)
    expect(tpls.data.items.length).toBeGreaterThanOrEqual(3)

    const stages = [
      { kind: "prompt_upsample", name: "PUP", enabled: true, core: { prompt: "red fox" } },
      { kind: "generate", name: "Base", enabled: true, tab: "txt2img", core: { prompt: "red fox", cfg_scale: 5 }, scripts: { sampler: { steps: 20 } } },
      { kind: "refine", name: "Refine", enabled: true, tab: "img2img", core: { denoising_strength: 0.35 } },
      { kind: "finish", name: "Skip Finish", enabled: false, tab: "extras", core: { upscaling_resize: 2 } },
    ]

    const preview1 = await call("POST", "/chains/preview", { stages })
    expect(preview1.status).toBe(200)
    expect(preview1.data.stages.every((s: { cache_hit: boolean }) => !s.cache_hit)).toBe(true)

    const created = await call("POST", "/chains", { name: "Fox pipeline", stages })
    expect(created.status).toBe(200)
    const chainId = created.data.id as string
    expect(created.data.status).toBe("completed")
    expect(created.data.stages[0].output_prompt).toContain("red fox")
    expect(created.data.stages[3].status).toBe("skipped")

    // Preview after tweaking ONLY stage 2 (Refine) -> stages 0 & 1 hit cache, stage 2 misses!
    const tweaked = structuredClone(stages)
    tweaked[2]!.core = { denoising_strength: 0.5 }
    const preview2 = await call("POST", "/chains/preview", { stages: tweaked })
    expect(preview2.data.stages[0].cache_hit).toBe(true)
    expect(preview2.data.stages[1].cache_hit).toBe(true)
    expect(preview2.data.stages[2].cache_hit).toBe(false)

    // Rerun reuses cached upstream stages automatically
    const rerun = await call("POST", `/chains/${chainId}/rerun`, { stages: tweaked })
    expect(rerun.status).toBe(200)
    expect(rerun.data.stages[0].cached).toBe(true)
    expect(rerun.data.stages[1].cached).toBe(true)
    expect(rerun.data.stages[2].cached).toBe(false)

    expect((await call("GET", "/chains?status=completed")).data.total).toBe(1)
    expect((await call("GET", `/chains/${chainId}`)).status).toBe(200)
    expect((await call("POST", `/chains/${chainId}/pause`)).data.status).toBe("paused")
    expect((await call("POST", `/chains/${chainId}/resume`)).data.status).toBe("completed")
    // a finished run cannot be cancelled (it would overwrite its result), as on the server
    expect((await call("POST", `/chains/${chainId}/cancel`)).status).toBe(409)
    expect((await call("DELETE", `/chains/${chainId}`)).status).toBe(204)
    expect((await call("GET", `/chains/${chainId}`)).status).toBe(404)

    // Templates CRUD
    const newTpl = await call("POST", "/chains/templates", { name: "Custom 2-step", stages: stages.slice(1, 3), tags: ["fast"] })
    expect(newTpl.status).toBe(200)
    const tplId = newTpl.data.id as string
    expect((await call("GET", `/chains/templates/${tplId}`)).status).toBe(200)
    expect((await call("DELETE", `/chains/templates/${tplId}`)).status).toBe(204)
    expect((await call("GET", `/chains/templates/${tplId}`)).status).toBe(404)
  })

  it("supports Gallery Boards, ?board_id= filtering, Asset Lineage, and Structured Compare Diff", async () => {
    const listed = await call("GET", "/boards")
    expect(listed.status).toBe(200)
    expect(listed.data.items.length).toBe(2)

    // Filtering gallery by seeded board returns only that board's items
    const filtered = await call("GET", "/gallery?board_id=board-characters")
    expect(filtered.status).toBe(200)
    expect(filtered.data.total).toBe(3)

    // Create, rename, add/remove items, delete board
    const created = await call("POST", "/boards", { name: "Lighting studies", description: "Rim light tests", color: "#5b6ee1" })
    expect(created.status).toBe(200)
    const boardId = created.data.id as string
    expect((await call("POST", "/boards", { name: "Lighting studies" })).status).toBe(409)
    expect((await call("GET", `/boards/${boardId}`)).data.name).toBe("Lighting studies")

    const added = await call("POST", `/boards/${boardId}/items`, { file_ids: ["mock-lib5-185", "mock-lib6-222"] })
    expect(added.data).toEqual({ board_id: boardId, count: 2 })
    expect((await call("GET", `/gallery?board_id=${boardId}`)).data.total).toBe(2)

    const removed = await call("POST", `/boards/${boardId}/items/remove`, { file_ids: ["mock-lib5-185"] })
    expect(removed.data).toEqual({ board_id: boardId, count: 1 })
    expect((await call("DELETE", `/boards/${boardId}/items/mock-lib6-222`)).status).toBe(204)
    expect((await call("PATCH", `/boards/${boardId}`, { name: "Lighting v2" })).data.name).toBe("Lighting v2")
    expect((await call("DELETE", `/boards/${boardId}`)).status).toBe(204)
    expect((await call("GET", `/boards/${boardId}`)).status).toBe(404)

    // Lineage
    const lineage = await call("GET", "/gallery/mock-lib2-74/lineage")
    expect(lineage.status).toBe(200)
    expect(lineage.data.item.file_id).toBe("mock-lib2-74")
    expect(lineage.data.ancestors.length).toBe(1)
    expect(lineage.data.descendants.length).toBe(1)
    expect((await call("GET", "/gallery/nonexistent-id/lineage")).status).toBe(404)

    // Compare by gallery IDs & by raw infotext
    const cmpIds = await call("POST", "/gallery/compare", { id_a: "mock-lib0-0", id_b: "mock-lib1-37" })
    expect(cmpIds.status).toBe(200)
    expect(cmpIds.data.identical).toBe(false)
    expect(cmpIds.data.core_diffs.some((d: { key: string }) => d.key === "Seed")).toBe(true)
    expect(cmpIds.data.adopt_b_patch.scripts.seed.seed).toBe(100001)

    const cmpText = await call("POST", "/gallery/compare", {
      infotext_a: "a fox\nSteps: 20, Sampler: Euler, CFG scale: 5, Size: 512x512",
      infotext_b: "a fox, sunset\nSteps: 30, Sampler: Euler a, Schedule type: Karras, CFG scale: 4, Denoising strength: 0.4, Size: 1024x1024",
    })
    expect(cmpText.status).toBe(200)
    expect(cmpText.data.prompt_diff.added).toContain("sunset")
    expect(cmpText.data.adopt_b_patch.core).toMatchObject({ cfg_scale: 4, denoising_strength: 0.4, width: 1024, height: 1024 })
    expect(cmpText.data.adopt_b_patch.scripts.sampler).toMatchObject({ steps: 30, sampler_name: "Euler a", scheduler: "Karras" })
    expect((await call("POST", "/gallery/compare", {})).status).toBe(422)
    expect((await call("POST", "/gallery/compare", { id_a: "missing" })).status).toBe(404)
  })

  it("supports Extension Diagnostics, Workers status/restart, and Multimodal PUP upsample", async () => {
    const diag = await call("GET", "/extensions/diagnostics")
    expect(diag.status).toBe(200)
    expect(diag.data.api_version).toBe("1.0")
    expect(diag.data.summary.total).toBe(4)

    const workers = await call("GET", "/workers")
    expect(workers.status).toBe(200)
    expect(workers.data.workers.length).toBe(1)
    const wid = workers.data.workers[0].worker_id as string
    expect((await call("GET", `/workers/${wid}`)).data.worker_id).toBe(wid)

    const restarted = await call("POST", `/workers/${wid}/restart`)
    expect(restarted.status).toBe(200)
    expect(restarted.data.ok).toBe(true)
    expect(restarted.data.worker.restarts).toBe(1)
    expect((await call("GET", "/workers/missing")).status).toBe(404)
    expect((await call("POST", "/workers/missing/restart")).status).toBe(404)

    // PUP upsample across modes & families
    const pupQwen = await call("POST", "/prompt-tools/upsample", {
      prompt: "girl with umbrella",
      model_family: "qwen21",
      reference_file_ids: ["mock-lib0-0"],
      aspect_ratio: "16:9",
    })
    expect(pupQwen.status).toBe(200)
    expect(pupQwen.data.mode).toBe("i2i")
    expect(pupQwen.data.prompt).toContain("<image_1>")
    expect(pupQwen.data.used_reference_count).toBe(1)

    const pupFibo = await call("POST", "/prompt-tools/upsample", { prompt: "ceramic vase", model_family: "fibo" })
    expect(pupFibo.status).toBe(200)
    expect(pupFibo.data.mode).toBe("vgl_json")
    expect(pupFibo.data.vgl_json.short_description).toBe("ceramic vase")

    const pupBbox = await call("POST", "/prompt-tools/upsample", { prompt: "two cats", model_family: "flux3", mode: "t2i_bbox", reference_images: ["data:image/png;base64,AA=="] })
    expect(pupBbox.status).toBe(200)
    expect(pupBbox.data.layout_rows).toHaveLength(1)
    expect(pupBbox.data.prompt).toContain("ref_image_0")

    const pupAnima = await call("POST", "/prompt-tools/upsample", { prompt: "1girl, forest", model_family: "anima", instructions: "soft sunlight" })
    expect(pupAnima.data.prompt).toContain("soft sunlight")

    expect((await call("POST", "/prompt-tools/upsample", { prompt: "   " })).status).toBe(422)
  })
})
