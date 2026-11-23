import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { queryClient } from "@/api/queries"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { captureFormOverrides, countRecipeParams, recipeHighlights } from "./capture"
import { recipeFileName } from "./file-name"
import { RecipeManager } from "./RecipeManager"
import type { Recipe } from "./types"

const SAMPLE_RECIPE: Recipe = {
  id: "r1",
  name: "Qwen Turbo 8-step",
  description: "Fast 8-step setup",
  model_family: "qwen21",
  capabilities: ["distilled_cfg1"],
  tab: "txt2img",
  core_patch: { cfg_scale: 1, width: 1024, height: 1024 },
  scripts_patch: { sampler: { steps: 8, sampler_name: "Euler" } },
  tags: ["turbo"],
  created_at: "2026-11-22T00:00:00Z",
  updated_at: "2026-11-22T00:00:00Z",
}

describe("Recipes capture and manager UI", () => {
  beforeEach(() => {
    getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("captures only modified persistable overrides from a form store", () => {
    const store = getFormStore("txt2img")
    expect(captureFormOverrides("txt2img").count).toBe(0)
    store.getState().setValue("core", "cfg_scale", 2.5)
    store.getState().setValue("sampler", "steps", 12)

    const captured = captureFormOverrides("txt2img")
    expect(captured.count).toBe(2)
    expect(captured.core_patch).toEqual({ cfg_scale: 2.5 })
    expect(captured.scripts_patch).toEqual({ sampler: { steps: 12 } })
    expect(countRecipeParams(SAMPLE_RECIPE)).toBe(5)
    expect(recipeHighlights(SAMPLE_RECIPE)).toContain("8 steps")
    expect(recipeHighlights(SAMPLE_RECIPE)).toContain("CFG 1")
  })

  it("keeps the recipes when a refetch fails", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ items: [SAMPLE_RECIPE], total: 1 }), { status: 200, headers: { "Content-Type": "application/json" } }))
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Qwen Turbo 8-step")).toBeInTheDocument()
    fetch.mockImplementation(async () => new Response(JSON.stringify({ detail: "bad gateway" }), { status: 502, headers: { "Content-Type": "application/json" } }))
    await act(() => queryClient.refetchQueries())
    expect(queryClient.getQueryState(["recipes", "list", {}])?.status).toBe("error")
    // (the query notifies its observers on the next tick)
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)))
    expect(screen.getByText("Qwen Turbo 8-step")).toBeInTheDocument()
  })

  it("lists recipes, filters by search, and applies a recipe to the form", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/recipes/r1/apply")) {
        return new Response(
          JSON.stringify({
            recipe: SAMPLE_RECIPE,
            tab: "txt2img",
            patch: {},
            core: SAMPLE_RECIPE.core_patch,
            scripts: SAMPLE_RECIPE.scripts_patch,
            conflicts: [],
            warnings: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        )
      }
      return new Response(JSON.stringify({ items: [SAMPLE_RECIPE], total: 1 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    })

    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )

    expect(await screen.findByText("Qwen Turbo 8-step")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Apply recipe" }))
    await waitFor(() => expect(getFormStore("txt2img").getState().values.core?.cfg_scale).toBe(1))
    expect(getFormStore("txt2img").getState().values.sampler?.steps).toBe(8)
  })
  it("applies only the recipe's own values from the merged form the server answers, and counts those", async () => {
    const { toast } = await import("sonner")
    const success = vi.spyOn(toast, "success")
    const store = getFormStore("txt2img")
    const before = store.getState().values
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/recipes/r1/apply")) {
        const body = JSON.parse(String(init?.body ?? (await (input as Request).text())))
        // the server answers the whole form merged with the recipe, and the recipe's targets in `patch`
        const { core, ...scripts } = { core: { ...body.core, ...SAMPLE_RECIPE.core_patch }, ...body.scripts }
        for (const [owner, values] of Object.entries(SAMPLE_RECIPE.scripts_patch)) scripts[owner] = { ...scripts[owner], ...values }
        const patch: Record<string, unknown> = { "core.cfg_scale": { value: 1 }, "core.width": { value: 1024 }, "core.height": { value: 1024 }, "sampler.steps": { value: 8 }, "sampler.sampler_name": { value: "Euler" } }
        return new Response(JSON.stringify({ recipe: SAMPLE_RECIPE, tab: "txt2img", patch, core, scripts, conflicts: {}, warnings: [] }), { status: 200, headers: { "Content-Type": "application/json" } })
      }
      return new Response(JSON.stringify({ items: [SAMPLE_RECIPE], total: 1 }), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Qwen Turbo 8-step")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Apply recipe" }))
    await waitFor(() => expect(success).toHaveBeenCalledWith("Recipe “Qwen Turbo 8-step” applied (5 params)", expect.anything()))
    const after = store.getState().values
    expect(after.core?.cfg_scale).toBe(1)
    expect(after.sampler?.steps).toBe(8)
    expect(after.core?.prompt).toEqual(before.core?.prompt)
  })

  it("names forms and parameters in the current language", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ items: [{ ...SAMPLE_RECIPE, core_patch: { ...SAMPLE_RECIPE.core_patch, denoising_strength: 0.35 } }], total: 1 }), { status: 200, headers: { "Content-Type": "application/json" } }))
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    try {
      render(
        <QueryClientProvider client={queryClient}>
          <RecipeManager />
        </QueryClientProvider>,
      )
      expect(await screen.findByText("Qwen Turbo 8-step")).toBeInTheDocument()
      const row = screen.getByText("Qwen Turbo 8-step").closest("li")!
      expect(row).toHaveTextContent("文生图")
      expect(row).toHaveTextContent("8 步")
      expect(row).toHaveTextContent("重绘幅度 0.35")
      expect(row).not.toHaveTextContent("txt2img")
    } finally {
      await act(() => i18n.changeLanguage("en"))
    }
  })

  it("tells a search that matched nothing from an empty library, keeps the list while a search loads, and clears the filters", async () => {
    let release: (() => void) | null = null
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(typeof input === "string" ? input : (input as Request).url, "http://x")
      const q = url.searchParams.get("q")
      if (q === "zzz") {
        await new Promise<void>((resolve) => {
          release = resolve
        })
        return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { "Content-Type": "application/json" } })
      }
      return new Response(JSON.stringify({ items: q ? [] : [SAMPLE_RECIPE], total: q ? 0 : 1 }), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Qwen Turbo 8-step")).toBeInTheDocument()
    fireEvent.change(screen.getByRole("searchbox", { name: "Search recipes…" }), { target: { value: "zzz" } })
    // while the search is in flight the previous list stays (no skeleton flash)
    await waitFor(() => expect(release).not.toBeNull())
    expect(screen.getByText("Qwen Turbo 8-step")).toBeInTheDocument()
    expect(document.querySelector("[aria-busy=true]")).toBeNull()
    act(() => release!())
    expect(await screen.findByText("No recipes match")).toBeInTheDocument()
    expect(screen.queryByText("No recipes yet")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }))
    expect(await screen.findByText("Qwen Turbo 8-step")).toBeInTheDocument()
    expect(screen.getByRole("searchbox", { name: "Search recipes…" })).toHaveValue("")
  })

  it("names the export when it fails", async () => {
    const { toast } = await import("sonner")
    const error = vi.spyOn(toast, "error")
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/export")) return new Response(JSON.stringify({ detail: "nope" }), { status: 404, headers: { "Content-Type": "application/json" } })
      return new Response(JSON.stringify({ items: [SAMPLE_RECIPE], total: 1 }), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    await screen.findByText("Qwen Turbo 8-step")
    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }))
    await waitFor(() => expect(error).toHaveBeenCalled())
    expect(error.mock.calls[0]![0]).toBe("Could not export the recipes")
  })

  it("names a recipe's export after the recipe, whatever its script", () => {
    expect(recipeFileName({ ...SAMPLE_RECIPE, name: "夏日人像" })).toBe("夏日人像.json")
    expect(recipeFileName({ ...SAMPLE_RECIPE, name: "Portrait / 夏: v2?" })).toBe("Portrait - 夏- v2.json")
    expect(recipeFileName({ ...SAMPLE_RECIPE, name: " /// " })).toBe("r1.json")
  })

  it("says a picked file is not a recipes file instead of blaming the server", async () => {
    const { toast } = await import("sonner")
    const error = vi.spyOn(toast, "error")
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ items: [SAMPLE_RECIPE], total: 1 }), { status: 200, headers: { "Content-Type": "application/json" } }))
    const { container } = render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    await screen.findByText("Qwen Turbo 8-step")
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!
    fireEvent.change(input, { target: { files: [new File(["{not json"], "bad.json", { type: "application/json" })] } })
    await waitFor(() => expect(error).toHaveBeenCalled())
    expect(error.mock.calls[0]![1]).toMatchObject({ description: "This file is not a recipes export." })
  })

  it("does not report a recipe as applied when the server could not apply it", async () => {
    const { toast } = await import("sonner")
    const success = vi.spyOn(toast, "success")
    const error = vi.spyOn(toast, "error")
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/apply")) return new Response(JSON.stringify({ detail: "gone" }), { status: 404, headers: { "Content-Type": "application/json" } })
      return new Response(JSON.stringify({ items: [SAMPLE_RECIPE], total: 1 }), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    await screen.findByText("Qwen Turbo 8-step")
    getFormStore("txt2img").getState().setValue("core", "cfg_scale", 7)
    fireEvent.click(screen.getByRole("button", { name: "Apply recipe" }))
    await waitFor(() => expect(error).toHaveBeenCalled())
    expect(success).not.toHaveBeenCalled()
    expect(getFormStore("txt2img").getState().values.core?.cfg_scale).toBe(7)
  })
})
