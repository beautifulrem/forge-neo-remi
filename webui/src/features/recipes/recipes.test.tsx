import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { queryClient } from "@/api/queries"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { captureFormOverrides, countRecipeParams, recipeHighlights } from "./capture"
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
})
