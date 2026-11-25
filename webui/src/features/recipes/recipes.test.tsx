import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { queryClient } from "@/api/queries"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { captureFormOverrides, captureTabOverrides, countRecipeParams, recipeHighlights } from "./capture"
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

  it("reads a form the app has not opened yet against its schema, so only its drafted changes count", async () => {
    const schema = tabSchema("img2img")
    const defaults = tabOwners(schema).find((spec) => spec.owner === "core")!.defaults
    // the drafts hydrated from localStorage before the img2img page ever loaded: every value, one of them changed
    getFormStore("img2img").setState({ values: { core: { ...defaults, cfg_scale: 3.5 } } })
    expect(captureFormOverrides("img2img").count).toBe(0) // no defaults yet: nothing reads as changed
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify(schema), { status: 200, headers: { "Content-Type": "application/json" } }))
    const captured = await captureTabOverrides("img2img")
    expect(captured.core_patch).toEqual({ cfg_scale: 3.5 })
    expect(captured.scripts_patch).toEqual({})
    expect(captured.count).toBe(1)
    expect(fetch).toHaveBeenCalledTimes(1) // the schema
    expect(getFormStore("img2img").getState().revision).toBe(1)
  })

  // Loading the schema re-derives the conflict locks with a server action; the capture read the
  // form before that action's patch was written.
  it("waits for the conflicts a first load derives before reading the form", async () => {
    const schema = tabSchema("img2img")
    const owners = tabOwners(schema)
    const defaults = (owner: string) => owners.find((spec) => spec.owner === owner)!.defaults
    getFormStore("img2img").setState({ revision: 0, values: { core: { ...defaults("core") }, sampler: { ...defaults("sampler"), sampler_name: "LCM" } } })
    const calls: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      calls.push(new URL(url, "http://x").pathname)
      const body = url.includes("/actions/core/__conflicts__") ? { "sampler.scheduler": { value: "SGM Uniform" } } : schema
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    const captured = await captureTabOverrides("img2img")
    expect(calls.some((path) => path.endsWith("/actions/core/__conflicts__"))).toBe(true)
    expect(captured.scripts_patch.sampler).toEqual({ sampler_name: "LCM", scheduler: "SGM Uniform" })
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

  it("shows no count when the recipes could not load", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ detail: "nope" }), { status: 404, headers: { "Content-Type": "application/json" } }))
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole("button", { name: "Retry" })).toBeInTheDocument()
    // "0" would say the library is empty: nothing is known
    expect(screen.getByRole("heading", { name: /^Cross-model parameter recipes/ })).toHaveTextContent(/^Cross-model parameter recipes$/)
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

  it("says which of the recipe's values a lock overrides, and leaves them out of the count", async () => {
    const { toast } = await import("sonner")
    const success = vi.spyOn(toast, "success")
    const store = getFormStore("txt2img")
    // a toggle that locks the width (`locks: "core.width"`) is on, over a width the recipe changes
    store.setState((s) => ({ locks: [{ owner: "keep", name: "size", target: ["core", "width"] }], values: { ...s.values, core: { ...s.values.core, width: 768 }, keep: { size: true } } }))
    const width = store.getState().values.core?.width
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/recipes/r1/apply")) {
        const patch = { "core.cfg_scale": { value: 1 }, "core.width": { value: 1024 }, "core.height": { value: 1024 }, "sampler.steps": { value: 8 }, "sampler.sampler_name": { value: "Euler" } }
        // the conflict locks hold the steps at 20 (keys: core params bare, the others "owner.param")
        const conflicts = { cfg_scale: { value: 1, enabled: true }, "sampler.steps": { value: 20, enabled: false }, "sampler.sampler_name": { value: "Euler" }, __toast__: { level: "warning", message: "Compatibility: …" }, __advice__: [] }
        const warnings = ["Active model lacks required capabilities: distilled_cfg1"]
        return new Response(JSON.stringify({ recipe: SAMPLE_RECIPE, tab: "txt2img", patch, core: {}, scripts: {}, conflicts, warnings }), { status: 200, headers: { "Content-Type": "application/json" } })
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
    await waitFor(() =>
      expect(success).toHaveBeenCalledWith("Recipe “Qwen Turbo 8-step” applied (3 params)", {
        description: "Active model lacks required capabilities: distilled_cfg1\nOverridden by a lock: Width and Sampling steps",
      }),
    )
    expect(store.getState().values.core?.width).toBe(width)
  })


  // The server formatted the warning in English ("Active model lacks required capabilities: …");
  // it now sends the template and the capability names too, which the toast translates.
  it("says a capability warning in the current language", async () => {
    const { toast } = await import("sonner")
    const success = vi.spyOn(toast, "success")
    const template = "Active model lacks required capabilities: {{capabilities}}"
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/recipes/r1/apply")) {
        const warning_notes = [{ message: template, values: { capabilities: ["Distilled CFG 1", "Spectrum forecasting"] } }]
        const body = { recipe: SAMPLE_RECIPE, tab: "txt2img", patch: { "core.cfg_scale": { value: 1 } }, core: {}, scripts: {}, conflicts: {}, warnings: ["English"], warning_notes }
        return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } })
      }
      return new Response(JSON.stringify({ items: [SAMPLE_RECIPE], total: 1 }), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    i18n.addResourceBundle("zh-CN", "server", { [template]: "当前模型缺少所需能力：{{capabilities}}", "Distilled CFG 1": "CFG 1 蒸馏模式", "Spectrum forecasting": "Spectrum 特征预测" }, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    try {
      render(
        <QueryClientProvider client={queryClient}>
          <RecipeManager />
        </QueryClientProvider>,
      )
      fireEvent.click(await screen.findByRole("button", { name: "应用配方" }))
      await waitFor(() => expect(success).toHaveBeenCalledWith(expect.any(String), { description: "当前模型缺少所需能力：CFG 1 蒸馏模式和Spectrum 特征预测" }))
    } finally {
      await act(() => i18n.changeLanguage("en"))
    }
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
    const noMatch = await screen.findByText("No recipes match")
    expect(screen.queryByText("No recipes yet")).toBeNull()
    // the placeholder pattern: a dimmed glyph over one muted line, and the way out
    expect(noMatch).toHaveClass("text-muted-foreground")
    expect(noMatch.parentElement?.querySelector("svg")).not.toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }))
    expect(await screen.findByText("Qwen Turbo 8-step")).toBeInTheDocument()
    expect(screen.getByRole("searchbox", { name: "Search recipes…" })).toHaveValue("")
    // the button went away with the state: the focus goes to the search, not to the page
    expect(screen.getByRole("searchbox", { name: "Search recipes…" })).toHaveFocus()
  })

  it("keeps the capture dialog's result in one fixed slot while it reads the form", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { "Content-Type": "application/json" } }))
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    fireEvent.click(await screen.findByRole("button", { name: "Capture from form" }))
    // reading (the slot is there, empty), then read: the dialog keeps its height across them
    const slot = () => screen.getByRole("dialog").querySelector("[data-capture-result]")
    expect(slot()).toHaveClass("h-28")
    // (what was read depends on the form: its changes, or a line saying there are none)
    await waitFor(() => expect(slot()?.textContent).not.toBe(""), { timeout: 5_000 })
    expect(slot()).toHaveClass("h-28")
  })

  // Retry emptied the slot while the form was read again: the failure, and the Retry under the
  // focus, went away until the read failed again.
  it("keeps a failed read of the form up while its retry runs, its Retry busy", async () => {
    getFormStore("img2img").setState({ revision: 0, values: {} })
    let schema: "fail" | "hold" = "fail"
    let release = () => {}
    const held = new Promise<void>((resolve) => (release = resolve))
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(typeof input === "string" ? input : (input as Request).url, "http://x")
      if (url.pathname.endsWith("/schema/img2img")) {
        // a 4xx is not retried: the failure shows at once
        if (schema === "fail") return Response.json({ detail: "gone" }, { status: 404 })
        await held
        return Response.json(tabSchema("img2img"))
      }
      return Response.json({ items: [], total: 0 })
    })
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    fireEvent.click(await screen.findByRole("button", { name: "Capture from form" }))
    const dialog = await screen.findByRole("dialog", { name: "Capture recipe" })
    fireEvent.click(within(dialog).getByRole("radio", { name: "img2img" }))
    const alert = await within(dialog).findByRole("alert")
    schema = "hold"
    const retry = within(alert).getByRole("button", { name: "Retry" })
    fireEvent.click(retry)
    await waitFor(() => expect(retry).toHaveAttribute("aria-disabled", "true"))
    expect(alert).toBeInTheDocument()
    await act(async () => release())
    await waitFor(() => expect(within(dialog).queryByRole("alert")).toBeNull())
    expect(dialog.querySelector("[data-capture-result]")?.textContent).not.toBe("")
  })

  it("stacks the model family and tags fields on a phone, where two columns cut their placeholders", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { "Content-Type": "application/json" } }))
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    fireEvent.click(await screen.findByRole("button", { name: "Capture from form" }))
    const row = screen.getByRole("textbox", { name: "Model family" }).closest("label")!.parentElement!
    expect(row).toContainElement(screen.getByRole("textbox", { name: "Tags (comma-separated)" }))
    expect(row).toHaveClass("grid", "grid-cols-1", "sm:grid-cols-2")
    expect(row).not.toHaveClass("grid-cols-2")
  })

  // The toolbar's export, with a search or a family filter on, still saved every recipe.
  it("exports only the recipes a filter shows", async () => {
    const other: Recipe = { ...SAMPLE_RECIPE, id: "r2", name: "SDXL portrait", model_family: "sdxl", tags: [] }
    const exportAll = vi.fn()
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(typeof input === "string" ? input : (input as Request).url, "http://x")
      if (url.pathname.endsWith("/export")) {
        exportAll()
        return Response.json({ version: 1, recipes: [SAMPLE_RECIPE, other] })
      }
      const items = url.searchParams.get("q") === "turbo" ? [SAMPLE_RECIPE] : [SAMPLE_RECIPE, other]
      return Response.json({ items, total: items.length })
    })
    const saved: Blob[] = []
    vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
      saved.push(blob as Blob)
      return "blob:recipes"
    })
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("SDXL portrait")).toBeInTheDocument()
    fireEvent.change(screen.getByRole("searchbox", { name: "Search recipes…" }), { target: { value: "turbo" } })
    await waitFor(() => expect(screen.queryByText("SDXL portrait")).toBeNull())
    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }))
    await waitFor(() => expect(saved).toHaveLength(1))
    const bundle = JSON.parse(await saved[0]!.text()) as { recipes: Recipe[] }
    expect(bundle.recipes.map((recipe) => recipe.id)).toEqual(["r1"])
    expect(exportAll).not.toHaveBeenCalled()
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

  // "Imported 1 recipes (0 updated)"
  it("counts one imported recipe in the singular", async () => {
    const { toast } = await import("sonner")
    const success = vi.spyOn(toast, "success")
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      if (init?.method === "POST") return Response.json({ added: 1, updated: 0, skipped: 0, items: [SAMPLE_RECIPE] })
      return Response.json({ items: [SAMPLE_RECIPE], total: 1 })
    })
    const { container } = render(
      <QueryClientProvider client={queryClient}>
        <RecipeManager />
      </QueryClientProvider>,
    )
    await screen.findByText("Qwen Turbo 8-step")
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!
    fireEvent.change(input, { target: { files: [new File([JSON.stringify({ version: 1, recipes: [SAMPLE_RECIPE] })], "one.json", { type: "application/json" })] } })
    await waitFor(() => expect(success).toHaveBeenCalledWith("Imported 1 recipe (0 updated)"))
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
