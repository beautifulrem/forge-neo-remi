/**
 * End-to-end workbench integration tests (`VITE_MOCK=1`):
 * - WorkerHealthPill + DiagnosticsView shared cache consistency & restart
 * - LivePreview 2-slot double-buffer frame retention
 * - Header ErrorBoundary lazy-chunk failure recovery ("Try again")
 * - WorkflowAdviceBar reactive manual edits (cfg_scale / steps) & clearing
 * - ScriptsPanel inapplicable drawer disabled controls
 * - ModelBadges reactive route transitions (#/txt2img <-> #/img2img)
 * - CompareDialog numeric/parameter diff rendering
 * - PipelineBuilder stage cache & PUP mode labels
 * - RecipeManager schema-settled cross-tab apply, family filter & edit
 */
import { expect, test } from "@playwright/test"
import { mod, openGenerate, openRoute, watchConsole } from "./helpers"

test.describe("Workbench integration", () => {
  test("WorkerHealthPill and DiagnosticsView share worker status and reflect restarts", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await page.setViewportSize({ width: 1600, height: 900 })
    await openGenerate(page, "txt2img")

    const workerButton = page.getByRole("button", { name: "Worker status" })
    await expect(workerButton).toBeVisible()
    await workerButton.click()
    const popover = page.getByRole("dialog", { name: "Inference workers" })
    await expect(popover).toBeVisible()
    await expect(popover).toContainText("Idle")
    await expect(popover).toContainText("32.0 GB")
    await expect(popover).toContainText("14 done")
    await page.keyboard.press("Escape")
    await expect(popover).toBeHidden()

    await openRoute(page, "extensions?tab=diagnostics")
    await expect(page.getByText("worker-0")).toBeVisible()
    await expect(page.getByText("Optional mflux sidecar not installed")).toBeVisible()
    await expect(page.getByText(/Jobs completed:\s*14/)).toBeVisible()
    await expect(page.getByText(/Failed:\s*0/)).toBeVisible()
    await expect(page.getByText(/Restarts:\s*0/)).toBeVisible()

    await page.getByRole("button", { name: "Restart worker" }).click()
    await expect(page.locator("[data-sonner-toast]").filter({ hasText: "Worker restarted" })).toBeVisible()
    await expect(page.getByText(/Restarts:\s*1/)).toBeVisible()

    // Returning to generate reads the shared ["workers"] cache consistently.
    await openGenerate(page, "txt2img")
    await page.getByRole("button", { name: "Worker status" }).click()
    await expect(page.getByRole("dialog", { name: "Inference workers" })).toContainText("14 done")
    consoleWatch.assertClean()
  })

  test("LivePreview double-buffers current and previous blob frames without broken images", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openGenerate(page, "txt2img")

    // Set 9 steps so the mock worker emits multiple binary preview frames (steps 3, 6, 9).
    const stepsInput = page.getByLabel("Sampling steps", { exact: true }).first()
    await stepsInput.fill("9")
    await stepsInput.blur()

    await page.keyboard.press(`${mod}+Enter`)
    const current = page.locator('[data-frame="current"]')
    const previous = page.locator('[data-frame="previous"]')
    await expect(current).toBeVisible({ timeout: 15_000 })
    await expect(previous).toBeVisible({ timeout: 15_000 })

    const srcs = await Promise.all([current.getAttribute("src"), previous.getAttribute("src")])
    expect(srcs[0]).toMatch(/^blob:/)
    expect(srcs[1]).toMatch(/^blob:/)
    expect(srcs[0]).not.toBe(srcs[1])

    const decoded = await Promise.all([
      current.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0),
      previous.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0),
    ])
    expect(decoded).toEqual([true, true])

    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 20_000 })
    consoleWatch.assertClean()
  })

  test("Header ErrorBoundary isolates lazy control failures and recovers on Try again", async ({ page }) => {
    const pageErrors: string[] = []
    page.on("pageerror", (error) => pageErrors.push(error.message))

    let blockMenu = true
    await page.route("**/AppearanceMenu*", (route) => (blockMenu ? route.abort() : route.continue()))

    await page.goto("/#/txt2img")
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible({ timeout: 20_000 })

    const retryButton = page.getByRole("button", { name: /^Appearance \(Try again\)/ })
    await expect(retryButton).toBeVisible({ timeout: 10_000 })

    blockMenu = false
    await page.unroute("**/AppearanceMenu*")
    await retryButton.click()

    const appearanceButton = page.getByRole("button", { name: "Appearance", exact: true })
    await expect(appearanceButton).toBeVisible({ timeout: 10_000 })
    await appearanceButton.click()
    await expect(page.getByRole("menu")).toBeVisible()
    expect(pageErrors).toEqual([])
  })

  test("WorkflowAdviceBar reacts to manual cfg_scale and steps edits and clears when fixed", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await page.setViewportSize({ width: 1600, height: 900 })
    await openGenerate(page, "txt2img")

    // Switch to Qwen-Image 2.1 (defaults: 8 steps, CFG 1 -> no turbo_mismatch advice initially).
    await page.getByRole("combobox", { name: "UI preset" }).click()
    await page.getByRole("option", { name: "Qwen-Image 2.1", exact: true }).click()

    const adviceRegion = page.getByRole("region", { name: "Workflow advice" })
    await expect(adviceRegion.locator('[data-advice-code="qwen21.turbo_mismatch"]')).toHaveCount(0)

    // Manually raise CFG scale (non-conflict param) -> advice appears reactively.
    const cfgInput = page.getByRole("textbox", { name: "CFG Scale", exact: true }).first()
    await cfgInput.fill("7")
    await cfgInput.blur()
    await expect(adviceRegion.locator('[data-advice-code="qwen21.turbo_mismatch"]')).toBeVisible({ timeout: 10_000 })

    // Manually lower CFG scale back to 1 -> advice disappears reactively.
    await cfgInput.fill("1")
    await cfgInput.blur()
    await expect(adviceRegion.locator('[data-advice-code="qwen21.turbo_mismatch"]')).toHaveCount(0, { timeout: 10_000 })

    // Manually raise Sampling steps -> advice reappears, then clearing back to 8 removes it.
    const stepsInput = page.getByRole("textbox", { name: "Sampling steps", exact: true }).first()
    await stepsInput.fill("24")
    await stepsInput.blur()
    await expect(adviceRegion.locator('[data-advice-code="qwen21.turbo_mismatch"]')).toBeVisible({ timeout: 10_000 })

    await stepsInput.fill("8")
    await stepsInput.blur()
    await expect(adviceRegion.locator('[data-advice-code="qwen21.turbo_mismatch"]')).toHaveCount(0, { timeout: 10_000 })
    consoleWatch.assertClean()
  })

  test("ScriptsPanel disables controls inside inapplicable scripts drawer", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openGenerate(page, "txt2img")

    // Switch to Qwen-Image 2.1 where Anima lacks `anima_hooks` capability.
    await page.getByRole("combobox", { name: "UI preset" }).click()
    await page.getByRole("option", { name: "Qwen-Image 2.1", exact: true }).click()

    const drawer = page.locator('[data-slot="inapplicable-scripts"]')
    await expect(drawer).toBeVisible()
    await drawer.getByRole("button", { name: /Not applicable for active model/ }).click()
    await expect(drawer).toContainText("Requires Anima engine hooks")

    const animaToggle = drawer.getByRole("switch", { name: "Enable Anima" })
    await expect(animaToggle).toHaveAttribute("aria-disabled", "true")
    consoleWatch.assertClean()
  })

  test("ModelBadges updates reactively across txt2img and img2img route transitions", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await page.setViewportSize({ width: 1600, height: 900 })
    await openGenerate(page, "txt2img")

    await page.getByRole("combobox", { name: "UI preset" }).click()
    await page.getByRole("option", { name: "Qwen-Image 2.1", exact: true }).click()

    const badges = page.locator('[data-slot="model-badges"]').first()
    await expect(badges).toContainText("Turbo")
    await expect(badges).toContainText("RGBA")

    const modeNav = page.getByRole("navigation", { name: "Mode" })
    await modeNav.getByRole("link", { name: "img2img", exact: true }).click()
    await expect(page.getByRole("button", { name: "Add a source image" })).toBeVisible({ timeout: 15_000 })
    await expect(badges).toContainText("Turbo")
    await expect(badges).toContainText("RGBA")

    await modeNav.getByRole("link", { name: "txt2img", exact: true }).click()
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
    await expect(badges).toContainText("Turbo")
    consoleWatch.assertClean()
  })

  test("CompareDialog renders parameter diffs cleanly for numeric and scalar differences", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openRoute(page, "gallery")
    const grid = page.getByRole("list", { name: "Images" })
    const tiles = grid.getByRole("listitem")
    await expect(tiles.nth(1)).toBeVisible()

    for (const index of [0, 1]) {
      await tiles.nth(index).hover()
      await tiles.nth(index).getByRole("checkbox").click()
    }
    await page.getByRole("toolbar", { name: "Selected images" }).getByRole("button", { name: "Compare" }).click()

    const dialog = page.getByRole("dialog", { name: "Compare images" })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByText("Parameters diff")).toBeVisible()
    await expect(dialog.getByText("Prompt diff")).toBeVisible()
    await expect(dialog.getByText("Core parameters")).toBeVisible()
    await expect(dialog.getByText("Seed")).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(dialog).toBeHidden()
    consoleWatch.assertClean()
  })

  test("PipelineBuilder runs stages and surfaces PUP mode labels", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openRoute(page, "queue")
    await page.getByRole("tab", { name: "Pipelines" }).click()

    await page.getByRole("button", { name: "Add stage" }).click()
    await page.getByRole("menuitem", { name: "Prompt upsample" }).click()
    await expect(page.getByText("PUP mode")).toBeVisible()

    await page.getByRole("button", { name: "Run pipeline" }).click()
    await expect(page.locator('[data-cache-decision="cached"]').first()).toBeVisible({ timeout: 10_000 })
    consoleWatch.assertClean()
  })

  test("RecipeManager settles unopened tab schema on apply, filters families, and edits recipes", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openRoute(page, "queue?view=recipes")

    // Filter to flux3 (img2img recipe) on a fresh session where #/img2img was never opened.
    await page.getByRole("group", { name: "Model family" }).getByRole("button", { name: "flux3", exact: true }).click()
    await expect(page.getByText("FLUX.3 Klein · Detail Refine")).toBeVisible()

    // Edit flow while on #/queue?view=recipes
    const editButton = page.getByRole("button", { name: /^Edit FLUX\.3 Klein/ })
    await expect(editButton).toBeVisible()
    await editButton.click()
    const editDialog = page.getByRole("dialog")
    await expect(editDialog).toBeVisible()
    await editDialog.getByPlaceholder("Description").fill("Updated refine pass")
    await editDialog.getByRole("button", { name: "Save", exact: true }).click()
    await expect(editDialog).toBeHidden()
    await expect(page.getByText("Updated refine pass")).toBeVisible()

    // Apply img2img recipe while #/img2img has never been opened: settles schema, applies, and opens #/img2img.
    await page.getByRole("button", { name: "Apply recipe" }).first().click()
    await expect(page.locator("[data-sonner-toast]").filter({ hasText: "FLUX.3 Klein · Detail Refine" })).toBeVisible()
    await expect(page.getByRole("textbox", { name: "CFG Scale", exact: true }).first()).toHaveValue("2.5", { timeout: 15_000 })
    await expect(page.getByRole("textbox", { name: "Denoising strength", exact: true }).first()).toHaveValue("0.35")
    consoleWatch.assertClean()
  })
})
