/**
 * End-to-end coverage of Forge Neo Remi's creative studio workbench (`VITE_MOCK=1`):
 * - Model capability badges, Worker health pill, and pre-submit Workflow Advice one-click fix
 * - Gallery Boards (switch + create), CompareDialog structured Parameter Diff & RGBA 4-backdrop,
 *   Asset Lineage strip & Lightbox RGBA 4-backdrop toggle
 * - Queue linear multi-stage Pipelines (cache-hit reuse on downstream edits) & Cross-Model Recipes
 * - Extensions Health Diagnostics & Workers monitor (+ restart)
 * - JSON Builder Bria Fibo 2.0 VGL schema mode & CC BY-NC 4.0 license badge
 * Every flow asserts zero console/page errors and zero serious/critical axe-core violations.
 */
import AxeBuilder from "@axe-core/playwright"
import { expect, type Page, test } from "@playwright/test"
import { openGenerate, openRoute, presetStorage, watchConsole } from "./helpers"

async function expectNoA11yViolations(page: Page, context: string) {
  await page.waitForFunction(
    () =>
      document
        .getAnimations()
        .every((animation) => animation.playState !== "running" || animation.effect?.getTiming().iterations === Infinity),
    null,
    { timeout: 5_000 },
  )
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"])
    .analyze()
  const serious = results.violations
    .filter((violation) => violation.impact === "serious" || violation.impact === "critical")
    .map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      help: violation.help,
      nodes: violation.nodes.slice(0, 5).map((node) => `${node.target.join(" ")} :: ${node.failureSummary?.split("\n").slice(1, 3).join(" ")}`),
    }))
  expect(serious, context).toEqual([])
}

test.describe("studio workbench", () => {
  test("generate: worker health pill, workflow advice one-click fix, and model capability badges", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    // Wide desktop viewport so ModelBadges (hidden below 2xl) and WorkerHealthPill are visible in the header.
    await page.setViewportSize({ width: 1600, height: 900 })
    await openGenerate(page, "txt2img")

    // 1. Worker health pill popover
    const workerButton = page.getByRole("button", { name: "Worker status" })
    await expect(workerButton).toBeVisible()
    await workerButton.click()
    const workerDialog = page.getByRole("dialog", { name: "Inference workers" })
    await expect(workerDialog).toBeVisible()
    await expect(workerDialog).toContainText("MLX sidecar")
    await page.keyboard.press("Escape")
    await expect(workerDialog).toBeHidden()

    // 2. Workflow advice one-click fix: pick LCM at default CFG 5 -> advice chip appears -> click "Tune for LCM"
    await page.getByRole("main").getByRole("combobox", { name: "Sampling method" }).click()
    await page.getByRole("option", { name: "LCM", exact: true }).click()

    const adviceRegion = page.getByRole("region", { name: "Workflow advice" })
    await expect(adviceRegion).toBeVisible()
    await expect(adviceRegion).toContainText("LCM works best at low CFG")
    await adviceRegion.getByRole("button", { name: "Tune for LCM" }).click()

    await expect(page.getByRole("textbox", { name: "CFG Scale", exact: true }).first()).toHaveValue("1.5")
    await expect(page.getByRole("textbox", { name: "Sampling steps", exact: true }).first()).toHaveValue("6")
    await expect(adviceRegion.locator('[data-advice-code="sampler.lcm_cfg"]')).toHaveCount(0)

    // 3. Switch UI preset to Qwen-Image 2.1 -> capability badges appear
    await page.getByRole("combobox", { name: "UI preset" }).click()
    await page.getByRole("option", { name: "Qwen-Image 2.1", exact: true }).click()

    const badges = page.locator('[data-slot="model-badges"]').first()
    await expect(badges).toBeVisible()
    await expect(badges).toContainText("Turbo")
    await expect(badges).toContainText("RGBA")

    await expectNoA11yViolations(page, "generate workbench")
    consoleWatch.assertClean()
  })

  test("gallery: boards, compare parameter diff & backdrop, asset lineage, and lightbox 4-backdrop", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openRoute(page, "gallery")
    const grid = page.getByRole("list", { name: "Images" })
    await expect(grid.getByRole("listitem").first()).toBeVisible()

    // 1. Seeded Boards switch & new Board creation
    const boardsTablist = page.getByRole("tablist", { name: "Boards" })
    await expect(boardsTablist.getByRole("tab", { name: /Character references/ })).toBeVisible()
    await boardsTablist.getByRole("tab", { name: /Character references/ }).click()
    await expect(grid.getByRole("listitem")).toHaveCount(3)
    await expect(page.getByRole("button", { name: "Send board to Reference Images" })).toBeVisible()

    await page.getByRole("button", { name: "New board" }).click()
    await page.getByPlaceholder("Board name").fill("E2E Sprint")
    await page.getByRole("button", { name: "Create board" }).click()
    await expect(boardsTablist.getByRole("tab", { name: /E2E Sprint/ })).toHaveAttribute("aria-selected", "true")

    // Return to All boards so every gallery item is visible.
    await boardsTablist.getByRole("tab", { name: "All boards" }).click()
    await expect(grid.getByRole("listitem").nth(1)).toBeVisible()

    // 2. Select 2 images -> CompareDialog structured Parameter Diff & RGBA backdrop toggle
    const tiles = grid.getByRole("listitem")
    const bulk = page.getByRole("toolbar", { name: "Selected images" })
    for (const index of [0, 1]) {
      await tiles.nth(index).hover()
      await tiles.nth(index).getByRole("checkbox").click()
      await expect(bulk).toContainText(`${index + 1} selected`)
    }
    await bulk.getByRole("button", { name: "Compare" }).click()

    const compareDialog = page.getByRole("dialog", { name: "Compare images" })
    await expect(compareDialog).toBeVisible()
    await expect(compareDialog.getByText("Parameters diff")).toBeVisible()
    await expect(compareDialog.getByText("Prompt diff")).toBeVisible()
    await expect(compareDialog.getByRole("button", { name: "Apply A to form" })).toBeVisible()
    await expect(compareDialog.getByRole("button", { name: "Apply B to form" })).toBeVisible()

    // Toggle Compare RGBA backdrop
    await compareDialog.getByRole("group", { name: "RGBA backdrop" }).getByRole("button", { name: "Green screen" }).click()
    await expect(compareDialog.locator('[data-backdrop="green"]').first()).toBeVisible()
    await expectNoA11yViolations(page, "compare dialog")

    await page.keyboard.press("Escape")
    await expect(compareDialog).toBeHidden()
    await bulk.getByRole("button", { name: "Clear selection" }).click()

    // 3. Open second item (has ancestor + descendant lineage) in Lightbox -> Lineage Strip & RGBA 4-backdrop
    await tiles.nth(1).getByRole("button").first().click()
    const viewer = page.getByRole("dialog", { name: "Image viewer" })
    await expect(viewer).toBeVisible()

    const details = page.getByRole("complementary", { name: "Image details" })
    const lineage = details.getByRole("region", { name: "Asset lineage" })
    await expect(lineage).toBeVisible()
    await expect(lineage.getByText("Current")).toBeVisible()
    await expect(lineage.getByText("init_img")).toBeVisible()
    await expect(lineage.getByText("upscale")).toBeVisible()

    const viewerBackdrops = viewer.getByRole("group", { name: "RGBA backdrop" })
    await viewerBackdrops.getByRole("button", { name: "Pure white" }).click()
    await expect(viewer.locator('[data-backdrop="white"]')).toBeVisible()
    await viewerBackdrops.getByRole("button", { name: "Green screen" }).click()
    await expect(viewer.locator('[data-backdrop="green"]')).toBeVisible()
    await expectNoA11yViolations(page, "lightbox with lineage and backdrop")

    await page.keyboard.press("Escape")
    await expect(viewer).toBeHidden()
    consoleWatch.assertClean()
  })

  test("queue: linear multi-stage pipelines cache-hit reuse and cross-model recipes", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openRoute(page, "queue")

    // 1. Pipelines tab: run -> all stages cached -> tweak stage 3 -> stages 1-2 stay cached, stage 3 executes
    await page.getByRole("tab", { name: "Pipelines" }).click()
    await expect(page.locator('[data-cache-decision="execute"]')).toHaveCount(3)

    await page.getByRole("button", { name: "Run pipeline" }).click()
    await expect(page.locator('[data-cache-decision="cached"]')).toHaveCount(3)
    await expect(page.getByRole("region", { name: "Recent pipeline runs" })).toContainText("Generate → Refine → Upscale")

    const factorInput = page.getByLabel("Upscale factor")
    await factorInput.fill("4")
    await expect(page.locator('[data-cache-decision="cached"]')).toHaveCount(2)
    await expect(page.locator('[data-cache-decision="execute"]')).toHaveCount(1)
    await expect(page.getByRole("button", { name: "Re-run modified" })).toBeVisible()

    await page.getByRole("button", { name: "Re-run modified" }).click()
    await expect(page.locator('[data-cache-decision="cached"]')).toHaveCount(3)
    await expectNoA11yViolations(page, "queue pipelines")

    // 2. Recipes tab: seeded cross-model recipes, family filter, and one-click apply
    await page.getByRole("tab", { name: "Recipes" }).click()
    await expect(page.getByText("Anima 2B · Balanced Illustration")).toBeVisible()
    await expect(page.getByText("Qwen-Image-2.1 · Turbo 8-Step RGBA")).toBeVisible()

    await page.getByRole("group", { name: "Model family" }).getByRole("button", { name: "qwen21", exact: true }).click()
    await expect(page.getByText("Qwen-Image-2.1 · Turbo 8-Step RGBA")).toBeVisible()
    await expect(page.getByText("Anima 2B · Balanced Illustration")).toHaveCount(0)

    await page.getByRole("button", { name: "Apply recipe" }).first().click()
    await expect(page.locator("[data-sonner-toast]").filter({ hasText: "Qwen-Image-2.1" })).toBeVisible()
    await expectNoA11yViolations(page, "queue recipes")
    consoleWatch.assertClean()
  })

  test("extensions: health diagnostics, filter, and worker restart", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await openRoute(page, "extensions?tab=diagnostics")

    await expect(page.getByText(/Remi API version:\s*1\.0/)).toBeVisible()
    await expect(page.getByText("worker-0")).toBeVisible()
    await expect(page.getByText("Apple Silicon MLX sidecar")).toBeVisible()
    await expect(page.getByText("sd_forge_image_stitch")).toBeVisible()
    await expect(page.getByText("sd_forge_spectrum")).toBeVisible()
    await expect(page.getByText("legacy_gradio_plugin")).toBeVisible()

    // Filter to issues only hides healthy extensions
    await page.getByRole("button", { name: "Issues only" }).click()
    await expect(page.getByText("sd_forge_spectrum")).toBeVisible()
    await expect(page.getByText("sd_forge_image_stitch")).toHaveCount(0)
    await page.getByRole("button", { name: "All", exact: true }).click()

    // Restart worker increments restart count
    await page.getByRole("button", { name: "Restart worker" }).click()
    await expect(page.locator("[data-sonner-toast]").filter({ hasText: "Worker restarted" })).toBeVisible()
    await expect(page.getByText(/Restarts:\s*1/)).toBeVisible()

    await expectNoA11yViolations(page, "extensions diagnostics")
    consoleWatch.assertClean()
  })

  test("json-builder: Bria Fibo 2.0 VGL schema mode & dark-theme a11y", async ({ page }) => {
    const consoleWatch = watchConsole(page)
    await presetStorage(page, { theme: "dark" })
    await openRoute(page, "ext/json-builder")

    await page.getByRole("radiogroup", { name: "Prompt format" }).getByRole("radio", { name: "Fibo VGL" }).click()
    const vglControls = page.locator('[data-slot="vgl-schema-controls"]')
    await expect(vglControls).toBeVisible()
    await expect(vglControls).toContainText("CC BY-NC 4.0")
    await expect(vglControls.getByLabel("Background setting")).toBeVisible()
    await vglControls.getByLabel("Lighting").fill("golden hour rim light")
    await expect(vglControls.getByLabel("Lighting")).toHaveValue("golden hour rim light")

    await expectNoA11yViolations(page, "json-builder vgl dark")
    consoleWatch.assertClean()
  })
})
