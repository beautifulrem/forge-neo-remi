/** Gallery, PNG Info and the Queue against the mock backend. */
import { expect, type Page, test } from "@playwright/test"
import { injectFaults } from "./helpers"

async function openGallery(page: Page) {
  await page.goto("/#/gallery")
  const grid = page.getByRole("list", { name: "Images" })
  await expect(grid.getByRole("listitem").first()).toBeVisible()
  return grid
}

test.describe("gallery", () => {
  test("a failing server shows the error, and the toolbar stops saying Loading…", async ({ page }) => {
    await injectFaults(page, [{ path: "^/gallery(/facets)?$", status: 500 }])
    await page.goto("/#/gallery")
    await expect(page.getByText("Gallery unavailable")).toBeVisible({ timeout: 20_000 })
    await expect(page.getByRole("main").getByText("Loading…")).toHaveCount(0)
    await expect(page.getByRole("main").locator("[aria-busy=true]")).toHaveCount(0)
  })

  // A dialog opened from a menu item lost the focus to the menu's button once the menu had faded
  // out: typing went nowhere and Ctrl+A selected the gallery behind the dialog.
  test("a dialog opened from a menu keeps the focus", async ({ page }) => {
    await openGallery(page)
    await page.getByLabel("Boards").getByText("Character references").click()
    await page.getByRole("button", { name: "Board actions" }).click()
    await page.getByRole("menuitem", { name: "Rename board" }).click()
    const dialog = page.getByRole("alertdialog")
    await expect(dialog).toBeVisible()
    await expect(page.getByRole("menu")).toHaveCount(0)
    await page.waitForTimeout(500) // past the menu's fade
    expect(await page.evaluate(() => Boolean(document.activeElement?.closest("[role=alertdialog]")))).toBe(true)
    await page.keyboard.type(" v2")
    await expect(dialog.getByRole("textbox")).toHaveValue(/ v2$/)
    await page.keyboard.press("Escape")
    await expect(dialog).toBeHidden()
    await expect(page.getByRole("button", { name: "Board actions" })).toBeFocused()

    // the second time, once the dialog's code is loaded, it opens within the menu's fade
    for (let time = 0; time < 2; time++) {
      await page.getByRole("button", { name: "Appearance" }).click()
      await page.getByRole("menuitem", { name: /Keyboard shortcuts/ }).click()
      await expect(page.getByRole("dialog")).toBeVisible()
      await page.waitForTimeout(500)
      expect(await page.evaluate(() => Boolean(document.activeElement?.closest("[role=dialog]"))), `time ${time + 1}`).toBe(true)
      await page.keyboard.press("Escape")
      await expect(page.getByRole("dialog")).toBeHidden()
    }
  })

  test("filters by search and opens the viewer with the parameters beside the image", async ({ page }) => {
    const grid = await openGallery(page)
    await page.getByRole("searchbox", { name: "Search the gallery" }).fill("lighthouse")
    await expect(page.getByText(/images match/)).toBeVisible()
    await expect(grid.getByRole("button", { name: /^lighthouse at dusk/ }).first()).toBeVisible()
    await expect(grid.getByRole("button", { name: /^a red fox/ })).toHaveCount(0)

    await grid.getByRole("button", { name: /^lighthouse at dusk/ }).first().click()
    const details = page.getByRole("complementary", { name: "Image details" })
    await expect(details).toBeVisible()
    await expect(details.getByRole("button", { name: "PNG Info" })).toBeVisible()
    await expect(details.getByText("lighthouse at dusk, volumetric light").first()).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(details).toBeHidden()
  })

  test("selects two images and compares them", async ({ page }) => {
    const grid = await openGallery(page)
    const tiles = grid.getByRole("listitem")
    const bulk = page.getByRole("toolbar", { name: "Selected images" })
    for (const index of [0, 1]) {
      await tiles.nth(index).hover()
      await tiles.nth(index).getByRole("checkbox").click()
      await expect(bulk).toContainText(`${index + 1} selected`)
    }
    await bulk.getByRole("button", { name: "Compare" }).click()
    const dialog = page.getByRole("dialog")
    const slider = dialog.getByRole("slider", { name: "Comparison divider" })
    await expect(slider).toBeVisible()
    await expect(slider).toHaveAttribute("aria-valuenow", "50")
    // The dialog moves the focus when it opens: retry until the key reaches the handle.
    await expect(async () => {
      await slider.press("Home")
      await expect(slider).toHaveAttribute("aria-valuenow", "0", { timeout: 500 })
    }).toPass()
    // the close button in the corner covers none of the header's controls (a click there closed the dialog)
    for (const width of [1440, 1024]) {
      await page.setViewportSize({ width, height: 900 })
      const covered = await dialog.evaluate((root) => {
        const close = [...root.querySelectorAll<HTMLElement>("[data-slot=dialog-close]")].at(-1)!.getBoundingClientRect()
        return [...root.querySelectorAll<HTMLElement>("button, [role=radio]")]
          .filter((element) => element.getAttribute("data-slot") !== "dialog-close")
          .map((element) => ({ name: element.getAttribute("aria-label") || element.textContent?.trim() || "", box: element.getBoundingClientRect() }))
          .filter(({ box }) => box.width > 0 && Math.min(box.right, close.right) - Math.max(box.left, close.left) > 0.5 && Math.min(box.bottom, close.bottom) - Math.max(box.top, close.top) > 0.5)
          .map(({ name }) => name)
      })
      expect(covered, `at ${width} px`).toEqual([])
    }
  })

  test("sends an image's parameters to txt2img through PNG Info", async ({ page }) => {
    const grid = await openGallery(page)
    await grid.getByRole("button", { name: /^a red fox/ }).first().click()
    await page.getByRole("complementary", { name: "Image details" }).getByRole("button", { name: "PNG Info" }).click()
    await expect(page).toHaveURL(/#\/png-info\?file=/)
    await expect(page.getByRole("tab", { name: /All metadata/ })).toBeVisible()
    await page.getByRole("button", { name: "To txt2img" }).click()
    await expect(page).toHaveURL(/#\/txt2img$/)
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toContainText("a red fox in fresh snow")
  })
})

/** Client-side navigation: the mock backend lives in the page, a reload would reset it. */
async function go(page: Page, hash: string) {
  await page.evaluate((target) => (location.hash = target), hash)
}

test.describe("queue", () => {
  // The history filters were the page's own state: a reload, or Back after opening a job in its
  // form, showed every job again.
  test("keeps the history filters through a reload and Back", async ({ page }) => {
    await page.goto("/#/queue")
    const history = page.getByRole("region", { name: "History" })
    await history.getByRole("radio", { name: "Failed" }).click()
    await history.getByRole("searchbox", { name: "Search the history" }).fill("cat")
    await expect(page).toHaveURL(/#\/queue\?status=failed&q=cat$/)
    await page.reload()
    await expect(history.getByRole("radio", { name: "Failed" })).toBeChecked()
    await expect(history.getByRole("searchbox", { name: "Search the history" })).toHaveValue("cat")
    await go(page, "#/gallery")
    await expect(page.getByRole("list", { name: "Images" })).toBeVisible()
    await page.goBack()
    await expect(history.getByRole("radio", { name: "Failed" })).toBeChecked()
    await expect(history.getByRole("searchbox", { name: "Search the history" })).toHaveValue("cat")
  })

  test("lines jobs up while paused, reorders them from the keyboard and resumes", async ({ page }) => {
    await page.goto("/#/queue")
    await page.getByRole("button", { name: "Pause" }).click()
    await expect(page.getByRole("button", { name: "Resume" })).toBeVisible()

    await go(page, "#/txt2img")
    const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
    for (const text of ["first job", "second job"]) {
      await prompt.click()
      await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A")
      await page.keyboard.type(text)
      await page.getByRole("button", { name: "Add to queue" }).click()
      await expect(page.getByText(/Added to the queue as #/).last()).toBeVisible()
    }

    await go(page, "#/queue")
    const queued = page.getByRole("list", { name: "Queued jobs" }).getByRole("listitem")
    await expect(queued).toHaveCount(2)
    await expect(queued.nth(0)).toContainText("first job")

    const grip = queued.nth(1).getByRole("button", { name: /^Move job/ })
    await grip.focus()
    await page.keyboard.press("Space")
    await page.keyboard.press("ArrowUp")
    await page.keyboard.press("Space")
    await expect(queued.nth(0)).toContainText("second job")

    await page.getByRole("button", { name: "Resume" }).click()
    await expect(page.getByRole("button", { name: "Pause" })).toBeVisible()
    await expect(page.getByRole("list", { name: "Job history" }).getByText("second job")).toBeVisible({ timeout: 20_000 })
  })
})
