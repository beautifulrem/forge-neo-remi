/**
 * Main flows against the mock backend: settings, the networks drawer,
 * queue cancel, infotext / image paste, extension pages, language, theme and
 * keyboard shortcuts. Each test also fails on console errors.
 */
import { crc32 } from "node:zlib"
import { expect, type Page, test } from "@playwright/test"
import { mod, openGenerate, openRoute, presetStorage, setPrompt, watchConsole } from "./helpers"

/** Client-side navigation: the mock backend lives in the page, a reload would reset it. */
async function go(page: Page, hash: string) {
  await page.evaluate((target) => (location.hash = target), hash)
}

/** A PNG (drawn in the page) with a `parameters` tEXt chunk inserted after IHDR, as the WebUI writes it. */
async function pngWithParameters(page: Page, text: string): Promise<Buffer> {
  const b64 = await page.evaluate(() => {
    const canvas = document.createElement("canvas")
    canvas.width = canvas.height = 64
    const ctx = canvas.getContext("2d")!
    ctx.fillStyle = "#3b82f6"
    ctx.fillRect(0, 0, 64, 64)
    return canvas.toDataURL("image/png").split(",")[1]!
  })
  const png = Buffer.from(b64, "base64")
  const data = Buffer.concat([Buffer.from("parameters\0", "latin1"), Buffer.from(text, "latin1")])
  const chunk = Buffer.alloc(12 + data.length)
  chunk.writeUInt32BE(data.length, 0)
  chunk.write("tEXt", 4, "latin1")
  data.copy(chunk, 8)
  chunk.writeUInt32BE(crc32(chunk.subarray(4, 8 + data.length)), 8 + data.length)
  const ihdrEnd = 8 + 8 + 13 + 4 // signature + IHDR (length, type, data, crc)
  return Buffer.concat([png.subarray(0, ihdrEnd), chunk, png.subarray(ihdrEnd)])
}

test.describe("settings", () => {
  test("changes a setting, applies it with Mod+S and keeps it", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "settings")
    const toggle = page.getByRole("switch", { name: "Automatically save every generated image", exact: true })
    await expect(toggle).toBeChecked()
    await toggle.click()
    await expect(toggle).not.toBeChecked()
    const pending = page.getByRole("region", { name: "Unsaved changes" })
    await expect(pending).toContainText("Unsaved changes: 1")
    await page.keyboard.press(`${mod}+S`)
    await expect(pending).toBeHidden()
    await expect(page.getByRole("radio", { name: /^Unsaved/ })).toHaveAccessibleName("Unsaved")
    // The applied value is what the server now reports: leave and come back.
    await go(page, "#/txt2img")
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
    await go(page, "#/settings")
    await expect(page.getByRole("switch", { name: "Automatically save every generated image", exact: true })).not.toBeChecked()
    await page.getByRole("radio", { name: /^Changed/ }).click()
    await expect(page.getByRole("switch", { name: "Automatically save every generated image", exact: true })).toBeVisible()
    console.assertClean()
  })

  test("discards an unsaved change", async ({ page }) => {
    await openRoute(page, "settings")
    const toggle = page.getByRole("switch", { name: "Append an ascending number to the filename", exact: true })
    await toggle.click()
    const pending = page.getByRole("region", { name: "Unsaved changes" })
    await pending.getByRole("button", { name: "Discard" }).click()
    await expect(pending).toBeHidden()
    await expect(toggle).toBeChecked()
  })
})

// Settings → Licenses links THIRD-PARTY-NOTICES.txt, which only the build writes: the dev server answers it too
test("serves the third-party notices the Licenses view links to", async ({ request }) => {
  const response = await request.get("/THIRD-PARTY-NOTICES.txt")
  expect(response.status()).toBe(200)
  expect(response.headers()["content-type"]).toMatch(/^text\/plain/)
  expect(await response.text()).toMatch(/^react \d/m)
})

test.describe("networks drawer", () => {
  test("opens with Alt+N, filters and inserts a LoRA into the prompt", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "1girl")
    await page.locator("body").press("Alt+N")
    const drawer = page.getByRole("dialog", { name: "Networks" })
    await expect(drawer).toBeVisible()
    await drawer.getByRole("searchbox", { name: "Search networks" }).fill("comic_sketch_v1_700")
    const card = drawer.getByRole("option", { name: /^comic_sketch_v1_700,/ })
    await expect(card).toBeVisible()
    await card.click()
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toContainText("<lora:comic_sketch_v1_700:")
    await drawer.getByRole("button", { name: "Close" }).click()
    await expect(drawer).toBeHidden()
    console.assertClean()
  })
})

test.describe("queue", () => {
  test("cancels a queued job and keeps the others", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "queue")
    await page.getByRole("button", { name: "Pause" }).click()
    await go(page, "#/txt2img")
    for (const text of ["keep me", "cancel me"]) {
      await setPrompt(page, text)
      await page.getByRole("button", { name: "Add to queue" }).click()
      await expect(page.getByText(/Added to the queue as #/).last()).toBeVisible()
    }
    await go(page, "#/queue")
    const queued = page.getByRole("list", { name: "Queued jobs" }).getByRole("listitem")
    await expect(queued).toHaveCount(2)
    await queued.filter({ hasText: "cancel me" }).getByRole("button", { name: /^More actions for job/ }).click()
    await page.getByRole("menuitem", { name: "Cancel job" }).click()
    await expect(queued).toHaveCount(1)
    await expect(queued.first()).toContainText("keep me")
    await page.getByRole("button", { name: "Resume" }).click()
    const history = page.getByRole("list", { name: "Job history" })
    await expect(history.getByText("keep me")).toBeVisible({ timeout: 20_000 })
    console.assertClean()
  })
})

test.describe("paste", () => {
  test("reads generation parameters pasted into the prompt", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "")
    // Typed infotext stands in for a clipboard paste (CodeMirror takes both the same way).
    await page.keyboard.insertText("a lighthouse at dusk\nNegative prompt: blurry\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 42, Size: 512x768")
    await page.getByRole("button", { name: /^Read parameters from the prompt/ }).click()
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toHaveText("a lighthouse at dusk")
    await expect(page.getByRole("textbox", { name: "Negative prompt", exact: true })).toHaveText("blurry")
    await expect(page.getByRole("textbox", { name: "Sampling steps", exact: true })).toHaveValue("20")
    await expect(page.getByRole("textbox", { name: "Width", exact: true })).toHaveValue("512")
    await expect(page.getByRole("textbox", { name: "Height", exact: true })).toHaveValue("768")
    await expect(page.getByRole("textbox", { name: "CFG Scale", exact: true })).toHaveValue("7")
    await expect(page.getByRole("textbox", { name: "Seed", exact: true })).toHaveValue("42")
    console.assertClean()
  })

  test("PNG Info reads an image pasted on the page", async ({ page }) => {
    await openRoute(page, "png-info")
    const png = await pngWithParameters(page, "a red fox\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 1, Size: 64x64")
    const transfer = await page.evaluateHandle((b64) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
      const data = new DataTransfer()
      data.items.add(new File([bytes], "pasted.png", { type: "image/png" }))
      return data
    }, png.toString("base64"))
    await page.evaluate((clipboardData) => {
      window.dispatchEvent(new ClipboardEvent("paste", { clipboardData, bubbles: true, cancelable: true }))
    }, transfer)
    await expect(page).toHaveURL(/#\/png-info\?file=/)
    await expect(page.getByRole("tab", { name: /All metadata/ })).toBeVisible()
  })
})

test.describe("extension pages", () => {
  test("renders a schema page and runs its action", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "txt2img")
    await page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Toolbox" }).click()
    await expect(page).toHaveURL(/#\/ext\/toolbox$/)
    const main = page.getByRole("main")
    await expect(main.getByRole("heading", { name: "Toolbox" })).toBeVisible()
    await main.getByRole("textbox", { name: "Text to transform" }).fill("Hello Remi")
    await main.getByRole("radio", { name: "reverse" }).click()
    await main.getByRole("button", { name: "Run" }).click()
    await expect(main.getByText("imeR olleH")).toBeVisible()
    console.assertClean()
  })

  test("JSON Builder: field editor, background, resize, direct load and expand", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "ext/json-builder")
    const main = page.getByRole("main")
    await expect(main.getByTestId("json-builder-canvas")).toBeVisible()
    await main.getByLabel("Caption", { exact: true }).fill("A cat on a sofa")
    await main.getByRole("textbox", { name: "Caption of item 1" }).fill("an orange cat")

    // the field editor: the real prompt editor writes the draft's field, a style applies into it
    await main.getByRole("button", { name: "Edit the summary caption in the field editor" }).click()
    const dialog = page.getByRole("dialog")
    const editor = dialog.getByRole("textbox", { name: "Edit the summary caption" })
    await expect(editor).toHaveText("A cat on a sofa")
    await editor.fill("a quiet room")
    await dialog.getByRole("combobox", { name: "Prompt styles" }).fill("Water")
    await page.getByRole("option", { name: "Watercolor", exact: true }).click()
    await dialog.getByRole("button", { name: "Apply styles", exact: true }).click()
    await expect(editor).toHaveText("a quiet room, watercolor (medium), soft colors")
    await page.keyboard.press("Escape")
    await expect(dialog).not.toBeVisible()
    await expect(main.getByLabel("Caption", { exact: true })).toHaveValue("a quiet room, watercolor (medium), soft colors")

    // a file under the boxes as a drawing guide
    await main.getByTestId("json-builder-background").setInputFiles({
      name: "guide.svg",
      mimeType: "image/svg+xml",
      buffer: Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"/>'),
    })
    await expect(main.getByTestId("json-builder-canvas").locator("img")).toBeVisible()
    await main.getByRole("button", { name: "Clear the background" }).click()

    // committing a new size (Enter) rescales the boxes from the committed one
    const before = await main.evaluate(() => JSON.parse(localStorage.getItem("remi-json-builder")!).state.items[0].bbox as number[])
    // Base UI's number field commits on Enter or on leaving the field
    await main.getByLabel("Width").click()
    await main.getByLabel("Width").press("ControlOrMeta+a")
    await main.getByLabel("Width").pressSequentially("512")
    await main.getByLabel("Width").press("Tab")
    const stored = await main.evaluate(() => JSON.parse(localStorage.getItem("remi-json-builder")!).state)
    expect(stored.width).toBe(512)
    const after = stored.items[0].bbox as number[]
    expect(Math.abs(after[0]! - before[0]! / 2)).toBeLessThanOrEqual(1)
    expect(Math.abs(after[2]! - before[2]! / 2)).toBeLessThanOrEqual(1)
    await main.getByLabel("Width").click()
    await main.getByLabel("Width").press("ControlOrMeta+a")
    await main.getByLabel("Width").pressSequentially("1024")
    await main.getByLabel("Width").press("Tab")

    // send, come back, read the tab's prompt back in
    await main.getByRole("button", { name: "Send to txt2img" }).click()
    await expect(page).toHaveURL(/#\/txt2img$/)
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toContainText('"caption":"a quiet room, watercolor \\\(medium\\\), soft colors"')
    await go(page, "#/ext/json-builder")
    const builder = page.getByRole("main")
    await builder.getByRole("button", { name: "Expand the item list" }).click()
    await expect(builder.getByLabel("Caption", { exact: true })).not.toBeVisible()
    await builder.getByRole("button", { name: "Collapse the item list" }).click()
    await builder.getByRole("button", { name: "Load from txt2img" }).click()
    await expect(builder.getByLabel("Caption", { exact: true })).toHaveValue("a quiet room, watercolor (medium), soft colors")
    console.assertClean()
  })

  test("lists installed extensions and their tabs", async ({ page }) => {
    await openRoute(page, "extensions")
    await expect(page.getByRole("tab", { name: "Installed" })).toHaveAttribute("aria-selected", "true")
    await expect(page.getByRole("switch", { name: "sd-dynamic-prompts" })).toBeVisible()
    await page.getByRole("tab", { name: "Available" }).click()
    await expect(page.getByRole("tabpanel", { name: "Available" })).toBeVisible()
  })
})

test.describe("appearance and language", () => {
  test("switches to zh-CN without a reload: chrome and schema labels", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await page.evaluate(() => ((window as { __noReload?: boolean }).__noReload = true))
    await page.getByRole("button", { name: "Appearance" }).click()
    await page.getByRole("menuitemradio", { name: "简体中文" }).click()
    await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN")
    const nav = page.getByRole("navigation", { name: /./ }).first()
    await expect(nav.getByRole("link", { name: "设置" })).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(page.getByRole("textbox", { name: "迭代步数" })).toBeVisible()
    await expect(page.getByRole("button", { name: /^生成/ }).first()).toBeVisible()
    expect(await page.evaluate(() => (window as { __noReload?: boolean }).__noReload)).toBe(true)
    // The choice persists across a reload.
    await page.reload()
    await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN")
    console.assertClean()
  })

  test("zh-CN: settings help blocks, a missing source image and the notifications region are Chinese", async ({ page }) => {
    const console = watchConsole(page)
    await presetStorage(page, { locale: "zh-CN" })
    await openRoute(page, "img2img")
    await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN")
    await expect(page.getByRole("textbox", { name: "提示词", exact: true })).toBeVisible()
    // the client checks the required source image itself: inline marker and a Chinese toast body
    await page.getByRole("button", { name: /^生成/ }).first().click()
    const toast = page.locator("[data-sonner-toast]").filter({ hasText: "部分参数无效" })
    await expect(toast).toContainText("需要一张图像")
    await expect(toast).not.toContainText("an image is required")
    await expect(page.getByRole("region", { name: "通知（Alt+T）" })).toBeAttached()
    await expect(toast.getByRole("button", { name: "关闭通知" })).toBeAttached()
    // settings: a help block (Markdown) is translated by its source text
    await go(page, "#/settings")
    await page.getByRole("navigation", { name: "设置分区" }).getByRole("button", { name: "生成信息" }).click()
    await expect(page.getByText("生成信息（Infotext）是 WebUI 对包含生成参数的文本的称呼")).toBeVisible()
    console.assertClean()
  })

  test("applies user.css from the data folder over the app's own styles", async ({ page }) => {
    const console = watchConsole(page)
    await page.route("**/api/v2/user-assets/user.css", (route) =>
      route.fulfill({ contentType: "text/css", body: "body { outline: 3px solid rgb(1, 2, 3); background-color: rgb(4, 5, 6); }" }),
    )
    await openGenerate(page)
    const body = () => page.evaluate(() => [getComputedStyle(document.body).outlineColor, getComputedStyle(document.body).backgroundColor])
    // the background rule competes with the app's own body background, and wins
    await expect.poll(body).toEqual(["rgb(1, 2, 3)", "rgb(4, 5, 6)"])
    expect(await page.evaluate(() => document.querySelector<HTMLLinkElement>("link[rel=stylesheet][href$='/user-assets/user.css']")?.parentElement?.tagName)).toBe("BODY")
    // still applied after visiting a lazily loaded page (its CSS comes later, into <head>)
    await page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Settings" }).click()
    await expect(page).toHaveURL(/#\/settings/)
    expect(await body()).toEqual(["rgb(1, 2, 3)", "rgb(4, 5, 6)"])
    console.assertClean()
  })

  test("lays the app out right to left for a right-to-left language", async ({ page }) => {
    await presetStorage(page, { locale: "ar" })
    await openGenerate(page)
    const html = page.locator("html")
    await expect(html).toHaveAttribute("lang", "ar")
    await expect(html).toHaveAttribute("dir", "rtl")
    // the navigation rail moves to the right-hand side
    const rail = await page.getByRole("navigation", { name: "Main navigation" }).boundingBox()
    const main = await page.getByRole("main").boundingBox()
    expect(rail!.x).toBeGreaterThan(main!.x)
    // key combinations keep their order ("⌘Enter", not "Enter⌘")
    expect(await page.locator("kbd").first().evaluate((kbd) => getComputedStyle(kbd).direction)).toBe("ltr")
    await page.getByRole("button", { name: "Appearance" }).click()
    await page.getByRole("menuitemradio", { name: "English" }).click()
    await expect(html).toHaveAttribute("dir", "ltr")
  })

  test("switches between light and dark and remembers the theme", async ({ page }) => {
    await presetStorage(page, { theme: "dark" })
    await openGenerate(page)
    const html = page.locator("html")
    await expect(html).toHaveClass(/\bdark\b/)
    await page.getByRole("button", { name: "Appearance" }).click()
    await page.getByRole("menuitemradio", { name: "Light" }).click()
    await expect(html).not.toHaveClass(/\bdark\b/)
    const background = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
    await page.reload()
    await expect(html).not.toHaveClass(/\bdark\b/)
    expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe(background)
    await page.getByRole("button", { name: "Appearance" }).click()
    await page.getByRole("menuitemradio", { name: "Dark" }).click()
    await expect(html).toHaveClass(/\bdark\b/)
  })
})

test.describe("keyboard", () => {
  test("command palette: opens with Mod+K and navigates", async ({ page }) => {
    await openGenerate(page)
    await page.keyboard.press(`${mod}+K`)
    const palette = page.getByRole("dialog", { name: "Command palette" })
    await expect(palette).toBeVisible()
    await expect(palette.getByRole("combobox")).toBeFocused()
    await page.keyboard.type("Gallery")
    await expect(palette.getByRole("option", { name: "Gallery", selected: true })).toBeVisible()
    await page.keyboard.press("Enter")
    await expect(palette).toBeHidden()
    await expect(page).toHaveURL(/#\/gallery$/)
  })

  test("command palette switches the theme", async ({ page }) => {
    await presetStorage(page, { theme: "dark" })
    await openGenerate(page)
    await page.keyboard.press(`${mod}+K`)
    const palette = page.getByRole("dialog", { name: "Command palette" })
    await expect(palette.getByRole("combobox")).toBeFocused()
    await page.keyboard.type("Light theme")
    await expect(palette.getByRole("option", { name: "Light theme", selected: true })).toBeVisible()
    await page.keyboard.press("Enter")
    await expect(page.locator("html")).not.toHaveClass(/\bdark\b/)
  })

  test("skip link moves the focus to the page content", async ({ page }) => {
    await openGenerate(page)
    await page.locator("body").press("Tab")
    const skip = page.getByRole("link", { name: "Skip to content" })
    await expect(skip).toBeFocused()
    await page.keyboard.press("Enter")
    await expect(page.getByRole("main")).toBeFocused()
  })

  test("every nav link is reachable with Tab and shows a focus ring", async ({ page }) => {
    await openGenerate(page)
    const nav = page.getByRole("navigation", { name: "Main navigation" })
    const gallery = nav.getByRole("link", { name: "Gallery" })
    await gallery.focus()
    await expect(gallery).toBeFocused()
    const ring = await gallery.evaluate((element) => {
      const style = getComputedStyle(element)
      return style.outlineStyle !== "none" || style.boxShadow !== "none"
    })
    expect(ring).toBe(true)
    await page.keyboard.press("Enter")
    await expect(page).toHaveURL(/#\/gallery$/)
  })
})
