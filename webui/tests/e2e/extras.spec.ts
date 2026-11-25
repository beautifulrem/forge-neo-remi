/**
 * Extras against the mock backend: upload a source, pick the upscaler and the scale, process,
 * then the result (its size, the before / after comparison, the viewer) and its infotext; a
 * generated image sent to Extras; Hires. fix and Refiner values in a txt2img job and its infotext.
 */
import { expect, type Page, test } from "@playwright/test"
import { mod, openGenerate, openRoute, presetStorage, setPrompt, watchConsole } from "./helpers"

type MockModule = { handleMockRequest: (request: Request) => Promise<Response> }

/** The newest job of `tab` as the in-page mock server stored it (its named values). */
async function lastJob(page: Page, tab: string): Promise<{ core: Record<string, unknown>; scripts?: Record<string, Record<string, unknown>>; outputs: { id: string; width: number; height: number; infotext?: string }[] }> {
  return page.evaluate(async (jobTab) => {
    // The app's own instance of the in-browser mock server: the exact module URL it loaded.
    const loaded = performance.getEntriesByType("resource").map((entry) => entry.name)
    const path = loaded.find((name) => new URL(name).pathname.endsWith("/src/api/mock/server.ts"))
    if (!path) throw new Error("mock server module not loaded")
    const { handleMockRequest } = (await import(/* @vite-ignore */ path)) as MockModule
    const get = async (url: string) => (await handleMockRequest(new Request(`${location.origin}/api/v2${url}`))).json()
    const list = (await get(`/jobs?tab=${jobTab}&limit=1`)) as { items: { id: string }[] }
    return get(`/jobs/${list.items[0]!.id}`)
  }, tab)
}

/** A PNG of the given size, drawn in the page. */
async function makePng(page: Page, width: number, height: number): Promise<Buffer> {
  const data = await page.evaluate(
    ([w, h]) => {
      const canvas = document.createElement("canvas")
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext("2d")!
      ctx.fillStyle = "#16a34a"
      ctx.fillRect(0, 0, w, h)
      ctx.fillStyle = "#f59e0b"
      ctx.fillRect(w / 4, h / 4, w / 2, h / 2)
      return canvas.toDataURL("image/png").split(",")[1]!
    },
    [width, height] as const,
  )
  return Buffer.from(data, "base64")
}

// The mock server's module URL is read from the resource timings: the dev server loads a few hundred
// modules, more than the browser's default buffer of 250 entries keeps.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => performance.setResourceTimingBufferSize(5000))
})

test.describe("extras", () => {
  test("upscales an uploaded image with the chosen upscaler and scale", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "extras")
    const form = page.getByRole("form", { name: "Parameters" })
    await expect(page.getByText("Results appear here, next to their source.")).toBeVisible()

    // Upload the source
    const png = await makePng(page, 300, 200)
    const chooser = page.waitForEvent("filechooser")
    await form.getByRole("button", { name: "Source" }).click()
    await (await chooser).setFiles({ name: "photo.png", mimeType: "image/png", buffer: png })
    await expect(form.getByRole("button", { name: "Remove Source" })).toBeVisible()

    // Upscaler and scale
    await form.getByRole("combobox", { name: "Upscaler" }).click()
    await page.getByRole("option", { name: "R-ESRGAN 4x+" }).click()
    await expect(form.getByRole("combobox", { name: "Upscaler" })).toHaveText(/R-ESRGAN 4x\+/)
    const scale = form.getByRole("textbox", { name: "Scale by" })
    await scale.fill("3")
    await scale.press("Tab")
    await expect(form.getByRole("slider", { name: "Scale by" })).toHaveAttribute("aria-valuenow", "3")

    // Process: progress, then the result next to its source
    await page.getByRole("button", { name: /^Process/ }).click()
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeHidden({ timeout: 15_000 })
    const result = page.getByRole("region", { name: "Result" })
    await expect(result.getByRole("slider", { name: "Comparison divider" })).toBeVisible()
    await expect(result.getByText("900×600")).toBeVisible()
    await expect(result.getByText("Postprocess upscale by: 3, Postprocess upscaler: R-ESRGAN 4x+")).toBeVisible()

    // The job carried the form's values by name
    const job = await lastJob(page, "extras")
    expect(job.core).toMatchObject({ upscaler_1: "R-ESRGAN 4x+", upscaling_resize: 3 })
    expect(job.core.image).toMatch(/^upload-/)
    expect(job.outputs).toHaveLength(1)
    expect(job.outputs[0]).toMatchObject({ width: 900, height: 600, infotext: "Postprocess upscale by: 3, Postprocess upscaler: R-ESRGAN 4x+" })

    // Result only, then the viewer
    await result.getByRole("radio", { name: "Result" }).click()
    await expect(result.getByRole("img", { name: "Processed image" })).toHaveAttribute("src", new RegExp(`/files/${job.outputs[0]!.id}`))
    await result.getByRole("button", { name: "Open in the viewer" }).first().click()
    const viewer = page.getByRole("dialog")
    await expect(viewer).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(viewer).toBeHidden()
    console.assertClean()
  })

  test("processes an image sent from a txt2img result", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a lighthouse at dusk")
    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const generated = (await lastJob(page, "txt2img")).outputs[0]!
    await page.getByRole("button", { name: "Send to" }).click()
    await page.getByRole("menuitem", { name: "Extras", exact: true }).click()
    await expect(page).toHaveURL(/#\/extras$/)

    const form = page.getByRole("form", { name: "Parameters" })
    await expect(form.getByRole("button", { name: "Remove Source" })).toBeVisible()
    await page.keyboard.press(`${mod}+Enter`)
    const result = page.getByRole("region", { name: "Result" })
    await expect(result.getByText("Postprocess upscale by: 2, Postprocess upscaler: Lanczos")).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "extras")
    expect(job.core.image).toBe(generated.id) // the output itself, no new upload
    expect(job.outputs[0]).toMatchObject({ width: generated.width * 2, height: generated.height * 2 })
    await expect(result.getByText(`${generated.width * 2}×${generated.height * 2}`)).toBeVisible()
    console.assertClean()
  })
})

test.describe("hires fix and refiner", () => {
  test("generates with Hires. fix and its values reach the job and the infotext", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a castle on a hill")
    // The section header carries the enable switch; switching it on opens the section.
    const toggle = page.getByRole("switch", { name: "Enable Hires. fix", exact: true })
    await toggle.click()
    await expect(toggle).toBeChecked()
    const section = page.getByRole("region", { name: "Hires. fix", exact: true })
    const upscaleBy = section.getByRole("textbox", { name: "Upscale by" })
    await upscaleBy.fill("1.5")
    await upscaleBy.press("Tab")
    await section.getByRole("combobox", { name: "Upscaler" }).click()
    await page.getByRole("option", { name: "R-ESRGAN 4x+" }).click()

    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "txt2img")
    expect(job.core).toMatchObject({ enable_hr: true, hr_scale: 1.5, hr_upscaler: "R-ESRGAN 4x+" })
    const width = Number(job.core.width)
    const height = Number(job.core.height)
    expect(job.outputs[0]).toMatchObject({ width: Math.round(width * 1.5), height: Math.round(height * 1.5) })
    expect(job.outputs[0]!.infotext).toContain("Hires upscale: 1.5, Hires upscaler: R-ESRGAN 4x+")
    // The result's parameters list shows the Hires. fix values from the infotext
    const result = page.getByRole("region", { name: "Result" })
    const value = (term: string) => result.getByRole("term").filter({ hasText: new RegExp(`^${term}$`) }).locator("xpath=following-sibling::*[1]")
    await expect(value("Hires upscale")).toHaveText("1.5")
    await expect(value("Hires upscaler")).toHaveText("R-ESRGAN 4x+")
    console.assertClean()
  })

  test("Hires. fix dims the target controls that do not apply, and a target size wins over the factor", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a castle on a hill")
    await page.getByRole("switch", { name: "Enable Hires. fix", exact: true }).click()
    const section = page.getByRole("region", { name: "Hires. fix", exact: true })
    const field = (name: string) => section.getByRole("textbox", { name }).locator("xpath=ancestor::div[contains(@class, 'group/field')][1]")
    const inactive = async (name: string) => (await field(name).getAttribute("data-inactive")) === "true"
    const states = async () => [await inactive("Upscale by"), await inactive("Resize width to"), await inactive("Resize height to")]
    expect(await states()).toEqual([false, true, true])

    // a dimmed control stays editable: typing a width makes it the one in use
    const resizeWidth = section.getByRole("textbox", { name: "Resize width to" })
    await expect(resizeWidth).toBeEditable()
    await resizeWidth.fill("1536")
    await resizeWidth.press("Tab")
    await expect.poll(states).toEqual([true, false, true])
    await expect(section.getByText(/to 1536×1536/)).toBeVisible()
    await expect.poll(() => field("Upscale by").evaluate((element) => getComputedStyle(element).opacity)).toBe("0.5")

    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "txt2img")
    expect(job.core).toMatchObject({ enable_hr: true, hr_resize_x: 1536, hr_resize_y: 0 })
    expect(job.outputs[0]).toMatchObject({ width: 1536, height: 1536 })
    expect(job.outputs[0]!.infotext).toContain("Hires resize: 1536x1536")
    console.assertClean()
  })

  test("generates with the Refiner and its checkpoint and switch point reach the job and the infotext", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a lighthouse at dusk")
    const toggle = page.getByRole("switch", { name: "Enable Refiner", exact: true })
    await toggle.click()
    await expect(toggle).toBeChecked()
    const section = page.getByRole("region", { name: "Refiner", exact: true })
    await section.getByRole("combobox", { name: "Checkpoint" }).click()
    await page.getByRole("option", { name: "noobai-xl-vpred-1.0.safetensors" }).click()
    const switchAt = section.getByRole("textbox", { name: "Switch at" })
    await switchAt.fill("0.6")
    await switchAt.press("Tab")

    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "txt2img")
    expect(job.scripts?.refiner).toMatchObject({ enable: true, checkpoint: "noobai-xl-vpred-1.0.safetensors", switch_at: 0.6 })
    expect(job.outputs[0]!.infotext).toContain("Refiner: noobai-xl-vpred-1.0.safetensors, Refiner switch at: 0.6")
    const result = page.getByRole("region", { name: "Result" })
    const value = (term: string) => result.getByRole("term").filter({ hasText: new RegExp(`^${term}$`) }).locator("xpath=following-sibling::*[1]")
    await expect(value("Refiner")).toHaveText("noobai-xl-vpred-1.0.safetensors")
    await expect(value("Refiner switch at")).toHaveText("0.6")
    console.assertClean()
  })
})

// The strip of this session's runs scrolls sideways, and a scroll box cuts what is drawn outside
// it: the selected run's ring and the focus ring lost their top and left sides.
for (const { name, size } of [
  { name: "a laptop", size: { width: 1280, height: 720 } },
  { name: "a phone", size: { width: 390, height: 844 } },
]) {
  test(`keeps room for the rings around this session's runs on ${name}`, async ({ page }) => {
    await page.setViewportSize(size)
    await openRoute(page, "extras")
    const form = page.getByRole("form", { name: "Parameters" })
    const chooser = page.waitForEvent("filechooser")
    await form.getByRole("button", { name: "Source" }).click()
    await (await chooser).setFiles({ name: "photo.png", mimeType: "image/png", buffer: await makePng(page, 300, 200) })
    await expect(form.getByRole("button", { name: "Remove Source" })).toBeVisible()
    const runs = page.getByRole("listbox", { name: "Recent runs" })
    for (const run of [1, 2]) {
      await page.getByRole("button", { name: /^Process/ }).click()
      await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible()
      await expect(page.getByRole("button", { name: "Interrupt" })).toBeHidden({ timeout: 15_000 })
      if (run === 2) await expect(runs.getByRole("option")).toHaveCount(2)
    }
    const cut = await runs.evaluate((strip) => {
      const box = strip.getBoundingClientRect()
      const inner = { left: box.left + strip.clientLeft, top: box.top + strip.clientTop }
      const bottom = inner.top + strip.clientHeight
      const ring = 3 // the focus ring, the widest of the two
      return [...strip.querySelectorAll("[role=option]")].flatMap((option, index) => {
        const r = option.getBoundingClientRect()
        const sides = [r.left - ring < inner.left - 0.5 && "left", r.top - ring < inner.top - 0.5 && "top", r.bottom + ring > bottom + 0.5 && "bottom"]
        return sides.filter(Boolean).map((side) => `run ${index + 1}: ${side}`)
      })
    })
    expect(cut).toEqual([])
    // the room is taken from the strip's own margins: nothing scrolls sideways for it
    const sideways = await page.evaluate(() =>
      [document.scrollingElement!, ...document.querySelectorAll("main, main *")].flatMap((element) => {
        const style = getComputedStyle(element)
        const scrolls = element === document.scrollingElement || /auto|scroll/.test(style.overflowX)
        return scrolls && element.getAttribute("role") !== "listbox" && element.scrollWidth > element.clientWidth + 1 ? [element.tagName + "." + String(element.className).slice(0, 40)] : []
      }),
    )
    expect(sideways).toEqual([])
  })
}

// The result's toolbar never folded: at 768 px, and on a phone held sideways, "Send to" sat past the
// edge of the screen (or of its safe area) where no tap reached it. The route sweeps ran Extras
// without a result, so nothing showed it.
test.describe("the result toolbar", () => {
  for (const { name, size, insets } of [
    { name: "a tablet", size: { width: 768, height: 1024 }, insets: { top: 0, left: 0, bottom: 0, right: 0 } },
    { name: "a phone held sideways", size: { width: 852, height: 393 }, insets: { top: 0, left: 59, bottom: 21, right: 59 } },
  ]) {
    test(`keeps every action on screen on ${name}`, async ({ page }) => {
      await page.setViewportSize(size)
      const cdp = await page.context().newCDPSession(page)
      await cdp.send("Emulation.setSafeAreaInsetsOverride" as never, { insets } as never)
      await openRoute(page, "extras")
      const form = page.getByRole("form", { name: "Parameters" })
      const chooser = page.waitForEvent("filechooser")
      await form.getByRole("button", { name: "Source" }).click()
      await (await chooser).setFiles({ name: "photo.png", mimeType: "image/png", buffer: await makePng(page, 300, 200) })
      await expect(form.getByRole("button", { name: "Remove Source" })).toBeVisible()
      await page.keyboard.press(`${mod}+Enter`)
      const result = page.getByRole("region", { name: "Result" })
      await expect(result.getByRole("slider", { name: "Comparison divider" })).toBeVisible({ timeout: 15_000 })
      const sendTo = result.getByRole("button", { name: "Send to" })
      await sendTo.scrollIntoViewIfNeeded()
      const unreachable = await result.getByRole("button").evaluateAll(
        (buttons, { size, insets }) =>
          buttons.flatMap((button) => {
            const box = button.getBoundingClientRect()
            if (box.width === 0) return []
            const hit = button.contains(document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2))
            const inside = box.left >= insets.left - 0.5 && box.right <= size.width - insets.right + 0.5
            return hit && inside ? [] : [`${button.getAttribute("aria-label") ?? button.textContent?.trim()} at ${Math.round(box.left)}..${Math.round(box.right)}`]
          }),
        { size, insets },
      )
      expect(unreachable).toEqual([])
      await sendTo.click()
      await expect(page.getByRole("menuitem", { name: "img2img" })).toBeVisible()
    })
  }

  // It wrapped wherever it ran out of room: the 208 px column of a 768 px tablet put the view
  // switch, the size and the actions on three ragged rows. A narrow column gives the actions a row
  // of their own and lets the size go; a wide one keeps all of it on one row.
  for (const { name, size, rows } of [
    { name: "a tablet", size: { width: 768, height: 1024 }, rows: [["view"], ["actions"]] },
    { name: "a phone", size: { width: 375, height: 812 }, rows: [["view", "size"], ["actions"]] },
    // the actions unfold from a 544 px column and need about 550 px beside the switch and the size
    { name: "a window where the actions unfold", size: { width: 1106, height: 720 }, rows: [["view", "actions"]] },
    { name: "a laptop", size: { width: 1280, height: 720 }, rows: [["view", "size", "actions"]] },
  ]) {
    test(`keeps to planned rows on ${name}`, async ({ page }) => {
      await page.setViewportSize(size)
      await openRoute(page, "extras")
      const form = page.getByRole("form", { name: "Parameters" })
      const chooser = page.waitForEvent("filechooser")
      await form.getByRole("button", { name: "Source" }).click()
      await (await chooser).setFiles({ name: "photo.png", mimeType: "image/png", buffer: await makePng(page, 300, 200) })
      await expect(form.getByRole("button", { name: "Remove Source" })).toBeVisible()
      await page.keyboard.press(`${mod}+Enter`)
      const result = page.getByRole("region", { name: "Result" })
      await expect(result.getByRole("slider", { name: "Comparison divider" })).toBeVisible({ timeout: 15_000 })
      const bar = result.getByRole("radiogroup", { name: "View" }).locator("xpath=..")
      const shown = await bar.evaluate((row) => {
        const named = [...row.children].flatMap((child) => {
          const box = child.getBoundingClientRect()
          if (box.width === 0) return []
          const part = child.getAttribute("role") === "radiogroup" ? "view" : child.tagName === "SPAN" ? "size" : "actions"
          return [{ part, middle: box.top + box.height / 2 }]
        })
        const lines: { middle: number; parts: string[] }[] = []
        for (const { part, middle } of named) {
          const line = lines.find((entry) => Math.abs(entry.middle - middle) < 8)
          if (line) line.parts.push(part)
          else lines.push({ middle, parts: [part] })
        }
        return lines.sort((a, b) => a.middle - b.middle).map((line) => line.parts)
      })
      expect(shown).toEqual(rows)
    })
  }
})

// Held sideways (844×390) the comparison of a portrait image is a narrow frame, and its two corner
// labels ("源图" / "结果") ran into each other: each keeps to its half, cut with a title.
for (const locale of ["en", "zh-CN"]) {
  test(`the comparison's labels keep apart in a narrow frame at 844×390 (${locale})`, async ({ page }) => {
    await presetStorage(page, { locale })
    await page.setViewportSize({ width: 844, height: 390 })
    await openRoute(page, "extras")
    const form = page.getByRole("form", { name: /^(Parameters|参数)$/ })
    const chooser = page.waitForEvent("filechooser")
    await form.getByRole("button", { name: /^(Source|源图)$/ }).first().click()
    await (await chooser).setFiles({ name: "portrait.png", mimeType: "image/png", buffer: await makePng(page, 200, 400) })
    await page.keyboard.press(`${mod}+Enter`)
    const result = page.getByRole("region", { name: /^(Result|结果)$/ })
    const divider = result.getByRole("slider")
    await expect(divider).toBeVisible({ timeout: 15_000 })
    const overlap = await divider.evaluate((slider) => {
      const labels = [...slider.parentElement!.querySelectorAll(":scope > span[title]")].map((label) => label.getBoundingClientRect())
      const [a, b] = labels
      return labels.length === 2 && a!.right > b!.left && b!.right > a!.left && a!.bottom > b!.top && b!.bottom > a!.top
    })
    expect(overlap).toBe(false)
  })
}

// The selected image of a run is marked by a layer over the thumbnail, which also covered the
// focus ring drawn inside it: Tab landed on the selected thumbnail first, and nothing showed it.
test("the selected image of a run shows the focus", async ({ page }) => {
  await page.addInitScript(() => {
    if (sessionStorage.getItem("remi-extras-runs")) return
    const outputs = [0, 1, 2].map((i) => ({ id: `mock-run-${i}-${i * 120}`, kind: "image", width: 512, height: 512, infotext: null }))
    const result = { jobId: "job-1", outputs, source: null, info: null, at: Date.now() }
    sessionStorage.setItem("remi-extras-runs", JSON.stringify({ state: { results: [result], jobs: [], sources: {} }, version: 1 }))
  })
  await openRoute(page, "extras")
  await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
  const strip = page.getByRole("listbox", { name: "Images of this run" })
  const selected = strip.getByRole("option", { selected: true })
  await expect(strip.getByRole("option")).toHaveCount(3)
  await selected.focus()
  expect(await selected.evaluate((element) => element.matches(":focus-visible"))).toBe(true)
  // the middle of the left edge, 3 px either side of it (not the corners, whose antialiasing differs
  // either way): something there changes when the focus arrives
  const edge = async () => {
    const box = (await selected.boundingBox())!
    return page.screenshot({ clip: { x: box.x - 3, y: box.y + box.height / 3, width: 6, height: box.height / 3 } })
  }
  const focused = await edge()
  await selected.evaluate((element: HTMLElement) => element.blur())
  expect(focused.equals(await edge())).toBe(false)
})
