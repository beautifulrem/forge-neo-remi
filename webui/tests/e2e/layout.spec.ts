/**
 * Layout and feedback checks that need a real browser: the header at
 * 1280×720 (idle and while a job runs), the Generate panes' hide / show
 * toggles (real pane sizes), tab strips on a phone, boxes that scroll sideways, platform
 * shortcut hints and generation warnings.
 */
import { expect, type Page, test } from "@playwright/test"
import { mod, openGenerate, openRoute, presetStorage, setPrompt } from "./helpers"

/** Pairs of header controls whose boxes overlap, and kbd hints cut by their button. */
async function headerProblems(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const header = document.querySelector("header")!
    const boxes = [...header.querySelectorAll<HTMLElement>("button, [role=combobox], h1")]
      .map((element) => ({ element, rect: element.getBoundingClientRect(), name: (element.getAttribute("aria-label") || element.textContent || "").trim() }))
      .filter(({ rect }) => rect.width > 0)
    const problems: string[] = []
    for (const [i, a] of boxes.entries()) {
      for (const b of boxes.slice(i + 1)) {
        if (a.element.contains(b.element) || b.element.contains(a.element)) continue
        const x = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left)
        const y = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top)
        if (x > 0.5 && y > 0.5) problems.push(`"${a.name}" overlaps "${b.name}" by ${x.toFixed(1)}px`)
      }
      if (a.rect.right > innerWidth + 0.5) problems.push(`"${a.name}" leaves the window`)
    }
    for (const kbd of document.querySelectorAll<HTMLElement>("header button kbd, main button kbd")) {
      const k = kbd.getBoundingClientRect()
      const b = kbd.closest("button")!.getBoundingClientRect()
      if (k.width > 0 && (k.right > b.right - 2 || k.left < b.left)) problems.push(`kbd "${kbd.textContent}" is cut by its button`)
    }
    for (const label of document.querySelectorAll<HTMLElement>("main button .truncate")) {
      if (label.classList.contains("sr-only")) continue // read out, not shown: nothing to cut
      if (label.scrollWidth > label.clientWidth + 1) problems.push(`button label "${label.textContent}" is cut`)
    }
    return problems
  })
}

test.describe("layout at 1280×720", () => {
  test.use({ viewport: { width: 1280, height: 720 } })

  for (const locale of ["en", "zh-CN"]) {
    test(`the header and the generate bar fit, idle and while a job runs (${locale})`, async ({ page }) => {
      await presetStorage(page, { locale })
      // Quick settings in the header too (the most crowded case).
      await page.goto("/#/settings")
      await page.getByRole("button", { name: /^(Pin .*quick settings|将.*固定)/ }).first().click()
      await expect(page.getByRole("button", { name: /^(Unpin .*quick settings|从快捷设置中取消固定)/ }).first()).toBeVisible()
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await expect(page.getByRole("textbox", { name: /^(Prompt|提示词)$/ })).toBeVisible()
      await expect(page.locator("header").getByRole("button", { name: /^(Quick settings|快捷设置)$/ })).toBeVisible()
      await expect.poll(() => headerProblems(page)).toEqual([])

      await page.getByRole("button", { name: /^(Generate|生成)/ }).first().click()
      await expect(page.getByRole("button", { name: /^(Interrupt|中断)$/ })).toBeVisible()
      expect(await headerProblems(page)).toEqual([])
    })
  }
})

test.describe("generate panes", () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test("hiding a pane gives its room to the stage and showing it restores the layout", async ({ page }) => {
    await openGenerate(page)
    const sizes = () => page.evaluate(() => ["params", "stage", "history"].map((id) => Math.round(Number(document.getElementById(id)?.style.flexGrow))))
    const initial = await sizes()
    expect(initial.every((size) => size > 0)).toBe(true)
    const generate = page.getByRole("region", { name: "Result" }).getByRole("button", { name: /^Generate/ })
    await expect(generate).toHaveCount(0)

    await page.getByRole("button", { name: "Hide parameters" }).click()
    await expect.poll(sizes).toEqual([0, initial[0]! + initial[1]!, initial[2]])
    // the run controls move onto the stage while the parameters are hidden
    await expect(generate.first()).toBeVisible()
    await page.getByRole("button", { name: "Show parameters" }).click()
    await expect.poll(sizes).toEqual(initial)
    await expect(generate).toHaveCount(0)

    await page.getByRole("button", { name: "Hide history" }).click()
    await expect.poll(sizes).toEqual([initial[0], initial[1]! + initial[2]!, 0])
    await page.getByRole("button", { name: "Show history" }).click()
    await expect.poll(sizes).toEqual(initial)
  })
})

test.describe("tab strips on a phone", () => {
  test.use({ viewport: { width: 420, height: 860 } })

  for (const [route, label] of [
    ["extensions", "Extensions"],
    ["networks", "Network kinds"],
  ] as const) {
    test(`#/${route} shows its tabs as a select instead of cutting them`, async ({ page }) => {
      await page.goto(`/#/${route}`)
      const select = page.getByRole("main").getByRole("combobox", { name: label })
      await expect(select).toBeVisible()
      const overflowing = await page.evaluate(() =>
        [...document.querySelectorAll<HTMLElement>("[role=tablist]")]
          .filter((list) => list.getBoundingClientRect().width > 0 && getComputedStyle(list).visibility !== "hidden")
          .filter((list) => list.scrollWidth > list.clientWidth + 1)
          .map((list) => list.getAttribute("aria-label")),
      )
      expect(overflowing).toEqual([])
      // The select switches tabs like the strip.
      await select.click()
      const second = page.locator("[data-slot=select-item]").nth(1)
      const name = (await second.textContent())!.trim()
      await second.click()
      await expect(select).toHaveText(new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")))
    })
  }

  test("the tabs come back when there is room", async ({ page }) => {
    await page.goto("/#/extensions")
    await expect(page.getByRole("main").getByRole("combobox", { name: "Extensions" })).toBeVisible()
    await page.setViewportSize({ width: 1280, height: 860 })
    await expect(page.getByRole("tab", { name: "Backup and restore" })).toBeVisible()
    await expect(page.getByRole("main").getByRole("combobox", { name: "Extensions" })).toHaveCount(0)
  })
})

test.describe("hints and warnings", () => {
  test("the prompt placeholder names this platform's shortcuts", async ({ page }) => {
    await openGenerate(page)
    const prompt = await setPrompt(page, "x")
    await prompt.press("Backspace") // the placeholder shows in an empty prompt
    const generate = process.platform === "darwin" ? "⌘Enter" : "Ctrl+Enter"
    await expect(page.locator(".cm-placeholder").first()).toContainText(`${generate} to Generate`)
    await expect(page.getByRole("button", { name: /^Generate/ }).first()).toContainText(generate)
  })

  test("a LoRA that is not there ends the job with a warning", async ({ page }) => {
    await openGenerate(page)
    await setPrompt(page, "1girl, silver hair, <lora:doesnotexist:1>")
    await page.keyboard.press(`${mod}+Enter`)
    const toast = page.locator("[data-sonner-toast]").filter({ hasText: "Finished with warnings" })
    await expect(toast).toBeVisible({ timeout: 20_000 })
    await expect(toast).toContainText("LoRA not found or failed to load: doesnotexist")
  })
})

test.describe("schema rows on extension pages", () => {
  const actions = ["Build prompt", "Send to txt2img Prompt", "Replace txt2img Prompt", "Clear all fields"]

  test("an action row is one left-aligned group of same-sized buttons; status lines use the app's status style", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 })
    await page.goto("/#/ext/toolbox")
    const main = page.getByRole("main")
    const boxes = []
    for (const name of actions) boxes.push((await main.getByRole("button", { name, exact: true }).boundingBox())!)
    expect(new Set(boxes.map((box) => Math.round(box.height))).size).toBe(1)
    for (const [index, box] of boxes.slice(1).entries()) expect(box.x - (boxes[index]!.x + boxes[index]!.width)).toBeLessThan(16) // side by side, not spread out
    await main.getByRole("button", { name: "Build prompt" }).click()
    const status = main.locator("[data-tone=success]")
    await expect(status).toHaveText("Prompt built") // the "✅" became the icon and tone
  })

  test("two columns of fields stack on a phone", async ({ page }) => {
    for (const [width, sideBySide] of [
      [1280, true],
      [420, false],
    ] as const) {
      await page.setViewportSize({ width, height: 900 })
      await page.goto("/#/ext/toolbox")
      const series = page.getByRole("textbox", { name: "Series" })
      await expect(series).toBeVisible()
      // both boxes in one frame: measured one after the other they can straddle a late layout change
      const sameLine = async () =>
        page
          .getByRole("textbox", { name: "Quality" })
          .evaluate((quality, other) => Math.abs(quality.getBoundingClientRect().top - other!.getBoundingClientRect().top) < 2, await series.elementHandle())
      await expect.poll(sameLine, { message: `${width}px` }).toBe(sideBySide)
    }
  })
})

/** Toasts against the controls pinned to the bottom edge (`data-toast-obstacle`) they overlap, as messages. */
async function toastOverlaps(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const toasts = document.querySelectorAll<HTMLElement>("[data-sonner-toast]")
    // (a toast that timed out proves nothing)
    const problems: string[] = toasts.length ? [] : ["no toast"]
    for (const toast of toasts) {
      const t = toast.getBoundingClientRect()
      for (const obstacle of document.querySelectorAll<HTMLElement>("[data-toast-obstacle]")) {
        const o = obstacle.getBoundingClientRect()
        if (o.width === 0 || o.height === 0) continue
        // the close button sticks out of the toast's top-left corner
        const x = Math.min(t.right, o.right) - Math.max(t.left - 8, o.left)
        const y = Math.min(t.bottom, o.bottom) - Math.max(t.top - 8, o.top)
        if (x > 0 && y > 0) problems.push(`toast covers ${obstacle.getAttribute("aria-label") ?? obstacle.tagName}`)
      }
    }
    return problems
  })
}

test.describe("toasts keep clear of the bottom controls", () => {
  for (const viewport of [
    { width: 420, height: 860 },
    { width: 1280, height: 720 },
  ]) {
    test(`at ${viewport.width}×${viewport.height}`, async ({ page }) => {
      await page.setViewportSize(viewport)
      await openGenerate(page)
      await setPrompt(page, "1girl, <lora:doesnotexist:1>")
      await page.keyboard.press(`${mod}+Enter`)
      const toast = page.locator("[data-sonner-toast]").filter({ hasText: "Finished with warnings" })
      await expect(toast).toBeVisible({ timeout: 20_000 })
      await toast.hover() // keeps it open while it is measured
      if (viewport.width < 768) await expect(page.getByRole("navigation", { name: "Main navigation" })).toBeVisible()
      await expect.poll(() => toastOverlaps(page)).toEqual([])
    })
  }
})

test.describe("form rows", () => {
  test.use({ viewport: { width: 2560, height: 1440 } })

  test("a slider beside selects lines up its label and its track with theirs", async ({ page }) => {
    await openGenerate(page)
    const box = async (name: string) => (await page.locator("label", { hasText: new RegExp(`^${name}$`) }).first().boundingBox())!
    const [method, schedule, steps] = [await box("Sampling method"), await box("Schedule type"), await box("Sampling steps")]
    // one row: the three fields side by side, their labels on one line
    expect(Math.abs(method.y - schedule.y)).toBeLessThan(1)
    expect(Math.abs(method.y - steps.y)).toBeLessThan(1)
    expect(steps.x).toBeGreaterThan(schedule.x)
    const select = (await page.getByRole("combobox", { name: "Sampling method" }).boundingBox())!
    const track = (await page.locator("[data-slot=slider-track]").first().boundingBox())!
    expect(Math.abs(select.y + select.height / 2 - (track.y + track.height / 2))).toBeLessThan(2)
  })
})

test.describe("page container", () => {
  test.use({ viewport: { width: 1920, height: 1080 } })

  test("extension pages share the Extensions page's column", async ({ page }) => {
    const column = async (hash: string) => {
      await page.goto(`/#/${hash}`)
      const root = page.locator("main [data-scroll-root] > div").first()
      await expect(root).toBeVisible()
      const box = (await root.boundingBox())!
      const padding = await root.evaluate((element) => getComputedStyle(element).paddingLeft)
      return { left: Math.round(box.x), width: Math.round(box.width), padding }
    }
    expect(await column("ext/toolbox")).toEqual(await column("extensions"))
  })
})

test.describe("toasts keep clear of the canvas toolbars", () => {
  test.use({ viewport: { width: 1280, height: 720 } })

  test("img2img at 1280×720", async ({ page }) => {
    await openGenerate(page, "img2img")
    const png = await page.evaluate(() => {
      const canvas = document.createElement("canvas")
      canvas.width = 640
      canvas.height = 832
      const ctx = canvas.getContext("2d")!
      ctx.fillStyle = "#2a6df4"
      ctx.fillRect(0, 0, 640, 832)
      return canvas.toDataURL("image/png").split(",")[1]!
    })
    const chooser = page.waitForEvent("filechooser")
    await page.getByRole("button", { name: "Add a source image" }).click()
    await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") })
    await expect(page.getByRole("img", { name: /^Image editor, 640×832/ })).toBeVisible()
    const toolbar = page.getByRole("toolbar", { name: "Image", exact: true })
    await expect(toolbar).toBeVisible()
    // an action's toast (the mock's LCM compatibility note) while the source is on the stage
    await page.getByRole("combobox", { name: "Sampling method" }).click()
    await page.getByRole("option", { name: "LCM", exact: true }).click()
    const toast = page.locator("[data-sonner-toast]").filter({ hasText: "LCM uses the SGM Uniform schedule" })
    await expect(toast).toBeVisible({ timeout: 20_000 })
    await toast.hover()
    // the toolbar is under the toast column: without the lift the toast would cover it
    const bar = (await toolbar.boundingBox())!
    expect(bar.x + bar.width).toBeGreaterThan(1280 - 24 - 356)
    await expect.poll(() => toastOverlaps(page)).toEqual([])
  })
})


test.describe("toggles and control rows", () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test("the extensions filter's Modified toggle is a field like the search beside it, and shows when it is on", async ({ page }) => {
    await openGenerate(page)
    const filter = page.getByRole("searchbox", { name: "Filter extensions" })
    if (!(await filter.isVisible())) await page.getByRole("button", { name: /^Extensions/ }).first().click()
    await expect(filter).toBeVisible()
    const toggle = page.getByRole("button", { name: "Only modified" })
    const look = () =>
      toggle.evaluate((element) => {
        const style = getComputedStyle(element)
        const probe = document.createElement("span")
        probe.style.color = "var(--primary)"
        document.body.append(probe)
        const primary = getComputedStyle(probe).color
        probe.remove()
        return { background: style.backgroundColor, border: style.borderTopColor, color: style.color, primary }
      })
    const fieldFill = await filter.evaluate((element) => getComputedStyle(element).backgroundColor)
    const off = await look()
    expect(off.background).toBe(fieldFill) // the same flat fill as the search field, not an outline
    expect(off.border).toBe("rgba(0, 0, 0, 0)")
    await toggle.click()
    await expect(toggle).toHaveAttribute("aria-pressed", "true")
    await page.mouse.move(0, 0) // hover has its own look; the colours also transition
    await expect.poll(async () => { const on = await look(); return on.color === on.primary }).toBe(true) // pressed wears the selection colour
    expect((await look()).background).not.toBe(off.background)
  })

  test("every pressed toggle wears the selection tint, hovered or not", async ({ page }) => {
    await page.goto("/#/queue")
    const toggle = page.getByRole("main").getByRole("button", { name: "Starred only" })
    const tinted = () =>
      toggle.evaluate((element) => {
        const probe = document.createElement("span")
        probe.style.color = "var(--primary)"
        document.body.append(probe)
        const primary = getComputedStyle(probe).color
        probe.remove()
        return getComputedStyle(element).color === primary
      })
    await toggle.click()
    await expect(toggle).toHaveAttribute("aria-pressed", "true")
    await expect.poll(tinted).toBe(true) // still under the pointer
    await page.mouse.move(0, 0)
    await expect.poll(tinted).toBe(true)
    // the header's Networks button is a lone toggle too (it opens the drawer)
    await page.goto("/#/txt2img")
    const networks = page.getByRole("banner").getByRole("button", { name: /Networks/ })
    await networks.click()
    await expect(networks).toHaveAttribute("aria-pressed", "true")
    await page.mouse.move(0, 0)
    await expect.poll(() => networks.evaluate((element) => {
      const probe = document.createElement("span")
      probe.style.color = "var(--primary)"
      document.body.append(probe)
      const primary = getComputedStyle(probe).color
      probe.remove()
      return getComputedStyle(element).color === primary
    })).toBe(true)
  })

  test("the checkpoint hashing row's select and button share one height", async ({ page }) => {
    await page.goto("/#/settings?view=server")
    const main = page.getByRole("main")
    const select = main.getByRole("combobox", { name: "Checkpoints hashed at once" })
    const button = main.getByRole("button", { name: "Hash all checkpoints" })
    await expect(button).toBeVisible()
    expect(Math.round((await select.boundingBox())!.height)).toBe(Math.round((await button.boundingBox())!.height))
  })
})

/** img2img with a source image on the canvas and a finished run (`batch` images) on the Result view. */
async function generateOnCanvas(page: Page, width: number, batch = 1) {
  if (batch > 1) {
    await page.addInitScript((size) => {
      localStorage.setItem("remi-form:img2img", JSON.stringify({ state: { values: { core: { batch_size: size } }, open: {}, tabs: {} }, version: 1 }))
    }, batch)
  }
  await page.setViewportSize({ width, height: 900 })
  await openGenerate(page, "img2img")
  const png = await page.evaluate(() => {
    const canvas = document.createElement("canvas")
    canvas.width = 640
    canvas.height = 480
    canvas.getContext("2d")!.fillRect(0, 0, 640, 480)
    return canvas.toDataURL("image/png").split(",")[1]!
  })
  const chooser = page.waitForEvent("filechooser")
  await page.getByRole("button", { name: "Add a source image" }).click()
  await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") })
  await expect(page.getByRole("img", { name: /^Image editor/ })).toBeVisible()
  await page.keyboard.press(`${mod}+Enter`)
  const view = page.getByRole("tablist", { name: "Image view" })
  await expect(view.getByRole("tab", { name: "Result" })).toHaveAttribute("aria-selected", "true", { timeout: 20_000 })
  await expect(page.getByRole("button", { name: "Stage latest result" })).toBeVisible()
  return view
}

test.describe("the img2img stage toolbar", () => {
  // 768 and 1280 give the result actions a row of their own; from 1680 the bar is one row, where
  // the longer English labels once pushed "Parameters" under "Hide history" (a bar of 740 to 830 px).
  for (const width of [768, 1280, 1680, 1760, 1840, 1920]) {
    test(`at ${width} px keeps its controls apart and whole, on at most two rows`, async ({ page }) => {
      const view = await generateOnCanvas(page, width)
      const layout = await view.evaluate((switcher) => {
        const bar = switcher.parentElement!
        const right = bar.getBoundingClientRect().right
        // a row is a vertical centre (the segmented tabs sit 2 px inside their track)
        const row = (element: Element) => {
          const box = element.getBoundingClientRect()
          return Math.round((box.top + box.bottom) / 8)
        }
        const name = (element: Element) => element.getAttribute("aria-label") || element.textContent?.trim()
        const controls = [...bar.querySelectorAll("button, [role=tab]")].filter((element) => element.getBoundingClientRect().width > 0)
        const cut = controls.filter((element) => element.getBoundingClientRect().right > right + 1).map(name)
        const overlaps: string[] = []
        for (const [index, a] of controls.entries()) {
          for (const b of controls.slice(index + 1)) {
            if (a.contains(b) || b.contains(a)) continue
            const [boxA, boxB] = [a.getBoundingClientRect(), b.getBoundingClientRect()]
            const across = Math.min(boxA.right, boxB.right) - Math.max(boxA.left, boxB.left)
            const down = Math.min(boxA.bottom, boxB.bottom) - Math.max(boxA.top, boxB.top)
            if (across > 0.5 && down > 0.5) overlaps.push(`${name(a)} × ${name(b)}`)
          }
        }
        const actions = bar.querySelector("[role=toolbar]")!
        const parameters = controls.find((button) => button.textContent?.trim() === "Parameters")!
        return { cut, overlaps, rows: new Set([...bar.children].filter((element) => element.getBoundingClientRect().width > 0).map(row)).size, actionsWithParameters: row(actions) === row(parameters) }
      })
      expect(layout.cut).toEqual([])
      expect(layout.overlaps).toEqual([])
      expect(layout.rows).toBeLessThanOrEqual(2)
      expect(layout.actionsWithParameters).toBe(true) // the result's own row, not split across two
    })
  }
})

test.describe("the canvas staging bar", () => {
  for (const [width, batch] of [
    [390, 1],
    [390, 4],
    [768, 4],
    [1024, 4],
    [1440, 4],
    [1920, 4],
  ] as const) {
    test(`at ${width} px with ${batch} candidate(s) is in view, takes the focus and keeps to one row`, async ({ page }) => {
      await generateOnCanvas(page, width, batch)
      // staged from the keyboard: the button goes away with the Result view
      await page.getByRole("button", { name: "Stage latest result" }).focus()
      await page.keyboard.press("Enter")
      const bar = page.getByRole("toolbar", { name: "Canvas staging" })
      await expect(bar).toBeAttached()
      await page.mouse.move(0, 0) // not hovering the canvas
      await page.waitForTimeout(400) // the overlay's fade
      const state = await bar.evaluate((element) => {
        let opacity = 1
        for (let node: Element | null = element; node; node = node.parentElement) opacity *= Number(getComputedStyle(node).opacity)
        const shown = [...element.children].filter((child) => child.getBoundingClientRect().width > 0)
        const rows = new Set(shown.map((child) => Math.round((child.getBoundingClientRect().top + child.getBoundingClientRect().bottom) / 8))).size
        const track = element.querySelector("[data-slot=slider-track]")
        const trackWidth = track && track.getBoundingClientRect().width > 0 ? track.getBoundingClientRect().width : null
        const focus = document.activeElement
        return { opacity, rows, trackWidth, focusInBar: element.contains(focus), focusOnAction: focus instanceof HTMLButtonElement }
      })
      expect(state.opacity).toBe(1)
      expect(state.focusInBar).toBe(true)
      expect(state.focusOnAction).toBe(false) // a held Enter must not accept or discard
      expect(state.rows).toBe(1)
      if (state.trackWidth !== null) expect(state.trackWidth).toBeGreaterThanOrEqual(60) // a fader, not a stub
    })
  }

  // Enter anywhere in the editor accepted the candidate: on "Discard candidate" it replaced the
  // source and dropped the rest. Escape interrupted a running job instead of discarding, when the
  // editor had appeared after the page (as here: the Source view after "Stage latest result").
  test("leaves Enter to a focused button and Escape to the candidates while a job runs", async ({ page }) => {
    await generateOnCanvas(page, 1440, 4)
    await page.getByRole("button", { name: "Stage latest result" }).focus()
    await page.keyboard.press("Enter")
    const bar = page.getByRole("toolbar", { name: "Canvas staging" })
    await expect(bar.getByText("Candidate 1 of 4")).toBeAttached()
    await bar.getByRole("button", { name: /Discard candidate/ }).focus()
    await page.keyboard.press("Enter")
    await expect(bar.getByText("Candidate 1 of 3")).toBeAttached()
    await expect(page.getByText("Accepted candidate as new base layer")).toHaveCount(0)
    // a new job shows the Result view; staging the last batch while it runs brings the editor back
    await page.keyboard.press(`${mod}+Enter`)
    const interrupt = page.getByRole("button", { name: "Interrupt" }).first()
    await expect(interrupt).toBeVisible()
    await page.getByRole("button", { name: "Stage latest result" }).focus()
    await page.keyboard.press("Enter")
    await expect(bar.getByText("Candidate 1 of 4")).toBeAttached()
    await page.keyboard.press("Escape")
    await expect(bar.getByText("Candidate 1 of 3")).toBeAttached()
    await page.waitForTimeout(300)
    await expect(interrupt).toBeVisible()
  })
})

test.describe("the img2img stage on a phone held sideways", () => {
  // The editor kept its 16rem floor in a column that had less room: it overflowed onto the history
  // strip, whose thumbnails then covered the editor's history and zoom controls.
  for (const [width, height] of [
    [844, 390],
    [932, 430],
  ] as const) {
    test(`at ${width}×${height} keeps the history strip below the editor`, async ({ page }) => {
      await page.setViewportSize({ width, height })
      await openGenerate(page, "img2img")
      const history = page.getByRole("region", { name: "History" })
      const editor = page.locator("#img2img-stage-source > *").first()
      const apart = async () => {
        const [top, bottom] = await Promise.all([editor.boundingBox(), history.boundingBox()])
        return Math.round(bottom!.y - (top!.y + top!.height))
      }
      expect(await apart(), "empty editor").toBeGreaterThanOrEqual(0)
      const png = await page.evaluate(() => {
        const canvas = document.createElement("canvas")
        canvas.width = 640
        canvas.height = 480
        canvas.getContext("2d")!.fillRect(0, 0, 640, 480)
        return canvas.toDataURL("image/png").split(",")[1]!
      })
      const chooser = page.waitForEvent("filechooser")
      await page.getByRole("button", { name: "Add a source image" }).click()
      await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") })
      await expect(page.getByRole("img", { name: /^Image editor/ })).toBeVisible()
      expect(await apart(), "with a source image").toBeGreaterThanOrEqual(0)
      // the editor keeps room to work in, and the history is a scroll away, not under it
      expect((await editor.boundingBox())!.height).toBeGreaterThanOrEqual(256)
      const heading = history.getByRole("heading", { name: "History" })
      await heading.scrollIntoViewIfNeeded()
      const hit = await heading.evaluate((element) => {
        const box = element.getBoundingClientRect()
        return element.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2))
      })
      expect(hit).toBe(true)
      // the result view: its picture scales down to the room there is
      await page.keyboard.press(`${mod}+Enter`)
      await expect(page.getByRole("tab", { name: "Result" })).toHaveAttribute("aria-selected", "true", { timeout: 20_000 })
      const picture = page.getByRole("button", { name: "Open image viewer" }).filter({ has: page.getByRole("img", { name: "Generated image" }) })
      await expect(picture).toBeVisible()
      const [shown, strip] = await Promise.all([picture.boundingBox(), history.boundingBox()])
      expect(Math.round(strip!.y - (shown!.y + shown!.height)), "result view").toBeGreaterThanOrEqual(0)
    })
  }
})

// Held sideways, the rail's links shrank to 20 px to fit and the last one was cut off, with nothing
// to say the rail scrolls: a tap where it showed opened Settings instead.
test("the rail on a phone held sideways keeps its links whole and fades where it scrolls", async ({ page }) => {
  await page.setViewportSize({ width: 844, height: 390 })
  await openGenerate(page)
  const rail = page.getByRole("navigation", { name: "Main navigation" })
  const heights = await rail.getByRole("link").evaluateAll((links) => links.map((link) => Math.round(link.getBoundingClientRect().height)))
  expect(Math.min(...heights)).toBeGreaterThanOrEqual(40)
  const scroller = rail.locator(".overflow-y-auto")
  await expect.poll(() => scroller.evaluate((element) => getComputedStyle(element).maskImage)).toContain("linear-gradient")
  // scrolled to its end, the last link is whole and takes its own taps
  const last = scroller.getByRole("link").last()
  await scroller.evaluate((element) => (element.scrollTop = element.scrollHeight))
  const hit = await last.evaluate((link) => {
    const box = link.getBoundingClientRect()
    return link.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2))
  })
  expect(hit).toBe(true)
})

// The palette opens a third of the way down with a list up to 60% of the window tall: in a short
// window (a phone held sideways, a 600 px laptop) it ran off the bottom and cut its last command.
for (const [width, height] of [
  [844, 390],
  [1280, 600],
] as const) {
  test(`the command palette fits a ${width}×${height} window`, async ({ page }) => {
    await page.setViewportSize({ width, height })
    await openGenerate(page)
    await page.keyboard.press(`${mod}+K`)
    const palette = page.getByRole("dialog", { name: "Command palette" })
    await expect(palette.getByRole("option").first()).toBeVisible()
    await page.waitForTimeout(300) // the opening zoom
    const box = (await palette.boundingBox())!
    expect(Math.round(box.y + box.height), "the palette's bottom").toBeLessThanOrEqual(height)
    // its last command scrolls into the list and shows whole
    const last = palette.getByRole("option").last()
    await last.scrollIntoViewIfNeeded()
    const [item, list] = await Promise.all([last.boundingBox(), palette.getByRole("listbox").boundingBox()])
    expect(Math.round(item!.y + item!.height)).toBeLessThanOrEqual(Math.round(Math.min(list!.y + list!.height, height)) + 1)
  })
}

// JSON Builder measured its canvas after the first paint: a load could show a 2 px stage, then the
// whole one (a layout shift of 0.136). The size it has when the canvas is first inserted, before
// anything can paint (a microtask after the commit), is the size the first frame shows.
test("JSON Builder's canvas opens at its size", async ({ page }) => {
  await page.addInitScript(() => {
    const found = new MutationObserver(() => {
      const stage = document.querySelector("[data-testid=json-builder-canvas] > div")
      if (!stage) return
      found.disconnect()
      ;(window as unknown as { firstStage: number }).firstStage = stage.getBoundingClientRect().height
    })
    found.observe(document, { childList: true, subtree: true })
  })
  await openRoute(page, "ext/json-builder")
  const canvas = page.getByRole("main").getByTestId("json-builder-canvas")
  await expect(canvas).toBeVisible()
  const settled = await canvas.locator(":scope > div").first().evaluate((stage) => stage.getBoundingClientRect().height)
  expect(settled).toBeGreaterThan(100)
  expect(await page.evaluate(() => (window as unknown as { firstStage: number }).firstStage)).toBe(settled)
})

// A dialog taller than a short window ran off both edges: held sideways (844×390), JSON Builder's
// "Load prompt" dialog had its title, Close and Load buttons outside the window.
test("a dialog fits a phone held sideways and scrolls inside", async ({ page }) => {
  await page.setViewportSize({ width: 844, height: 390 })
  await openRoute(page, "ext/json-builder")
  await page.getByRole("button", { name: "Load prompt…" }).click()
  const dialog = page.getByRole("dialog", { name: "Load prompt" })
  await expect(dialog).toBeVisible()
  await page.waitForTimeout(300) // the opening zoom
  const box = (await dialog.boundingBox())!
  expect(Math.round(box.y), "the dialog's top").toBeGreaterThanOrEqual(0)
  expect(Math.round(box.y + box.height), "the dialog's bottom").toBeLessThanOrEqual(390)
  const load = dialog.getByRole("button", { name: "Load", exact: true })
  await load.scrollIntoViewIfNeeded()
  const hit = await load.evaluate((button) => {
    const rect = button.getBoundingClientRect()
    return button.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2))
  })
  expect(hit).toBe(true)
})

// A popover taller than the room beside its trigger ran off the window: held sideways (844×390),
// the VAE / Text encoders picker had its title and search above the top edge, out of reach.
test("a popover fits a phone held sideways and scrolls inside", async ({ page }) => {
  await page.setViewportSize({ width: 844, height: 390 })
  await openGenerate(page)
  const trigger = page.getByRole("button", { name: "VAE / Text encoders" })
  if (!(await trigger.isVisible())) await page.getByRole("button", { name: "Model settings" }).click()
  await trigger.click()
  const popover = page.getByRole("dialog", { name: "VAE / Text encoders" })
  await expect(popover).toBeVisible()
  await page.waitForTimeout(300) // the opening zoom
  const box = (await popover.boundingBox())!
  expect(Math.round(box.y), "the popover's top").toBeGreaterThanOrEqual(0)
  expect(Math.round(box.y + box.height), "the popover's bottom").toBeLessThanOrEqual(390)
  await expect(popover.getByRole("textbox", { name: "Search modules" })).toBeInViewport()
})

// A popover scrolls up and down only: the Layers popover's switches reach 12 px past their row
// (their hit boxes) into its 10 px padding, and it scrolled sideways by 2 px.
test("the canvas Layers popover does not scroll sideways", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 })
  await openGenerate(page, "img2img")
  const png = await page.evaluate(() => {
    const canvas = document.createElement("canvas")
    canvas.width = 64
    canvas.height = 64
    return canvas.toDataURL("image/png").split(",")[1]!
  })
  const chooser = page.waitForEvent("filechooser")
  await page.getByRole("button", { name: "Add a source image" }).click()
  await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") })
  await page.getByRole("button", { name: "Layers and view" }).click()
  const popover = page.getByRole("dialog", { name: "Layers and view" })
  await expect(popover).toBeVisible()
  await expect(popover.getByRole("switch").first()).toBeVisible()
  // scrollable by the user: an axis that scrolls (auto / scroll) and more to scroll to
  const scrolls = await popover.evaluate((element) => ["auto", "scroll"].includes(getComputedStyle(element).overflowX) && element.scrollWidth > element.clientWidth)
  expect(scrolls, "scrolls sideways").toBe(false)
})

// While a job runs the Generate button shares its row with Skip and Interrupt: from 768 to 932 px
// its "Queue another" was cut to "Qu…", with nothing to read the rest from. The label shows when
// it fits beside the plus and only then: a fixed breakpoint hid it at 1280 px, where it fits.
for (const [width, height] of [
  [768, 1024],
  [900, 1024],
  [844, 390],
  [932, 430],
  [1024, 768],
  [1280, 720],
  [1440, 900],
] as const) {
  test(`the running Generate button is never a cut label at ${width}×${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height })
    await openGenerate(page)
    await page.keyboard.press(`${mod}+Enter`)
    const queue = page.getByRole("button", { name: "Queue another" })
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible() // Skip and Interrupt have joined the row
    await expect(queue).toBeVisible()
    const label = await queue.evaluate((button) => {
      const text = [...button.querySelectorAll("span")].find((span) => span.textContent === "Queue another")!
      const hidden = getComputedStyle(text).position === "absolute" // read out, not shown
      // the room beside the plus: the button's content box less the plus and the gap
      const style = getComputedStyle(button)
      const plus = button.querySelector("svg")!.getBoundingClientRect().width
      const room = button.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) - plus - parseFloat(style.columnGap)
      return { cut: !hidden && text.scrollWidth > text.clientWidth + 1, hidden, fits: text.scrollWidth <= room + 0.5, title: button.getAttribute("title") }
    })
    expect(label.cut, "a cut label").toBe(false)
    expect(label.hidden, label.fits ? "the label fits but is hidden" : "the label does not fit but is shown").toBe(!label.fits)
    if (label.hidden) expect(label.title).toBe("Queue another")
    await page.getByRole("button", { name: "Interrupt" }).click()
  })
}

// In a short window the empty stage's frame kept its minimum height and the stage cut off what
// was below it: held sideways, the next step ("Write a prompt, then press …") was not shown.
for (const [width, height] of [
  [844, 390],
  [932, 430],
  [1280, 720],
] as const) {
  test(`the empty stage shows its next step at ${width}×${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height })
    await openGenerate(page)
    const hint = page.getByText(/^Write a prompt, then press/)
    await expect(hint).toBeVisible()
    const cut = await hint.evaluate((node) => {
      const rect = node.getBoundingClientRect()
      let box = node.parentElement
      while (box && getComputedStyle(box).overflowY === "visible") box = box.parentElement
      const frame = box!.getBoundingClientRect()
      return { top: Math.round(rect.top - frame.top), bottom: Math.round(frame.bottom - rect.bottom) }
    })
    expect(cut.top, "room above the hint").toBeGreaterThanOrEqual(0)
    expect(cut.bottom, "room below the hint").toBeGreaterThanOrEqual(0)
  })
}

// The reset dot (a changed value's way back to its default) was a 24 × 20 px target between a
// label and the row's controls: it takes the room up to them, never reaching one.
for (const [width, height] of [
  [1440, 900],
  [390, 844],
] as const) {
  test(`reset dots are 32 px targets that reach no neighbouring control at ${width}×${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height })
    await openGenerate(page)
    const steps = page.getByRole("textbox", { name: "Sampling steps", exact: true }).first()
    await steps.fill("31")
    await steps.press("Tab")
    await page.getByRole("switch", { name: "Enable Hires. fix", exact: true }).click()
    await setPrompt(page, "a changed prompt")
    const dots = page.getByRole("button", { name: /^Reset .* to default$/ })
    await expect(dots.first()).toBeVisible()
    const boxes = await dots.evaluateAll((buttons) =>
      buttons
        .filter((button) => button.getBoundingClientRect().width > 0)
        .map((button) => {
          const rect = button.getBoundingClientRect()
          const after = getComputedStyle(button, "::after")
          const box = { left: rect.left - -parseFloat(after.left), top: rect.top - -parseFloat(after.top), right: rect.right - parseFloat(after.right), bottom: rect.bottom - parseFloat(after.bottom) }
          const controls = [...document.querySelectorAll("button, input, textarea, label, a, [role=switch], [role=slider], [contenteditable], [tabindex]:not([tabindex='-1'])")].filter(
            (element) => element !== button && !element.contains(button) && !button.contains(element),
          )
          const reached = controls
            .filter((element) => {
              const other = element.getBoundingClientRect()
              return other.width > 0 && other.left < box.right - 0.5 && other.right > box.left + 0.5 && other.top < box.bottom - 0.5 && other.bottom > box.top + 0.5
            })
            .map((element) => element.getAttribute("aria-label") ?? element.textContent?.slice(0, 30))
          return { name: button.getAttribute("aria-label"), width: Math.round(box.right - box.left), height: Math.round(box.bottom - box.top), reached }
        }),
    )
    expect(boxes.length).toBeGreaterThanOrEqual(3)
    for (const box of boxes) {
      expect(Math.min(box.width, box.height), `${box.name}: its target`).toBeGreaterThanOrEqual(32)
      expect(box.reached, `${box.name}: controls under its target`).toEqual([])
    }
  })
}

// Tabbing down the rail in a short window parked a link flush against the scroller's edge, under
// the fade, with the bottom of its focus ring cut off.
test("the rail keeps a focused link out of its faded edges", async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 340 })
  await openGenerate(page)
  const links = page.getByRole("navigation", { name: "Main navigation" }).locator(".overflow-y-auto").getByRole("link")
  const count = await links.count()
  await links.first().focus()
  const parked: string[] = []
  for (let i = 1; i < count; i++) {
    await page.keyboard.press("Tab")
    const room = await page.evaluate(() => {
      const link = document.activeElement as HTMLElement
      const scroller = link.closest(".overflow-y-auto") as HTMLElement
      const [a, s] = [link.getBoundingClientRect(), scroller.getBoundingClientRect()]
      const startFades = scroller.scrollTop > 1
      const endFades = scroller.scrollTop + scroller.clientHeight < scroller.scrollHeight - 1
      return { name: link.getAttribute("aria-label"), top: a.top - s.top, bottom: s.bottom - a.bottom, startFades, endFades }
    })
    // the fade is 1.5rem (24 px)
    if ((room.startFades && room.top < 24) || (room.endFades && room.bottom < 24)) parked.push(`${room.name}: ${Math.round(room.top)} px / ${Math.round(room.bottom)} px`)
  }
  expect(parked).toEqual([])
})

/** Boxes that scroll sideways though only meant to scroll up and down, and the page itself. */
async function sidewaysScrolls(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const found = [...document.querySelectorAll<HTMLElement>("body *")]
      .filter((element) => {
        const classes = element.getAttribute("class") ?? ""
        // strips meant to scroll sideways say so
        if (!["auto", "scroll"].includes(getComputedStyle(element).overflowX) || /overflow-x-auto|\[scrollbar-width:none\]/.test(classes)) return false
        return element.clientWidth > 0 && element.scrollWidth > element.clientWidth
      })
      .map((element) => `"${(element.getAttribute("class") ?? "").slice(0, 60)}" scrolls sideways by ${element.scrollWidth - element.clientWidth}px`)
    const page = document.documentElement.scrollWidth - innerWidth
    return page > 0 ? [...found, `the page scrolls sideways by ${page}px`] : found
  })
}

test.describe("no sideways scrolling", () => {
  // The copy button's hit area reaches 8 px past it, the panel left it 4 px: the parameters
  // panel scrolled sideways under every result (a scrollbar under it where scrollbars show).
  test("the parameters under a result", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    await openGenerate(page)
    await page.keyboard.press(`${mod}+Enter`)
    // the parameters panel shows once the run is done
    await expect(page.getByRole("region", { name: "Result" }).getByRole("heading", { name: "Prompt", exact: true })).toBeVisible({ timeout: 20_000 })
    expect(await sidewaysScrolls(page)).toEqual([])
  })

  // Below two columns the page was one column as wide as its widest line: the three section
  // tabs (or a long file name) pushed the page past a phone's window.
  test("PNG Info on a phone", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await openRoute(page, "png-info")
    const png = await page.evaluate(() => {
      const canvas = document.createElement("canvas")
      canvas.width = 64
      canvas.height = 48
      canvas.getContext("2d")!.fillRect(0, 0, 64, 48)
      return canvas.toDataURL("image/png").split(",")[1]!
    })
    const chooser = page.waitForEvent("filechooser")
    await page.getByRole("button", { name: "Choose an image to read" }).click()
    await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") })
    await expect(page.getByRole("main").locator("section h2")).toBeVisible()
    expect(await sidewaysScrolls(page)).toEqual([])
    // the section tabs do not fit: a select stands in for them
    const sections = page.getByRole("main").getByRole("combobox", { name: "Metadata" })
    await expect(sections).toBeVisible()

    await page.getByRole("main").locator("section h2").first().evaluate((title) => (title.textContent = `${"a".repeat(120)}.png`))
    expect(await sidewaysScrolls(page), "a long file name").toEqual([])

    // the select switches sections as the tabs do
    await sections.click()
    await page.locator("[data-slot=select-item]").filter({ hasText: "All metadata" }).click()
    await expect(page.getByRole("rowheader", { name: "Software" })).toBeVisible()
    expect(await sidewaysScrolls(page)).toEqual([])
  })

  // Below lg the backups page was one column as wide as its longest backup name.
  test("extension backups on a phone", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await openRoute(page, "extensions")
    await page.getByRole("main").getByRole("combobox", { name: "Extensions" }).click()
    await page.locator("[data-slot=select-item]").filter({ hasText: "Backup" }).first().click()
    await page.getByRole("textbox", { name: "Backup name" }).fill("Before trying the new ControlNet and the regional prompter extensions together")
    await page.getByRole("button", { name: "Back up" }).click()
    await expect(page.getByRole("option", { name: /^Before trying/ })).toBeVisible()
    expect(await sidewaysScrolls(page)).toEqual([])
  })

  // The model metadata form was one column as wide as its longest module name.
  test("the model metadata editor on a phone", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await openRoute(page, "networks")
    await page.getByRole("main").getByRole("combobox", { name: "Network kinds" }).click()
    await page.locator("[data-slot=select-item]").filter({ hasText: "Checkpoint" }).first().click()
    await page.getByRole("main").getByRole("option").first().click({ button: "right" })
    await page.getByRole("menuitem", { name: "Edit metadata" }).click()
    const modules = page.getByRole("dialog").getByRole("group", { name: "Preferred VAE / Text encoders" })
    await expect(modules.getByRole("checkbox").first()).toBeVisible()
    expect(await sidewaysScrolls(page)).toEqual([])
    await modules.locator("label").first().evaluate((label) => (label.textContent = "umt5_xxl_fp8_e4m3fn_scaled_with_a_much_longer_suffix.safetensors"))
    expect(await sidewaysScrolls(page), "a long module name").toEqual([])
  })
})
