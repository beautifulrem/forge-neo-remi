/**
 * Layout and feedback checks that need a real browser: the header at
 * 1280×720 (idle and while a job runs), the Generate panes' hide / show
 * toggles (real pane sizes), tab strips on a phone, boxes that scroll sideways, platform
 * shortcut hints and generation warnings.
 */
import { expect, type Locator, type Page, test } from "@playwright/test"
import { injectFaults, mod, openGenerate, openRoute, presetStorage, setPrompt } from "./helpers"

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

/** Pins the first setting on the settings page to the quick settings (the mock keeps it for the visit). */
async function pinQuickSetting(page: Page) {
  await page.goto("/#/settings")
  const pin = page.getByRole("button", { name: /^(Pin .*quick settings|将.*固定)/ }).first()
  await pin.click()
  await expect(pin).toHaveAttribute("aria-pressed", "true")
}

/** Where the header's last control ends past its content box (into its right padding), in px. */
async function pastPadding(page: Page): Promise<number> {
  return page.locator("header").first().evaluate((header) => {
    const contentRight = header.getBoundingClientRect().right - parseFloat(getComputedStyle(header).paddingRight)
    const rights = [...header.querySelectorAll<HTMLElement>("button, a[href]")].map((element) => element.getBoundingClientRect()).filter((rect) => rect.width > 0).map((rect) => rect.right)
    return Math.max(...rights) - contentRight
  })
}

/** Whether the word beside the connection dot is on the dot's line (a word that has no room wraps out of its box). */
async function besideDot(word: Locator): Promise<boolean> {
  return word.evaluate((element) => {
    const rect = element.getBoundingClientRect()
    const box = element.parentElement!.getBoundingClientRect()
    return rect.top >= box.top - 0.5 && rect.bottom <= box.bottom + 0.5 && rect.left >= box.left - 0.5 && rect.right <= box.right + 0.5
  })
}

test.describe("layout at 1280×720", () => {
  test.use({ viewport: { width: 1280, height: 720 } })

  for (const locale of ["en", "zh-CN"]) {
    test(`the header and the generate bar fit, idle and while a job runs (${locale})`, async ({ page }) => {
      await presetStorage(page, { locale })
      // Quick settings in the header too (the most crowded case).
      await page.goto("/#/settings")
      const pin = page.getByRole("button", { name: /^(Pin .*quick settings|将.*固定)/ }).first()
      await pin.click()
      await expect(pin).toHaveAttribute("aria-pressed", "true")
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

// The word "Offline" beside the connection dot pushed the appearance menu past a 390 px window:
// the header had a few pixels to spare there. The page title yields to it on a phone.
test.describe("the header on a phone while offline", () => {
  /** Puts the app's own connection store (the module URL it loaded) in the lost state. */
  async function goOffline(page: Page) {
    await page.evaluate(async () => {
      const loaded = performance.getEntriesByType("resource").map((entry) => entry.name)
      const path = loaded.find((name) => new URL(name).pathname.endsWith("/src/api/ws.ts"))
      if (!path) throw new Error("the connection module is not loaded")
      const { useWs } = (await import(/* @vite-ignore */ path)) as { useWs: { setState: (state: { status: string }) => void } }
      useWs.setState({ status: "closed" })
    })
  }

  for (const width of [390, 360]) {
    for (const [locale, offline] of [
      ["en", "Offline"],
      ["zh-CN", "离线"],
    ] as const) {
      test(`fits at ${width} px (${locale})`, async ({ page }) => {
        await page.setViewportSize({ width, height: 800 })
        await presetStorage(page, { locale })
        // Generate holds the most in the header: the page title, both modes and the networks button
        await page.goto("/#/txt2img")
        await expect(page.getByRole("textbox", { name: /^(Prompt|提示词)$/ })).toBeVisible()
        await goOffline(page)
        const header = page.locator("header").first()
        await expect(header.getByText(offline, { exact: true })).toBeVisible()
        await expect.poll(() => headerProblems(page)).toEqual([])
        expect(await header.evaluate((element) => element.scrollWidth <= element.clientWidth + 0.5), "the header overflows").toBe(true)
        const word = await header.getByText(offline, { exact: true }).evaluate((element) => element.scrollWidth <= element.clientWidth + 0.5)
        expect(word, "the word is cut").toBe(true)
        expect(await besideDot(header.getByText(offline, { exact: true })), "the word is not beside the dot").toBe(true)
      })
    }
  }

  // At 320 px in English with a quick setting pinned even the page's tabs leave no room for the word:
  // it took the last 13 px and the appearance menu left the window. The dot shows alone there.
  test("gives the word's room to the controls at 320 px with a quick setting pinned (en)", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 800 })
    await presetStorage(page, { locale: "en" })
    await pinQuickSetting(page)
    await page.evaluate(() => (location.hash = "#/txt2img"))
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
    const header = page.locator("header").first()
    await expect(header.getByRole("button", { name: "Quick settings" })).toBeVisible()
    await goOffline(page)
    await expect(header.getByRole("status").filter({ hasText: "Disconnected" })).toBeAttached()
    await expect.poll(() => besideDot(header.getByText("Offline", { exact: true }))).toBe(false)
    await expect.poll(() => headerProblems(page)).toEqual([])
    expect(await header.evaluate((element) => element.scrollWidth <= element.clientWidth + 0.5), "the header overflows").toBe(true)
    expect(await pastPadding(page), "a control is in the header's padding").toBeLessThanOrEqual(0.5)
  })
})

// Beside Generate's txt2img / img2img tabs the page title was cut to a stub on a phone ("G…" at
// 360 px in English), and a pinned quick setting pushed the appearance menu past the window. The
// title gives way to the page's own tabs there (still the page's heading); a page without them
// keeps its title whole.
test.describe("the header on a phone with a quick setting pinned", () => {
  for (const [locale, title, gallery] of [
    ["en", "Generate", "Gallery"],
    ["zh-CN", "生成", "图库"],
  ] as const) {
    test(`the title is whole or gives way to the page's tabs at 360 px (${locale})`, async ({ page }) => {
      await page.setViewportSize({ width: 360, height: 800 })
      await presetStorage(page, { locale })
      await pinQuickSetting(page)
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await expect(page.getByRole("textbox", { name: /^(Prompt|提示词)$/ })).toBeVisible()
      const header = page.locator("header").first()
      await expect(header.getByRole("button", { name: /^(Quick settings|快捷设置)$/ })).toBeVisible()

      // still the page's heading, shown whole or not at all: never a stub
      const heading = header.getByRole("heading", { level: 1, name: title })
      await expect(heading).toBeAttached()
      const shown = await heading.evaluate((element) => ({ width: element.getBoundingClientRect().width, cut: element.scrollWidth > element.clientWidth + 0.5 }))
      expect(shown.width <= 1 || !shown.cut, `the title is cut to ${shown.width.toFixed(0)} px`).toBe(true)
      await expect.poll(() => headerProblems(page)).toEqual([])
      expect(await header.evaluate((element) => element.scrollWidth <= element.clientWidth + 0.5), "the header overflows").toBe(true)
      expect(await pastPadding(page), "a control is in the header's padding").toBeLessThanOrEqual(0.5)

      // a page without tabs of its own in the header keeps its title, whole
      await page.evaluate(() => (location.hash = "#/gallery"))
      // the title changes with the address; Generate's tabs leave once the gallery's chunk is in
      await expect(header.locator("[data-page-tabs]")).toHaveCount(0)
      const galleryTitle = header.getByRole("heading", { level: 1, name: gallery })
      await expect(galleryTitle).toBeVisible()
      expect(await galleryTitle.evaluate((element) => element.scrollWidth <= element.clientWidth + 0.5), "the gallery's title is cut").toBe(true)
      await expect.poll(() => headerProblems(page)).toEqual([])
    })
  }
})

test.describe("the header's icon buttons on a phone", () => {
  test("draw their icons at one size, whether the label is hidden or never there", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 800 })
    await presetStorage(page, { locale: "en" })
    await pinQuickSetting(page)
    const sizes = new Map<string, string>()
    for (const [route, own] of [
      ["txt2img", /^Networks$/],
      ["gallery", /^Rescan$/],
      ["queue", /^(Pause|Resume)$/],
    ] as const) {
      await page.evaluate((hash) => (location.hash = hash), `#/${route}`)
      // the page's own header button is there once its chunk is in
      await expect(page.locator("header").first().getByRole("button", { name: own })).toBeVisible()
      // a 28 px square header button: search, appearance, and the buttons whose labels hide here
      const seen = await page.locator("header").first().evaluate((header) =>
        [...header.querySelectorAll<HTMLElement>("button")]
          .map((button) => ({ button, rect: button.getBoundingClientRect(), icon: button.querySelector("svg")?.getBoundingClientRect() }))
          .filter(({ rect, icon }) => icon && Math.round(rect.width) === 28 && Math.round(rect.height) === 28)
          // named as read out: aria-label, its rendered text (sr-only included), else the title (the networks button's label is display: none here)
          .map(({ button, icon }) => {
            // ghost, as search and the appearance menu: no fill and no edge until hovered or pressed
            const style = getComputedStyle(button)
            const look = /rgba\(.*, 0\)|transparent/.test(style.backgroundColor) && /rgba\(.*, 0\)|transparent/.test(style.borderTopColor) ? "" : " filled"
            return [(button.getAttribute("aria-label") || button.innerText.trim() || button.title.replace(/ \(.*\)$/, "")).trim(), `${icon!.width}×${icon!.height}${look}`] as const
          }),
      )
      for (const [name, size] of seen) sizes.set(name, size)
    }
    for (const name of ["Search", "Appearance", "Networks", "Quick settings", "Rescan", "Pause"]) {
      expect(sizes.get(name), `"${name}" is not an icon square`).toBeDefined()
    }
    expect(Object.fromEntries(sizes)).toEqual(Object.fromEntries([...sizes.keys()].map((name) => [name, "16×16"])))

    // a toggle keeps its one name, pressed or not (its title says what a press does)
    await page.evaluate(() => (location.hash = "#/txt2img"))
    const networks = page.locator("header").first().getByRole("button", { name: "Networks", exact: true })
    await expect(networks).toHaveAttribute("aria-pressed", "false")
    await networks.click()
    await expect(networks).toHaveAttribute("aria-pressed", "true")
  })
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
    // the hidden pane's controls are out of the Tab order and the accessibility tree (inert): its
    // Generate button takes no focus
    await expect(page.locator("#params")).toHaveAttribute("inert")
    const focusable = await page.locator("#params button").first().evaluate((button: HTMLElement) => (button.focus(), document.activeElement === button))
    expect(focusable).toBe(false)
    await page.getByRole("button", { name: "Show parameters" }).click()
    await expect.poll(sizes).toEqual(initial)
    await expect(generate).toHaveCount(0)

    await page.getByRole("button", { name: "Hide history" }).click()
    await expect.poll(sizes).toEqual([initial[0], initial[1]! + initial[2]!, 0])
    await expect(page.locator("#history")).toHaveAttribute("inert")
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

  test("#/queue keeps its modes on one line or turns them into a select at 360 px", async ({ page }) => {
    await page.setViewportSize({ width: 360, height: 800 })
    await page.goto("/#/queue")
    const main = page.getByRole("main")
    const select = main.getByRole("combobox", { name: "Queue" })
    const tabs = main.getByRole("tablist", { name: "Queue" })
    await expect(select.or(tabs)).toBeVisible()
    if (await select.isVisible()) {
      await select.click()
      await page.locator("[data-slot=select-item]", { hasText: "Recipes" }).click()
      await expect(page).toHaveURL(/view=recipes/)
      return
    }
    // one row of tabs, nothing scrolled or cut
    expect((await tabs.boundingBox())!.height).toBeLessThanOrEqual(40)
    expect(await tabs.evaluate((list) => list.scrollWidth <= list.clientWidth + 1)).toBe(true)
  })

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

test.describe("a deferred view's placeholder", () => {
  // A tabs node's placeholder was a 96 px block, where img2img's resize tabs (their strip over the
  // width and height sliders) are taller: what came after them jumped when the code arrived.
  // Measured from the strip, not the window: the Source card above loads on its own and grows 4 px
  // as it lands (its 160 px stand-in, a 164 px panel), at any time around the tabs.
  test("a tabs placeholder is as tall as the tabs", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    let release = () => {}
    const held = new Promise<void>((resolve) => (release = resolve))
    await page.route("**/src/ui/layout/Tabs.tsx*", async (route) => {
      await held
      await route.continue()
    })
    await page.goto("/#/img2img")
    const main = page.getByRole("main")
    await expect(main.locator("[data-part=tab-strip]").first()).toBeVisible()
    const next = main.getByRole("textbox", { name: "CFG Scale", exact: true })
    await expect(next).toBeVisible()
    // how far below the strip the next field starts, both read in one frame
    const below = async (strip: Locator) =>
      next.evaluate((field, top) => field.getBoundingClientRect().top - top!.getBoundingClientRect().top, await strip.elementHandle())
    const before = await below(main.locator("[data-part=tab-strip]").first())
    release()
    await expect(main.getByRole("tab", { name: "Resize to" })).toBeVisible()
    await expect(main.locator("[data-part=tab-strip]")).toHaveCount(0)
    const tabs = main.getByRole("tablist").filter({ has: page.getByRole("tab", { name: "Resize to" }) })
    expect(Math.abs((await below(tabs)) - before)).toBeLessThan(1)
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

  // An output's placeholder was a 36 px bar where the output (its label over "No output yet") is
  // 61 px: what came after it jumped down when its code arrived (CLS 0.11 on this page).
  test("an output's placeholder is as tall as the output", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    let release = () => {}
    const held = new Promise<void>((resolve) => (release = resolve))
    await page.route("**/src/ui/layout/Outputs.tsx*", async (route) => {
      await held
      await route.continue()
    })
    await page.goto("/#/ext/toolbox")
    const main = page.getByRole("main")
    const next = main.getByRole("textbox", { name: "Consolidated prompt" })
    await expect(main.locator("[data-slot=skeleton]").first()).toBeVisible()
    await expect(next).toBeVisible()
    const before = (await next.boundingBox())!.y
    release()
    await expect(main.getByText("No output yet")).toBeVisible()
    expect(Math.abs((await next.boundingBox())!.y - before)).toBeLessThan(1)
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

// With a keyboard (no touch screen) the image toolbar holds a shortcuts button too: in an editor
// narrower than about 385 px its last buttons spilled out of its panel to the right, where the
// editor's edge cut them off. It scrolls instead; every button can be scrolled to and takes its
// own taps.
test.describe("the editor's bottom toolbars in a narrow editor with a keyboard", () => {
  for (const width of [390, 360]) {
    test(`at ${width} px stay apart, the image toolbar scrolling`, async ({ page }) => {
      await page.setViewportSize({ width, height: 800 })
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
      const history = page.getByRole("toolbar", { name: "History and zoom" })
      const image = page.getByRole("toolbar", { name: "Image", exact: true })
      await expect(image.getByRole("button", { name: "Keyboard shortcuts" })).toBeAttached()
      const editor = page.locator("#img2img-stage-source > *").first()
      const [left, right, frame] = await Promise.all([history.boundingBox(), image.boundingBox(), editor.boundingBox()])
      expect(left!.x + left!.width, "the toolbars overlap").toBeLessThanOrEqual(right!.x + 0.5)
      expect(right!.x + right!.width, "the image toolbar leaves the editor").toBeLessThanOrEqual(frame!.x + frame!.width + 0.5)
      expect(left!.x, "the history toolbar leaves the editor").toBeGreaterThanOrEqual(frame!.x - 0.5)
      // each button, scrolled into the toolbar, is whole and takes its own taps
      for (const button of await image.getByRole("button").all()) {
        await button.scrollIntoViewIfNeeded()
        const hit = await button.evaluate((element) => {
          const box = element.getBoundingClientRect()
          const bar = element.closest("[role=toolbar]")!.getBoundingClientRect()
          const whole = box.left >= bar.left - 0.5 && box.right <= bar.right + 0.5
          return whole && element.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2))
        })
        expect(hit, (await button.getAttribute("aria-label")) ?? (await button.textContent()) ?? "a button").toBe(true)
      }
    })
  }
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

// A wide dialog was centred on the window, not on its safe area: with the notch on one side only
// (left 59, right 0) the shortcuts dialog, narrowed by both insets, sat 45 px from the left edge,
// under the notch, and 16 + 59 px from the right one.
test("a wide dialog keeps clear of a notch on one side", async ({ page }) => {
  const size = { width: 844, height: 390 }
  const insets = { top: 0, left: 59, bottom: 0, right: 0 }
  await page.setViewportSize(size)
  const cdp = await page.context().newCDPSession(page)
  await cdp.send("Emulation.setSafeAreaInsetsOverride" as never, { insets } as never)
  await openGenerate(page)
  await page.locator("body").press("?")
  const dialog = page.getByRole("dialog", { name: "Keyboard shortcuts" })
  await expect(dialog.getByRole("button", { name: "Close" })).toBeVisible()
  await page.waitForTimeout(300) // the opening zoom
  const box = (await dialog.boundingBox())!
  expect(Math.round(box.x), "its left edge: past the notch and the 1rem gutter").toBeGreaterThanOrEqual(insets.left + 16)
  expect(Math.round(box.x + box.width), "its right edge").toBeLessThanOrEqual(size.width - insets.right - 16)
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
  // the list keeps room for a few rows; the help text below it scrolls with the popover
  const list = popover.getByRole("group", { name: "VAE / Text encoders" })
  expect(Math.round((await list.boundingBox())!.height), "the modules list's height").toBeGreaterThanOrEqual(96)
})

// The modules list's last checkbox reached 2 px below the list with its hit box, so a list that fit
// still scrolled by 2 px.
test("the VAE / Text encoders list does not scroll when its rows fit", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await openGenerate(page)
  const trigger = page.getByRole("button", { name: "VAE / Text encoders" })
  if (!(await trigger.isVisible())) await page.getByRole("button", { name: "Model settings" }).click()
  await trigger.click()
  const popover = page.getByRole("dialog", { name: "VAE / Text encoders" })
  const list = popover.getByRole("group", { name: "VAE / Text encoders" })
  await expect(list.getByRole("checkbox").first()).toBeVisible()
  const { scrollHeight, clientHeight } = await list.evaluate((element) => ({ scrollHeight: element.scrollHeight, clientHeight: element.clientHeight }))
  expect(scrollHeight).toBe(clientHeight)
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

// Held sideways the stage is short, and the progress block over its lower part covered the middle of
// the picture: in a short window the progress is a strip along the stage's bottom edge.
for (const [width, height] of [
  [844, 390],
  [932, 430],
  [1280, 720],
] as const) {
  test(`a running job's progress leaves the middle of the stage clear at ${width}×${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height })
    await openGenerate(page)
    await page.keyboard.press(`${mod}+Enter`)
    // the stage's own (the header has a progress line of the same name)
    const progress = page.locator("[data-slot=stage-progress]").getByRole("progressbar", { name: "Generation progress" })
    await expect(progress).toBeVisible()
    const covers = await progress.evaluate((bar) => {
      const overlay = bar.closest("[data-slot=stage-progress]")!
      const stage = overlay.parentElement!.getBoundingClientRect()
      const box = overlay.getBoundingClientRect()
      const middle = stage.top + stage.height / 2
      return box.top <= middle && box.bottom >= middle
    })
    expect(covers, "the progress covers the middle of the stage").toBe(false)
    await page.getByRole("button", { name: "Interrupt" }).click()
  })
}

/** Puts a 640×480 picture on img2img's canvas. */
async function addSourceImage(page: Page) {
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
}

// Held sideways, the stage column gave the history strip 132 px and the picture being made 142 px
// (110 px on img2img's Result view, under its view tabs), a third of the window or less. In a short
// window the stage takes the column less the strip's title row: the strip is a scroll away.
for (const [width, height] of [
  [844, 390],
  [932, 430],
] as const) {
  for (const tab of ["txt2img", "img2img"] as const) {
    test(`the picture being made takes half the window or more at ${width}×${height} on ${tab}`, async ({ page }) => {
      await page.setViewportSize({ width, height })
      await openGenerate(page, tab)
      if (tab === "img2img") await addSourceImage(page)
      await page.keyboard.press(`${mod}+Enter`)
      const progress = page.locator("[data-slot=stage-progress]")
      await expect(progress).toBeVisible()
      // the stage's picture area: the live preview, under the progress strip along its bottom edge
      const area = (await progress.locator("xpath=..").boundingBox())!
      expect(area.height, "the picture's area").toBeGreaterThanOrEqual(height / 2)
      expect(area.y, "the picture's top").toBeGreaterThanOrEqual(0)
      expect(area.y + area.height, "the picture's bottom").toBeLessThanOrEqual(height)
      // the history strip's title stays in sight, and the strip is a scroll away
      const history = page.getByRole("region", { name: "History" })
      const titleHit = () =>
        history.getByRole("heading", { name: "History" }).evaluate((element) => {
          const box = element.getBoundingClientRect()
          return element.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2))
        })
      expect(await titleHit(), "the history's title is in sight").toBe(true)
      const strip = history.getByText("Results of this session appear here")
      await strip.scrollIntoViewIfNeeded()
      await expect(strip).toBeInViewport()
      expect(await titleHit(), "the history's title, scrolled to").toBe(true)
      await page.getByRole("button", { name: "Interrupt" }).click()
    })
  }
}

// Held sideways narrower than 768 px (667×375, 740×360) the page is the phone layout, whose stage
// was 70vh tall and ran under the run bar pinned to the bottom: the bar covered the progress strip
// along the stage's bottom edge and the picture's foot. The stage ends clear of the bar
// (`--pinned-bottom`); upright, where 70vh already does, it keeps its height.
for (const [width, height] of [
  [667, 375],
  [740, 360],
  [390, 844],
  [360, 640],
] as const) {
  for (const tab of ["txt2img", "img2img"] as const) {
    test(`the picture being made and its progress are clear of the run bar at ${width}×${height} on ${tab}`, async ({ page }) => {
      await page.setViewportSize({ width, height })
      await openGenerate(page, tab)
      if (tab === "img2img") await addSourceImage(page)
      await page.keyboard.press(`${mod}+Enter`)
      const progress = page.locator("[data-slot=stage-progress]")
      await expect(progress).toBeVisible()
      const runBar = page.locator("[data-toast-obstacle]").filter({ has: page.getByRole("button", { name: "Interrupt" }) })
      const [area, bar, main] = await Promise.all([progress.locator("xpath=..").boundingBox(), runBar.boundingBox(), page.getByRole("main").boundingBox()])
      expect(area!.y, "the picture's top, under the header").toBeGreaterThanOrEqual(main!.y - 0.5)
      expect(area!.y + area!.height, "the picture's foot, above the run bar").toBeLessThanOrEqual(bar!.y + 0.5)
      // the strip is the progress at both ends and in the middle, not the bar over it
      const shown = await progress.evaluate((strip) => {
        const box = strip.getBoundingClientRect()
        const y = box.y + box.height / 2
        return [box.x + 2, box.x + box.width / 2, box.right - 2].every((x) => strip.contains(document.elementFromPoint(x, y)))
      })
      expect(shown, "the progress is covered").toBe(true)
      if (height > width) {
        const stage = page.getByRole("region", { name: tab === "img2img" ? "Image area" : "Result", exact: true })
        const stageHeight = (await stage.boundingBox())!.height
        expect(Math.abs(stageHeight - Math.min(height * 0.7, 576)), "upright, the stage keeps 70vh").toBeLessThanOrEqual(1)
      }
      await page.getByRole("button", { name: "Interrupt" }).click()
    })
  }
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

// A lone small icon button (the server page's Refresh of the weights in memory) takes taps on an
// invisible 40 px box around it that reaches no other control.
test("the weights in memory refresh is a 40 px target", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto("/#/settings?view=server")
  await page.getByRole("button", { name: "Show the weights in memory" }).click()
  const refresh = page.getByRole("main").locator("[aria-live=polite]").filter({ hasText: "Weights in memory" }).getByRole("button", { name: "Refresh", exact: true })
  await expect(refresh).toBeVisible()
  const box = await refresh.evaluate((button) => {
    const rect = button.getBoundingClientRect()
    const after = getComputedStyle(button, "::after")
    const box = { left: rect.left + parseFloat(after.left), top: rect.top + parseFloat(after.top), right: rect.right - parseFloat(after.right), bottom: rect.bottom - parseFloat(after.bottom) }
    const reached = [...document.querySelectorAll("button, input, select, textarea, a, [role=switch], [tabindex]:not([tabindex='-1'])")]
      .filter((element) => element !== button && !element.contains(button) && !button.contains(element))
      .filter((element) => {
        const other = element.getBoundingClientRect()
        return other.width > 0 && other.left < box.right - 0.5 && other.right > box.left + 0.5 && other.top < box.bottom - 0.5 && other.bottom > box.top + 0.5
      })
      .map((element) => element.getAttribute("aria-label") ?? element.textContent?.slice(0, 30))
    return { width: Math.round(box.right - box.left), height: Math.round(box.bottom - box.top), reached }
  })
  expect(Math.min(box.width, box.height)).toBeGreaterThanOrEqual(40)
  expect(box.reached).toEqual([])
})

// On a phone the licences page's one-line links (the notices link, each Show license) are 20 px of
// text: each takes taps on an invisible box at least 40 px tall that reaches no other control.
test("the licences page's small links are 40 px tall targets on a phone", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto("/#/settings?view=licenses")
  const main = page.getByRole("main")
  await expect(main.getByRole("button", { name: "Show license" }).first()).toBeVisible()
  // an entry's name links to its project: one more line of text, its box shared with the toggle below
  const targets = [main.getByRole("link", { name: /Web interface libraries/ }), main.getByRole("button", { name: "Show license" }).first(), main.getByRole("link", { name: "ComfyUI" })]
  const boxes: { top: number; bottom: number }[] = []
  for (const target of targets) {
    const box = await target.evaluate((element) => {
      const rect = element.getBoundingClientRect()
      const after = getComputedStyle(element, "::after")
      const box = { left: rect.left + parseFloat(after.left), top: rect.top + parseFloat(after.top), right: rect.right - parseFloat(after.right), bottom: rect.bottom - parseFloat(after.bottom) }
      const reached = [...document.querySelectorAll("button, input, select, textarea, a, [tabindex]:not([tabindex='-1'])")]
        .filter((other) => other !== element && !other.contains(element) && !element.contains(other))
        .filter((other) => {
          const r = other.getBoundingClientRect()
          return r.width > 0 && r.left < box.right - 0.5 && r.right > box.left + 0.5 && r.top < box.bottom - 0.5 && r.bottom > box.top + 0.5
        })
        .map((other) => other.getAttribute("aria-label") ?? other.textContent?.slice(0, 30))
      return { name: element.textContent, top: box.top, bottom: box.bottom, width: Math.round(box.right - box.left), ownWidth: Math.round(rect.width), height: Math.round(box.bottom - box.top), reached }
    })
    boxes.push(box)
    expect(box.height, `${box.name}: its target`).toBeGreaterThanOrEqual(40)
    // an after: box given only its top and bottom is as wide as its (empty) content: none
    expect(box.width, `${box.name}: its target's width`).toBeGreaterThanOrEqual(box.ownWidth)
    expect(box.reached, `${box.name}: controls under its target`).toEqual([])
  }
  // ComfyUI's name and its Show license toggle: the two boxes meet, neither takes the other's taps
  expect(boxes[2]!.bottom, "the name's box reaches into the toggle's").toBeLessThanOrEqual(boxes[1]!.top + 0.5)
})

// The recipe capture dialog changed height as it read a form (nothing), found no changes (a line) and
// failed (a line and a button): its result has one slot of a fixed height.
test("the recipe capture dialog keeps its height while it reads a form, and when that fails", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await injectFaults(page, [
    { path: "^/schema/txt2img$", delay: 1500 },
    { path: "^/schema/img2img$", status: 404 },
  ])
  await openRoute(page, "queue?view=recipes")
  await page.getByRole("button", { name: "Capture from form" }).click()
  const dialog = page.getByRole("dialog")
  await expect(dialog).toBeVisible()
  await page.waitForTimeout(300) // the opening zoom
  const reading = (await dialog.boundingBox())!.height
  await expect(dialog.getByText(/No modified parameters found/)).toBeVisible({ timeout: 5000 })
  const read = (await dialog.boundingBox())!.height
  await dialog.getByRole("radio", { name: "img2img" }).click()
  await expect(dialog.getByRole("alert")).toBeVisible()
  const failed = (await dialog.boundingBox())!.height
  expect(Math.abs(read - reading), "found no changes").toBeLessThanOrEqual(1)
  expect(Math.abs(failed - reading), "failed").toBeLessThanOrEqual(1)
})

// At 390 the capture dialog's model family and tags fields shared a row and cut their placeholders.
test("the recipe capture dialog's fields show their placeholders on a phone", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await openRoute(page, "queue?view=recipes")
  await page.getByRole("button", { name: "Capture from form" }).click()
  const dialog = page.getByRole("dialog")
  for (const name of ["Model family", "Tags (comma-separated)"]) {
    const field = dialog.getByRole("textbox", { name })
    await expect(field).toBeVisible()
    const cut = await field.evaluate((input: HTMLInputElement) => {
      const probe = document.createElement("span")
      const style = getComputedStyle(input)
      probe.style.font = style.font
      probe.style.position = "absolute"
      probe.style.whiteSpace = "pre"
      probe.textContent = input.placeholder
      document.body.append(probe)
      const text = probe.getBoundingClientRect().width
      probe.remove()
      return text - (input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight))
    })
    expect(cut, `${name}: px of its placeholder cut off`).toBeLessThanOrEqual(0)
  }
})

// A section's enable switch has a wide after: box; it reached 4 px into the section's title, so a tap at
// the title's end flipped the switch instead of opening the section.
test("an enable switch's target stops short of the section title", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await openGenerate(page)
  const toggle = page.getByRole("switch", { name: "Enable Hires. fix", exact: true })
  await expect(toggle).toBeVisible()
  const overlap = await toggle.evaluate((element) => {
    const rect = element.getBoundingClientRect()
    const left = rect.left + parseFloat(getComputedStyle(element, "::after").left)
    const title = element.closest("section")!.querySelector("button[aria-expanded]")!.getBoundingClientRect()
    return title.right - left
  })
  expect(overlap, "px of the title under the switch's target").toBeLessThanOrEqual(0.5)
})

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

  // The preview dialog's three source tabs were a plain strip, whose labels grow once translated.
  test("the preview dialog's source tabs at 320 px in Chinese", async ({ page }) => {
    await presetStorage(page, { locale: "zh-CN" })
    await page.setViewportSize({ width: 320, height: 720 })
    await openRoute(page, "networks")
    await page.getByRole("main").getByRole("option").first().click({ button: "right" })
    await page.getByRole("menuitem", { name: "设置预览图" }).click()
    const dialog = page.getByRole("dialog")
    // the strip, or the select it turns into, names the sources
    await expect(dialog.getByRole("tablist", { name: "预览图来源" }).or(dialog.getByRole("combobox", { name: "预览图来源" }))).toBeVisible()
    await expect
      .poll(() =>
        dialog.evaluate((root) =>
          [...root.querySelectorAll<HTMLElement>("[role=tablist], [role=tab]")]
            .filter((element) => element.getBoundingClientRect().width > 0 && getComputedStyle(element).visibility !== "hidden")
            .filter((element) => element.scrollWidth > element.clientWidth + 1)
            .map((element) => element.getAttribute("aria-label") ?? element.textContent),
        ),
      )
      .toEqual([])
  })

  // The phone's "Insert into" row (its label and two segmented controls) was 400 px wide in English.
  test("the insert targets fit a phone's row", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await openRoute(page, "networks")
    const row = page.getByRole("main").getByRole("group", { name: "Insert into" }).first()
    await expect(row.getByRole("radio", { name: "Negative" })).toBeVisible()
    const { scrollWidth, clientWidth } = await row.evaluate((element) => ({ scrollWidth: element.scrollWidth, clientWidth: element.clientWidth }))
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth)
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

// The viewer, the comparison and the networks sheet are portalled outside the app's shell, which pads
// for a notch and the home indicator: on a phone held sideways their arrows, counter and Close button
// sat under the notch.
test.describe("overlays on a phone held sideways", () => {
  const size = { width: 852, height: 393 }
  const insets = { top: 0, left: 59, bottom: 21, right: 59 }

  test.beforeEach(async ({ page }) => {
    await page.setViewportSize(size)
    const cdp = await page.context().newCDPSession(page)
    await cdp.send("Emulation.setSafeAreaInsetsOverride" as never, { insets } as never)
  })

  /** The buttons of `overlay` shown on screen (not scrolled out of their list) that reach into the insets. */
  const unsafe = (overlay: Locator) =>
    overlay.getByRole("button").evaluateAll(
      (buttons, { size, insets }) =>
        buttons.flatMap((button) => {
          const box = button.getBoundingClientRect()
          if (box.width === 0 || !button.contains(document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2))) return []
          const inside = box.left >= insets.left - 0.5 && box.right <= size.width - insets.right + 0.5 && box.bottom <= size.height - insets.bottom + 0.5
          return inside ? [] : [`${button.getAttribute("aria-label") ?? button.textContent?.trim()} at ${Math.round(box.left)},${Math.round(box.top)}`]
        }),
      { size, insets },
    )

  /** Whether `box` lies inside the safe area (the inset sides only). */
  const outside = (box: { x: number; y: number; width: number; height: number }) =>
    [
      box.x < insets.left - 0.5 ? `left ${Math.round(box.x)}` : "",
      box.x + box.width > size.width - insets.right + 0.5 ? `right ${Math.round(box.x + box.width)}` : "",
      box.y + box.height > size.height - insets.bottom + 0.5 ? `bottom ${Math.round(box.y + box.height)}` : "",
    ].filter(Boolean)

  // The safe area was kept by the dialogs and sheets only (O14-1): menus, selects and popovers took
  // Base UI's 5 px from the screen's edge, toasts sat 24 px from it, the nav rail ran onto the home
  // indicator, and the command palette and the selection bar reached 5 px into it.
  test("a menu", async ({ page }) => {
    await openGenerate(page)
    await page.getByRole("button", { name: "Appearance" }).click()
    const menu = page.getByRole("menu")
    await expect(menu.getByRole("menuitemradio").first()).toBeVisible()
    await page.waitForTimeout(300) // its opening animation
    expect(outside((await menu.boundingBox())!)).toEqual([])
  })

  test("a select and a popover at the screen's edges", async ({ page }) => {
    await openGenerate(page)
    const main = page.getByRole("main")
    // the lowest select of the form, scrolled to the bottom of the window
    const select = main.locator("[data-slot=select-trigger]:visible").last()
    await select.evaluate((element) => element.scrollIntoView({ block: "end" }))
    await select.click()
    const options = page.locator("[data-slot=select-content]")
    await expect(options.getByRole("option").first()).toBeVisible()
    await page.waitForTimeout(300)
    expect(outside((await options.boundingBox())!), "the select").toEqual([])
    await page.keyboard.press("Escape")

    // the right-most popover trigger on the page
    const triggers = page.locator("[data-slot=popover-trigger]:visible")
    const lefts = await triggers.evaluateAll((elements) => elements.map((element) => element.getBoundingClientRect().right))
    expect(lefts.length, "a popover on the page").toBeGreaterThan(0)
    await triggers.nth(lefts.indexOf(Math.max(...lefts))).click()
    const popover = page.locator("[data-slot=popover-content]")
    await expect(popover).toBeVisible()
    await page.waitForTimeout(300)
    expect(outside((await popover.boundingBox())!), "the popover").toEqual([])
  })

  test("a toast", async ({ page }) => {
    // the browser records 250 resources by default; the dev server loads more modules than that
    await page.addInitScript(() => performance.setResourceTimingBufferSize(10_000))
    await openGenerate(page)
    // the app's own sonner (the dev server's module of it), as any feature calls it
    await page.evaluate(async () => {
      const url = performance.getEntriesByType("resource").map((entry) => entry.name).find((name) => /\/sonner\.js/.test(new URL(name).pathname))
      if (!url) throw new Error("sonner not loaded")
      const { toast } = (await import(/* @vite-ignore */ url)) as { toast: { success: (text: string, options: object) => void } }
      toast.success("Queued job #4 updated", { action: { label: "Open queue", onClick: () => {} }, duration: 60_000 })
    })
    const item = page.locator("[data-sonner-toast]").first()
    await expect(item.getByRole("button", { name: "Open queue" })).toBeVisible()
    await page.waitForTimeout(500) // it slides in
    expect(outside((await item.boundingBox())!)).toEqual([])
  })

  test("the nav rail, the command palette and the selection bar", async ({ page }) => {
    await page.goto("/#/gallery")
    const rail = page.getByRole("navigation", { name: "Main navigation" })
    await expect(rail.getByRole("link").last()).toBeVisible()
    expect(outside((await rail.getByRole("link").last().boundingBox())!), "the rail's last item").toEqual([])

    const tiles = page.getByRole("list", { name: "Images" }).getByRole("listitem")
    await tiles.first().hover()
    await tiles.first().getByRole("checkbox").click()
    const bulk = page.getByRole("toolbar", { name: "Selected images" })
    await expect(bulk).toContainText("1 selected")
    await page.waitForTimeout(300)
    expect(outside((await bulk.boundingBox())!), "the selection bar").toEqual([])
    await page.keyboard.press("Escape")

    await page.keyboard.press(`${mod}+K`)
    const palette = page.getByRole("dialog", { name: "Command palette" })
    await expect(palette).toBeVisible()
    await page.waitForTimeout(300)
    expect(outside((await palette.boundingBox())!), "the command palette").toEqual([])
  })

  test("the image viewer", async ({ page }) => {
    await page.goto("/#/gallery")
    const tile = page.getByRole("list", { name: "Images" }).getByRole("listitem").first()
    await tile.getByRole("button").first().press("Enter")
    const viewer = page.getByRole("dialog", { name: "Image viewer" })
    await expect(viewer.getByRole("button", { name: "Next image" })).toBeVisible()
    expect(await unsafe(viewer)).toEqual([])
  })

  test("the comparison", async ({ page }) => {
    await page.goto("/#/gallery")
    const tiles = page.getByRole("list", { name: "Images" }).getByRole("listitem")
    const bulk = page.getByRole("toolbar", { name: "Selected images" })
    for (const index of [0, 1]) {
      await tiles.nth(index).hover()
      await tiles.nth(index).getByRole("checkbox").click()
      await expect(bulk).toContainText(`${index + 1} selected`)
    }
    await bulk.getByRole("button", { name: "Compare" }).click()
    const dialog = page.getByRole("dialog", { name: "Compare images" })
    await expect(dialog.getByRole("slider", { name: "Comparison divider" })).toBeVisible()
    expect(await unsafe(dialog)).toEqual([])
  })

  test("the networks sheet", async ({ page }) => {
    await openGenerate(page)
    await page.locator("body").press("Alt+N")
    const drawer = page.getByRole("dialog", { name: "Networks" })
    await expect(drawer.getByRole("option").first()).toBeVisible()
    // its last row: the list scrolled to its end
    await drawer.locator(".overflow-y-auto").last().evaluate((list) => (list.scrollTop = list.scrollHeight))
    const bottom = await drawer.evaluate((element) => element.getBoundingClientRect().bottom - parseFloat(getComputedStyle(element).paddingBottom))
    expect(bottom).toBeLessThanOrEqual(size.height - insets.bottom + 0.5)
    expect(await unsafe(drawer)).toEqual([])
  })

  // A centred dialog was sized from the whole viewport: a wide one (the shortcuts, 48rem) reached under
  // the notch with its Close button, and a tall one ran onto the home indicator.
  test("a dialog", async ({ page }) => {
    await openGenerate(page)
    await page.locator("body").press("?")
    const dialog = page.getByRole("dialog", { name: "Keyboard shortcuts" })
    await expect(dialog.getByRole("button", { name: "Close" })).toBeVisible()
    await page.waitForTimeout(300) // the opening zoom
    const box = (await dialog.boundingBox())!
    expect(Math.round(box.x), "its left edge").toBeGreaterThanOrEqual(insets.left)
    expect(Math.round(box.x + box.width), "its right edge").toBeLessThanOrEqual(size.width - insets.right)
    expect(Math.round(box.y + box.height), "its bottom").toBeLessThanOrEqual(size.height - insets.bottom)
    expect(await unsafe(dialog)).toEqual([])
  })

  test("the settings sections", async ({ page }) => {
    await page.goto("/#/settings")
    await page.getByRole("button", { name: "Settings sections" }).first().click()
    const sheet = page.getByRole("dialog", { name: "Settings sections" })
    await expect(sheet.getByRole("button").nth(2)).toBeVisible()
    await page.waitForTimeout(300) // the slide in
    await sheet.evaluate((element) => (element.scrollTop = element.scrollHeight))
    expect(await unsafe(sheet)).toEqual([])
  })

  test("the image editor in full screen", async ({ page }) => {
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
    await page.getByRole("button", { name: "Edit in full screen" }).click()
    const editor = page.getByRole("dialog")
    await expect(editor.getByRole("button", { name: "Exit full screen" })).toBeVisible()
    await page.waitForTimeout(300)
    expect(await unsafe(editor)).toEqual([])
  })

  // From md a bar pinned to the bottom of a page stands on the screen's edge (the phone's bottom
  // navigation keeps that room below md): only the gallery's selection bar cleared the home indicator.
  test("the bars pinned to the bottom of a page", async ({ page }) => {
    await openRoute(page, "settings")
    await page.getByRole("switch", { name: "Automatically save every generated image", exact: true }).click()
    const unsaved = page.getByRole("region", { name: "Unsaved changes" })
    await expect(unsaved.getByRole("button").first()).toBeVisible()
    await page.waitForTimeout(300) // the slide in
    expect(await unsafe(unsaved)).toEqual([])

    await openRoute(page, "extensions")
    await page.getByRole("main").getByRole("switch").first().click()
    const pending = page.getByRole("region", { name: "Pending extension changes" })
    await expect(pending.getByRole("button").first()).toBeVisible()
    await page.waitForTimeout(300)
    expect(await unsafe(pending)).toEqual([])

    await openRoute(page, "ext/json-builder")
    const send = page.getByRole("button", { name: "Send to txt2img" })
    await expect(send).toBeVisible()
    const box = (await send.boundingBox())!
    expect(outside(box)).toEqual([])
  })
})

// The JSON Builder's send bar was pinned within the column under the canvas, which starts low on a
// phone held sideways: the page opened with the send buttons half below the screen.
test("the JSON Builder's send buttons are on screen when it opens on a phone held sideways", async ({ page }) => {
  await page.setViewportSize({ width: 852, height: 393 })
  await openRoute(page, "ext/json-builder")
  const send = page.getByRole("button", { name: "Send to txt2img" })
  await expect(send).toBeVisible()
  const box = (await send.boundingBox())!
  expect(Math.round(box.y + box.height)).toBeLessThanOrEqual(393)
  expect(await send.evaluate((button) => button.contains(document.elementFromPoint(button.getBoundingClientRect().left + 8, button.getBoundingClientRect().top + 8)))).toBe(true)
})

// The comparison's frame took 60 % of the screen's height whatever its row had left: with the
// parameters diff open (its default) it spilled out of the row over the header (the description,
// "Side by side") and the diff, and side by side showed both images 0 px high on a phone held
// sideways. The route sweeps never opened it.
test.describe("the comparison fits its frame to its row", () => {
  for (const { name, size, insets } of [
    { name: "a laptop", size: { width: 1280, height: 720 }, insets: { top: 0, left: 0, bottom: 0, right: 0 } },
    { name: "a phone held sideways", size: { width: 844, height: 390 }, insets: { top: 0, left: 47, bottom: 21, right: 47 } },
  ]) {
    test(`on ${name}`, async ({ page }) => {
      await page.setViewportSize(size)
      const cdp = await page.context().newCDPSession(page)
      await cdp.send("Emulation.setSafeAreaInsetsOverride" as never, { insets } as never)
      await page.goto("/#/gallery")
      const tiles = page.getByRole("list", { name: "Images" }).getByRole("listitem")
      const bulk = page.getByRole("toolbar", { name: "Selected images" })
      for (const index of [0, 1]) {
        await tiles.nth(index).hover()
        await tiles.nth(index).getByRole("checkbox").click()
        await expect(bulk).toContainText(`${index + 1} selected`)
      }
      await bulk.getByRole("button", { name: "Compare" }).click()
      const dialog = page.getByRole("dialog", { name: "Compare images" })
      const slider = dialog.getByRole("slider", { name: "Comparison divider" })
      await expect(slider).toBeVisible()
      await expect(dialog.getByRole("button", { name: /Parameters diff/ })).toHaveAttribute("aria-expanded", "true")
      await page.waitForTimeout(300) // the opening zoom

      // the frame stays inside its row, and nothing it would cover is hidden under it
      const spill = await slider.evaluate((handle) => {
        const frame = handle.parentElement!
        const own = frame.getBoundingClientRect()
        const row = frame.parentElement!.getBoundingClientRect()
        return [own.top - row.top, row.bottom - own.bottom, own.left - row.left, row.right - own.right].map((gap) => Math.round(gap))
      })
      expect(Math.min(...spill), `frame inside its row (gaps ${spill.join(", ")})`).toBeGreaterThanOrEqual(0)
      for (const control of [dialog.locator("[data-slot=dialog-description]"), dialog.getByRole("radio", { name: "Side by side" }), dialog.getByRole("button", { name: /Parameters diff/ })]) {
        await control.scrollIntoViewIfNeeded()
        const hit = await control.evaluate((element) => {
          const box = element.getBoundingClientRect()
          return element.contains(document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2))
        })
        expect(hit, `${await control.textContent()} can be clicked`).toBe(true)
      }

      // side by side: both images have room
      await dialog.getByRole("radio", { name: "Side by side" }).click()
      const heights = await dialog.locator("figure img").evaluateAll((images) => images.map((image) => Math.round(image.getBoundingClientRect().height)))
      expect(heights).toHaveLength(2)
      for (const height of heights) expect(height).toBeGreaterThanOrEqual(100)
    })
  }
})

// Upright, the page ends at the bottom navigation, which keeps the home indicator's room itself: the
// selection bar keeps its 16 px above it, the inset is not counted twice.
test.describe("the selection bar on a phone held upright", () => {
  test("stands 16 px above the bottom navigation", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    const cdp = await page.context().newCDPSession(page)
    await cdp.send("Emulation.setSafeAreaInsetsOverride" as never, { insets: { top: 47, left: 0, bottom: 34, right: 0 } } as never)
    await page.goto("/#/gallery")
    const tile = page.getByRole("list", { name: "Images" }).getByRole("listitem").first()
    await tile.hover()
    await tile.getByRole("checkbox").click()
    const bulk = page.getByRole("toolbar", { name: "Selected images" })
    await expect(bulk).toContainText("1 selected")
    await page.waitForTimeout(300)
    const nav = page.getByRole("navigation", { name: "Main navigation" }).last()
    const gap = (await nav.boundingBox())!.y - ((await bulk.boundingBox())!.y + (await bulk.boundingBox())!.height)
    expect(Math.round(gap)).toBe(16)
  })
})

// Settings > Server: while the status loaded, the model's place beside "Model memory" held a short
// skeleton; the model's name then took more room, the description beside it wrapped once more and
// every panel below jumped (CLS 0.243 on each load in Chinese at 1440×900).
// When the status failed, its error and Retry sat on a line of their own above the panels and
// pushed them down 48 px; they take the version's place on the header line instead.
// On a phone the failure, its Retry and the API documentation link did not fit one line: the line
// wrapped and the panels moved down all the same; there the link has a line of its own.
for (const width of [1440, 390, 360]) {
  test(`the server view's status failure leaves the panels where they are at ${width} px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    const firstPanelTop = async () => {
      const reload = page.getByRole("main").getByRole("button", { name: "Reload UI" })
      await expect(reload).toBeVisible()
      return (await reload.boundingBox())!.y
    }
    await page.goto("/#/settings?view=server")
    await expect(page.getByText(/^Version /)).toBeVisible()
    const loaded = await firstPanelTop()
    await injectFaults(page, [{ path: "^/server/status$", status: 404 }])
    await page.reload()
    await expect(page.getByRole("alert").filter({ hasText: "Could not read the server status" })).toBeVisible()
    expect(Math.abs((await firstPanelTop()) - loaded), "the panels moved").toBeLessThanOrEqual(0.5)
  })
}

test("the server view's panels stay put on a phone when the version arrives", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await injectFaults(page, [{ path: "^/server/status$", delay: 1500 }])
  await page.goto("/#/settings?view=server")
  const reload = page.getByRole("main").getByRole("button", { name: "Reload UI" })
  await expect(reload).toBeVisible()
  await expect(page.getByText(/^Version /)).toHaveCount(0)
  const before = (await reload.boundingBox())!.y
  await expect(page.getByText(/^Version /)).toBeVisible()
  expect(Math.abs((await reload.boundingBox())!.y - before), "the panels moved").toBeLessThanOrEqual(0.5)
})

test.describe("the server view while its status loads", () => {
  test("keeps the panels below the model in place", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await presetStorage(page, { locale: "zh-CN" })
    await injectFaults(page, [{ path: "^/server/status$", delay: 1500 }])
    await page.goto("/#/settings?view=server")
    const unload = page.getByRole("button", { name: "卸载模型" })
    await expect(unload).toBeVisible()
    const before = (await unload.boundingBox())!.y
    // the loaded model's name (the mock's checkpoint), its hash in full in the title
    await expect(page.getByRole("main").locator("[data-slot=badge][title]")).toBeVisible()
    const after = (await unload.boundingBox())!.y
    expect(Math.abs(after - before), "the panel's buttons moved").toBeLessThanOrEqual(0.5)
  })
})

// A cold load of the settings: while `/options` loaded, the section nav held a title and eight rows
// with the System views under them, and the sections that came pushed those off the bottom of the
// nav (a layout shift of 0.024 at 1440×900, 0.047 at 1024×600; none on a phone, where the nav is in
// a closed sheet); on a system view the nav then scrolled the current one back into sight. Showing
// every setting, the nav's placeholder fills it as its sections will, and the current system view is
// in sight while they load and as they arrive.
test.describe("a cold load of the settings", () => {
  for (const [route, width, height] of [
    ["settings", 1440, 900],
    ["settings", 1024, 600],
    ["settings", 390, 844],
    ["settings?view=licenses", 1024, 600],
  ] as const) {
    test(`shifts the layout by less than 0.05, and nothing on the page, at ${width}×${height} on #/${route}`, async ({ page }) => {
      await page.setViewportSize({ width, height })
      await injectFaults(page, [{ path: "^/options$", delay: 1500 }])
      await page.addInitScript(() => {
        const record = window as unknown as { layoutShift: number; pageShifts: string[] }
        record.layoutShift = 0
        record.pageShifts = []
        new PerformanceObserver((list) => {
          for (const entry of list.getEntries() as (PerformanceEntry & { value: number; hadRecentInput: boolean; sources: { node: Node | null }[] })[]) {
            if (entry.hadRecentInput) continue
            record.layoutShift += entry.value
            // the shell's header and rail settle on every page; the settings page is in `main`
            for (const { node } of entry.sources) {
              const element = node instanceof Element ? node : (node?.parentElement ?? null)
              if (element?.closest("main")) record.pageShifts.push(`${entry.value.toFixed(4)} ${element.tagName.toLowerCase()}.${element.className} "${element.textContent?.slice(0, 40)}"`)
            }
          }
        }).observe({ type: "layout-shift", buffered: true })
      })
      await page.goto(`/#/${route}`)
      const sections = page.getByRole("main").getByRole("navigation", { name: "Settings sections" })
      await expect(page.getByRole("main").locator("[aria-busy=true]").first()).toBeVisible()
      if (route === "settings") await expect(page.locator("[data-settings-scroller]")).toBeVisible({ timeout: 10_000 })
      else await expect(sections.getByRole("button", { name: "Saving Images/Grids" })).toBeAttached({ timeout: 10_000 })
      await page.waitForTimeout(500)
      const { layoutShift, pageShifts } = await page.evaluate(() => {
        const record = window as unknown as { layoutShift: number; pageShifts: string[] }
        return { layoutShift: record.layoutShift, pageShifts: record.pageShifts }
      })
      expect(pageShifts, "shifts on the settings page").toEqual([])
      expect(layoutShift).toBeLessThan(0.05)
    })
  }
})
