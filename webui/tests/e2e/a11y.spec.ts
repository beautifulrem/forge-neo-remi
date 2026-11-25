/**
 * axe-core scans of every page in light and dark: no serious or critical
 * violations (WCAG 2.x A/AA rules plus axe best practices).
 */
import AxeBuilder from "@axe-core/playwright"
import { expect, type Page, test } from "@playwright/test"
import { injectFaults, openRoute, presetStorage, type Theme } from "./helpers"

/** A 96×64 solid PNG. */
const SOURCE_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAaUlEQVR42u3QMQ0AAAgDsElDGtKQhgNujiZV0FQPhygQJEiQIEGCBAlCkCBBggQJEiQIQYIECRIkSJAgBAkSJEiQIEGCBCFIkCBBggQJEoQgQYIECRIkSBCCBAkSJEiQIEGCECRIkKB/Fn+AAh1bHed2AAAAAElFTkSuQmCC",
  "base64",
)

const PAGES = [
  "txt2img",
  "img2img",
  "extras",
  "png-info",
  "gallery",
  "networks",
  "queue",
  "settings",
  "extensions",
  "ext/toolbox",
  "ext/json-builder",
  "no-such-page",
] as const

async function scan(page: Page) {
  // Scan the settled page: colours read mid-transition (a toast fading in or
  // stacking behind a newer one, a popover opening) are not what anyone reads
  // and made the contrast rule fail at random. Endless animations (spinners)
  // are left running.
  await page.waitForFunction(
    () => document.getAnimations().every((animation) => animation.playState !== "running" || animation.effect?.getTiming().iterations === Infinity),
    null,
    { timeout: 5_000 },
  )
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"])
    .analyze()
  return results.violations
    .filter((violation) => violation.impact === "serious" || violation.impact === "critical")
    .map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      help: violation.help,
      nodes: violation.nodes.slice(0, 5).map((node) => `${node.target.join(" ")} :: ${node.failureSummary?.split("\n").slice(1, 3).join(" ")}`),
    }))
}

for (const theme of ["light", "dark"] satisfies Theme[]) {
  test.describe(`axe (${theme})`, () => {
    test.beforeEach(async ({ page }) => {
      await presetStorage(page, { theme })
    })

    for (const route of PAGES) {
      test(`#/${route} has no serious or critical violations`, async ({ page }) => {
        await openRoute(page, route)
        await expect(page.locator("html")).toHaveClass(theme === "dark" ? /\bdark\b/ : /^(?!.*\bdark\b)/)
        // Let lazy panels, fonts and query results settle before scanning.
        await page.waitForLoadState("networkidle")
        expect(await scan(page)).toEqual([])
      })
    }

    test("generate states: networks drawer, result, image viewer", async ({ page }) => {
      await openRoute(page, "txt2img")
      await page.locator("body").press("Alt+N")
      await expect(page.getByRole("dialog", { name: "Networks" }).getByRole("option").first()).toBeVisible()
      expect(await scan(page)).toEqual([])
      await page.getByRole("dialog", { name: "Networks" }).getByRole("button", { name: "Close" }).click()
      await page.getByRole("button", { name: /^Generate/ }).first().click()
      await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 20_000 })
      expect(await scan(page)).toEqual([])
      await page.getByRole("button", { name: "Open image viewer" }).first().click()
      await expect(page.getByRole("dialog", { name: "Image viewer" })).toBeVisible()
      expect(await scan(page)).toEqual([])
    })

    test("networks drawer while its list is still loading", async ({ page }) => {
      // the card list answers late: the scan sees the loading skeleton, not the cards
      await injectFaults(page, [{ path: "^/networks/[^/]+$", delay: 12_000 }])
      await openRoute(page, "txt2img")
      await page.locator("body").press("Alt+N")
      const drawer = page.getByRole("dialog", { name: "Networks" })
      const skeleton = drawer.locator("[aria-busy=true]").first()
      await expect(skeleton).toBeVisible()
      expect(await scan(page)).toEqual([])
      await expect(skeleton, "the scan saw the loading state").toBeVisible()
      await expect(drawer.getByRole("option").first()).toBeVisible({ timeout: 20_000 })
    })

    test("prompt highlighting keeps every token readable", async ({ page }) => {
      await openRoute(page, "txt2img")
      const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
      await prompt.click()
      await page.keyboard.press(`${process.platform === "darwin" ? "Meta" : "Control"}+A`)
      await page.keyboard.type("((a cat:1.2)), [day:night:0.5], <lora:watercolor_pastel_v1_0:0.8>, __colors__, BREAK, \\(paren\\), [[b]]")
      await page.locator("body").click({ position: { x: 5, y: 5 } }) // no selection highlight
      expect(await scan(page)).toEqual([])
    })

    test("library states: gallery details, PNG Info result, queue with jobs", async ({ page }) => {
      await openRoute(page, "gallery")
      await page.getByRole("list", { name: "Images" }).getByRole("button").first().click()
      const details = page.getByRole("complementary", { name: "Image details" })
      await expect(details).toBeVisible()
      expect(await scan(page)).toEqual([])
      await details.getByRole("button", { name: "PNG Info" }).click()
      await expect(page.getByRole("tab", { name: /All metadata/ })).toBeVisible()
      expect(await scan(page)).toEqual([])
      await page.evaluate(() => (location.hash = "#/queue"))
      await page.getByRole("button", { name: "Pause" }).click()
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await page.getByRole("button", { name: "Add to queue" }).click()
      await page.evaluate(() => (location.hash = "#/queue"))
      await expect(page.getByRole("list", { name: "Queued jobs" }).getByRole("listitem")).toHaveCount(1)
      expect(await scan(page)).toEqual([])
    })

    test("img2img editor in inpaint mode", async ({ page }) => {
      await openRoute(page, "img2img")
      const chooser = page.waitForEvent("filechooser")
      await page.getByRole("button", { name: "Add a source image" }).click()
      await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: SOURCE_PNG })
      await expect(page.getByRole("img", { name: /^Image editor, / })).toBeVisible()
      await page.getByRole("radiogroup", { name: "Mode" }).getByRole("radio", { name: "Inpaint", exact: true }).click()
      await expect(page.getByRole("toolbar", { name: "Tools" })).toBeVisible()
      expect(await scan(page)).toEqual([])
    })

    test("settings with unsaved changes", async ({ page }) => {
      await openRoute(page, "settings")
      await page.getByRole("switch", { name: "Automatically save every generated image", exact: true }).click()
      await expect(page.getByRole("region", { name: "Unsaved changes" })).toBeVisible()
      expect(await scan(page)).toEqual([])
    })

    test("open overlays have no serious or critical violations", async ({ page }) => {
      await openRoute(page, "txt2img")
      await page.getByRole("button", { name: "Appearance" }).click()
      await expect(page.getByRole("menu")).toBeVisible()
      expect(await scan(page)).toEqual([])
      await page.keyboard.press("Escape")
      await page.keyboard.press(process.platform === "darwin" ? "Meta+K" : "Control+K")
      await expect(page.getByRole("dialog")).toBeVisible()
      expect(await scan(page)).toEqual([])
    })

    test("header and form popups are named dialogs without violations", async ({ page }) => {
      // Quick settings appear once a setting is pinned.
      await openRoute(page, "settings")
      await page.getByRole("button", { name: "Pin Automatically save every generated image to quick settings" }).click()
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
      const popups = [
        { trigger: page.getByRole("button", { name: "Quick settings" }), dialog: "Quick settings" },
        { trigger: page.getByRole("button", { name: "VAE / Text encoders" }), dialog: "VAE / Text encoders" },
        { trigger: page.getByRole("combobox", { name: "Checkpoint" }), dialog: "Checkpoint" },
        { trigger: page.getByRole("main").getByRole("combobox", { name: "Sampling method" }), dialog: "Sampling method" },
      ]
      for (const popup of popups) {
        await popup.trigger.click()
        await expect(page.getByRole("dialog", { name: popup.dialog })).toBeVisible()
        expect(await scan(page), popup.dialog).toEqual([])
        await page.keyboard.press("Escape")
        await expect(page.getByRole("dialog", { name: popup.dialog })).toBeHidden()
      }
    })
  })
}

/** The user-selectable accents (src/lib/theme.ts ACCENTS). */
const ACCENTS = ["iris", "blue", "teal", "green", "amber", "coral", "rose"] as const

test.describe("axe with every accent", () => {
  for (const theme of ["light", "dark"] satisfies Theme[]) {
    for (const accent of ACCENTS) {
      test(`${theme} ${accent}: primary buttons (also hovered), links and tinted labels keep their contrast`, async ({ page }) => {
        await presetStorage(page, { theme, accent })
        await openRoute(page, "txt2img")
        await expect(page.locator("html")).toHaveAttribute("data-accent", accent)
        await page.waitForLoadState("networkidle")
        expect(await scan(page)).toEqual([])
        // the hover shade of a primary button keeps its label readable
        const generate = page.getByRole("button", { name: /^Generate/ }).first()
        await generate.hover()
        expect(await scan(page), "hovered Generate").toEqual([])
        // a pressed toggle under the pointer keeps its tinted label readable
        const filter = page.getByRole("searchbox", { name: "Filter extensions" })
        if (!(await filter.isVisible())) await page.getByRole("button", { name: /^Extensions/ }).first().click()
        const modified = page.getByRole("button", { name: "Only modified" })
        await modified.click()
        await expect(modified).toHaveAttribute("aria-pressed", "true")
        await modified.hover()
        await page.waitForTimeout(300) // the colours transition
        expect(await scan(page), "hovered pressed toggle").toEqual([])
        await modified.click()
        // settings: primary-coloured links in help texts
        await page.evaluate(() => (location.hash = "#/settings"))
        await page.getByRole("button", { name: "Saving Videos" }).first().click()
        // exact: the Saving Videos help's "Wiki", not Saving images' "wiki" links next to it
        await expect(page.getByRole("link", { name: "Wiki", exact: true })).toBeVisible()
        await page.mouse.move(0, 0)
        expect(await scan(page), "settings").toEqual([])
      })
    }
  }
})

/** Tabs through a page and lists the tab stops that show no focus indicator (outline or ring). */
async function stopsWithoutFocusRing(page: Page, presses: number): Promise<string[]> {
  const missing = new Set<string>()
  const check = () =>
    page.evaluate(() => {
      const element = document.activeElement as HTMLElement | null
      if (!element || element === document.body) return null
      // an inset ring is a box shadow, painted under the element's content: an image that fills the
      // element hides it (the gallery's tiles showed no ring)
      const covered = (node: Element) => {
        const box = node.getBoundingClientRect()
        return [...node.querySelectorAll("img, canvas, video")].some((media) => {
          const r = media.getBoundingClientRect()
          return r.left <= box.left + 4 && r.top <= box.top + 4 && r.right >= box.right - 4 && r.bottom >= box.bottom - 4
        })
      }
      const shows = (node: Element) => {
        const style = getComputedStyle(node)
        if (style.outlineStyle !== "none" && parseFloat(style.outlineWidth) > 0) return true
        // or on the layer over the content (`after:`), where an image cannot cover it
        const after = getComputedStyle(node, "::after")
        if (after.content !== "none" && after.outlineStyle !== "none" && parseFloat(after.outlineWidth) > 0) return true
        if (style.boxShadow === "none" || style.boxShadow === "") return false
        // the visible part of the shadow: drop transparent layers, then inset ones over covered content
        const layers = style.boxShadow.split(/,(?![^(]*\))/).filter((layer) => !/rgba\(0, 0, 0, 0\)/.test(layer))
        return layers.some((layer) => !/inset/.test(layer) || !covered(node))
      }
      // a ring may be drawn by a wrapper that has the focus inside it (inputs in a group, the prompt editor)
      for (let node: Element | null = element, depth = 0; node && depth < 5; node = node.parentElement, depth++) {
        if (shows(node) && (node === element || node.matches(":focus-within"))) return null
      }
      const name = element.getAttribute("aria-label") || element.textContent?.trim().slice(0, 40) || element.tagName
      return `${element.tagName.toLowerCase()}[${element.getAttribute("role") ?? ""}] ${name}`
    })
  for (let i = 0; i < presses; i++) {
    await page.keyboard.press("Tab")
    let problem = await check()
    if (problem) {
      // some widgets mark focus a frame later (CodeMirror's .cm-focused): look again
      await page.waitForTimeout(150)
      problem = await check()
    }
    if (problem) missing.add(problem)
  }
  return [...missing]
}

test.describe("visible focus", () => {
  for (const route of ["txt2img", "extensions", "settings", "gallery", "queue", "png-info", "networks"]) {
    test(`every tab stop on #/${route} shows where the focus is`, async ({ page }) => {
      await openRoute(page, route)
      await page.waitForLoadState("networkidle")
      // rings fade in (box-shadow transitions): read their end state
      await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
      expect(await stopsWithoutFocusRing(page, 70)).toEqual([])
    })
  }
})

// The focus ring is drawn inside the tile, over the image. The image zooms in under the pointer,
// which lifted it onto a layer of its own over the ring: a tile focused from the keyboard showed
// no ring while the pointer rested on it.
test("a gallery tile shows its focus ring under the pointer", async ({ page }) => {
  await openRoute(page, "gallery")
  await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
  const tile = page.locator('[aria-describedby="gallery-grid-keys"]').first()
  await expect.poll(() => tile.locator("img").evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true)
  await tile.focus()
  expect(await tile.evaluate((element) => element.matches(":focus-visible"))).toBe(true)
  await tile.hover()
  // the left edge, away from the corner controls: the ring is there only while the tile has the focus
  const edge = async () => {
    const box = (await tile.boundingBox())!
    return page.screenshot({ clip: { x: box.x, y: box.y + box.height / 3, width: 4, height: box.height / 3 } })
  }
  const focused = await edge()
  await tile.evaluate((element: HTMLElement) => element.blur())
  expect(focused.equals(await edge())).toBe(false)
})

// A bar pinned over a scroller (the run bar, the apply bars, the send bar) covered the control Tab
// moved to: the scroller brought it to the edge of its box, under the bar (WCAG 2.4.11).
/** The focused tile when it or its focus ring (3 px plus a 2 px offset) reaches under the gallery's selection bar. */
function focusUnderSelectionBar(): string | null {
  const element = document.activeElement
  const toolbar = document.querySelector('[role="toolbar"][aria-label="Selected images"]')
  if (!element || !toolbar) return null
  const r = element.getBoundingClientRect()
  const b = toolbar.getBoundingClientRect()
  const ring = 5
  const overlaps = r.left - ring < b.right && r.right + ring > b.left && r.top - ring < b.bottom && r.bottom + ring > b.top
  return overlaps ? `${element.getAttribute("aria-label") ?? element.tagName} (${Math.round(r.bottom + ring - b.top)} px under)` : null
}

test.describe("focus is not hidden under a pinned bar", () => {
  const cases: { route: string; width: number; height: number; pending?: boolean }[] = [
    { route: "txt2img", width: 390, height: 844 },
    { route: "txt2img", width: 1280, height: 720 },
    { route: "extras", width: 390, height: 844 },
    { route: "settings", width: 1280, height: 720, pending: true },
    { route: "settings", width: 852, height: 393, pending: true },
    { route: "extensions", width: 852, height: 393, pending: true },
    { route: "txt2img", width: 852, height: 393 },
    { route: "ext/json-builder", width: 390, height: 844 },
    { route: "ext/json-builder", width: 852, height: 393 },
    { route: "ext/json-builder", width: 1280, height: 720 },
  ]
  for (const { route, width, height, pending } of cases) {
    test(`#/${route} at ${width}×${height}`, async ({ page }) => {
      await page.setViewportSize({ width, height })
      await openRoute(page, route)
      await page.waitForLoadState("networkidle")
      await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
      // a pending change shows the bar (settings, extensions)
      if (pending) await page.getByRole("main").getByRole("switch").first().click()
      await page.locator("body").click({ position: { x: 1, y: 1 } })
      const hidden = new Set<string>()
      // forwards, then back up (a bar at the top hides what Shift+Tab scrolls up to)
      for (let i = 0; i < 120; i++) {
        await page.keyboard.press(i < 60 ? "Tab" : "Shift+Tab")
        const problem = await page.evaluate(() => {
          const element = document.activeElement
          if (!element || element === document.body) return null
          const r = element.getBoundingClientRect()
          if (r.width < 4 || r.height < 4) return null
          const points = [
            [r.left + r.width / 2, r.top + r.height / 2],
            [r.left + 2, r.top + 2],
            [r.right - 2, r.bottom - 2],
          ]
          const underBar = points.filter(([x, y]) => {
            const hit = document.elementFromPoint(x!, y!)
            const bar = hit?.closest(".sticky, [data-toast-obstacle]")
            return bar !== null && bar !== undefined && !bar.contains(element)
          })
          if (underBar.length < 2) return null
          return element.getAttribute("aria-label") || element.textContent?.trim().slice(0, 40) || element.tagName
        })
        if (problem) hidden.add(problem)
      }
      expect([...hidden]).toEqual([])
    })
  }

  // The gallery's selection bar floats over the grid, which scrolls a tile into view itself (virtualized).
  for (const [width, height] of [
    [390, 844],
    [1280, 720],
  ] as const) {
    test(`#/gallery selection bar at ${width}×${height}`, async ({ page }) => {
      await page.setViewportSize({ width, height })
      await openRoute(page, "gallery")
      await page.waitForLoadState("networkidle")
      await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
      const tile = page.getByTestId("gallery-scroller").locator("[data-gallery-tile] button").first()
      await tile.focus()
      await page.keyboard.press("Space")
      const bar = page.getByRole("toolbar", { name: "Selected images" })
      await expect(bar).toBeVisible()
      // start in the bar's column (ArrowDown keeps to it): the tile of the first row nearest its centre
      await page.evaluate(() => {
        const box = document.querySelector('[role="toolbar"][aria-label="Selected images"]')!.getBoundingClientRect()
        const centre = (box.left + box.right) / 2
        const tiles = [...document.querySelectorAll<HTMLElement>("[data-gallery-scroller] [data-gallery-tile] button")].map((element) => ({ element, r: element.getBoundingClientRect() }))
        const top = Math.min(...tiles.map(({ r }) => r.top))
        const row = tiles.filter(({ r }) => Math.abs(r.top - top) < 2)
        const off = ({ r }: { r: DOMRect }) => Math.abs((r.left + r.right) / 2 - centre)
        row.sort((a, b) => off(a) - off(b))
        row[0]!.element.focus()
      })
      const hidden: string[] = []
      for (let i = 0; i < 40; i++) {
        await page.keyboard.press("ArrowDown")
        const problem = await page.evaluate(focusUnderSelectionBar)
        if (problem) hidden.push(problem)
      }
      expect(hidden).toEqual([])
    })
  }

  // The bar comes up with the first selection, over a tile the arrow keys had left at the bottom edge.
  test("#/gallery selection bar rising over the focused tile", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await openRoute(page, "gallery")
    await page.waitForLoadState("networkidle")
    await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
    await page.getByTestId("gallery-scroller").locator("[data-gallery-tile] button").first().focus()
    for (let i = 0; i < 6; i++) await page.keyboard.press("ArrowDown")
    await page.keyboard.press("Space")
    await expect(page.getByRole("toolbar", { name: "Selected images" })).toBeVisible()
    await expect.poll(() => page.evaluate(focusUnderSelectionBar)).toBeNull()
  })

  // A new query remounts the grid; with the results cached the bar stays up over the new scroller.
  test("#/gallery selection bar over a grid rebuilt from cached results", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    await openRoute(page, "gallery")
    await page.waitForLoadState("networkidle")
    // all of them: the selection outlives the new order
    await page.getByTestId("gallery-scroller").locator("[data-gallery-tile] button").first().focus()
    await page.keyboard.press("ControlOrMeta+a")
    const bar = page.getByRole("toolbar", { name: "Selected images" })
    await expect(bar).toBeVisible()
    const sort = page.getByRole("combobox", { name: "Sort" })
    for (const option of ["Oldest first", "Newest first"]) {
      await sort.click()
      await page.getByRole("option", { name: option }).click()
      await page.waitForLoadState("networkidle")
    }
    await expect(bar).toBeVisible()
    const room = () => page.getByTestId("gallery-scroller").evaluate((element) => parseFloat(getComputedStyle(element).scrollPaddingBottom) || 0)
    await expect.poll(room).toBeGreaterThan(0)
  })
})

test.describe("axe on a phone", () => {
  for (const route of ["txt2img", "img2img", "gallery", "queue", "settings", "networks"]) {
    test(`#/${route} has no serious or critical violations @phone`, async ({ page }) => {
      await presetStorage(page, { theme: "dark" })
      await openRoute(page, route)
      await page.waitForLoadState("networkidle")
      expect(await scan(page)).toEqual([])
    })
  }
})

// A dialog whose text field took the focus as it mounted (`autoFocus`) recorded that field as the
// place to return to: closing it left the focus on the page itself, and the next Tab started over
// from the top of the page.
test.describe("focus after a dialog", () => {
  const cases: { name: string; route: string; before?: (page: Page) => Promise<void>; opener: string }[] = [
    { name: "JSON Builder's Load prompt", route: "ext/json-builder", opener: "Load prompt…" },
    { name: "Capture a recipe from the form", route: "queue", before: (page) => page.getByRole("tab", { name: "Recipes" }).click(), opener: "Capture from form" },
  ]
  for (const item of cases) {
    test(`${item.name} gives the focus back to its button`, async ({ page }) => {
      await openRoute(page, item.route)
      await item.before?.(page)
      const opener = page.getByRole("button", { name: item.opener })
      await opener.focus()
      await page.keyboard.press("Enter")
      const dialog = page.getByRole("dialog")
      await expect(dialog).toBeVisible()
      await expect(dialog.locator(":focus")).toHaveCount(1)
      await page.keyboard.press("Escape")
      await expect(dialog).toBeHidden()
      await expect(opener).toBeFocused()
    })
  }

  // the popovers whose name field takes the focus as they open
  const popovers: { route: string; before?: (page: Page) => Promise<void>; opener: string; field: string }[] = [
    { route: "gallery", opener: "New board", field: "Board name" },
    { route: "queue", before: (page) => page.getByRole("tab", { name: "Pipelines" }).click(), opener: "Save template", field: "Pipeline name" },
  ]
  for (const item of popovers) {
    test(`${item.opener} focuses its name field and gives the focus back`, async ({ page }) => {
      await openRoute(page, item.route)
      await item.before?.(page)
      const opener = page.getByRole("button", { name: item.opener, exact: true })
      await opener.click()
      await expect(page.getByRole("dialog", { name: item.opener }).getByRole("textbox", { name: item.field })).toBeFocused()
      await page.keyboard.press("Escape")
      await expect(opener).toBeFocused()
    })
  }
})
