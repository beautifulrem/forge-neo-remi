/**
 * axe-core scans of every page in light and dark: no serious or critical
 * violations (WCAG 2.x A/AA rules plus axe best practices).
 */
import AxeBuilder from "@axe-core/playwright"
import { expect, type Locator, type Page, test } from "@playwright/test"
import { injectFaults, mod, openGenerate, openRoute, presetStorage, setPrompt, type Theme } from "./helpers"

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

/**
 * Tabs through a page and lists the tab stops whose focus indicator is not the focus ring: none at
 * all, the browser's own outline (`outline-style: auto`, its width and colour the browser's), one
 * 1 or 2 px wide, one in another colour than --focus-ring, or an outline and a ring drawn together.
 * The focus ring is a 3 px box-shadow ring (`ring-3`) or a 3 px solid outline, in --focus-ring.
 */
async function focusIndicatorProblems(page: Page, presses: number): Promise<string[]> {
  const problems = new Set<string>()
  const check = () =>
    page.evaluate(() => {
      const element = document.activeElement as HTMLElement | null
      if (!element || element === document.body) return null
      const canvas = document.createElement("canvas").getContext("2d", { willReadFrequently: true })!
      /** The colour as 8-bit sRGB channels and alpha, whatever notation the computed style uses. */
      const rgba = (color: string) => {
        canvas.clearRect(0, 0, 1, 1)
        canvas.fillStyle = "#000"
        canvas.fillStyle = color
        canvas.fillRect(0, 0, 1, 1)
        return [...canvas.getImageData(0, 0, 1, 1).data] as [number, number, number, number]
      }
      /** --focus-ring as the element's own context resolves it (an accent scope re-derives it). */
      const focusRing = (node: Element) => {
        const probe = document.createElement("i")
        probe.style.color = "var(--focus-ring)"
        ;(node.parentElement ?? document.body).append(probe)
        const color = rgba(getComputedStyle(probe).color)
        probe.remove()
        return color
      }
      const same = (a: number[], b: number[]) => a.every((x, i) => Math.abs(x - b[i]!) <= 3)
      // an inset ring is a box shadow, painted under the element's content: an image that fills the
      // element hides it (the gallery's tiles showed no ring)
      const covered = (node: Element) => {
        const box = node.getBoundingClientRect()
        return [...node.querySelectorAll("img, canvas, video")].some((media) => {
          const r = media.getBoundingClientRect()
          return r.left <= box.left + 4 && r.top <= box.top + 4 && r.right >= box.right - 4 && r.bottom >= box.bottom - 4
        })
      }
      type Mark = { kind: "outline" | "ring"; width: number; color: number[]; auto: boolean }
      /** What a node draws around itself: its outline, its `after:` layer's outline, its rings. */
      const marks = (node: Element): Mark[] => {
        const found: Mark[] = []
        const after = getComputedStyle(node, "::after")
        for (const style of after.content === "none" ? [getComputedStyle(node)] : [getComputedStyle(node), after]) {
          const width = parseFloat(style.outlineWidth)
          const color = rgba(style.outlineColor)
          if (style.outlineStyle !== "none" && width > 0 && color[3] > 0) found.push({ kind: "outline", width, color, auto: style.outlineStyle === "auto" })
        }
        const shadow = getComputedStyle(node).boxShadow
        if (shadow && shadow !== "none") {
          for (const layer of shadow.split(/,(?![^(]*\))/)) {
            const color = rgba(/^\s*([a-z-]+\([^)]*\)|#\w+|[a-z]+)/i.exec(layer)?.[1] ?? "transparent")
            const [x = 0, y = 0, blur = 0, spread = 0] = [...layer.matchAll(/(-?[\d.]+)px/g)].map((m) => Number(m[1]))
            // a ring: no offset, no blur, a spread; an inset one only where no image covers it
            if (color[3] === 0 || x !== 0 || y !== 0 || blur !== 0 || spread <= 0) continue
            if (/\binset\b/.test(layer) && covered(node)) continue
            found.push({ kind: "ring", width: spread, color, auto: false })
          }
        }
        return found
      }
      const name = `${element.tagName.toLowerCase()}[${element.getAttribute("role") ?? ""}] ${element.getAttribute("aria-label") || element.textContent?.trim().slice(0, 40) || ""}`
      // a control drawn by its wrapper is often an input nobody sees (a slider's range input, a file
      // input behind its label, a colour input at opacity 0): it draws nothing, its wrapper does
      const style = getComputedStyle(element)
      const unseen = style.opacity === "0" || /^rect\(0px,? 0px,? 0px,? 0px\)$/.test(style.clip) || style.clipPath === "inset(50%)"
      const own = unseen ? [] : marks(element)
      if (own.some((mark) => mark.auto)) return `${name}: the browser's outline (outline-style: auto)`
      // a ring may be drawn by a wrapper that has the focus inside it (inputs in a group, the prompt editor)
      const seen: Mark[] = []
      for (let node: Element | null = element, depth = 0; node && depth < 5; node = node.parentElement, depth++) {
        const drawn = node === element ? own : marks(node)
        seen.push(...drawn)
        const ring = focusRing(node)
        const good = drawn.filter((mark) => mark.width >= 3 && same(mark.color, ring))
        if (good.length === 0) continue
        // one indicator: a 3 px outline and a ring together draw it twice (shadcn's 1 px outline
        // inside the tabs' ring is an edge, not a second ring)
        if (drawn.some((mark) => mark.kind === "outline" && mark.width >= 2) && drawn.some((mark) => mark.kind === "ring" && mark.width >= 3)) return `${name}: an outline and a ring`
        return null
      }
      if (seen.length === 0) return `${name}: no focus indicator`
      const widest = Math.max(...seen.map((mark) => mark.width))
      return widest < 3 ? `${name}: a ${widest} px indicator` : `${name}: not in --focus-ring (rgba ${seen.find((mark) => mark.width >= 3)!.color.join(",")})`
    })
  for (let i = 0; i < presses; i++) {
    await page.keyboard.press("Tab")
    let problem = await check()
    if (problem) {
      // some widgets mark focus a frame later (CodeMirror's .cm-focused): look again
      await page.waitForTimeout(150)
      problem = await check()
    }
    if (problem) problems.add(problem)
  }
  return [...problems]
}

// Tab stops with no focus class of their own fell back to the browser's outline (`outline: auto`):
// its width and colour are the browser's, not the 3 px --focus-ring every other stop draws.
test.describe("visible focus", () => {
  for (const route of ["txt2img", "img2img", "extras", "png-info", "gallery", "networks", "queue", "settings", "extensions", "ext/toolbox", "ext/json-builder"]) {
    test(`every tab stop on #/${route} shows the 3 px focus ring`, async ({ page }) => {
      await openRoute(page, route)
      await page.waitForLoadState("networkidle")
      // rings fade in (box-shadow transitions): read their end state
      await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
      expect(await focusIndicatorProblems(page, 70)).toEqual([])
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

/**
 * How a field draws its state: its outer ring (the focus ring or an invalid field's halo) and its edge (its border, or
 * the inset 1 px ring of a field without one: the prompt editor), each named after the token it is drawn in
 * (`focus-ring`, `destructive`, `ring`) as the field resolves it.
 */
async function fieldMarks(field: Locator): Promise<{ ring: string; edge: string }> {
  return field.evaluate((element) => {
    const canvas = document.createElement("canvas").getContext("2d", { willReadFrequently: true })!
    /** The colour as 8-bit sRGB channels and alpha, whatever notation the computed style uses. */
    const rgba = (color: string) => {
      canvas.clearRect(0, 0, 1, 1)
      canvas.fillStyle = "#000"
      canvas.fillStyle = color
      canvas.fillRect(0, 0, 1, 1)
      return [...canvas.getImageData(0, 0, 1, 1).data]
    }
    const token = (name: string) => {
      const probe = document.createElement("i")
      probe.style.color = `var(${name})`
      ;(element.parentElement ?? document.body).append(probe)
      const color = rgba(getComputedStyle(probe).color)
      probe.remove()
      return color
    }
    const near = (a: number[], b: number[], channels: number, by: number) => a.slice(0, channels).every((x, i) => Math.abs(x - b[i]!) <= by)
    // the focus ring is --ring at 80 % (60 % dark): its alpha tells it from an edge in --ring; the halo and an edge
    // in --destructive are a share of it (20 % to 50 %): their channels tell them
    const tokens = { "focus-ring": token("--focus-ring"), ring: token("--ring"), destructive: token("--destructive") }
    const name = (color: number[]) =>
      near(color, tokens["focus-ring"], 4, 4) ? "focus-ring" : near(color, tokens.destructive, 3, 6) ? "destructive" : near(color, tokens.ring, 4, 4) ? "ring" : `rgba(${color.join(", ")})`
    const style = getComputedStyle(element)
    const layers = (style.boxShadow === "none" ? [] : style.boxShadow.split(/,(?![^(]*\))/))
      .map((layer) => {
        const [x = 0, y = 0, blur = 0, spread = 0] = [...layer.matchAll(/(-?[\d.]+)px/g)].map((m) => Number(m[1]))
        const color = rgba(/^\s*([a-z-]+\([^)]*\)|#\w+|[a-z]+)/i.exec(layer)?.[1] ?? "transparent")
        return { inset: /\binset\b/.test(layer), color, spread, drawn: x === 0 && y === 0 && blur === 0 && spread > 0 && color[3]! > 0 }
      })
      .filter((layer) => layer.drawn)
    const ring = layers.filter((layer) => !layer.inset).sort((a, b) => b.spread - a.spread)[0]
    const inset = layers.find((layer) => layer.inset)
    const border = parseFloat(style.borderTopWidth) > 0 && style.borderTopStyle !== "none" ? rgba(style.borderTopColor) : undefined
    const edge = border ?? inset?.color
    return { ring: ring ? `${ring.spread} px ${name(ring.color)}` : "none", edge: edge && edge[3]! > 0 ? name(edge) : "none" }
  })
}

// An invalid field drew its halo (3 px of 20 % --destructive) where the focus ring goes: `aria-invalid:ring-destructive/20`
// comes after `focus-visible:ring-focus-ring` in Tailwind's CSS, so a focused invalid field looked like an unfocused one.
// Focus wins: focused, it draws the focus ring, and its red edge tells it from a valid field with the focus. The prompt
// editor showed nothing invalid at all once focused (its own focus ring wins over the classes), and no edge.
test.describe("invalid fields", () => {
  for (const theme of ["light", "dark"] satisfies Theme[]) {
    test(`${theme}: draw the focus ring once focused, and keep their red edge`, async ({ page }) => {
      await presetStorage(page, { theme })
      await openGenerate(page)
      await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
      // the size field of the aspect ratio popover is invalid while its text is not a size
      await page.getByRole("button", { name: /^Aspect ratio/ }).first().click()
      const size = page.getByRole("textbox", { name: "Size as width x height" })
      await size.focus()
      await page.keyboard.type("12x")
      await expect(size).toHaveAttribute("aria-invalid", "true")
      const sizeFocused = await fieldMarks(size)
      await page.keyboard.press(`${mod}+A`)
      await page.keyboard.type("1216x832")
      await expect(size).not.toHaveAttribute("aria-invalid")
      const sizeValid = await fieldMarks(size)
      await page.keyboard.press(`${mod}+A`)
      await page.keyboard.type("12x")
      await size.evaluate((element: HTMLElement) => element.blur())
      await expect(size).toHaveAttribute("aria-invalid", "true")
      const sizeBlurred = await fieldMarks(size)
      await page.keyboard.press("Escape")
      // the prompt editor, refused by the server (the mock refuses "#invalid")
      const prompt = await setPrompt(page, "a cat #invalid")
      await page.getByRole("button", { name: /^Generate/ }).first().click()
      await expect(prompt).toHaveAttribute("aria-invalid", "true")
      const editor = page.locator(".cm-editor").filter({ has: prompt })
      const promptBlurred = await fieldMarks(editor)
      await prompt.focus()
      await expect(editor).toHaveClass(/\bcm-focused\b/)
      const promptFocused = await fieldMarks(editor)
      await page.keyboard.type(" ") // editing a field clears its error
      await expect(prompt).not.toHaveAttribute("aria-invalid")
      const promptValid = await fieldMarks(editor)
      expect({ sizeFocused, sizeValid, sizeBlurred, promptBlurred, promptFocused, promptValid }).toEqual({
        sizeFocused: { ring: "3 px focus-ring", edge: "destructive" },
        sizeValid: { ring: "3 px focus-ring", edge: "ring" },
        sizeBlurred: { ring: "3 px destructive", edge: "destructive" },
        promptBlurred: { ring: "3 px destructive", edge: "destructive" },
        promptFocused: { ring: "3 px focus-ring", edge: "destructive" },
        promptValid: { ring: "3 px focus-ring", edge: "ring" },
      })
    })
  }
})

/**
 * How a segmented control or tab strip marks its selected segment: the contrast of the segment's
 * tone against its track, and of the edge drawn inside it (an inset box shadow) against the track
 * and against that tone. A WCAG 1.4.11 state indicator needs 3:1 against what is next to it.
 */
async function selectedSegment(group: Locator) {
  await group.page().waitForFunction(() => document.getAnimations().every((animation) => animation.playState !== "running" || animation.effect?.getTiming().iterations === Infinity))
  return group.evaluate((element) => {
    const canvas = document.createElement("canvas").getContext("2d", { willReadFrequently: true })!
    /** The colour as sRGB channels (0..1) and alpha, whatever notation the computed style uses. */
    const rgba = (color: string) => {
      canvas.clearRect(0, 0, 1, 1)
      canvas.fillStyle = "#000"
      canvas.fillStyle = color
      canvas.fillRect(0, 0, 1, 1)
      const [r, g, b, a] = canvas.getImageData(0, 0, 1, 1).data
      return { rgb: [r!, g!, b!].map((x) => x / 255), alpha: a! / 255 }
    }
    const over = (color: string, under: number[]) => {
      const { rgb, alpha } = rgba(color)
      return rgb.map((x, i) => alpha * x + (1 - alpha) * under[i]!)
    }
    const luminance = (rgb: number[]) => {
      const [r, g, b] = rgb.map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4)) as [number, number, number]
      return 0.2126 * r + 0.7152 * g + 0.0722 * b
    }
    const contrast = (a: number[], b: number[]) => {
      const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number]
      return Math.round(((hi + 0.05) / (lo + 0.05)) * 100) / 100
    }
    // the track: the group's own fill, or the first filled box around it
    let trackColor = "rgba(0, 0, 0, 0)"
    for (let node: Element | null = element; node && rgba(trackColor).alpha === 0; node = node.parentElement) trackColor = getComputedStyle(node).backgroundColor
    const track = rgba(trackColor).rgb
    const selected = element.querySelector<HTMLElement>('[aria-selected="true"], [aria-checked="true"]')
    if (!selected) return { name: element.getAttribute("aria-label"), selected: null }
    const style = getComputedStyle(selected)
    const tone = over(style.backgroundColor, track)
    // the inset layers of its box shadow that draw something
    const edges = style.boxShadow
      .split(/,(?![^(]*\))/)
      .filter((layer) => /\binset\b/.test(layer))
      .map((layer) => ({ color: /^\s*([a-z-]+\([^)]*\)|#\w+|[a-z]+)/i.exec(layer)?.[1] ?? "transparent", lengths: [...layer.matchAll(/(-?[\d.]+)px/g)].map((m) => Number(m[1])) }))
      .filter(({ color, lengths }) => rgba(color).alpha > 0 && (lengths[3] ?? 0) > 0)
    const edge = edges.map(({ color }) => over(color, tone))
    const edgeContrast = edge.length ? Math.max(...edge.map((painted) => Math.min(contrast(painted, track), contrast(painted, tone)))) : 0
    return { name: element.getAttribute("aria-label"), selected: selected.textContent?.trim(), tone: contrast(tone, track), edge: edgeContrast, indicator: Math.max(contrast(tone, track), edgeContrast) }
  })
}

// The selected segment was a tone a step off its track: 1.18:1 in the light theme and 1.10:1 in the
// dark one, too close to tell which segment is on (the img2img stage's Source / Result switch, every
// Segmented, the kit's tab strips, the img2img mode tiles).
test.describe("selected segments", () => {
  for (const theme of ["light", "dark"] satisfies Theme[]) {
    test(`${theme}: stand out from their track by 3:1`, async ({ page }) => {
      await presetStorage(page, { theme })
      await openRoute(page, "img2img")
      const groups = [
        page.getByRole("tablist", { name: "Image view" }),
        page.getByRole("tablist", { name: "Resize" }),
        page.getByRole("radiogroup").filter({ has: page.getByRole("radio", { name: "Balanced" }) }).first(),
        page.getByRole("radiogroup").filter({ has: page.getByRole("radio", { name: "img2img" }) }).first(),
      ]
      const marks = []
      for (const group of groups) {
        await expect(group).toBeVisible()
        marks.push(await selectedSegment(group))
      }
      // a track inside a dialog (the page tone in the dark theme)
      await page.evaluate(() => (location.hash = "#/gallery"))
      const tiles = page.getByRole("list", { name: "Images" }).getByRole("listitem")
      const bulk = page.getByRole("toolbar", { name: "Selected images" })
      for (const index of [0, 1]) {
        await tiles.nth(index).hover()
        await tiles.nth(index).getByRole("checkbox").click()
        await expect(bulk).toContainText(`${index + 1} selected`)
      }
      await bulk.getByRole("button", { name: "Compare" }).click()
      const view = page.getByRole("dialog", { name: "Compare images" }).getByRole("radiogroup", { name: "Comparison view" })
      await expect(view).toBeVisible()
      marks.push(await selectedSegment(view))
      expect(marks.every((mark) => mark.selected)).toBe(true)
      expect(marks.filter((mark) => !("indicator" in mark) || (mark.indicator ?? 0) < 3)).toEqual([])
    })
  }

  // The folder tree's selected row was the same kind of mark: --accent behind it, a step off the
  // page or the popover around the tree, the only thing telling it from the other rows.
  for (const theme of ["light", "dark"] satisfies Theme[]) {
    test(`${theme}: the folder tree's selected row stands out by 3:1`, async ({ page }) => {
      await presetStorage(page, { theme })
      // the networks page shows the tree in a pane of its own
      await openRoute(page, "networks")
      const pane = page.getByRole("complementary", { name: "Folders" }).getByRole("tree", { name: "Folders" })
      await expect(pane.getByRole("treeitem", { selected: true })).toBeVisible()
      const marks = [await selectedSegment(pane)]
      // the networks drawer shows it in a popover
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
      await page.locator("body").press("Alt+N")
      const drawer = page.getByRole("dialog", { name: "Networks" })
      await expect(drawer.getByRole("option").first()).toBeVisible()
      await drawer.getByRole("button", { name: "Folders" }).click()
      const popover = page.getByRole("dialog", { name: "Folders" }).getByRole("tree", { name: "Folders" })
      await expect(popover.getByRole("treeitem", { selected: true })).toBeVisible()
      marks.push(await selectedSegment(popover))
      expect(marks.every((mark) => mark.selected)).toBe(true)
      expect(marks.filter((mark) => !("indicator" in mark) || (mark.indicator ?? 0) < 3)).toEqual([])
    })
  }
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

  // Only the focused tile is lifted clear when the bar comes up: a check mark clicked in a scrolled
  // grid (a mouse press focuses it) leaves the grid where it is, not at the roving tab stop.
  test("#/gallery selection bar rising after a check mark is clicked", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    await openRoute(page, "gallery")
    await page.waitForLoadState("networkidle")
    const scroller = page.getByTestId("gallery-scroller")
    await scroller.evaluate((element) => element.scrollTo({ top: 900 }))
    await page.waitForTimeout(200)
    await scroller.evaluate((element) => {
      const view = element.getBoundingClientRect()
      const tile = [...element.querySelectorAll<HTMLElement>("[data-gallery-tile]")].find((candidate) => candidate.getBoundingClientRect().top > view.top + 40)
      tile!.dataset.probe = ""
    })
    const tile = scroller.locator("[data-gallery-tile][data-probe]")
    const before = await scroller.evaluate((element) => element.scrollTop)
    await tile.hover()
    await tile.getByRole("checkbox").click()
    await expect(page.getByRole("toolbar", { name: "Selected images" })).toBeVisible()
    await page.waitForTimeout(300)
    expect(await scroller.evaluate((element) => element.scrollTop)).toBe(before)
    await expect(tile).toBeInViewport()
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
      await expect(bar).toBeVisible() // the case holds only while the selection lives through each view
    }
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

/**
 * Presses Tab `count` times as fast as the browser takes key events (a held Tab with no repeat
 * delay): the presses are sent at once, none waits for the page to answer the one before.
 */
async function burstOfTabs(page: Page, count: number) {
  const cdp = await page.context().newCDPSession(page)
  const key = { key: "Tab", code: "Tab", windowsVirtualKeyCode: 9, nativeVirtualKeyCode: 9 }
  await Promise.all(Array.from({ length: count }, (_, i) => [cdp.send("Input.dispatchKeyEvent", { type: "rawKeyDown", autoRepeat: i > 0, ...key }), cdp.send("Input.dispatchKeyEvent", { type: "keyUp", ...key })]).flat())
  await cdp.detach()
}

/**
 * The elements behind an open modal that could still take the focus: every focusable element
 * outside the modal's portal that is not inert. The toasts stay live (they are still read out).
 */
function reachableBehind(modal: Locator) {
  return modal.evaluate((popup) => {
    const portal = popup.closest("[data-base-ui-portal]") ?? popup
    const focusable = 'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"]), [contenteditable="true"]'
    return [...document.querySelectorAll<HTMLElement>(focusable)]
      .filter((element) => !portal.contains(element) && !element.closest("[inert], [data-react-aria-top-layer]") && element.getClientRects().length > 0)
      .map((element) => `${element.tagName.toLowerCase()} ${element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 30) ?? ""}`)
  })
}

/** How many elements of the page are inert (not counting inside shadow roots: the token counter's animated digits). */
function inertCount(page: Page) {
  return page.evaluate(() => document.querySelectorAll("[inert]").length)
}

/** Records every element outside the open modal's portal that takes the focus from now on. */
async function watchFocusLeaving(modal: Locator) {
  await modal.evaluate((popup) => {
    const portal = popup.closest("[data-base-ui-portal]") ?? popup
    const strays: string[] = []
    ;(window as unknown as { __strays: string[] }).__strays = strays
    document.addEventListener("focusin", (event) => {
      const target = event.target as Element
      if (!portal.contains(target)) strays.push(`${target.tagName.toLowerCase()} ${target.getAttribute("aria-label") ?? target.textContent?.trim().slice(0, 30) ?? ""}`)
    })
  })
  return () => modal.page().evaluate(() => (window as unknown as { __strays: string[] }).__strays)
}

// Holding Tab with no key repeat delay took the focus out of an open dialog: Base UI's focus guards
// send it back inside a frame later, and the next press, sooner than that, moved on from the guard
// to the page behind, where nothing was inert.
test.describe("focus stays in a modal", () => {
  test("a dialog keeps it under a burst of Tab presses", async ({ page }) => {
    await openRoute(page, "txt2img")
    const inertBefore = await inertCount(page)
    await page.locator("body").press("?")
    const dialog = page.getByRole("dialog", { name: "Keyboard shortcuts" })
    await expect(dialog.getByRole("button", { name: "Close" })).toBeFocused()
    // the page behind takes no focus and no clicks while the dialog is open
    expect(await reachableBehind(dialog)).toEqual([])
    const strays = await watchFocusLeaving(dialog)
    for (const backwards of [false, true]) {
      if (backwards) await page.keyboard.down("Shift")
      await burstOfTabs(page, 60)
      if (backwards) await page.keyboard.up("Shift")
      await page.waitForTimeout(100) // the guards' frame
      expect(await strays()).toEqual([])
      expect(await dialog.evaluate((popup) => popup.contains(document.activeElement))).toBe(true)
    }
    await page.keyboard.press("Escape")
    await expect(dialog).toBeHidden()
    await expect.poll(() => inertCount(page)).toBe(inertBefore)
  })

  test("an alert dialog over a dialog holds both, and a popup inside the dialog still works", async ({ page }) => {
    await openRoute(page, "txt2img")
    const inertBefore = await inertCount(page)
    const opener = page.getByRole("button", { name: "Edit styles" })
    await opener.click()
    const dialog = page.getByRole("dialog", { name: "Styles" })
    expect(await reachableBehind(dialog)).toEqual([])
    // the style select's list opens inside the dialog's portal, and takes the pointer and the keys
    await dialog.getByRole("combobox").click()
    const option = page.getByRole("option").first()
    const name = (await option.textContent())!.trim()
    await option.click()
    await expect(dialog.getByRole("combobox")).toContainText(name)
    const remove = dialog.getByRole("button", { name: "Delete" })
    await remove.click()
    const alert = page.getByRole("alertdialog", { name: `Delete style “${name}”?` })
    await expect(alert.getByRole("button", { name: "Cancel" })).toBeFocused()
    // the dialog under it is behind it now
    expect(await reachableBehind(alert)).toEqual([])
    const strays = await watchFocusLeaving(alert)
    await burstOfTabs(page, 60)
    await page.waitForTimeout(100)
    expect(await strays()).toEqual([])
    // closed, it gives the focus back to the button in the dialog, which is live again
    await alert.getByRole("button", { name: "Cancel" }).click()
    await expect(remove).toBeFocused()
    expect(await reachableBehind(dialog)).toEqual([])
    await page.keyboard.press("Escape")
    await expect(dialog).toBeHidden()
    await expect(opener).toBeFocused()
    await expect.poll(() => inertCount(page)).toBe(inertBefore)
  })

  test("the Networks drawer leaves the page live", async ({ page }) => {
    await openGenerate(page)
    const inertBefore = await inertCount(page)
    await page.locator("body").press("Alt+N")
    const drawer = page.getByRole("dialog", { name: "Networks" })
    await expect(drawer).toHaveAttribute("aria-modal", "false")
    await expect(drawer.getByRole("searchbox").first()).toBeFocused()
    await expect.poll(() => inertCount(page)).toBe(inertBefore)
    const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
    await prompt.click()
    await expect(prompt).toBeFocused()
  })

  test("a sheet keeps it under a burst of Tab presses @phone", async ({ page }) => {
    await openRoute(page, "txt2img")
    const inertBefore = await inertCount(page)
    await page.getByRole("button", { name: "More", exact: true }).click()
    const sheet = page.getByRole("dialog", { name: "More" })
    await expect(sheet).toBeVisible()
    expect(await reachableBehind(sheet)).toEqual([])
    const strays = await watchFocusLeaving(sheet)
    await burstOfTabs(page, 60)
    await page.waitForTimeout(100)
    expect(await strays()).toEqual([])
    await page.keyboard.press("Escape")
    await expect(sheet).toBeHidden()
    await expect.poll(() => inertCount(page)).toBe(inertBefore)
  })
})
