import { act, render, screen } from "@testing-library/react"
import { useTranslation } from "react-i18next"
import { afterEach, describe, expect, it, vi } from "vitest"
import { activateLocale, browserLocale, i18n, initialLocale, localeName, textDirection, useLocale } from "./index"

function Probe() {
  const { t } = useTranslation()
  return (
    <p>
      {t("Load more")} / {t("Sampling method")} / {t("Untranslated label")}
    </p>
  )
}

describe("locale switching", () => {
  afterEach(async () => {
    vi.restoreAllMocks()
    await activateLocale("en")
  })

  it("loads the chrome and server catalogs and re-renders without a reload", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ "Sampling method": "采样方法" }), { headers: { "Content-Type": "application/json" } }))
    render(<Probe />)
    expect(screen.getByText("Load more / Sampling method / Untranslated label")).toBeInTheDocument()

    await act(() => activateLocale("zh_CN"))
    expect(new URL((fetchSpy.mock.calls[0]![0] as Request).url).pathname).toBe("/api/v2/i18n/zh_CN")
    expect(i18n.language).toBe("zh-CN")
    expect(document.documentElement.lang).toBe("zh-CN")
    // chrome → server → the English key itself.
    expect(screen.getByText("加载更多 / 采样方法 / Untranslated label")).toBeInTheDocument()

    await act(() => activateLocale("en"))
    expect(screen.getByText("Load more / Sampling method / Untranslated label")).toBeInTheDocument()
  })

  it("keeps one English word with two meanings apart (context keys)", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { headers: { "Content-Type": "application/json" } }))
    expect(i18n.t("System", { context: "theme" })).toBe("System")
    await activateLocale("zh_CN")
    i18n.addResourceBundle("zh-CN", "server", { System: "系统", Start: "起始" }, true, true)
    expect(i18n.t("System", { context: "theme" })).toBe("跟随系统") // the theme option
    expect(i18n.t("System")).toBe("系统") // the settings category
    expect(i18n.t("Start", { context: "runs" })).toBe("开始")
    expect(i18n.t("Start")).toBe("起始") // a slider's start, from the server catalog
  })

  it("keeps the language chosen last when an earlier choice is still loading", async () => {
    let release: () => void = () => {}
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          release = () => resolve(new Response("{}", { headers: { "Content-Type": "application/json" } }))
        }),
    )
    const slow = activateLocale("ja_JP")
    await activateLocale("en")
    release()
    await slow
    expect(i18n.language).toBe("en")
    expect(useLocale.getState().locale).toBe("en")
  })

  it("uses English for a catalog whose name is no language tag, so dates still format", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { headers: { "Content-Type": "application/json" } }))
    await activateLocale("zh-Hans (Stable)")
    expect(i18n.language).toBe("en")
    expect(() => new Date(0).toLocaleString(i18n.language)).not.toThrow()
  })

  it("ignores a stored language choice of the wrong type", async () => {
    localStorage.setItem("remi-locale", JSON.stringify({ state: { chosen: 5 }, version: 1 }))
    await useLocale.persist.rehydrate()
    expect(useLocale.getState().chosen).toBeNull()
    expect(initialLocale()).toBe(browserLocale())
  })

  it("gives English counts their singular or plural form", () => {
    expect(i18n.t("Remove {{count}} images?", { count: 1 })).toBe("Remove 1 image?")
    expect(i18n.t("Remove {{count}} images?", { count: 3 })).toBe("Remove 3 images?")
    expect(i18n.t("Staged {{count}} candidates on canvas", { count: 1 })).toBe("Staged 1 candidate on canvas")
    expect(i18n.t("Staged {{count}} candidates on canvas", { count: 4 })).toBe("Staged 4 candidates on canvas")
    expect(i18n.t("Added {{count}} images to Reference Images ({{total}} total)", { count: 1, total: 3 })).toBe("Added 1 image to Reference Images (3 total)")
  })

  it("writes right-to-left languages right to left (<html dir>)", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { headers: { "Content-Type": "application/json" } }))
    for (const locale of ["ar", "he_IL", "fa-IR", "ur", "ckb", "pa-Arab"]) expect(textDirection(locale), locale).toBe("rtl")
    for (const locale of ["en", "zh_CN", "ja_JP", "pa", "not a locale"]) expect(textDirection(locale), locale).toBe("ltr")

    await activateLocale("ar_SA")
    expect(document.documentElement.dir).toBe("rtl")
    expect(useLocale.getState().direction).toBe("rtl")
    await activateLocale("en")
    expect(document.documentElement.dir).toBe("ltr")
    expect(useLocale.getState().direction).toBe("ltr")
  })

  it("names a catalog the server has no name for in its own language", () => {
    expect(localeName("zh_CN", "简体中文")).toBe("简体中文")
    expect(localeName("ar", "ar")).toBe("العربية")
    expect(localeName("he_IL", "he_IL")).toBe("עברית (ישראל)")
    expect(localeName("not a locale", "not a locale")).toBe("not a locale")
  })
})
