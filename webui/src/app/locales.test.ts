import { describe, expect, it } from "vitest"
import { languageTag } from "@/i18n"
import { availableLocales } from "./locales"

describe("availableLocales", () => {
  it("offers each catalog under the tag the active language is stored as, so the menu marks it", () => {
    const list = availableLocales([
      { id: "zh_CN", name: "简体中文" },
      { id: "pt_br", name: "Português" },
    ])
    expect(list.map((locale) => locale.id)).toEqual(["en", "zh-CN", "pt-BR"])
    // activateLocale stores languageTag(choice) as the current locale; the radio value must equal it
    for (const locale of list) expect(languageTag(locale.id)).toBe(locale.id)
  })

  it("lists a language once when two catalog names differ only in case", () => {
    const list = availableLocales([
      { id: "zh_CN", name: "简体中文" },
      { id: "zh_cn", name: "简体中文 (copy)" },
    ])
    expect(list.map((locale) => locale.id)).toEqual(["en", "zh-CN"])
  })
})
