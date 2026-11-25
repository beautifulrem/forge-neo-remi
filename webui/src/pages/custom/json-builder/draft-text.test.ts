import { afterEach, describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import { ApiError } from "@/api/http"
import { check, DraftError, word } from "./compile"
import { draftErrorText } from "./draft-text"

const thrown = (run: () => void): unknown => {
  try {
    run()
  } catch (error) {
    return error
  }
  throw new Error("nothing thrown")
}

describe("draftErrorText", () => {
  afterEach(() => {
    i18n.removeResourceBundle("zh-CN", "chrome")
  })

  it("says a refused draft in the language asked for, its values as they are", () => {
    i18n.addResourceBundle("zh-CN", "chrome", { "{{location}}: unknown fields {{fields}}": "{{location}}：未知字段 {{fields}}", "Caption draft": "描述草稿" }, true, true)
    const error = thrown(() => check(false, "{{location}}: unknown fields {{fields}}", { location: word("Caption draft"), fields: '["x"]' }))
    expect(error).toBeInstanceOf(DraftError)
    // the English message stays upstream's
    expect((error as DraftError).message).toBe('Caption draft: unknown fields ["x"]')
    expect(draftErrorText(error, "en")).toBe('Caption draft: unknown fields ["x"]')
    expect(draftErrorText(error, "zh-CN")).toBe('描述草稿：未知字段 ["x"]')
  })

  // A value comes from the user's own JSON (a duplicate key): neither interpolated nor nested again.
  it("never reads a value as a placeholder, a nesting or a replacement pattern", () => {
    const key = "{{location}} $t(Load) $& $1"
    const error = new DraftError("Duplicate JSON key: {{name}}", { name: key })
    expect(error.message).toBe(`Duplicate JSON key: ${key}`)
    expect(draftErrorText(error, "en")).toBe(`Duplicate JSON key: ${key}`)
  })

  it("says a text that is not JSON in one sentence, and any other failure as the app does", () => {
    expect(draftErrorText(new SyntaxError("Unexpected token 'n', \"not json\" is not valid JSON"), "en")).toBe("The text is not valid JSON.")
    expect(draftErrorText(new ApiError(503, "busy"), "en")).toBe("The server is busy; try again shortly.")
  })
})
