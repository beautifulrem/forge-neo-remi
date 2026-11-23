import { describe, expect, it } from "vitest"
import { isComposing } from "./ime"

const key = (init: KeyboardEventInit, keyCode = 13) => ({ nativeEvent: new KeyboardEvent("keydown", init), keyCode })

describe("isComposing", () => {
  it("is true while an IME composes, and for Safari's confirming Enter", () => {
    expect(isComposing(key({ key: "Enter", isComposing: true }))).toBe(true)
    expect(isComposing(key({ key: "Enter" }, 229))).toBe(true)
    expect(isComposing(key({ key: "Enter" }))).toBe(false)
  })
})
