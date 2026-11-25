import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import { shownReason } from "@/test/says-why"
import { ColorControl } from "./ColorControl"

const HEX = "Enter a hex color, such as #ff8800."

async function openHexField() {
  render(<ColorControl color="#112233" recent={[]} onChange={() => {}} onEyedropper={() => {}} />)
  fireEvent.click(screen.getByRole("button", { name: "Brush color" }))
  return screen.findByRole("textbox", { name: "Hex color" })
}

describe("the brush color's hex field", () => {
  afterEach(async () => {
    await act(() => i18n.changeLanguage("en"))
  })

  // Its red edge was the only sign that the text does not parse, and under the coral accent the
  // focus ring around it is the same red: it says what it takes, in text, while the text is wrong.
  it("says what it takes while its text is not a hex color", async () => {
    const hex = await openHexField()
    const live = hex.closest("[role=dialog]")!.querySelector("[aria-live=polite]")
    expect(live).toBeEmptyDOMElement()
    fireEvent.change(hex, { target: { value: "#ff88" } })
    expect(shownReason(hex)).toBe(HEX)
    // announced as it appears: it is written into a polite live region that was already there
    expect(live).toHaveTextContent(HEX)
    fireEvent.change(hex, { target: { value: "#ff8800" } })
    expect(hex).not.toHaveAttribute("aria-invalid")
    expect(hex).not.toHaveAttribute("aria-describedby")
    expect(screen.queryByText(HEX)).toBeNull()
  })

  it("says it in the UI language", async () => {
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    render(<ColorControl color="#112233" recent={[]} onChange={() => {}} onEyedropper={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: zhCN["Brush color"] }))
    const hex = await screen.findByRole("textbox", { name: zhCN["Hex color"] })
    fireEvent.change(hex, { target: { value: "red" } })
    expect(shownReason(hex)).toBe(zhCN[HEX])
  })
})
