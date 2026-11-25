import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import { shownReason } from "@/test/says-why"
import { FormProvider } from "@/ui/context"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { SizeTools } from "./SizeTools"
import { loadSchema } from "./use-form-loader"

const SIZE = "Enter width x height, such as 1216x832."

let store: FormStore

function renderTools() {
  render(
    <FormProvider formId="txt2img" store={store}>
      <SizeTools />
    </FormProvider>,
  )
}

describe("the aspect ratio popover's size field", () => {
  beforeEach(() => {
    store = createFormStore("size-tools-test", { persistDrafts: false })
    loadSchema(store, tabSchema("txt2img"))
  })
  afterEach(async () => {
    await act(() => i18n.changeLanguage("en"))
  })

  // Its red edge was the only sign that the text is not a size, and under the coral accent the
  // focus ring around it is the same red: it says what it takes, in text, while the text is wrong.
  it("says what it takes while its text is not a size", async () => {
    renderTools()
    fireEvent.click(screen.getByRole("button", { name: /^Aspect ratio/ }))
    const field = await screen.findByRole("textbox", { name: "Size as width x height" })
    const live = field.closest("[role=dialog]")!.querySelector("[aria-live=polite]")
    expect(live).toBeEmptyDOMElement()
    fireEvent.change(field, { target: { value: "12x" } })
    expect(shownReason(field)).toBe(SIZE)
    // announced as it appears: it is written into a polite live region that was already there
    expect(live).toHaveTextContent(SIZE)
    fireEvent.change(field, { target: { value: "1216x832" } })
    expect(field).not.toHaveAttribute("aria-invalid")
    expect(field).not.toHaveAttribute("aria-describedby")
    expect(screen.queryByText(SIZE)).toBeNull()
  })

  it("says it in the UI language", async () => {
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    renderTools()
    fireEvent.click(screen.getByRole("button", { name: new RegExp(`^${zhCN["Aspect ratio"]}`) }))
    const field = await screen.findByRole("textbox", { name: zhCN["Size as width x height"] })
    fireEvent.change(field, { target: { value: "wide" } })
    expect(shownReason(field)).toBe(zhCN[SIZE])
  })
})
