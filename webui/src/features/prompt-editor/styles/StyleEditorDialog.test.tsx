import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { keys, queryClient } from "@/api/queries"
import type { PromptStyle } from "@/api/types"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import type { MultiChoiceNode } from "@/ui/schema"
import { StyleEditorDialog } from "./StyleEditorDialog"
import { StylesField } from "./StylesField"

// The real editor is CodeMirror: a plain textarea stands in for it here.
vi.mock("@/features/prompt-editor", () => ({
  LazyPromptEditor: ({ value, onChange, ariaLabel }: { value: string; onChange: (v: string) => void; ariaLabel?: string }) => (
    <textarea aria-label={ariaLabel} value={value} onChange={(e) => onChange(e.target.value)} />
  ),
}))

const CINEMATIC: PromptStyle = { name: "Cinematic", prompt: "cinematic still, {prompt}", negative_prompt: "cartoon" }

function editor(open: boolean, initial: string | null = "Cinematic") {
  return (
    <QueryClientProvider client={queryClient}>
      <StyleEditorDialog open={open} onOpenChange={() => {}} currentPrompts={() => ({ prompt: "", negative: "" })} initial={initial} />
    </QueryClientProvider>
  )
}

function renderEditor(initial: string | null = "Cinematic") {
  return render(editor(true, initial))
}

const prompt = () => screen.getByRole("textbox", { name: "Style prompt" }) as HTMLTextAreaElement
const negative = () => screen.getByRole("textbox", { name: "Style negative prompt" }) as HTMLTextAreaElement
const saveButton = () => screen.getByRole("button", { name: "Save" }) as HTMLButtonElement

describe("style editor opened on the form's style", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("shows that style's prompts once the list arrives, and has nothing to save", async () => {
    let answer: (styles: PromptStyle[]) => void = () => {}
    vi.spyOn(api.styles, "list").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    renderEditor()
    // Until the list is known, saving could overwrite a style the editor does not show.
    expect(saveButton().disabled).toBe(true)
    answer([CINEMATIC])
    await waitFor(() => expect(prompt().value).toBe("cinematic still, {prompt}"))
    expect(negative().value).toBe("cartoon")
    expect(saveButton().disabled).toBe(true)
  })

  // Typing while the list was on its way derived the draft from a blank style: once the list
  // arrived, Save wrote that blank negative prompt over the style.
  it("takes no edits before the list arrives, and says it is loading", async () => {
    let answer: (styles: PromptStyle[]) => void = () => {}
    vi.spyOn(api.styles, "list").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    renderEditor()
    expect(screen.getByRole("status")).toHaveTextContent("Loading…")
    expect(screen.queryByRole("textbox", { name: "Style prompt" })).not.toBeInTheDocument()
    expect((screen.getByRole("button", { name: "Copy from prompts" }) as HTMLButtonElement).disabled).toBe(true)
    answer([CINEMATIC])
    await waitFor(() => expect(negative().value).toBe("cartoon"))
  })

  it("shows a style the list already holds at once", () => {
    queryClient.setQueryData(keys.styles, [CINEMATIC])
    vi.spyOn(api.styles, "list").mockResolvedValue([CINEMATIC])
    renderEditor()
    expect(prompt().value).toBe("cinematic still, {prompt}")
    expect(negative().value).toBe("cartoon")
  })

  it("keeps the user's edit when the list refreshes", async () => {
    queryClient.setQueryData(keys.styles, [CINEMATIC])
    vi.spyOn(api.styles, "list").mockResolvedValue([CINEMATIC])
    renderEditor()
    fireEvent.change(prompt(), { target: { value: "film grain, {prompt}" } })
    await act(() => queryClient.invalidateQueries({ queryKey: keys.styles }))
    expect(prompt().value).toBe("film grain, {prompt}")
    expect(negative().value).toBe("cartoon")
    await waitFor(() => expect(saveButton().disabled).toBe(false))
  })

  // Another client may have changed the style since the cached list was fetched (no event says
  // so): saving the stale copy wrote its old text back over the other client's change.
  it("asks for the list again on every opening and saves only once it has answered", async () => {
    queryClient.setQueryData(keys.styles, [CINEMATIC])
    let answer: (styles: PromptStyle[]) => void = () => {}
    const list = vi.spyOn(api.styles, "list").mockImplementation(() => new Promise((resolve) => (answer = resolve)))
    const { rerender } = render(editor(true))
    expect(list).toHaveBeenCalledTimes(1)
    fireEvent.change(prompt(), { target: { value: "film grain, {prompt}" } })
    expect(saveButton().disabled).toBe(true)
    await act(async () => answer([CINEMATIC]))
    await waitFor(() => expect(saveButton().disabled).toBe(false))

    rerender(editor(false))
    rerender(editor(true))
    expect(list).toHaveBeenCalledTimes(2)
    await waitFor(() => expect(saveButton().disabled).toBe(true))
    await act(async () => answer([CINEMATIC]))
    await waitFor(() => expect(saveButton().disabled).toBe(false))
  })

  it("starts on the form's style again after closing once the edit is saved", async () => {
    const PASTEL: PromptStyle = { name: "Pastel", prompt: "pastel, {prompt}", negative_prompt: "" }
    const stored = [CINEMATIC, PASTEL]
    queryClient.setQueryData(keys.styles, stored)
    vi.spyOn(api.styles, "list").mockImplementation(async () => structuredClone(stored))
    vi.spyOn(api.styles, "save").mockImplementation(async (style) => {
      stored[0] = style
      return style
    })
    const { rerender } = render(editor(true))
    fireEvent.change(prompt(), { target: { value: "film grain, {prompt}" } })
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    fireEvent.click(saveButton())
    // saved: nothing left to keep
    await waitFor(() => expect(saveButton().disabled).toBe(true))
    rerender(editor(false))
    rerender(editor(true, "Pastel"))
    await waitFor(() => expect(prompt().value).toBe("pastel, {prompt}"))
  })

  it("shows the fields' outline while the list loads, so the dialog keeps its height", () => {
    vi.spyOn(api.styles, "list").mockReturnValue(new Promise(() => {}))
    renderEditor()
    const status = screen.getByRole("status")
    expect(status.querySelectorAll("[data-slot=skeleton]")).toHaveLength(6)
  })

  it("puts the footer's buttons in one column on a phone", async () => {
    queryClient.setQueryData(keys.styles, [CINEMATIC])
    vi.spyOn(api.styles, "list").mockResolvedValue([CINEMATIC])
    renderEditor()
    const actions = saveButton().parentElement!
    expect(actions).toHaveClass("flex-col-reverse", "sm:flex-row")
  })

  it("offers a retry when the list does not load", async () => {
    queryClient.setQueryDefaults(keys.styles, { retry: false })
    try {
      const list = vi.spyOn(api.styles, "list").mockRejectedValue(new TypeError("Failed to fetch"))
      renderEditor()
      const retry = await screen.findByRole("button", { name: "Retry" })
      expect(saveButton().disabled).toBe(true)
      list.mockResolvedValue([CINEMATIC])
      retry.focus()
      fireEvent.click(retry)
      await waitFor(() => expect(prompt().value).toBe("cinematic still, {prompt}"))
      // the focus moves to the list it was waiting for, not to the dialog
      await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("combobox", { name: "Style" })))
    } finally {
      queryClient.setQueryDefaults(keys.styles, {})
    }
  })
})

describe("style editor of the form's styles field", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("keeps an unsaved edit when the editor is closed with Esc", async () => {
    queryClient.setQueryData(keys.styles, [CINEMATIC])
    vi.spyOn(api.styles, "list").mockResolvedValue([CINEMATIC])
    const store = createFormStore("styles-field", { persistDrafts: false })
    store.getState().setValues("core", { styles: ["Cinematic"] })
    const node: MultiChoiceNode = { type: "multichoice", name: "styles", label: "Styles", choices: [{ value: "Cinematic", label: "Cinematic" }] }
    render(
      <QueryClientProvider client={queryClient}>
        <FormProvider formId="txt2img" store={store}>
          <OwnerProvider owner="core">
            <StylesField node={node} />
          </OwnerProvider>
        </FormProvider>
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole("button", { name: "Edit styles" }))
    const field = await screen.findByRole("textbox", { name: "Style prompt" }, { timeout: 10_000 })
    fireEvent.change(field, { target: { value: "film grain, {prompt}" } })
    fireEvent.keyDown(field, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    fireEvent.click(screen.getByRole("button", { name: "Edit styles" }))
    expect(((await screen.findByRole("textbox", { name: "Style prompt" })) as HTMLTextAreaElement).value).toBe("film grain, {prompt}")
  }, 20_000)
})
