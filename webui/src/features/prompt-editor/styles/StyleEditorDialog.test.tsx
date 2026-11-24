import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { keys, queryClient } from "@/api/queries"
import type { PromptStyle } from "@/api/types"
import { StyleEditorDialog } from "./StyleEditorDialog"

// The real editor is CodeMirror: a plain textarea stands in for it here.
vi.mock("@/features/prompt-editor", () => ({
  LazyPromptEditor: ({ value, onChange, ariaLabel }: { value: string; onChange: (v: string) => void; ariaLabel?: string }) => (
    <textarea aria-label={ariaLabel} value={value} onChange={(e) => onChange(e.target.value)} />
  ),
}))

const CINEMATIC: PromptStyle = { name: "Cinematic", prompt: "cinematic still, {prompt}", negative_prompt: "cartoon" }

function renderEditor(initial: string | null = "Cinematic") {
  return render(
    <QueryClientProvider client={queryClient}>
      <StyleEditorDialog open onOpenChange={() => {}} currentPrompts={() => ({ prompt: "", negative: "" })} initial={initial} />
    </QueryClientProvider>,
  )
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
    await queryClient.invalidateQueries({ queryKey: keys.styles })
    expect(prompt().value).toBe("film grain, {prompt}")
    expect(negative().value).toBe("cartoon")
    expect(saveButton().disabled).toBe(false)
  })

  it("offers a retry when the list does not load", async () => {
    queryClient.setQueryDefaults(keys.styles, { retry: false })
    try {
      const list = vi.spyOn(api.styles, "list").mockRejectedValue(new TypeError("Failed to fetch"))
      renderEditor()
      const retry = await screen.findByRole("button", { name: "Retry" })
      expect(saveButton().disabled).toBe(true)
      list.mockResolvedValue([CINEMATIC])
      fireEvent.click(retry)
      await waitFor(() => expect(prompt().value).toBe("cinematic still, {prompt}"))
    } finally {
      queryClient.setQueryDefaults(keys.styles, {})
    }
  })
})
