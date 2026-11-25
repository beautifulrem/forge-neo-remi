import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { promptToolsApi } from "../api"
import { TEST_CONFIG } from "../testing"
import HistoryPanel from "./HistoryPanel"

const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe("prompt history", () => {
  it("sends a rename once, however often Enter is pressed while it is on its way", async () => {
    vi.spyOn(promptToolsApi.history, "list").mockResolvedValue({
      items: [{ id: "h1", tab: "txt2img", prompt: "a cat, sunset", negative_prompt: "", name: "", created: 1, last_used: Date.now() / 1000 - 60, uses: 1 }],
      next_cursor: null,
      total: 1,
    })
    let answer: () => void = () => {}
    const rename = vi.spyOn(promptToolsApi.history, "rename").mockImplementation(() => new Promise((resolve) => (answer = () => resolve({} as never))))
    render(<HistoryPanel value="" setValue={vi.fn()} kind="positive" label="Prompt" disabled={false} config={TEST_CONFIG} close={vi.fn()} />, { wrapper })
    fireEvent.click(await screen.findByRole("button", { name: "Rename" }))
    const name = screen.getByRole("textbox", { name: "Name" })
    fireEvent.change(name, { target: { value: "Sunset cat" } })
    fireEvent.submit(name.closest("form")!)
    fireEvent.submit(name.closest("form")!)
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    expect(rename).toHaveBeenCalledTimes(1)
    await act(async () => answer())
    expect(rename).toHaveBeenCalledTimes(1)
  })
})
