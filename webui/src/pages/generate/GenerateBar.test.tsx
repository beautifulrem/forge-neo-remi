import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { dispatchRun } from "@/features/progress/run-controller"
import { GenerateBar } from "./GenerateBar"

vi.mock("@/features/progress/run-controller", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/progress/run-controller")>()),
  dispatchRun: vi.fn(),
}))

describe("generate several times", () => {
  afterEach(() => {
    vi.mocked(dispatchRun).mockClear()
    queryClient.clear()
  })

  // Clearing the field wrote 1 into it at once, so typing 5 next gave 15.
  it("takes the count typed into a cleared field", async () => {
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={queryClient}>
        <GenerateBar tab="txt2img" showAdvice={false} />
      </QueryClientProvider>,
    )
    await user.click(screen.getByRole("button", { name: "Generate several times" }))
    const field = await screen.findByLabelText("Generate several times", { selector: "input" })
    await user.clear(field)
    await user.type(field, "5")
    expect(field).toHaveDisplayValue("5")
    await user.click(screen.getByRole("button", { name: "Start" }))
    expect(dispatchRun).toHaveBeenCalledWith("txt2img", { type: "generate", runs: 5 })
  })

  it("keeps the count within what a run takes", async () => {
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={queryClient}>
        <GenerateBar tab="txt2img" showAdvice={false} />
      </QueryClientProvider>,
    )
    await user.click(screen.getByRole("button", { name: "Generate several times" }))
    const field = await screen.findByLabelText("Generate several times", { selector: "input" })
    await user.clear(field)
    await user.type(field, "5000{Enter}")
    expect(dispatchRun).toHaveBeenCalledWith("txt2img", { type: "generate", runs: 999 })
  })
})
