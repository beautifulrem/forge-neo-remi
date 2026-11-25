/** The app's toaster: a description's line breaks (a list of values) show as lines. */
import { act, render, screen } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, describe, expect, it } from "vitest"
import { Toaster } from "./sonner"

describe("Toaster", () => {
  afterEach(() => act(() => void toast.dismiss()))

  it("keeps the line breaks of a description", async () => {
    render(<Toaster />)
    act(() => void toast.warning("Some values were skipped", { description: "width: too wide\nsteps: too many" }))
    const description = (await screen.findByText(/width: too wide/)).closest("[data-description]")
    expect(description).toHaveClass("whitespace-pre-line")
  })
})
