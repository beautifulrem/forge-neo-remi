import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { InsertTargets } from "./index"

// On a phone the targets have a row of their own, 390 px wide: the label and both switches at
// their own widths took 400 px in English.
describe("InsertTargets", () => {
  it("on a phone, reads its label out only and lets the switches share the row", () => {
    render(
      <div role="group" aria-label="Insert into">
        <InsertTargets phone />
      </div>,
    )
    expect(screen.getByText("Insert into")).toHaveClass("sr-only")
    for (const name of ["Form", "Insert into"]) expect(screen.getByRole("radiogroup", { name })).toHaveClass("min-w-0", "flex-1")
  })

  it("elsewhere keeps the label and the switches at their own widths", () => {
    render(<InsertTargets />)
    expect(screen.getByText("Insert into")).not.toHaveClass("sr-only")
    expect(screen.getByRole("radiogroup", { name: "Form" })).toHaveClass("shrink-0")
  })
})
