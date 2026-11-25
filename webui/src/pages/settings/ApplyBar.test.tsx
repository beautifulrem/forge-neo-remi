import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ApplyBar } from "./ApplyBar"

describe("ApplyBar", () => {
  it("counts the unsaved changes in tabular figures, so the bar does not shift as the count changes", () => {
    render(<ApplyBar count={12} onApply={() => {}} onDiscard={() => {}} onReview={() => {}} reviewing={false} />)
    expect(screen.getByText("Unsaved changes: 12")).toHaveClass("tabular-nums")
  })
})
