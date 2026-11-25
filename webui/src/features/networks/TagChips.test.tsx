import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { TagChips } from "./TagChips"

describe("TagChips", () => {
  it("gives a tag cut short by its chip its full text as a title", () => {
    const tag = "very_long_training_tag_that_does_not_fit_in_a_chip_of_sixteen_rem"
    render(<TagChips tags={[{ tag, count: 12 }]} onPick={() => {}} pickLabel="Add to the prompt" />)
    expect(screen.getByText(tag)).toHaveAttribute("title", tag)
  })
})
