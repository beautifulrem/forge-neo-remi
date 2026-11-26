import { describe, expect, it } from "vitest"
import { classConstants, withClassConstants } from "./jsx-tags"

describe("withClassConstants", () => {
  const constants = classConstants(['export const FIELD_BUTTON_CLASS = "bg-field hover:bg-field-hover"'])

  it("adds the classes of a constant the tag names", () => {
    expect(withClassConstants('<Button className={cn(FIELD_BUTTON_CLASS, "px-2")}>', constants)).toContain("hover:bg-field-hover")
  })

  it("adds a name the file gives a class string or a constant", () => {
    const source = 'const ring = "focus-visible:ring-3"\nconst tone = FIELD_BUTTON_CLASS'
    expect(withClassConstants("<Button className={cn(ring, tone)}>", constants, source)).toContain('"focus-visible:ring-3 bg-field hover:bg-field-hover"')
  })

  it("reads a name inside a template literal's expression", () => {
    expect(withClassConstants("<Button className={`px-2 ${FIELD_BUTTON_CLASS}`}>", constants)).toContain("hover:bg-field-hover")
  })

  // `\b` splits a class at its hyphen: `ring-3` in a class string read as the file's `ring`, so a tag
  // wore classes it does not have, which can hide a violation as well as invent one.
  it("does not read the words of a class string as names", () => {
    const source = 'const ring = "dark:hover:bg-muted"\nconst border = "border-2"'
    const tag = '<Button className="focus-visible:ring-3 border-x" title=\'ring\'>'
    expect(withClassConstants(tag, constants, source)).toBe(tag)
    expect(withClassConstants("<Button className={`ring-3 ${ring}`}>", constants, source)).toContain('"dark:hover:bg-muted"')
  })
})
