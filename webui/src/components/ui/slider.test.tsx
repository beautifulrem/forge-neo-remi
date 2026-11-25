import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { Slider } from "./slider"

const root = (container: HTMLElement) => container.querySelector("[data-slot=slider]")!

describe("Slider", () => {
  it("fills its row unless given a width", () => {
    const { container } = render(<Slider value={[50]} getAriaLabel={() => "Opacity"} />)
    expect(root(container).className.split(" ")).toContain("w-full")
  })

  // `data-horizontal:w-full` came after `w-18` in the stylesheet and won: the canvas staging
  // bar's opacity fader was 6 px wide.
  it("takes a width it is given in place of its own", () => {
    const { container } = render(<Slider value={[50]} getAriaLabel={() => "Opacity"} className="w-18" />)
    const classes = root(container).className.split(" ")
    expect(classes).toContain("w-18")
    expect(classes.filter((name) => /(^|:)w-full$/.test(name))).toEqual([])
  })

  // `data-vertical:h-full` came after a caller's `h-24` in the stylesheet and won, as the width did.
  it("takes a height it is given in place of its own when vertical", () => {
    const { container } = render(<Slider value={[50]} orientation="vertical" getAriaLabel={() => "Opacity"} className="h-24" />)
    const classes = root(container).className.split(" ")
    expect(classes).toContain("h-24")
    expect(classes.filter((name) => /(^|:)h-full$/.test(name))).toEqual([])
  })
})
