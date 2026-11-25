import { readFileSync } from "node:fs"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { Segmented } from "./Segmented"
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs"

const options = [
  { value: "a", label: "Alpha" },
  { value: "b", label: "Beta" },
] as const

describe("segmented track on a floating surface", () => {
  // Dark --muted sits level with --popover: the track takes the page tone there, as a well does.
  it("defines on-popover for the dialog, alert dialog, popover and sheet surfaces", () => {
    const css = readFileSync("src/index.css", "utf8")
    const variant = /@custom-variant on-popover \(([^;]+)\);/.exec(css)?.[1] ?? ""
    for (const slot of ["dialog-content", "alert-dialog-content", "popover-content", "sheet-content"]) {
      expect(variant).toContain(`[data-slot=${slot}] *`)
    }
  })

  it("gives the segmented control's track the page tone there in the dark theme", () => {
    render(<Segmented aria-label="Mode" options={options} value="a" onValueChange={() => {}} />)
    expect(screen.getByRole("radiogroup", { name: "Mode" })).toHaveClass("bg-muted", "dark:on-popover:bg-background")
  })

  it("gives the tab strip's track the page tone there in the dark theme", () => {
    render(
      <Tabs defaultValue="a">
        <TabsList aria-label="Mode">
          <TabsTrigger value="a">Alpha</TabsTrigger>
          <TabsTrigger value="b">Beta</TabsTrigger>
        </TabsList>
      </Tabs>,
    )
    expect(screen.getByRole("tablist", { name: "Mode" })).toHaveClass("bg-muted", "dark:on-popover:bg-background")
  })
})
