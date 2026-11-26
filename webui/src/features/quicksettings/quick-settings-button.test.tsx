/** The header's quick settings button. */
import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import QuickSettings from "./QuickSettings"

describe("quick settings button", () => {
  afterEach(() => queryClient.clear())

  // With a setting pinned the button (a bolt and the count) was 53 px wide, and on a 360 px phone it
  // pushed the appearance menu past the window. On a phone it is a 28 px square like the header's
  // other icon buttons; the popover lists the pinned settings.
  it("is a 28 px square on a phone, its count shown from sm up", () => {
    queryClient.setQueryData(keys.app, { ...structuredClone(bootstrap), options: { ...bootstrap.options, quicksettings_list: ["samples_save", "grid_save"] } })
    render(
      <QueryClientProvider client={queryClient}>
        <QuickSettings />
      </QueryClientProvider>,
    )
    const button = screen.getByRole("button", { name: /^Quick settings/ })
    expect(button).toHaveClass("max-sm:size-7", "max-sm:px-0")
    expect(screen.getByText("2")).toHaveClass("max-sm:hidden")
  })
})
