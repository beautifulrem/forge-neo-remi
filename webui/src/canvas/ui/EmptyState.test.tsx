import { render, screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { shownReason } from "@/test/says-why"
import { EmptyState } from "./EmptyState"

describe("the editor's empty stage", () => {
  // While the image loads again its failure stays, the Retry busy but still focusable.
  it("keeps a failed image's Retry focusable and busy while it runs", () => {
    for (const noUpload of [false, true]) {
      const view = render(<EmptyState busy={false} noUpload={noUpload} loadFailed retrying onRetry={() => {}} onFiles={() => {}} onBlank={() => {}} />)
      const alert = screen.getByRole("alert")
      expect(alert).toHaveTextContent("Could not load the image")
      expect(alert).toHaveAttribute("aria-busy", "true")
      const retry = within(alert).getByRole("button", { name: "Retry" })
      expect(retry).toHaveAttribute("aria-disabled", "true")
      expect(retry).not.toBeDisabled()
      view.unmount()
    }
  })

  // An upload that failed said so in an alert beside the drop target, which named only its hint.
  it("names a failed upload's message as the drop target's description", () => {
    render(<EmptyState busy={false} error="Could not upload the image" onFiles={() => {}} onBlank={() => {}} />)
    const target = screen.getByRole("button", { name: /Add a source image/ })
    expect(shownReason(target)).toBe("Drop an image, paste, or click to browse Could not upload the image")
  })
})
