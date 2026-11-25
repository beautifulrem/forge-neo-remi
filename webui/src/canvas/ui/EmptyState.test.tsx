import { render, screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"
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
})
