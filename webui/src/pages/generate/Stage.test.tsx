/** The stage on its own: what the empty viewfinder asks for, on a device with a keyboard or without one. */
import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { keys, queryClient } from "@/api/queries"
import { useSession } from "@/features/gallery/session-store"
import { TOUCH_ONLY_QUERY } from "@/lib/use-media-query"
import { bootstrapWith } from "@/test/api-fixtures"
import { Stage } from "./Stage"

/** Makes `matchMedia` match the given queries (jsdom matches none). */
function mediaMatching(...queries: string[]) {
  const matchMedia = window.matchMedia
  vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({ ...matchMedia(query), matches: queries.includes(query) }))
}

function renderStage() {
  render(
    <QueryClientProvider client={queryClient}>
      <Stage tab="txt2img" onOpenViewer={() => {}} />
    </QueryClientProvider>,
  )
}

describe("Stage", () => {
  beforeEach(() => {
    queryClient.setQueryData(keys.app, bootstrapWith({}))
    useSession.getState().clear()
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("asks for the keyboard shortcut where there is a keyboard", () => {
    renderStage()
    expect(screen.getByText(/^Write a prompt, then press (Ctrl\+Enter|⌘Enter|⌘↩) to generate$/)).toBeInTheDocument()
  })

  // A phone was told to press ⌘Enter (or Ctrl+Enter), keys it does not have.
  it("asks for a tap on Generate on a touch screen without a keyboard", () => {
    mediaMatching(TOUCH_ONLY_QUERY)
    renderStage()
    expect(screen.getByText("Write a prompt, then tap Generate")).toBeInTheDocument()
    expect(screen.queryByText(/press|⌘|Ctrl/)).toBeNull()
  })
})
