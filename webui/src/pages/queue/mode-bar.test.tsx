/**
 * The queue's mode tabs sit at the top of `main`, which clips what it holds (`overflow-clip`): a
 * focus ring drawn outside a tab would lose its top there. The tabs are line tabs in a tab list
 * whose padding is as wide as a tab's focus ring (frontend.md: room around controls at a clipping
 * edge), so the whole ring stays inside the bar.
 */
import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import QueuePage from "./index"

beforeEach(() => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 404 }))
})

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

const px = (classes: string[], pattern: RegExp) => classes.map((name) => pattern.exec(name)?.[1]).find(Boolean)

describe("the queue's mode tabs", () => {
  it("keep their whole focus ring inside the bar under main's top edge", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <QueuePage />
      </QueryClientProvider>,
    )
    const strip = screen.getByRole("tablist", { name: "Queue" })
    const room = px(strip.className.split(" "), /^p-\[(\d+)px\]$/)
    expect(room).toBeDefined()
    for (const tab of within(strip).getAllByRole("tab")) {
      const ring = px(tab.className.split(" "), /^focus-visible:ring-\[(\d+)px\]$/)
      expect(ring).toBeDefined()
      expect(Number(ring)).toBeLessThanOrEqual(Number(room))
    }
  })
})
