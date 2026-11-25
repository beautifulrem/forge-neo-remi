/**
 * A popover whose code could not be fetched (offline, the server restarting) closes again, so its
 * trigger does not stay announced as expanded over nothing, and opens on the next press. The
 * failure is also told to the chunk reload (a new build on the server has no such chunk any more).
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

afterEach(() => {
  vi.doUnmock("@/components/ui/popover-impl")
  vi.resetModules()
})

describe("a popover whose code failed to load", { timeout: 30_000 }, () => {
  it("closes again and loads on the next press", async () => {
    vi.resetModules()
    let failures = 1
    vi.doMock("@/components/ui/popover-impl", async (importOriginal) => {
      if (failures-- > 0) throw new TypeError("Failed to fetch dynamically imported module")
      return importOriginal()
    })
    const { onLoadFailed } = await import("@/lib/retryable-lazy")
    const told = vi.fn()
    const stop = onLoadFailed(told)
    const onOpenChange = vi.fn()
    const { Popover, PopoverContent, PopoverTrigger } = await import("./popover")
    render(
      <Popover onOpenChange={onOpenChange}>
        <PopoverTrigger>Aspect ratio</PopoverTrigger>
        <PopoverContent aria-label="Aspect ratio">
          <button type="button">3:2</button>
        </PopoverContent>
      </Popover>,
    )
    const trigger = screen.getByRole("button", { name: "Aspect ratio" })
    fireEvent.click(trigger)
    await waitFor(() => expect(told).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(onOpenChange).toHaveBeenLastCalledWith(false, undefined)
    expect(screen.queryByRole("dialog")).toBeNull()

    fireEvent.click(trigger)
    expect(await screen.findByRole("dialog", {}, { timeout: 15_000 })).toBeInTheDocument()
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(told).toHaveBeenCalledTimes(1)
    stop()
  })
})
