/**
 * A menu whose code could not be fetched (offline, the server restarting) opens on the next press:
 * the failed fetch is not kept for the rest of the session. The failure is also told to the chunk
 * reload (a new build on the server has no such chunk any more).
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

afterEach(() => {
  vi.doUnmock("@/components/ui/dropdown-menu-impl")
  vi.resetModules()
})

describe("a dropdown menu whose code failed to load", { timeout: 30_000 }, () => {
  it("closes again and loads on the next press", async () => {
    vi.resetModules()
    let failures = 1
    vi.doMock("@/components/ui/dropdown-menu-impl", async (importOriginal) => {
      if (failures-- > 0) throw new TypeError("Failed to fetch dynamically imported module")
      return importOriginal()
    })
    const { onLoadFailed } = await import("@/lib/retryable-lazy")
    const told = vi.fn()
    const stop = onLoadFailed(told)
    const { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } = await import("./dropdown-menu")
    render(
      <DropdownMenu>
        <DropdownMenuTrigger>Open</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Reload</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    const trigger = screen.getByRole("button", { name: "Open" })
    fireEvent.click(trigger)
    await waitFor(() => expect(told).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(screen.queryByRole("menu")).toBeNull()

    fireEvent.click(trigger)
    expect(await screen.findByRole("menu", {}, { timeout: 15_000 })).toBeInTheDocument()
    expect(told).toHaveBeenCalledTimes(1)
    stop()
  })
})
