import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ApiError } from "@/api/http"
import { activateLocale } from "@/i18n"
import { LoadFailed } from "./LoadFailed"

describe("load failed", () => {
  afterEach(async () => {
    await act(() => activateLocale("en"))
  })

  // The React Compiler keeps a render's text until the values it read change: the failure was made
  // with the global i18n, so after a language switch the Retry button changed and the reason did not.
  it("says the failure in the language switched to", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { headers: { "Content-Type": "application/json" } }))
    render(<LoadFailed error={new ApiError(503, "busy")} onRetry={() => {}} />)
    expect(screen.getByRole("alert")).toHaveTextContent("The server is busy; try again shortly.")
    await act(() => activateLocale("zh_CN"))
    expect(screen.getByRole("button", { name: "重试" })).toBeInTheDocument()
    expect(screen.getByRole("alert")).toHaveTextContent("服务器正忙，请稍后重试。")
  })
})
