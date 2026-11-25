import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import { useSession } from "@/api/session"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import { LicensesView } from "./LicensesView"

describe("LicensesView", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
    useSession.setState({ source: null })
  })

  function renderEmpty() {
    vi.spyOn(api.system, "licenses").mockReturnValue(new Promise(() => {}))
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
  }

  // GNU AGPL v3 section 13: the users of a modified program on a network are offered its source
  it("offers the source code of the running version, under the WebUI's own license", () => {
    useSession.setState({ source: "https://example.org/remi/tree/0123abc" })
    renderEmpty()
    expect(screen.getByText(/free software under the GNU Affero General Public License v3\.0/)).toBeInTheDocument()
    const link = screen.getByRole("link", { name: "Source code" })
    expect(link).toHaveAttribute("href", "https://example.org/remi/tree/0123abc")
    expect(link).toHaveAttribute("target", "_blank")
    // a 40 px target: a line of text lies between it and the notices link, so the boxes do not meet
    expect(link).toHaveClass("relative", "after:absolute", "after:inset-x-0", "after:-inset-y-2.5")
  })

  it("names the source code link in the UI language", async () => {
    useSession.setState({ source: "https://example.org/remi" })
    renderEmpty()
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    try {
      expect(screen.getByRole("link", { name: zhCN["Source code"] })).toHaveAttribute("href", "https://example.org/remi")
      expect(screen.getByText(zhCN["This WebUI is free software under the GNU Affero General Public License v3.0."], { exact: false })).toBeInTheDocument()
    } finally {
      await act(() => i18n.changeLanguage("en"))
    }
  })

  it("links no source before the server said where it is", () => {
    renderEmpty()
    expect(screen.queryByRole("link", { name: "Source code" })).toBeNull()
  })

  it("links the notices of the libraries bundled into the web interface", () => {
    vi.spyOn(api.system, "licenses").mockReturnValue(new Promise(() => {}))
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("link", { name: /Web interface libraries/ })).toHaveAttribute("href", "/THIRD-PARTY-NOTICES.txt")
  })

  it("gives the one-line notices link a 40 px tall target, like the Show license toggles", () => {
    vi.spyOn(api.system, "licenses").mockReturnValue(new Promise(() => {}))
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    expect(screen.getByRole("link", { name: /Web interface libraries/ })).toHaveClass("relative", "after:absolute", "after:inset-x-0", "after:-inset-y-2.5")
  })

  // An entry's name links to its project: a 20 px line of text with no target around it. Its box
  // reaches the panel's hairline above and stops where the Show license toggle's box starts.
  it("gives an entry's name link a target that stops short of its Show license toggle", async () => {
    vi.spyOn(api.system, "licenses").mockResolvedValue({
      groups: [
        {
          title: "Libraries",
          items: [
            { id: "a", name: "Described", url: "https://example.com/a", description: "A library", license_url: "https://example.com/a/LICENSE" },
            { id: "b", name: "Bare", url: "https://example.com/b", license_url: "https://example.com/b/LICENSE" },
          ],
        },
      ],
    } as never)
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    const described = await screen.findByRole("link", { name: "Described" })
    expect(described).toHaveClass("relative", "after:absolute", "after:inset-x-0", "after:-top-2.5", "after:-bottom-2.5")
    // the description keeps 20 px between the name and the toggle: each box takes 10 of it
    const toggles = screen.getAllByRole("button", { name: "Show license" })
    expect(toggles[0]).toHaveClass("after:-top-2.5", "after:-bottom-2.5")
    // without one the toggle sits 4 px under the name: 2 px each, 32 px boxes that do not overlap
    expect(screen.getByRole("link", { name: "Bare" })).toHaveClass("after:-top-2.5", "after:-bottom-0.5")
    expect(toggles[1]).toHaveClass("after:-top-0.5", "after:-bottom-2.5")
    expect(toggles[1]).not.toHaveClass("after:-inset-y-2.5")
  })

  // A license text that could not load offered only its page on the web: no Retry.
  it("offers a Retry for a license text that did not load, kept up while it runs", async () => {
    vi.spyOn(api.system, "licenses").mockResolvedValue({ groups: [{ title: "", items: [{ id: "a", name: "Some library", license_url: "https://example.com/LICENSE" }] }] } as never)
    let release = () => {}
    const held = new Promise<void>((resolve) => (release = resolve))
    // two failures: the text's own query tries once more before it gives up
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("", { status: 503 }))
      .mockResolvedValueOnce(new Response("", { status: 503 }))
      .mockImplementationOnce(async () => {
        await held
        return new Response("MIT License", { status: 200 })
      })
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    fireEvent.click(await screen.findByRole("button", { name: "Show license" }))
    const alert = await screen.findByRole("alert", {}, { timeout: 5_000 })
    expect(alert).toHaveTextContent("Could not load the license text.")
    // its page on the web stays a way out
    expect(within(alert).getByRole("link", { name: "Open it on the web" })).toHaveAttribute("href", "https://example.com/LICENSE")
    const retry = within(alert).getByRole("button", { name: "Retry" })
    fireEvent.click(retry)
    await waitFor(() => expect(retry).toHaveAttribute("aria-disabled", "true"))
    expect(alert).toBeInTheDocument()
    expect(fetch).toHaveBeenCalledTimes(3)
    await act(async () => release())
    expect(await screen.findByText("MIT License")).toBeInTheDocument()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  // The Retry under the focus went with the failure when the list arrived: the focus fell to the page.
  it("gives the focus to the list a retry brought", async () => {
    let release = () => {}
    const held = new Promise<void>((resolve) => (release = resolve))
    const licenses = vi.spyOn(api.system, "licenses").mockRejectedValueOnce(new ApiError(404, "boom", { detail: "boom" }))
    licenses.mockImplementationOnce(async () => {
      await held
      return { groups: [{ title: "", items: [{ id: "a", name: "Some library", url: "https://example.com/a" }] }] } as never
    })
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    const retry = within(await screen.findByRole("alert")).getByRole("button", { name: "Retry" })
    act(() => retry.focus())
    fireEvent.click(retry)
    await waitFor(() => expect(retry).toHaveAttribute("aria-disabled", "true"))
    await act(async () => release())
    const link = await screen.findByRole("link", { name: "Some library" })
    await waitFor(() => expect(link).toHaveFocus())
  })

  it("says the list did not load in the UI language, with a Retry", async () => {
    const licenses = vi.spyOn(api.system, "licenses").mockRejectedValueOnce(new ApiError(404, "boom", { detail: "boom" }))
    licenses.mockResolvedValueOnce({ groups: [{ title: "", items: [{ id: "a", name: "Some library" }] }] } as never)
    render(
      <QueryClientProvider client={queryClient}>
        <LicensesView />
      </QueryClientProvider>,
    )
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("It is no longer there.")
    expect(alert).not.toHaveTextContent("boom")
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(screen.getByText("Some library")).toBeInTheDocument())
  })
})
