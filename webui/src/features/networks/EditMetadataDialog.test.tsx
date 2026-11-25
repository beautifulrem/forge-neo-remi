import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { bootstrap, models } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import type { NetworkItem, NetworkKind, NetworkMeta } from "@/api/types"
import { EditMetadataDialog, PreferredModules } from "./EditMetadataDialog"

function renderModules(value: string[] = []) {
  render(
    <QueryClientProvider client={queryClient}>
      <PreferredModules id="preferred" value={value} onChange={() => {}} />
    </QueryClientProvider>,
  )
}

// A checkpoint's preferred VAE / text encoders, in its metadata editor (the phone has no model bar).
describe("preferred modules", () => {
  beforeEach(() => queryClient.setQueryData(keys.app, structuredClone(bootstrap)))
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("says the modules are on their way while the model list loads, not that there are none", async () => {
    vi.spyOn(api, "models").mockReturnValue(new Promise(() => {}))
    renderModules()
    expect(await screen.findByRole("status")).toHaveTextContent("Loading…")
    expect(screen.queryByText("No modules found")).not.toBeInTheDocument()
  })

  it("offers a retry when the model list does not load", async () => {
    const list = vi.spyOn(api, "models").mockRejectedValue(new ApiError(404, "boom"))
    renderModules()
    const retry = await screen.findByRole("button", { name: "Retry" })
    expect(screen.queryByText("No modules found")).not.toBeInTheDocument()
    // centred in the list, as its loading and empty states are
    expect(screen.getByRole("alert")).toHaveClass("justify-center")
    list.mockResolvedValue(models)
    fireEvent.click(retry)
    expect(await screen.findByRole("checkbox", { name: models.modules?.[0]?.title ?? models.modules?.[0]?.name ?? "" })).toBeInTheDocument()
  })

  it("does not strike a chosen module through while the list loads", async () => {
    vi.spyOn(api, "models").mockReturnValue(new Promise(() => {}))
    renderModules(["gone.safetensors"])
    await screen.findByRole("status")
    expect(screen.getByText("gone.safetensors")).not.toHaveClass("line-through")
  })

  it("does not strike a chosen module through when the list failed to load", async () => {
    vi.spyOn(api, "models").mockRejectedValue(new ApiError(404, "boom"))
    renderModules(["gone.safetensors"])
    await screen.findByRole("button", { name: "Retry" })
    expect(screen.getByText("gone.safetensors")).not.toHaveClass("line-through")
  })

  it("strikes through a chosen module the loaded list lacks", async () => {
    vi.spyOn(api, "models").mockResolvedValue(models)
    renderModules(["gone.safetensors"])
    await screen.findByRole("checkbox", { name: models.modules?.[0]?.title ?? models.modules?.[0]?.name ?? "" })
    expect(screen.getByText("gone.safetensors")).toHaveClass("line-through")
  })
})

describe("edit metadata dialog", () => {
  const kind = { id: "lora", title: "LoRA", action: "prompt", negative: true, fields: ["description", "notes"], builtin: true } as NetworkKind
  const item = (name: string) => ({ name, title: name, kind: "lora", path: `${name}.safetensors`, folder: "", tags: [], preferred_modules: [], search_terms: [], editable: true, hidden: false }) as NetworkItem
  const meta = (name: string) => ({ ...item(name), user_metadata: { description: name }, metadata: {}, dataset_tags: [] }) as unknown as NetworkMeta

  beforeEach(() => queryClient.setQueryData(keys.app, structuredClone(bootstrap)))
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("does not close the next item's editor when an earlier save answers late", async () => {
    vi.spyOn(api.networks, "meta").mockImplementation(async (_kind, name) => meta(name))
    let answer: (value: NetworkMeta) => void = () => {}
    vi.spyOn(api.networks, "editMeta").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    const onOpenChange = vi.fn()
    const view = (name: string, open: boolean) => (
      <QueryClientProvider client={queryClient}>
        <EditMetadataDialog item={item(name)} kind={kind} open={open} onOpenChange={onOpenChange} />
      </QueryClientProvider>
    )
    const { rerender } = render(view("ink", true))
    fireEvent.click(await screen.findByRole("button", { name: "Save" }))
    // the user closes A while its save is on its way, then opens B
    rerender(view("ink", false))
    rerender(view("pastel", true))
    expect(await screen.findByDisplayValue("pastel")).toBeInTheDocument()
    await act(async () => answer(meta("ink")))
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
    expect(screen.getByDisplayValue("pastel")).toBeInTheDocument()
  })
})
