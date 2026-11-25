import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { keys } from "@/api/queries"
import { bootstrapWith } from "@/test/api-fixtures"
import { FormProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { diffPromptTokens } from "./diff"
import { collectFormReferenceIds, detectPupFamily, formatAspectRatio } from "./pup"
import { PupTrigger } from "./PupTrigger"

describe("diffPromptTokens", () => {
  it("computes minimal additions and deletions preserving exact text", () => {
    const segs = diffPromptTokens("a cat sitting", "a calico cat sitting warmly")
    expect(segs.filter((s) => s.kind !== "del").map((s) => s.text).join("")).toBe("a calico cat sitting warmly")
    expect(segs.filter((s) => s.kind !== "add").map((s) => s.text).join("")).toBe("a cat sitting")
    expect(segs.some((s) => s.kind === "add" && s.text.includes("calico"))).toBe(true)
    expect(segs.some((s) => s.kind === "add" && s.text.includes("warmly"))).toBe(true)
  })

  it("handles identical and empty strings", () => {
    expect(diffPromptTokens("same", "same")).toEqual([{ kind: "equal", text: "same" }])
    expect(diffPromptTokens("", "new")).toEqual([{ kind: "add", text: "new" }])
    expect(diffPromptTokens("old", "")).toEqual([{ kind: "del", text: "old" }])
  })
})

describe("PUP helpers", () => {
  it("detects model family from capability, preset, or checkpoint", () => {
    expect(detectPupFamily("flux3", null, null)).toBe("flux3")
    expect(detectPupFamily("xl", "Qwen-Image-2.1-fp8.safetensors", null)).toBe("qwen21")
    expect(detectPupFamily("zit", null, null)).toBe("zimage")
    expect(detectPupFamily("fibo", null, null)).toBe("fibo")
    expect(detectPupFamily("xl", "plain.safetensors", "moss")).toBe("moss")
    expect(detectPupFamily("xl", "sdxl.safetensors", null)).toBe("auto")
  })

  it("formats aspect ratios cleanly", () => {
    expect(formatAspectRatio(1024, 1024)).toBe("1:1")
    expect(formatAspectRatio(1280, 720)).toBe("16:9")
    expect(formatAspectRatio(832, 1216)).toBe("13:19")
    expect(formatAspectRatio(0, 1024)).toBeNull()
  })

  it("collects reference image upload ids across form owners", () => {
    const store = createFormStore("pup-refs", { persistDrafts: false })
    store.getState().load(tabOwners(tabSchema("txt2img")))
    store.getState().setValue("batch-tools", "references", ["ref-1", "ref-2"])
    expect(collectFormReferenceIds(store.getState())).toEqual(["ref-1", "ref-2"])
  })
})

describe("PupTrigger & lazy PupUpsamplerDialog", () => {
  afterEach(() => vi.restoreAllMocks())

  it("opens lazy dialog, upsamples with references & ratio, previews diff, applies and reverts", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          prompt: "A sunlit orange cat resting on an oak windowsill",
          original_prompt: "orange cat",
          model_family: "flux3",
          mode: "i2i",
          backend: "openai_vlm",
          used_reference_count: 1,
          aspect_ratio: "1:1",
          layout_rows: null,
          vgl_json: null,
          warnings: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    const store = createFormStore("pup-ui", { persistDrafts: false })
    store.getState().load(tabOwners(tabSchema("txt2img")))
    store.getState().setValue("batch-tools", "references", ["up-ref-1"])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    client.setQueryData(keys.app, bootstrapWith({ forge_preset: "flux3", sd_model_checkpoint: "flux3-klein.safetensors" }))

    let current = "orange cat"
    const setValue = vi.fn((next: string) => {
      current = next
    })

    const { rerender } = render(
      <QueryClientProvider client={client}>
        <FormProvider formId="txt2img" store={store}>
          <PupTrigger value={current} setValue={setValue} />
        </FormProvider>
      </QueryClientProvider>,
    )

    fireEvent.click(screen.getByRole("button", { name: "Upsample prompt (PUP)" }))
    const dialog = await screen.findByRole("dialog", {}, { timeout: 15_000 })
    expect(dialog).toHaveTextContent("FLUX.3")
    expect(dialog).toHaveTextContent("Attach reference images (1)")
    expect(dialog).toHaveTextContent("Include aspect ratio (1:1)")

    fireEvent.click(screen.getByRole("button", { name: "Upsample" }))
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(1))
    const body = JSON.parse(String(fetchSpy.mock.calls[0]?.[1]?.body))
    expect(body).toMatchObject({ prompt: "orange cat", model_family: "flux3", reference_file_ids: ["up-ref-1"], aspect_ratio: "1:1" })

    expect(await screen.findByText("Before")).toBeVisible()
    expect(screen.getByText("After")).toBeVisible()
    expect(screen.getByText(/sunlit/)).toBeVisible()

    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    expect(setValue).toHaveBeenCalledWith("A sunlit orange cat resting on an oak windowsill")

    // Re-render with applied value: quick revert icon appears and restores exact original prompt
    rerender(
      <QueryClientProvider client={client}>
        <FormProvider formId="txt2img" store={store}>
          <PupTrigger value={current} setValue={setValue} />
        </FormProvider>
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole("button", { name: "Revert upsampled prompt" }))
    expect(setValue).toHaveBeenLastCalledWith("orange cat")
  })

  it("says why upsampling failed in the UI language, keeping the model's answer", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "local GGUF PUP execution failed: out of memory" }), { status: 502, headers: { "Content-Type": "application/json" } }),
    )
    const store = createFormStore("pup-ui-failed", { persistDrafts: false })
    store.getState().load(tabOwners(tabSchema("txt2img")))
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    client.setQueryData(keys.app, bootstrapWith({ forge_preset: "flux3", sd_model_checkpoint: "flux3-klein.safetensors" }))
    render(
      <QueryClientProvider client={client}>
        <FormProvider formId="txt2img" store={store}>
          <PupTrigger value="orange cat" setValue={vi.fn()} />
        </FormProvider>
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole("button", { name: "Upsample prompt (PUP)" }))
    await screen.findByRole("dialog", {}, { timeout: 15_000 })
    fireEvent.click(screen.getByRole("button", { name: "Upsample" }))
    expect(await screen.findByText("The service failed: local GGUF PUP execution failed: out of memory")).toBeVisible()
  }, 20_000)
})
