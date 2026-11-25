import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { keys } from "@/api/queries"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
import { bootstrapWith } from "@/test/api-fixtures"
import { FormProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { diffPromptTokens } from "./diff"
import { collectFormReferenceIds, detectPupFamily, formatAspectRatio, PUP_FAMILY_LABELS } from "./pup"
import { PupTrigger } from "./PupTrigger"
import PupUpsamplerDialog from "./PupUpsamplerDialog"

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

  it("names the result's model family and mode in the UI language", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ prompt: "a cat on a sill", original_prompt: "cat", model_family: "generic", mode: "t2i", backend: "openai_vlm", used_reference_count: 0, aspect_ratio: null, layout_rows: null, vgl_json: null, warnings: [] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )
    const store = createFormStore("pup-ui-zh", { persistDrafts: false })
    store.getState().load(tabOwners(tabSchema("txt2img")))
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    client.setQueryData(keys.app, bootstrapWith({ forge_preset: "sdxl", sd_model_checkpoint: "animagine-xl.safetensors" }))
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    try {
      render(
        <QueryClientProvider client={client}>
          <FormProvider formId="txt2img" store={store}>
            <PupTrigger value="cat" setValue={vi.fn()} />
          </FormProvider>
        </QueryClientProvider>,
      )
      fireEvent.click(screen.getByRole("button", { name: zhCN["Upsample prompt (PUP)"] }))
      const dialog = await screen.findByRole("dialog", {}, { timeout: 15_000 })
      fireEvent.click(within(dialog).getByRole("button", { name: zhCN["Upsample"] }))
      await within(dialog).findByText(zhCN["Generic"])
      expect(within(dialog).getByText(zhCN["Text to image"])).toBeInTheDocument()
      expect(within(dialog).queryByText("generic")).toBeNull()
      expect(within(dialog).queryByText("t2i")).toBeNull()
    } finally {
      await act(() => i18n.changeLanguage("en"))
    }
    // "auto" is never a family a result names; it is the picker's own choice, translated there
    expect("auto" in PUP_FAMILY_LABELS).toBe(false)
  }, 20_000)

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

  // On a phone the footer's own row of Close and Apply (and the Revert beside it) kept DialogFooter from
  // stacking its buttons full width: each group is a column below sm, as the footer is.
  describe("footer on a phone", () => {
    function renderDialog(previousValue: string | null) {
      const store = createFormStore(`pup-footer-${previousValue ?? "none"}`, { persistDrafts: false })
      store.getState().load(tabOwners(tabSchema("txt2img")))
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
      client.setQueryData(keys.app, bootstrapWith({ forge_preset: "flux3", sd_model_checkpoint: "flux3-klein.safetensors" }))
      render(
        <QueryClientProvider client={client}>
          <FormProvider formId="txt2img" store={store}>
            <PupUpsamplerDialog open onOpenChange={() => {}} value="orange cat" previousValue={previousValue} onApply={() => {}} onRevert={() => {}} />
          </FormProvider>
        </QueryClientProvider>,
      )
      const footer = screen.getByRole("dialog").querySelector<HTMLElement>("[data-slot=dialog-footer]")!
      const group = (slot: string) => footer.querySelector<HTMLElement>(`[data-slot=${slot}]`)!
      const classes = (slot: string) => group(slot).className.split(/\s+/)
      return { footer, group, classes }
    }

    it("stacks Close and Revert as full-width rows below sm", () => {
      const { footer, group, classes } = renderDialog("orange cat, older")
      expect(group("pup-footer-end")).toContainElement(within(footer).getByRole("button", { name: "Close" }))
      expect(classes("pup-footer-end")).toEqual(expect.arrayContaining(["flex", "flex-col-reverse", "sm:flex-row"]))
      expect(classes("pup-footer-end")).not.toContain("items-center")
      expect(group("pup-footer-start")).toContainElement(within(footer).getByRole("button", { name: "Revert" }))
      expect(classes("pup-footer-start")).toEqual(expect.arrayContaining(["flex", "flex-col", "sm:flex-row"]))
    })

    it("leaves no empty row for Revert when there is nothing to revert", () => {
      const { group, classes } = renderDialog(null)
      expect(group("pup-footer-start")).toBeEmptyDOMElement()
      expect(classes("pup-footer-start")).toContain("max-sm:hidden")
    })
  })
})
