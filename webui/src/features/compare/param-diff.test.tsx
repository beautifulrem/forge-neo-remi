import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { tabSchema } from "@/api/mock/fixtures"
import { queryClient } from "@/api/queries"
import { CompareDialog } from "@/features/library/CompareDialog"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { nextBackdrop } from "./backdrop-modes"
import { BackdropPicker } from "./BackdropPicker"
import { applyCompareDiffToForm, diffInfotext, diffPromptTokens, fetchCompareDiff, totalDiffCount } from "./param-diff"
import { ParamDiffPanel } from "./ParamDiffPanel"

const INFO_A = [
  "1girl, silver hair, moonlit garden, <lora:detail_v1:0.8>, <lora:lighting:0.5>",
  "Negative prompt: lowres, blurry",
  "Steps: 28, Sampler: Euler a, Schedule type: Karras, CFG scale: 5, Seed: 1001, Size: 832x1216, Model: anima-2b, ADetailer model: face_yolov8n.pt, ADetailer confidence: 0.3",
].join("\n")

const INFO_B = [
  "1girl, silver hair, starry sky, volumetric glow, <lora:detail_v1:1.0>, <lora:style_x:0.6>",
  "Negative prompt: lowres, watermark",
  "Steps: 32, Sampler: ER SDE, Schedule type: Karras, CFG scale: 4, Seed: 2002, Size: 832x1216, Model: anima-2b, ADetailer model: face_yolov8n.pt, ADetailer confidence: 0.5",
].join("\n")

describe("structured parameter diff math", () => {
  it("diffs prompt tokens preserving order while ignoring LoRA tags", () => {
    const diff = diffPromptTokens("1girl, red scarf, <lora:a:0.5>, snow", "1girl, blue scarf, snow, lantern")
    expect(diff.unchanged).toEqual(["1girl", "snow"])
    expect(diff.removed).toEqual(["red scarf"])
    expect(diff.added).toEqual(["blue scarf", "lantern"])
  })

  it("extracts exact differences across prompts, core params, LoRAs and extensions", () => {
    const diff = diffInfotext(INFO_A, INFO_B)
    expect(diff.identical).toBe(false)
    expect(diff.prompt_diff.removed).toEqual(["moonlit garden"])
    expect(diff.prompt_diff.added).toEqual(["starry sky", "volumetric glow"])
    expect(diff.negative_diff.removed).toEqual(["blurry"])
    expect(diff.negative_diff.added).toEqual(["watermark"])

    expect(diff.core_diffs.map((d) => [d.key, d.a, d.b])).toEqual([
      ["Steps", "28", "32"],
      ["Sampler", "Euler a", "ER SDE"],
      ["CFG scale", "5", "4"],
      ["Seed", "1001", "2002"],
    ])
    expect(diff.lora_diffs).toEqual([
      { name: "detail_v1", a: "0.8", b: "1.0", kind: "changed" },
      { name: "lighting", a: "0.5", b: null, kind: "removed" },
      { name: "style_x", a: null, b: "0.6", kind: "added" },
    ])
    expect(diff.extension_diffs).toEqual([
      { group: "ADetailer", key: "ADetailer confidence", a: "0.3", b: "0.5", kind: "changed" },
    ])
    expect(totalDiffCount(diff)).toBe(10)
  })

  it("reports identical infotexts with zero diffs", () => {
    const diff = diffInfotext(INFO_A, INFO_A)
    expect(diff.identical).toBe(true)
    expect(totalDiffCount(diff)).toBe(0)
  })

  it("applies only the differing parameters from the selected side to the form", async () => {
    getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
    const diff = diffInfotext(INFO_A, INFO_B)
    diff.adopt_b_patch = { core: { cfg_scale: 4 }, scripts: { sampler: { steps: 32 }, seed: { seed: 2002 } }, raw_params: {} }
    const { count } = await applyCompareDiffToForm(diff, "b", "txt2img")
    expect(count).toBe(3)
    const values = getFormStore("txt2img").getState().values
    expect(values.core?.cfg_scale).toBe(4)
    expect(values.sampler?.steps).toBe(32)
    expect(values.seed?.seed).toBe(2002)
  })

  it("falls back to parsing infotext when server patch is empty", async () => {
    getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
    vi.spyOn(api, "parseInfotext")
      .mockResolvedValueOnce({ tab: "txt2img", core: { cfg_scale: 4, width: 832 }, scripts: { sampler: { steps: 32 } } })
      .mockResolvedValueOnce({ tab: "txt2img", core: { cfg_scale: 5, width: 832 }, scripts: { sampler: { steps: 28 } } })
    const diff = diffInfotext(INFO_A, INFO_B)
    const { count } = await applyCompareDiffToForm(diff, "b", "txt2img")
    expect(count).toBe(2) // width was identical so skipped
    expect(getFormStore("txt2img").getState().values.core?.cfg_scale).toBe(4)
    expect(getFormStore("txt2img").getState().values.sampler?.steps).toBe(32)
  })

  it("names the differences a lock kept, and leaves them out of the count", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 404 }))
    const success = vi.spyOn(toast, "success")
    const store = getFormStore("txt2img")
    store.getState().load(tabOwners(tabSchema("txt2img")))
    // a toggle that locks the CFG scale (`locks: "core.cfg_scale"`) is on, over a value of the user's own
    store.setState((s) => ({ locks: [{ owner: "keep", name: "cfg", target: ["core", "cfg_scale"] }], values: { ...s.values, core: { ...s.values.core, cfg_scale: 7 }, keep: { cfg: true } } }))
    vi.spyOn(api, "parseInfotext")
      .mockResolvedValueOnce({ tab: "txt2img", core: { cfg_scale: 4 }, scripts: { sampler: { steps: 32 } } })
      .mockResolvedValueOnce({ tab: "txt2img", core: { cfg_scale: 5 }, scripts: { sampler: { steps: 28 } } })
    render(
      <QueryClientProvider client={queryClient}>
        <ParamDiffPanel a={{ id: "a1", name: "a.png", infotext: INFO_A }} b={{ id: "b1", name: "b.png", infotext: INFO_B }} />
      </QueryClientProvider>,
    )
    fireEvent.click(await screen.findByRole("button", { name: "Apply B to form" }))
    await waitFor(() => expect(success).toHaveBeenCalledWith("Applied B differences (1)", { description: "Overridden by a lock: CFG Scale" }))
    expect(store.getState().values.core?.cfg_scale).toBe(7)
    expect(store.getState().values.sampler?.steps).toBe(32)
    vi.restoreAllMocks()
  })

  it("rethrows AbortError from fetchCompareDiff instead of falling back to local diff", async () => {
    const controller = new AbortController()
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const request = input as Request
      return new Promise((_resolve, reject) => {
        const abort = () => reject(new DOMException("The operation was aborted.", "AbortError"))
        if (request.signal.aborted) {
          abort()
          return
        }
        request.signal.addEventListener("abort", abort, { once: true })
      })
    })
    try {
      const pending = fetchCompareDiff({ id: "a1", infotext: INFO_A }, { id: "b1", infotext: INFO_B }, controller.signal)
      controller.abort()
      await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    } finally {
      vi.restoreAllMocks()
    }
  })

  it("rethrows Error with name AbortError even without a signal, while still falling back on non-abort failures", async () => {
    const spy = vi.spyOn(api.gallery, "compare").mockRejectedValueOnce(Object.assign(new Error("aborted"), { name: "AbortError" }))
    try {
      await expect(fetchCompareDiff({ id: "a1", infotext: INFO_A }, { id: "b1", infotext: INFO_B })).rejects.toMatchObject({ name: "AbortError" })

      spy.mockRejectedValueOnce(new TypeError("network down"))
      const fallback = await fetchCompareDiff({ id: "a1", infotext: INFO_A }, { id: "b1", infotext: INFO_B })
      expect(fallback.identical).toBe(false)
      expect(totalDiffCount(fallback)).toBe(10)
    } finally {
      vi.restoreAllMocks()
    }
  })
})

describe("RGBA 4-backdrop & CompareDialog UI", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 404 }))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("cycles through all 4 backdrops and switches via BackdropPicker", () => {
    expect(nextBackdrop("checker")).toBe("black")
    expect(nextBackdrop("black")).toBe("white")
    expect(nextBackdrop("white")).toBe("green")
    expect(nextBackdrop("green")).toBe("checker")

    const onChange = vi.fn()
    render(<BackdropPicker value="checker" onChange={onChange} />)
    fireEvent.click(screen.getByRole("button", { name: "Green screen" }))
    expect(onChange).toHaveBeenCalledWith("green")
  })

  it("renders CompareDialog with RGBA backdrop switcher and collapsible Parameters diff", async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CompareDialog
          images={[
            { id: "a1", name: "a.png", width: 832, height: 1216, infotext: INFO_A },
            { id: "b1", name: "b.png", width: 832, height: 1216, infotext: INFO_B },
          ]}
          onOpenChange={() => {}}
        />
      </QueryClientProvider>,
    )

    expect(await screen.findByText("Parameters diff")).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText("10 differences")).toBeInTheDocument())
    expect(screen.getByText("ADetailer confidence")).toBeInTheDocument()
    expect(screen.getByText("detail_v1")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Pure white" }))
    expect(document.querySelector('[data-backdrop="white"]')).toBeInTheDocument()
  })

  // `truncate` on a cell of an auto-layout table does not truncate: the column grows to the name and
  // the table scrolls sideways. A fixed layout keeps the columns a third each; the name is cut
  // with an ellipsis and carries its full text in a title.
  it("cuts a long LoRA name in its column, with the full name in its title", async () => {
    const name = "an_extremely_long_lora_name_that_would_widen_the_table_v12"
    const info = (weight: string) => `a cat, <lora:${name}:${weight}>\nSteps: 28, Seed: 1`
    render(
      <QueryClientProvider client={queryClient}>
        <CompareDialog
          images={[
            { id: "a1", name: "a.png", width: 832, height: 1216, infotext: info("0.5") },
            { id: "b1", name: "b.png", width: 832, height: 1216, infotext: info("0.9") },
          ]}
          onOpenChange={() => {}}
        />
      </QueryClientProvider>,
    )
    const cell = await screen.findByTitle(name)
    expect(cell.className.split(" ")).toContain("truncate")
    expect(cell.closest("table")!.className.split(" ")).toContain("table-fixed")
  })

  // The extensions table had an auto layout too: a long value without spaces (a model file, a
  // prompt joined by commas) widened it and the panel scrolled sideways.
  it("keeps a long extension value in its column", async () => {
    const model = "an_extremely_long_adetailer_model_name_that_would_widen_the_table_v12.pt"
    const info = (value: string) => `a cat\nSteps: 28, Seed: 1, ADetailer model: ${value}`
    render(
      <QueryClientProvider client={queryClient}>
        <CompareDialog
          images={[
            { id: "a1", name: "a.png", width: 832, height: 1216, infotext: info(model) },
            { id: "b1", name: "b.png", width: 832, height: 1216, infotext: info("face_yolov8n.pt") },
          ]}
          onOpenChange={() => {}}
        />
      </QueryClientProvider>,
    )
    const cell = (await screen.findByText("ADetailer model")).closest("td")!.nextElementSibling as HTMLElement
    expect(cell.textContent).toBe(model)
    expect(cell.className).toMatch(/\[overflow-wrap:anywhere\]/)
    expect(cell.closest("table")!.className.split(" ")).toContain("table-fixed")
  })

  it("wraps model and file names between their parts, not at any letter", async () => {
    const info = (model: string) => `a cat\nSteps: 28, Seed: 1, Model: ${model}, ADetailer model: face_yolov8n.pt`
    render(
      <QueryClientProvider client={queryClient}>
        <CompareDialog
          images={[
            { id: "a1", name: "a.png", width: 832, height: 1216, infotext: info("anima-v1.safetensors") },
            { id: "b1", name: "b.png", width: 832, height: 1216, infotext: info("sdxl_base.safetensors") },
          ]}
          onOpenChange={() => {}}
        />
      </QueryClientProvider>,
    )
    const cell = await screen.findByText("anima-v1.safetensors")
    // a soft break after each separator; a part wider than the column alone breaks inside
    expect(cell.innerHTML).toBe("anima-v1.<wbr>safetensors")
    expect(screen.getByText("sdxl_base.safetensors").innerHTML).toBe("sdxl_<wbr>base.<wbr>safetensors")
    expect(cell.className).not.toMatch(/break-all/)
    expect(cell.className).toMatch(/\[overflow-wrap:anywhere\]/)
  })


  // The core parameters table showed the raw infotext keys ("Steps", "CFG scale") in every language;
  // the infotext panel and PNG Info show their translation, the key as a title.
  it("names the core parameters in the UI language, with the infotext key as a title", async () => {
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "Steps", "步数")
    await act(() => i18n.changeLanguage("zh-CN"))
    try {
      render(
        <QueryClientProvider client={queryClient}>
          <CompareDialog
            images={[
              { id: "a1", name: "a.png", width: 832, height: 1216, infotext: INFO_A },
              { id: "b1", name: "b.png", width: 832, height: 1216, infotext: INFO_B },
            ]}
            onOpenChange={() => {}}
          />
        </QueryClientProvider>,
      )
      const label = await screen.findByText("步数", { selector: "td" })
      expect(label).toHaveAttribute("title", "Steps")
    } finally {
      await act(() => i18n.changeLanguage("en"))
    }
  })

  it("renders numeric and boolean primitives including 0 and false from real /gallery/compare without crashing", async () => {
    vi.spyOn(api.gallery, "compare").mockResolvedValue({
      id_a: "a1",
      id_b: "b1",
      infotext_a: "a\nSteps: 20, CFG scale: 0, Clip skip: 2",
      infotext_b: "a\nSteps: 0, CFG scale: 5, Anima enabled: false",
      identical: false,
      prompt_diff: { a: "a", b: "a", added: [], removed: [], unchanged: ["a"] },
      negative_diff: { a: "", b: "", added: [], removed: [], unchanged: [] },
      core_diffs: [
        { key: "Steps", field: "steps", a: 20, b: 0, kind: "changed" },
        { key: "CFG scale", field: "cfg_scale", a: 0, b: 5, kind: "changed" },
        { key: "Clip skip", field: "clip_skip", a: 2, b: null, kind: "removed" },
        { key: "Seed", field: "seed", a: undefined, b: 42, kind: "added" },
      ],
      lora_diffs: [{ name: "detail", a: 0, b: 0.8, kind: "changed" }],
      extension_diffs: [
        { group: "Anima", key: "Anima enabled", a: true, b: false, kind: "changed" },
        { group: "Anima", key: "Anima mode", a: null, b: "auto", kind: "added" },
      ],
      adopt_a_patch: { core: {}, scripts: {}, raw_params: {} },
      adopt_b_patch: { core: {}, scripts: {}, raw_params: {} },
    })

    render(
      <QueryClientProvider client={queryClient}>
        <ParamDiffPanel
          a={{ id: "a1", name: "a.png", infotext: "a\nSteps: 20" }}
          b={{ id: "b1", name: "b.png", infotext: "a\nSteps: 0" }}
        />
      </QueryClientProvider>,
    )

    expect(await screen.findByText("7 differences")).toBeInTheDocument()
    const stepsRow = screen.getByText("Steps").closest("tr")!
    expect(stepsRow.children[1]).toHaveTextContent("20")
    expect(stepsRow.children[2]).toHaveTextContent("0")
    expect(stepsRow.children[2]).toHaveClass("bg-diff-add-bg/60")

    const cfgRow = screen.getByText("CFG scale").closest("tr")!
    expect(cfgRow.children[1]).toHaveTextContent("0")
    expect(cfgRow.children[1]).toHaveClass("bg-diff-del-bg/60")

    const clipRow = screen.getByText("Clip skip").closest("tr")!
    expect(clipRow.children[2]).toHaveTextContent("—")

    const seedRow = screen.getByText("Seed").closest("tr")!
    expect(seedRow.children[1]).toHaveTextContent("—")
    expect(seedRow.children[2]).toHaveTextContent("42")

    const loraRow = screen.getByText("detail").closest("tr")!
    expect(loraRow.children[1]).toHaveTextContent("0")
    expect(loraRow.children[1]).toHaveClass("bg-diff-del-bg/60")

    const extRow = screen.getByText("Anima enabled").closest("tr")!
    expect(extRow.children[1]).toHaveTextContent("true")
    expect(extRow.children[2]).toHaveTextContent("false")
    expect(extRow.children[2]).toHaveClass("bg-diff-add-bg/60")
  })
})
