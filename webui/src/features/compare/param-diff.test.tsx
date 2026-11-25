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
import { applyCompareDiffToForm, diffInfotext, diffPromptTokens, totalDiffCount } from "./param-diff"
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
})
