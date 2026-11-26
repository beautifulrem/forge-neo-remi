import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { tabSchema } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import { bootstrapWith } from "@/test/api-fixtures"
import { FormProvider } from "@/ui/context"
import { createFormStore, getFormStore } from "@/ui/form-store"
import type { AdviceItem, ScriptSchema, TabSchema } from "@/ui/schema"
import { tabOwners } from "@/ui/tab"
import { ScriptsPanel } from "./ScriptsPanel"
import { WorkflowAdviceBar } from "./WorkflowAdviceBar"

function makeSchemaWithCapabilities(): TabSchema {
  const base = tabSchema("txt2img")
  const animaOnly: ScriptSchema = {
    id: "anima_dave",
    title: "Anima DAVE",
    kind: "alwayson",
    section: null,
    order: 5,
    requires_capabilities: ["anima_hooks"],
    applicable: false,
    inapplicable_reason: "Requires Anima engine hooks (active model: Qwen-Image-2.1)",
    layout: {
      type: "section",
      title: "Anima DAVE",
      open: false,
      enable: { type: "toggle", name: "enabled", label: "Enable DAVE" },
      children: [{ type: "slider", name: "strength", label: "Strength", minimum: 0, maximum: 2, step: 0.1 }],
    },
    defaults: { enabled: false, strength: 1.0 },
  }
  return {
    ...base,
    scripts: [...base.scripts, animaOnly],
    capabilities: {
      preset: "qwen21",
      dit_name: "qwen21",
      engine_label: "Qwen-Image-2.1",
      capabilities: ["multi_reference", "rgba_output"],
      max_references: 10,
      reference_slot_format: "<image_{i}>",
      reference_slot_start: 1,
      supports_rgba: true,
      supports_control_blocks: true,
      badges: ["FP8", "Turbo", "RGBA"],
    },
  }
}

describe("capability-aware ScriptsPanel", () => {
  it("groups inapplicable scripts into a quiet collapsed drawer with their reason", () => {
    const schema = makeSchemaWithCapabilities()
    const store = createFormStore("cap-test", { persistDrafts: false })
    store.getState().load(tabOwners(schema), { capabilities: schema.capabilities })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <FormProvider formId="txt2img" store={store}>
          <ScriptsPanel schema={schema} scriptChoice={null} />
        </FormProvider>
      </QueryClientProvider>,
    )

    const drawer = document.querySelector("[data-slot=inapplicable-scripts]") as HTMLElement
    expect(drawer).toBeTruthy()
    const trigger = within(drawer).getByRole("button", { name: /Not applicable for active model/ })
    expect(trigger).toHaveTextContent("1")
    // Collapsed by default: the Anima DAVE reason is unmounted until opened
    expect(screen.queryByText(/Requires Anima engine hooks/)).not.toBeInTheDocument()

    fireEvent.click(trigger)
    expect(screen.getByText("Requires Anima engine hooks (active model: Qwen-Image-2.1)")).toBeVisible()
    expect(within(drawer).getByText("Anima DAVE")).toBeVisible()
  })
})

describe("server sentences in the UI language", () => {
  afterEach(async () => {
    const { i18n } = await import("@/i18n")
    await i18n.changeLanguage("en")
  })

  it("translates an advice chip and the reason a script does not apply by their templates", async () => {
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "Requires {{capabilities}} (active model: {{model}})", "需要 {{capabilities}}（当前模型：{{model}}）")
    i18n.addResource("zh-CN", "server", "Anima engine hooks", "Anima 引擎钩子")
    i18n.addResource("zh-CN", "server", "Bria Fibo 2.0 supports at most {{max}} reference images; {{count}} are attached.", "Bria Fibo 2.0 最多支持 {{max}} 张参考图，当前附加了 {{count}} 张。")
    await act(() => i18n.changeLanguage("zh-CN"))
    const schema = makeSchemaWithCapabilities()
    const anima = schema.scripts.find((script) => script.id === "anima_dave")!
    anima.inapplicable_template = "Requires {{capabilities}} (active model: {{model}})"
    anima.inapplicable_params = { capabilities: ["Anima engine hooks"], model: "Qwen-Image-2.1" }
    const store = getFormStore("txt2img")
    const advice: AdviceItem[] = [
      {
        code: "fibo.too_many_references",
        severity: "warning",
        message: "Bria Fibo 2.0 supports at most 2 reference images; 3 are attached.",
        template: "Bria Fibo 2.0 supports at most {{max}} reference images; {{count}} are attached.",
        params: { max: "2", count: "3" },
      },
    ]
    store.getState().load(tabOwners(schema), { advice, capabilities: schema.capabilities })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <FormProvider formId="txt2img" store={store}>
          <WorkflowAdviceBar tab="txt2img" />
          <ScriptsPanel schema={schema} scriptChoice={null} />
        </FormProvider>
      </QueryClientProvider>,
    )
    expect(screen.getByText("Bria Fibo 2.0 最多支持 2 张参考图，当前附加了 3 张。")).toBeVisible()
    const drawer = document.querySelector("[data-slot=inapplicable-scripts]") as HTMLElement
    fireEvent.click(within(drawer).getAllByRole("button")[0]!)
    expect(screen.getByText("需要 Anima 引擎钩子（当前模型：Qwen-Image-2.1）")).toBeVisible()
  })
})

describe("WorkflowAdviceBar", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("renders pre-submit advice chips and applies one-click form and option patches", async () => {
    const schema = makeSchemaWithCapabilities()
    const store = getFormStore("txt2img")
    const advice: AdviceItem[] = [
      {
        code: "qwen21.turbo_mismatch",
        severity: "warning",
        message: "Turbo model works best at 8 steps and CFG 1.0.",
        patch: { "sampler.steps": { value: 8 }, cfg_scale: { value: 1.0 } },
        action_label: "Set Turbo (8 steps · CFG 1.0)",
      },
      {
        code: "qwen21.rgba_jpeg",
        severity: "warning",
        message: "Output format is JPEG; switch to PNG to keep alpha transparency.",
        patch: { __options__: { samples_format: "png" } },
        action_label: "Switch Output to PNG",
      },
    ]
    store.getState().load(tabOwners(schema), { advice })
    queryClient.setQueryData(keys.app, bootstrapWith({ samples_format: "jpg" }))
    const patchOpt = vi.spyOn(api, "patchOptions").mockResolvedValue({} as never)

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <WorkflowAdviceBar tab="txt2img" />
      </QueryClientProvider>,
    )

    expect(screen.getByText("Turbo model works best at 8 steps and CFG 1.0.")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Set Turbo (8 steps · CFG 1.0)" }))
    await waitFor(() => expect(store.getState().values.sampler?.steps).toBe(8))
    expect(store.getState().values.core?.cfg_scale).toBe(1.0)
    // Resolved chip disappears cleanly
    expect(screen.queryByText("Turbo model works best at 8 steps and CFG 1.0.")).not.toBeInTheDocument()

    // Click option fix once the first fix has finished: the bar disables every fix while one runs,
    // and on a busy machine the test can get here before the bar has drawn the end of the first.
    const pngFix = () => screen.getByRole("button", { name: "Switch Output to PNG" })
    await waitFor(() => expect(pngFix()).toBeEnabled())
    fireEvent.click(pngFix())
    await waitFor(() => expect(patchOpt).toHaveBeenCalledWith({ samples_format: "png" }))
  })

  it("collapses extra chips beyond 2 and supports dismissing", () => {
    const store = getFormStore("img2img")
    const schema = tabSchema("img2img")
    const items: AdviceItem[] = [
      { code: "a", severity: "info", message: "Tip one", patch: null },
      { code: "b", severity: "warning", message: "Tip two", patch: null },
      { code: "c", severity: "error", message: "Tip three", patch: null },
    ]
    store.getState().load(tabOwners(schema), { advice: items })

    render(<WorkflowAdviceBar tab="img2img" />)
    expect(screen.getByText("Tip one")).toBeVisible()
    expect(screen.getByText("Tip two")).toBeVisible()
    expect(screen.queryByText("Tip three")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Show 1 more advice" }))
    expect(screen.getByText("Tip three")).toBeVisible()

    const dismissButtons = screen.getAllByRole("button", { name: "Dismiss advice" })
    act(() => {
      fireEvent.click(dismissButtons[0]!)
    })
    expect(screen.queryByText("Tip one")).not.toBeInTheDocument()
  })
})
