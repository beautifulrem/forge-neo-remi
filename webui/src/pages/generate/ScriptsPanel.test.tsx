import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { FormProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import type { ChoiceNode, ScriptSchema, TabSchema } from "@/ui/schema"
import { tabOwners } from "@/ui/tab"
import { ScriptsPanel } from "./ScriptsPanel"

function makeNonSelectableScript(overrides: Partial<ScriptSchema> & { id: string; title: string }): ScriptSchema {
  return {
    kind: "alwayson",
    section: null,
    order: 50,
    layout: {
      type: "section",
      title: overrides.title,
      open: false,
      enable: { type: "toggle", name: "enabled", label: `Enable ${overrides.title}` },
      children: [{ type: "slider", name: "strength", label: "Strength", minimum: 0, maximum: 2, step: 0.1 }],
    },
    defaults: { enabled: false, strength: 1.0 },
    ...overrides,
  }
}

function makeSelectableScript(overrides: Partial<ScriptSchema> & { id: string; title: string }): ScriptSchema {
  return {
    kind: "selectable",
    section: "scripts",
    order: 10,
    requires_capabilities: ["anima_hooks"],
    applicable: false,
    inapplicable_reason: `Requires Anima engine hooks for ${overrides.title}`,
    visible_if: { param: "core.script", eq: overrides.title },
    layout: {
      type: "column",
      children: [{ type: "toggle", name: "grid", label: `${overrides.title} grid` }],
    },
    defaults: { grid: true },
    ...overrides,
  }
}

function renderScriptsPanel(schema: TabSchema, scriptChoice: ChoiceNode | null = null, storeId = "scripts-panel-test") {
  const store = createFormStore(storeId, { persistDrafts: false })
  store.getState().load(tabOwners(schema), { capabilities: schema.capabilities })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(
    <QueryClientProvider client={client}>
      <FormProvider formId="txt2img" store={store}>
        <ScriptsPanel schema={schema} scriptChoice={scriptChoice} />
      </FormProvider>
    </QueryClientProvider>,
  )
  return { ...view, store, client }
}

describe("ScriptsPanel visibility and inapplicable hardening", () => {
  it("omits the inapplicable drawer completely when all inapplicable scripts have visible_if: false", () => {
    const base = tabSchema("txt2img")
    const hiddenInapp = makeNonSelectableScript({
      id: "hidden_inapp",
      title: "Hidden Inapplicable",
      applicable: false,
      visible_if: false,
      inapplicable_reason: "Requires missing capability",
      defaults: { enabled: true, strength: 1.0 },
    })
    const schema: TabSchema = { ...base, scripts: [...base.scripts, hiddenInapp] }
    renderScriptsPanel(schema, null, "sp-all-hidden-inapp")

    expect(document.querySelector("[data-slot=inapplicable-scripts]")).toBeNull()
    expect(screen.queryByText("Not applicable for active model")).not.toBeInTheDocument()
    expect(screen.queryByText("Hidden Inapplicable")).not.toBeInTheDocument()
  })

  it("excludes visible_if: false scripts from inapplicable drawer counts, active count, and rendered cards", () => {
    const base = tabSchema("txt2img")
    const visibleInapp = makeNonSelectableScript({
      id: "visible_inapp",
      title: "Visible Inapplicable",
      applicable: false,
      inapplicable_reason: "Requires Anima hooks",
    })
    const hiddenInapp = makeNonSelectableScript({
      id: "hidden_inapp",
      title: "Hidden Inapplicable",
      applicable: false,
      visible_if: false,
      inapplicable_reason: "Should never appear",
      defaults: { enabled: true, strength: 1.0 },
    })
    const schema: TabSchema = { ...base, scripts: [...base.scripts, visibleInapp, hiddenInapp] }
    renderScriptsPanel(schema, null, "sp-mixed-inapp")

    const drawer = document.querySelector("[data-slot=inapplicable-scripts]") as HTMLElement
    expect(drawer).toBeTruthy()
    const trigger = within(drawer).getByRole("button", { name: /Not applicable for active model/ })
    expect(trigger).toHaveTextContent("1")
    expect(trigger).not.toHaveTextContent("2")
    expect(within(trigger).queryByText(/1 on/)).not.toBeInTheDocument()

    fireEvent.click(trigger)
    expect(within(drawer).getByText("Visible Inapplicable")).toBeVisible()
    expect(within(drawer).queryByText("Hidden Inapplicable")).not.toBeInTheDocument()
  })

  it("updates inapplicable drawer count and visibility reactively when a visible_if condition changes", () => {
    const base = tabSchema("txt2img")
    const conditionalInapp = makeNonSelectableScript({
      id: "conditional_inapp",
      title: "Conditional Inapplicable",
      applicable: false,
      visible_if: { param: "core.enable_hr", eq: true },
      inapplicable_reason: "Requires Anima hooks",
    })
    const schema: TabSchema = { ...base, scripts: [...base.scripts, conditionalInapp] }
    const { store } = renderScriptsPanel(schema, null, "sp-reactive-inapp")

    expect(document.querySelector("[data-slot=inapplicable-scripts]")).toBeNull()

    act(() => store.getState().setValue("core", "enable_hr", true))
    const drawer = document.querySelector("[data-slot=inapplicable-scripts]") as HTMLElement
    expect(drawer).toBeTruthy()
    expect(within(drawer).getByRole("button", { name: /Not applicable for active model/ })).toHaveTextContent("1")

    act(() => store.getState().setValue("core", "enable_hr", false))
    expect(document.querySelector("[data-slot=inapplicable-scripts]")).toBeNull()
  })

  it("excludes visible_if: false scripts from listed Extensions count and active count", () => {
    const base = tabSchema("txt2img")
    const visibleExt = makeNonSelectableScript({ id: "visible_ext", title: "Visible Ext", applicable: true })
    const hiddenExt = makeNonSelectableScript({
      id: "hidden_ext",
      title: "Hidden Ext",
      applicable: true,
      visible_if: false,
      defaults: { enabled: true, strength: 1.0 },
    })
    const schema: TabSchema = { ...base, scripts: [visibleExt, hiddenExt] }
    renderScriptsPanel(schema, null, "sp-listed-visible-if")

    const trigger = screen.getByRole("button", { name: /^Extensions/ })
    expect(trigger).toHaveTextContent("1")
    expect(trigger).not.toHaveTextContent("2")
    expect(within(trigger).queryByText(/1 on/)).not.toBeInTheDocument()
    expect(screen.getByText("Visible Ext")).toBeVisible()
    expect(screen.queryByText("Hidden Ext")).not.toBeInTheDocument()
  })

  it("does not duplicate inapplicable notice when an inapplicable selectable script is already selected in ScriptCard", () => {
    const base = tabSchema("txt2img")
    const animaXyz = makeSelectableScript({ id: "anima_xyz", title: "Anima XYZ Matrix" })
    const schema: TabSchema = { ...base, scripts: [...base.scripts, animaXyz] }
    const scriptChoice: ChoiceNode = {
      type: "choice",
      name: "script",
      label: "Script",
      choices: [
        { value: "None", label: "None" },
        { value: "Anima XYZ Matrix", label: "Anima XYZ Matrix" },
      ],
    }
    const { store } = renderScriptsPanel(schema, scriptChoice, "sp-selectable-dedup")
    const scriptSection = screen.getByRole("region", { name: "Script" })

    // When unselected, ScriptCard returns null so the notice renders once in the footer list.
    expect(scriptSection.querySelectorAll('[data-script="anima_xyz"]')).toHaveLength(1)
    expect(scriptSection.querySelector("[data-slot=inapplicable-selectable-scripts]")).toBeTruthy()
    expect(within(scriptSection).getAllByText(/Requires Anima engine hooks for Anima XYZ Matrix/)).toHaveLength(1)

    // When selected (e.g. restored from a draft), ScriptCard renders it with its reason and
    // the duplicate notice below must not render a second copy.
    act(() => store.getState().setValue("core", "script", "Anima XYZ Matrix"))
    expect(scriptSection.querySelectorAll('[data-script="anima_xyz"]')).toHaveLength(1)
    expect(scriptSection.querySelector("[data-slot=inapplicable-selectable-scripts]")).toBeNull()
    expect(within(scriptSection).getAllByText(/Requires Anima engine hooks for Anima XYZ Matrix/)).toHaveLength(1)
    expect(within(scriptSection).getByRole("switch", { name: "Anima XYZ Matrix grid" })).toHaveAttribute("aria-disabled", "true")
  })

  it("keeps unselected inapplicable selectable scripts in notice while omitting selected and visible_if: false ones", () => {
    const base = tabSchema("txt2img")
    const selectedInapp = makeSelectableScript({ id: "anima_xyz", title: "Anima XYZ Matrix" })
    const unselectedInapp = makeSelectableScript({ id: "anima_sweep", title: "Anima Sweep" })
    const hiddenInapp = makeSelectableScript({
      id: "anima_hidden",
      title: "Anima Hidden",
      visible_if: false,
    })
    const schema: TabSchema = { ...base, scripts: [...base.scripts, selectedInapp, unselectedInapp, hiddenInapp] }
    const scriptChoice: ChoiceNode = {
      type: "choice",
      name: "script",
      label: "Script",
      choices: [
        { value: "None", label: "None" },
        { value: "Anima XYZ Matrix", label: "Anima XYZ Matrix" },
        { value: "Anima Sweep", label: "Anima Sweep" },
        { value: "Anima Hidden", label: "Anima Hidden" },
      ],
    }
    const { store } = renderScriptsPanel(schema, scriptChoice, "sp-selectable-multi")
    act(() => store.getState().setValue("core", "script", "Anima XYZ Matrix"))

    const scriptSection = screen.getByRole("region", { name: "Script" })
    const notice = scriptSection.querySelector("[data-slot=inapplicable-selectable-scripts]") as HTMLElement
    expect(notice).toBeTruthy()
    expect(within(notice).queryByText("Anima XYZ Matrix")).not.toBeInTheDocument()
    expect(within(notice).queryByText("Anima Hidden")).not.toBeInTheDocument()
    expect(within(notice).getByText("Anima Sweep")).toBeVisible()
    expect(scriptSection.querySelectorAll('[data-script="anima_xyz"]')).toHaveLength(1)
    expect(scriptSection.querySelectorAll('[data-script="anima_sweep"]')).toHaveLength(1)
    expect(scriptSection.querySelectorAll('[data-script="anima_hidden"]')).toHaveLength(0)
  })
})
