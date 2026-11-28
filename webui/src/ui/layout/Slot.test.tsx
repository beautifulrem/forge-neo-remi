import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { FormProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import type { ScriptSchema, TabSchema } from "@/ui/schema"
import { tabOwners } from "@/ui/tab"
import { ScriptsProvider, Slot } from "./Slot"

function makeSlottedScript(overrides: Partial<ScriptSchema> & { id: string; title: string; section: string }): ScriptSchema {
  return {
    kind: "alwayson",
    order: 10,
    layout: {
      type: "section",
      title: overrides.title,
      open: true,
      children: [{ type: "toggle", name: "enabled", label: `${overrides.title} toggle` }],
    },
    defaults: { enabled: false },
    ...overrides,
  }
}

function renderSlot(section: string, scripts: ScriptSchema[], storeId: string) {
  const base = tabSchema("txt2img")
  const coreLayout = base.core.layout
  if (coreLayout.type !== "column") throw new Error("expected column core layout")
  const schema: TabSchema = {
    ...base,
    core: {
      ...base.core,
      layout: { ...coreLayout, children: [...coreLayout.children, { type: "slot", section }] },
    },
    scripts,
  }
  const store = createFormStore(storeId, { persistDrafts: false })
  store.getState().load(tabOwners(schema), { capabilities: schema.capabilities })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(
    <QueryClientProvider client={client}>
      <FormProvider formId="txt2img" store={store}>
        <ScriptsProvider scripts={scripts}>
          <Slot section={section} />
        </ScriptsProvider>
      </FormProvider>
    </QueryClientProvider>,
  )
  return { ...view, store }
}

describe("Slot visible_if filtering", () => {
  it("returns null without rendering an empty wrapper when all slotted scripts evaluate visible_if to false", () => {
    const hiddenStatic = makeSlottedScript({
      id: "hidden_static",
      title: "Hidden Static",
      section: "custom_slot",
      visible_if: false,
    })
    const hiddenCond = makeSlottedScript({
      id: "hidden_cond",
      title: "Hidden Conditional",
      section: "custom_slot",
      visible_if: { param: "core.enable_hr", eq: true },
    })
    const { container } = renderSlot("custom_slot", [hiddenStatic, hiddenCond], "slot-all-hidden")

    expect(container.querySelector("[data-slot-section='custom_slot']")).toBeNull()
    expect(container.firstChild).toBeNull()
    expect(screen.queryByText("Hidden Static")).not.toBeInTheDocument()
    expect(screen.queryByText("Hidden Conditional")).not.toBeInTheDocument()
  })

  it("renders the slot container reactively when visible_if turns true and removes it when false again", () => {
    const conditional = makeSlottedScript({
      id: "cond_script",
      title: "Conditional Slotted",
      section: "custom_slot",
      visible_if: { param: "core.enable_hr", eq: true },
    })
    const { container, store } = renderSlot("custom_slot", [conditional], "slot-reactive")

    expect(container.querySelector("[data-slot-section='custom_slot']")).toBeNull()
    expect(container.firstChild).toBeNull()

    act(() => store.getState().setValue("core", "enable_hr", true))
    const wrapper = container.querySelector("[data-slot-section='custom_slot']")
    expect(wrapper).not.toBeNull()
    expect(wrapper?.querySelector("[data-script='cond_script']")).not.toBeNull()
    expect(screen.getByText("Conditional Slotted")).toBeVisible()

    act(() => store.getState().setValue("core", "enable_hr", false))
    expect(container.querySelector("[data-slot-section='custom_slot']")).toBeNull()
    expect(container.firstChild).toBeNull()
  })

  it("renders only visible applicable scripts when a slot mixes visible and hidden scripts", () => {
    const hidden = makeSlottedScript({
      id: "hidden_slot",
      title: "Hidden Slotted",
      section: "custom_slot",
      visible_if: false,
    })
    const visible = makeSlottedScript({
      id: "visible_slot",
      title: "Visible Slotted",
      section: "custom_slot",
      visible_if: true,
    })
    const inapplicable = makeSlottedScript({
      id: "inapp_slot",
      title: "Inapplicable Slotted",
      section: "custom_slot",
      applicable: false,
      visible_if: true,
    })
    const { container } = renderSlot("custom_slot", [hidden, visible, inapplicable], "slot-mixed")

    const wrapper = container.querySelector("[data-slot-section='custom_slot']")
    expect(wrapper).not.toBeNull()
    expect(wrapper?.querySelectorAll("[data-script]")).toHaveLength(1)
    expect(screen.getByText("Visible Slotted")).toBeVisible()
    expect(screen.queryByText("Hidden Slotted")).not.toBeInTheDocument()
    expect(screen.queryByText("Inapplicable Slotted")).not.toBeInTheDocument()
  })
})
