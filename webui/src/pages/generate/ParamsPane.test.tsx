import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import type { ReactNode } from "react"
import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { getFormStore } from "@/ui/form-store"
import type { TabSchema } from "@/ui/schema"
import { ParamsPane } from "./ParamsPane"

const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })

function wrap(children: ReactNode) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

/** The schema as the server sends it after a settings change: other defaults, a new object. */
function refetched(schema: TabSchema): TabSchema {
  return { ...schema, core: { ...schema.core, defaults: { ...schema.core.defaults, cfg_scale: 6 } as TabSchema["core"]["defaults"] } }
}

describe("params pane", () => {
  // A refetched schema showed skeletons for one commit, which unmounted the whole form: the
  // field being typed in lost focus (and every section its open state).
  it("keeps the form mounted, and the focused field focused, when the schema is refetched", () => {
    const schema = tabSchema("txt2img")
    const { rerender } = render(wrap(<ParamsPane schema={schema} tab="txt2img" />))
    const steps = screen.getByRole("textbox", { name: "Sampling steps" })
    steps.focus()

    rerender(wrap(<ParamsPane schema={refetched(schema)} tab="txt2img" />))
    expect(screen.getByRole("textbox", { name: "Sampling steps" })).toBe(steps)
    expect(steps).toHaveFocus()
    expect(getFormStore("txt2img").getState().defaults.core?.cfg_scale).toBe(6) // and the new schema is loaded
  })
})
