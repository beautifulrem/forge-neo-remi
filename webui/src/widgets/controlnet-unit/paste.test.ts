/** Pasted units merge into the unit they replace; a locked unit is not written, so nothing merges or reports twice. */
import { toast } from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { applyPaste } from "@/features/infotext/paste"
import { createFormStore } from "@/ui/form-store"
import { watchPastes } from "./paste"
import { PASTE_MARKER } from "./unit"

function unitForm(locked: boolean) {
  const store = createFormStore(`cn-paste-${locked}`, { persistDrafts: false })
  store.getState().load([
    { owner: "controlnet", layout: { type: "custom", name: "unit_0", label: "Unit 0", widget: "controlnet-unit", props: {} }, defaults: { unit_0: { enabled: true, image: "img", weight: 1 } } },
    { owner: "keep", layout: { type: "toggle", name: "unit", label: "Keep unit", locks: "controlnet.unit_0" }, defaults: { unit: locked } },
  ])
  watchPastes(store)
  return store
}

describe("pasted ControlNet units", () => {
  afterEach(() => vi.restoreAllMocks())

  it("merge into the unit they replace, and a lock keeps the unit as the paste reports", async () => {
    const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
    const pasted = { [PASTE_MARKER]: true, weight: 0.4 }

    const free = unitForm(false)
    expect(applyPaste({ store: free, tab: "txt2img" }, [{ owner: "controlnet", values: { unit_0: pasted } }])).toEqual({ count: 1, locked: [] })
    expect(free.getState().values.controlnet?.unit_0).toMatchObject({ enabled: true, image: "img", weight: 0.4 })

    const kept = unitForm(true)
    expect(applyPaste({ store: kept, tab: "txt2img" }, [{ owner: "controlnet", values: { unit_0: pasted } }])).toEqual({ count: 0, locked: ["controlnet.unit_0"] })
    expect(kept.getState().values.controlnet?.unit_0).toEqual({ enabled: true, image: "img", weight: 1 })
    // the paste's own toast names it: the merge writes nothing a lock keeps
    await Promise.resolve()
    expect(info).not.toHaveBeenCalled()
  })
})
