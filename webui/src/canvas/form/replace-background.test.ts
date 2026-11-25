/** Replacing the image a form's layers are painted over: the layers go with it, or neither changes. */
import { toast } from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createFormStore } from "@/ui/form-store"
import { replaceBackground } from "./replace-background"

function paintedForm(locked: boolean) {
  const store = createFormStore(`background-${locked}`, { persistDrafts: false })
  store.getState().load([
    {
      owner: "core",
      layout: {
        type: "group",
        children: [
          { type: "image", name: "init_img", label: "Image" },
          { type: "image", name: "mask", label: "Mask" },
        ],
      },
      defaults: { init_img: null, mask: null },
    },
    { owner: "keep", layout: { type: "toggle", name: "image", label: "Keep the image", locks: "core.init_img" }, defaults: { image: locked } },
  ])
  store.getState().setValue("core", "init_img", "old")
  store.getState().setValue("core", "mask", "paint")
  return store
}

const target = (store: ReturnType<typeof paintedForm>) => ({ store, formId: "background", owner: "core", background: "init_img", painted: ["mask"] })

describe("replacing the background", () => {
  afterEach(() => vi.restoreAllMocks())

  it("clears the painted layers with it", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({}))
    const store = paintedForm(false)
    await replaceBackground(target(store), "new")
    expect(store.getState().values.core).toEqual({ init_img: "new", mask: null })
  })

  it("changes nothing while a lock keeps the image, and names it", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({}))
    const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
    const store = paintedForm(true)
    await replaceBackground(target(store), "new")
    // the paint stays with the image it was painted over
    expect(store.getState().values.core).toEqual({ init_img: "old", mask: "paint" })
    await Promise.resolve()
    expect(info).toHaveBeenCalledExactlyOnceWith("Overridden by a lock: Image", { id: "locks:background-true" })
  })
})
