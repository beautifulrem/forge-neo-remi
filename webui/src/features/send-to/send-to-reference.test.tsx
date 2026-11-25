import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap, TabSchema } from "@/api/types"
import { navigate } from "@/app/router"
import type { ResultItem } from "@/features/gallery/session-store"
import { bootstrapWith } from "@/test/api-fixtures"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { ResultActions } from "./ResultActions"
import { sendToReferenceWithFeedback } from "./result-actions"
import { findReferenceTarget, sendToReference } from "./send-to"

function schemaWithImageStitch(tab: "txt2img" | "img2img"): TabSchema {
  const base = tabSchema(tab)
  return {
    ...base,
    scripts: [
      ...base.scripts,
      {
        id: "imagestitch_integrated",
        title: "ImageStitch Integrated",
        kind: "alwayson",
        order: 529,
        layout: {
          type: "section",
          title: "ImageStitch Integrated",
          enable: { type: "toggle", name: "enable", default: false, label: "ImageStitch Integrated" },
          children: [{ type: "image_list", name: "references", label: "Reference Image(s)", widget: "image-list" }],
        },
        defaults: { enable: false, references: [] },
        args: ["enable", "references"],
      },
    ],
  }
}

beforeEach(() => {
  navigate({ name: "generate", tab: "txt2img" }, { replace: true })
  getFormStore("txt2img").getState().load(tabOwners(schemaWithImageStitch("txt2img")))
  queryClient.setQueryData<Bootstrap>(keys.app, bootstrapWith({}))
})

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe("send-to Reference Image (ImageStitch)", () => {
  it("prefers imagestitch_integrated owner over other image_list scripts", () => {
    const target = findReferenceTarget(getFormStore("txt2img").getState().index)
    expect(target).toEqual({ owner: "imagestitch_integrated", param: "references", enableParam: "enable" })
  })

  it("appends references cleanly without clobbering existing ones and enables ImageStitch", async () => {
    const store = getFormStore("txt2img")
    store.getState().setValues("imagestitch_integrated", { references: ["existing-1"], enable: false })

    const first = await sendToReference([{ fileId: "new-2", width: 512, height: 512 }], "txt2img")
    expect(first).toEqual({ tab: "txt2img", owner: "imagestitch_integrated", added: 1, total: 2, locked: [] })
    expect(store.getState().values.imagestitch_integrated).toEqual({
      enable: true,
      references: ["existing-1", "new-2"],
    })

    // Sending duplicates + a third image only appends the unseen one without clobbering.
    const second = await sendToReference([{ fileId: "existing-1" }, { fileId: "new-3" }], "txt2img")
    expect(second).toEqual({ tab: "txt2img", owner: "imagestitch_integrated", added: 1, total: 3, locked: [] })
    expect(store.getState().values.imagestitch_integrated?.references).toEqual(["existing-1", "new-2", "new-3"])
  })

  it("names the references a lock kept, in the send's own toast, and counts none added", async () => {
    const store = getFormStore("txt2img")
    // ImageStitch is off: nothing added, so it stays off
    store.getState().setValues("imagestitch_integrated", { references: ["existing-1"], enable: false })
    // a toggle that locks the reference list is on
    store.setState((s) => ({ locks: [{ owner: "keep", name: "refs", target: ["imagestitch_integrated", "references"] }], values: { ...s.values, keep: { refs: true } } }))
    const info = vi.spyOn(toast, "info")
    const success = vi.spyOn(toast, "success")
    expect(await sendToReference([{ fileId: "new-2" }], "txt2img")).toEqual({ tab: "txt2img", owner: "imagestitch_integrated", added: 0, total: 1, locked: ["imagestitch_integrated.references"] })
    expect(store.getState().values.imagestitch_integrated).toEqual({ references: ["existing-1"], enable: false })
    sendToReferenceWithFeedback([{ fileId: "new-3" }], "txt2img")
    await waitFor(() => expect(info).toHaveBeenCalledWith("Overridden by a lock: Reference Image(s)"))
    expect(info).toHaveBeenCalledOnce() // not "already in Reference Images", nor a second note
    expect(success).not.toHaveBeenCalled()
  })

  it("waits for schema load when target form has not loaded yet", async () => {
    getFormStore("img2img").setState({ revision: 0 })
    const promise = sendToReference([{ fileId: "ref-a" }, { fileId: "ref-b" }], "img2img")
    expect(location.hash).toBe("#/img2img")
    getFormStore("img2img").getState().load(tabOwners(schemaWithImageStitch("img2img")))
    const res = await promise
    expect(res).toEqual({ tab: "img2img", owner: "imagestitch_integrated", added: 2, total: 2, locked: [] })
    expect(getFormStore("img2img").getState().values.imagestitch_integrated?.references).toEqual(["ref-a", "ref-b"])
  })

  it("exposes Reference image (ImageStitch) inside ResultActions Send to menu", async () => {
    const item: ResultItem = { key: "j:0", fileId: "out-99", jobId: "j", tab: "txt2img", index: 0, infotext: null, createdAt: 1 }
    const onSent = vi.fn()
    render(
      <QueryClientProvider client={queryClient}>
        <ResultActions item={item} tab="txt2img" onSent={onSent} />
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole("button", { name: /Send to/ }))
    const menu = await screen.findByRole("menu")
    fireEvent.click(within(menu).getByRole("menuitem", { name: /Reference image \(ImageStitch\)/ }))
    expect(onSent).toHaveBeenCalledTimes(1)
    expect(getFormStore("txt2img").getState().values.imagestitch_integrated?.references).toEqual(["out-99"])
  })
})
