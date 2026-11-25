import { toast } from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { anyModified, createFormStore, draftKey, isModified } from "./form-store"
import type { OwnerSpec } from "./form-index"
import { inputsIn } from "./form-index"
import { tabOwners } from "./tab"

const owners: OwnerSpec[] = [
  {
    owner: "core",
    layout: {
      type: "column",
      children: [
        { type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1, integer: true },
        { type: "choice", name: "sampler", label: "Sampler", choices: [{ value: "Euler", label: "Euler" }, { value: "DDIM", label: "DDIM" }] },
        { type: "image", name: "init", label: "Init" },
        { type: "text", name: "prompt", label: "Prompt" },
      ],
    },
    defaults: { steps: 20, sampler: "Euler", init: null, prompt: "" },
  },
  {
    owner: "ext",
    layout: {
      type: "section",
      title: "Ext",
      enable: { type: "toggle", name: "enable", label: "Enable" },
      children: [{ type: "multichoice", name: "tags", label: "Tags", choices: [{ value: "a", label: "a" }] }],
    },
    defaults: { enable: false, tags: [] },
  },
]

describe("form store", () => {
  it("names the values a lock kept out of a programmatic write", () => {
    const store = createFormStore("t-locks", { persistDrafts: false })
    store.getState().load([
      ...owners,
      { owner: "pin", layout: { type: "toggle", name: "steps", label: "Pin steps", locks: "core.steps" }, defaults: { steps: false } },
    ])
    expect(store.getState().setValues("core", { steps: 30, prompt: "a cat" })).toEqual([])
    store.getState().setValue("pin", "steps", true)
    expect(store.getState().setValues("core", { steps: 40, prompt: "a dog" })).toEqual(["steps"])
    expect(store.getState().values.core).toMatchObject({ steps: 30, prompt: "a dog" })
    // the value it already has: nothing was kept from changing
    expect(store.getState().setValues("core", { steps: 30 })).toEqual([])
    expect(
      store.getState().applyPatches([
        ["core", "steps", { value: 50, maximum: 200 }],
        ["core", "prompt", { value: "a fox" }],
        ["pin", "steps", { enabled: false }],
      ]),
    ).toEqual(["core.steps"])
    expect(store.getState().applyPatches([["core", "steps", { value: 30 }]])).toEqual([])
    expect(store.getState().values.core).toMatchObject({ steps: 30, prompt: "a fox" })
  })

  // the form's own helpers (size presets, styles, a card into the prompt, …) have no toast of their own
  describe("what a lock kept from a write", () => {
    afterEach(() => vi.restoreAllMocks())

    const locked = () => {
      const store = createFormStore("t-lock-notes", { persistDrafts: false })
      store.getState().load([
        ...owners,
        { owner: "pin", layout: { type: "toggle", name: "steps", label: "Pin steps", locks: "core.steps" }, defaults: { steps: true } },
        { owner: "keep", layout: { type: "toggle", name: "prompt", label: "Keep prompt", locks: "core.prompt" }, defaults: { prompt: true } },
      ])
      return store
    }

    it("is named in a toast of its own, once for the writes of one task", async () => {
      const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
      const store = locked()
      store.getState().setValues("core", { steps: 40, sampler: "DDIM" })
      store.getState().applyPatches([["core", "prompt", { value: "a dog" }]])
      expect(info).not.toHaveBeenCalled()
      await Promise.resolve()
      // one id per form: a later write (a drag writes on every move) replaces the toast
      expect(info).toHaveBeenCalledExactlyOnceWith("Overridden by a lock: Steps and Prompt", { id: "locks:t-lock-notes" })
      expect(store.getState().values.core).toMatchObject({ steps: 20, sampler: "DDIM", prompt: "" })
    })

    it("is left to a caller that names it in its own toast, or to an automatic fill (quiet)", async () => {
      const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
      const store = locked()
      expect(store.getState().setValues("core", { steps: 40 }, { quiet: true })).toEqual(["steps"])
      expect(store.getState().applyPatches([["core", "prompt", { value: "a dog" }]], { quiet: true })).toEqual(["core.prompt"])
      // a write giving a locked param the value it has kept nothing
      store.getState().setValues("core", { steps: 20 })
      await Promise.resolve()
      expect(info).not.toHaveBeenCalled()
    })
  })

  it("loads defaults per owner, including section enable toggles", () => {
    const store = createFormStore("t1", { persistDrafts: false })
    store.getState().load(owners)
    const s = store.getState()
    expect(s.values.core).toEqual({ steps: 20, sampler: "Euler", init: null, prompt: "" })
    expect(s.values.ext).toEqual({ enable: false, tags: [] })
    expect(s.index.inputs.get("ext")?.has("enable")).toBe(true)
  })

  it("tracks modified state and resets fields and sections", () => {
    const store = createFormStore("t2", { persistDrafts: false })
    store.getState().load(owners)
    store.getState().setValue("core", "steps", 30)
    store.getState().setValue("ext", "enable", true)
    store.getState().setValue("ext", "tags", ["a"])
    expect(isModified(store.getState(), "core", "steps")).toBe(true)
    expect(isModified(store.getState(), "core", "sampler")).toBe(false)

    store.getState().reset("core", ["steps"])
    expect(store.getState().values.core!.steps).toBe(20)

    const sectionNames = inputsIn(owners[1]!.layout).map((n) => n.name)
    expect(sectionNames).toEqual(["enable", "tags"])
    expect(anyModified(store.getState(), "ext", sectionNames)).toBe(true)
    store.getState().reset("ext", sectionNames)
    expect(store.getState().values.ext).toEqual({ enable: false, tags: [] })
  })

  it("restores compatible drafts and drops incompatible or upload values", () => {
    localStorage.setItem(
      draftKey("t3"),
      JSON.stringify({
        state: {
          values: {
            core: { steps: 999, sampler: "Gone", init: "upload-1", prompt: "a cat" },
            ext: { enable: "yes", tags: ["a", "zzz"] },
            removed: { x: 1 },
          },
          open: { "ext::enable": true },
          tabs: {},
        },
        version: 1,
      }),
    )
    const store = createFormStore("t3")
    store.getState().load(owners)
    const s = store.getState()
    expect(s.values.core!.steps).toBe(150) // clamped to the slider range
    expect(s.values.core!.sampler).toBe("Euler") // not a choice any more
    expect(s.values.core!.init).toBeNull() // uploads are never restored
    expect(s.values.core!.prompt).toBe("a cat")
    expect(s.values.ext!.enable).toBe(false) // wrong type
    expect(s.values.ext!.tags).toEqual(["a"]) // unknown choice filtered
    expect(s.values.removed).toBeUndefined()
    expect(s.open["ext::enable"]).toBe(true)
  })

  // A multichoice that allows custom entries took a stored list as it was: a number or null from a
  // damaged draft reached the chips widget, which expects text.
  it("restores only text entries of a draft's custom multichoice", () => {
    const custom: OwnerSpec[] = [
      { owner: "core", layout: { type: "multichoice", name: "tags", label: "Tags", allow_custom: true, choices: [{ value: "a", label: "a" }] }, defaults: { tags: [] } },
    ]
    localStorage.setItem(draftKey("t-custom"), JSON.stringify({ state: { values: { core: { tags: [1, "mine", null, "a", { x: 1 }] } } }, version: 1 }))
    const store = createFormStore("t-custom")
    store.getState().load(custom)
    expect(store.getState().values.core!.tags).toEqual(["mine", "a"])
  })

  it.each([
    ["values: null", { values: null, open: {}, tabs: {} }],
    ["values: a string", { values: "x", open: null, tabs: 3 }],
    ["an owner's values: null", { values: { core: null, ext: 5 } }],
    ["state: null", null],
  ])("loads the defaults from a damaged draft (%s)", (_name, state) => {
    localStorage.setItem(draftKey("t-damaged"), JSON.stringify({ state, version: 1 }))
    const store = createFormStore("t-damaged")
    expect(() => store.getState().load(owners)).not.toThrow()
    expect(store.getState().values.core!.prompt).toBe("")
    expect(store.getState().open).toEqual(expect.any(Object))
    expect(store.getState().tabs).toEqual(expect.any(Object))
  })

  // A selected tab stored as a number (0, not "0") selected no tab: the strip showed no panel.
  it("keeps only a draft's tab choices and open states of their own kind", () => {
    localStorage.setItem(draftKey("t-kinds"), JSON.stringify({ state: { values: {}, open: { a: true, b: "yes", c: 1 }, tabs: { x: "1", y: 0, z: null, w: "01", v: "-1" } }, version: 1 }))
    const store = createFormStore("t-kinds")
    expect(store.getState().open).toEqual({ a: true })
    expect(store.getState().tabs).toEqual({ x: "1" })
  })

  it("persists drafts (debounced) without upload ids", async () => {
    const store = createFormStore("t4")
    store.getState().load(owners)
    store.getState().setValue("core", "prompt", "hello")
    store.getState().setValue("core", "init", "upload-9")
    expect(localStorage.getItem(draftKey("t4"))).toBeNull()
    await new Promise((resolve) => setTimeout(resolve, 400))
    const saved = JSON.parse(localStorage.getItem(draftKey("t4")) ?? "{}")
    expect(saved.state.values.core!.prompt).toBe("hello")
    expect(saved.state.values.core).not.toHaveProperty("init")
  })

  it("applies patches: values and display overrides", () => {
    const store = createFormStore("t5", { persistDrafts: false })
    store.getState().load(owners)
    store.getState().applyPatches([
      ["core", "steps", { value: 40, maximum: 80 }],
      ["ext", "tags", { visible: false, choices: [{ value: "b", label: "B" }] }],
    ])
    const s = store.getState()
    expect(s.values.core!.steps).toBe(40)
    expect(s.overrides.core!.steps).toEqual({ maximum: 80 })
    expect(s.overrides.ext!.tags).toEqual({ visible: false, choices: [{ value: "b", label: "B" }] })
  })

  it("keeps current values when the schema reloads", () => {
    const store = createFormStore("t6", { persistDrafts: false })
    store.getState().load(owners)
    store.getState().setValue("core", "prompt", "keep me")
    store.getState().load(owners)
    expect(store.getState().values.core!.prompt).toBe("keep me")
    expect(store.getState().revision).toBe(2)
  })

  it("loads the full mock tab schema", () => {
    const store = createFormStore("t7", { persistDrafts: false })
    const schema = tabSchema("txt2img")
    store.getState().load(tabOwners(schema))
    const s = store.getState()
    expect(s.values.anima!.preset).toBe("balanced")
    expect(s.values["misc-ext"]?.pair).toEqual([0.5, "auto"])
    expect(s.values.sampler).toEqual({ steps: 28, sampler_name: "Euler a", scheduler: "Automatic" })
    expect(s.index.slots).toEqual(new Set(["sampler", "seed", "accordions", "scripts"]))
  })

  it("rounds and clamps numeric drafts like the server and never restores hidden state", () => {
    const numbers: OwnerSpec[] = [
      {
        owner: "core",
        layout: {
          type: "column",
          children: [
            { type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1, integer: true },
            { type: "number", name: "shift", label: "Shift", minimum: 0, maximum: 10, precision: 2 },
            { type: "state", name: "internal", label: "" },
          ],
        },
        defaults: { steps: 20, shift: 3, internal: { v: 1 } },
      },
    ]
    localStorage.setItem(
      draftKey("t8"),
      JSON.stringify({ state: { values: { core: { steps: 27.6, shift: 2.34567, internal: { v: 99 } } }, open: {}, tabs: {} }, version: 1 }),
    )
    const store = createFormStore("t8")
    store.getState().load(numbers)
    expect(store.getState().values.core).toEqual({ steps: 28, shift: 2.35, internal: { v: 1 } })
  })

  it("keeps validation errors per field, reveals them and clears them on edit", () => {
    const nested: OwnerSpec[] = [
      {
        owner: "core",
        layout: {
          type: "column",
          children: [
            {
              type: "section",
              title: "Hires",
              children: [
                {
                  type: "tabs",
                  children: [
                    { type: "tab", title: "A", children: [] },
                    { type: "tab", title: "B", children: [{ type: "group", title: "More", open: false, children: [{ type: "text", name: "deep", label: "Deep" }] }] },
                  ],
                },
              ],
            },
            { type: "text", name: "top", label: "Top" },
          ],
        },
        defaults: { deep: "", top: "" },
      },
    ]
    const store = createFormStore("t9", { persistDrafts: false })
    store.getState().load(nested)
    store.getState().setErrors([
      { owner: "core", param: "deep", message: "too deep" },
      { owner: "core", param: "deep", message: "really" },
      { owner: "core", param: "top", message: "bad" },
      { owner: "sampler", param: "steps", message: "no such field here" },
    ])
    const s = store.getState()
    expect(s.errors.core).toEqual({ deep: [{ message: "too deep" }, { message: "really" }], top: [{ message: "bad" }] })
    expect(s.errors.sampler).toEqual({ steps: [{ message: "no such field here" }] })
    // The section, the titled group and the second tab are opened.
    expect(s.open).toEqual({ "core::Hires": true, "core::More::group": true })
    expect(s.tabs).toEqual({ "core::tabs::A|B": "1" })

    store.getState().setValue("core", "top", "fixed")
    expect(store.getState().errors.core).toEqual({ deep: [{ message: "too deep" }, { message: "really" }] })
    store.getState().reset("core", ["deep"])
    expect(store.getState().errors.core).toEqual({})
    store.getState().clearErrors()
    expect(store.getState().errors).toEqual({})
  })

  it("does not persist validation errors and clears them on load", () => {
    const store = createFormStore("t10", { persistDrafts: false })
    store.getState().load(owners, { conflictParams: ["core.sampler"] })
    expect(store.getState().conflictParams.has("core.sampler")).toBe(true)
    store.getState().setErrors([{ owner: "core", param: "steps", message: "x" }])
    store.getState().load(owners)
    expect(store.getState().errors).toEqual({})
    expect(store.getState().conflictParams.size).toBe(0)
  })
})
