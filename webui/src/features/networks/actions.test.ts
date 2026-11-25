import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type { NetworkItem } from "@/api/types"
import { getFormStore } from "@/ui/form-store"
import { applyItem } from "./actions"

const checkpoint = (extra: Partial<NetworkItem> = {}): NetworkItem => ({
  name: "anima",
  title: "anima",
  kind: "checkpoints",
  path: "anima.safetensors",
  folder: "",
  tags: [],
  preferred_modules: [],
  search_terms: [],
  editable: true,
  hidden: false,
  value: "anima.safetensors [abc]",
  ...extra,
})
const KIND = { action: "checkpoint", negative: false } as const

describe("applying a prompt card", () => {
  afterEach(() => getFormStore("img2img").setState({ values: {}, locks: [] }))

  // a LoRA card with negative text: its two texts go in together or not at all
  it("adds nothing, and claims nothing, while a lock keeps a prompt it would change", () => {
    const store = getFormStore("img2img")
    store.setState({ values: { core: { prompt: "a cat", negative_prompt: "" }, keep: { prompt: true } }, locks: [{ owner: "keep", name: "prompt", target: ["core", "prompt"] }] })
    const lora = checkpoint({ kind: "lora", name: "moss", title: "moss", prompt: "<lora:moss:1>", negative_prompt: "blurry", value: undefined })
    expect(applyItem(lora, { action: "prompt", negative: true }, "img2img", "positive")).toBeNull()
    expect(store.getState().values.core).toEqual({ prompt: "a cat", negative_prompt: "" })
  })
})

describe("applying a checkpoint card", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("loads the model and keeps the modules when it prefers none", () => {
    const patch = vi.spyOn(api, "patchOptions").mockResolvedValue({} as never)
    expect(applyItem(checkpoint(), KIND, "txt2img", "positive")).toEqual({ added: true })
    expect(patch).toHaveBeenCalledWith({ sd_model_checkpoint: "anima.safetensors [abc]" })
  })

  it("selects its preferred VAE / text encoders in the same change (upstream vae_te)", () => {
    const patch = vi.spyOn(api, "patchOptions").mockResolvedValue({} as never)
    applyItem(checkpoint({ preferred_modules: ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"] }), KIND, "txt2img", "positive")
    expect(patch).toHaveBeenCalledTimes(1)
    expect(patch).toHaveBeenCalledWith({ sd_model_checkpoint: "anima.safetensors [abc]", forge_additional_modules: ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"] })
  })
})
