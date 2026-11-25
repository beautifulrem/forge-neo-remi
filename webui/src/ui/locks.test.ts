/** Naming the values a lock kept: one note for every path that writes values programmatically. */
import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { buildIndex } from "./form-index"
import { lockedLabels, lockNote } from "./locks"
import { tabOwners } from "./tab"

const index = buildIndex(tabOwners(tabSchema("txt2img")))

describe("lock notes", () => {
  it("name locked params by their labels, joined the language's way", () => {
    expect(lockedLabels(index, ["core.cfg_scale", "sampler.steps", "gone.x"], "en")).toBe("CFG Scale, Sampling steps, and gone.x")
  })

  it("say what a lock overrode, and nothing when no lock did", () => {
    expect(lockNote(index, ["core.width"], "en")).toBe("Overridden by a lock: Width")
    expect(lockNote(index, [], "en")).toBeUndefined()
  })
})
