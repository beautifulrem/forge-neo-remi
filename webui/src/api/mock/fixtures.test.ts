import { describe, expect, it } from "vitest"
import type { PageSchema } from "@/api/types"
import { toolboxPage } from "./fixtures"

/** The labels of a layout's `primary` buttons. */
function primaries(node: unknown): string[] {
  if (!node || typeof node !== "object") return []
  const { type, variant, label, children } = node as { type?: string; variant?: string; label?: string; children?: unknown[] }
  const own = type === "button" && variant === "primary" ? [label ?? ""] : []
  return [...own, ...(children ?? []).flatMap(primaries)]
}

describe("mock fixtures", () => {
  // A screen has one filled button (frontend.md). The renderer demotes an extension's primary only in
  // a generate form; the Toolbox page showed two, "Run" and the prompt kit's "Build prompt".
  it.each<[string, PageSchema]>([["toolbox", toolboxPage]])("the %s page has one filled button", (_, page) => {
    expect(primaries(page.layout)).toEqual(["Run"])
  })
})
