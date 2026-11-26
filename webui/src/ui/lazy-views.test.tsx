import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { createFormStore } from "./form-store"
import { TabForm } from "./SchemaForm"
import type { Node, OwnerDefaults, TabSchema } from "./schema"

// No preloadViews() here: this file checks the deferred path itself.

function renderNode(node: Node, defaults: OwnerDefaults) {
  const schema: TabSchema = { version: 1, tab: "txt2img", core: { owner: "core", layout: { type: "column", children: [node] }, defaults }, scripts: [] }
  const store = createFormStore(`lazy-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  return render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
}

const slow = { timeout: 15_000 }

describe("deferred node views", { timeout: 30_000 }, () => {
  it("show a placeholder until their code has loaded, then render at once on later mounts", async () => {
    const { container } = renderNode({ type: "color", name: "tint", label: "Tint" }, { tint: "#ff0000" })
    // first render: the view's chunk is still loading
    expect(screen.queryByLabelText("Pick a color for Tint")).toBeNull()
    expect(container.querySelector('[data-slot="skeleton"]')).not.toBeNull()
    expect(await screen.findByLabelText("Pick a color for Tint", {}, slow)).toHaveAttribute("type", "color")
    cleanup()
    // loaded once: a new form renders it without the placeholder
    renderNode({ type: "color", name: "tint", label: "Tint" }, { tint: "#00ff00" })
    expect(screen.getByLabelText("Pick a color for Tint")).toHaveAttribute("type", "color")
  })

  // An image field's placeholder was a 160 px block where the empty field (its label over the drop
  // zone) is 122 px: what came after it jumped up when the code arrived.
  it("stand in for an empty image field with its label over the drop zone", async () => {
    // A mask painted over its source renders nothing (the editor shows it): it took a 160 px block
    // next to the editor's while their code loaded.
    const editor = renderNode(
      {
        type: "column",
        children: [
          { type: "image", name: "init", label: "Image" },
          { type: "image", name: "mask", label: "Mask", kind: "mask", source: "init" },
        ],
      },
      { init: null, mask: null },
    )
    expect(editor.container.querySelectorAll("[data-slot=skeleton]")).toHaveLength(1)
    editor.unmount()
    const { container } = renderNode({ type: "image", name: "source", label: "Source" }, { source: null })
    const classes = (element: Element) => element.className.split(/\s+/)
    expect(classes(container.querySelector("[data-part=label]")!)).toContain("h-5")
    const zone = container.querySelector("[data-part=drop-zone]")!
    expect(classes(zone)).toEqual(expect.arrayContaining(["rounded-lg", "border", "py-6"]))
    expect(container.querySelector(".h-40")).toBeNull()
    expect(await screen.findByRole("button", { name: /^Source / }, slow)).toBeInTheDocument()
    expect(container.querySelector("[data-part=drop-zone]")).toBeNull()
  })

  // A tabs node held a 96 px block while its code loaded, where the strip (32 px) over the selected
  // tab's content is often twice that (Extras: the source image under its four tabs): the strip's
  // stand-in is as tall as the strip, and the selected tab's content renders under it as it will.
  it("stand in for tabs with the strip over the selected tab's content", async () => {
    const tabs = (bind?: string) => ({
      type: "tabs" as const,
      ...(bind ? { bind } : {}),
      children: [
        { type: "tab" as const, title: "First", value: "A", children: [{ type: "toggle" as const, name: "alpha", label: "Alpha" }] },
        { type: "tab" as const, title: "Second", value: "B", children: [{ type: "toggle" as const, name: "beta", label: "Beta" }] },
      ],
    })
    const mode = { type: "choice" as const, name: "mode", label: "Mode", hidden: true, choices: [{ value: "A", label: "A" }, { value: "B", label: "B" }] }
    const bound = renderNode({ type: "column", children: [mode, tabs("mode")] }, { mode: "B", alpha: false, beta: true })
    const strip = bound.container.querySelector("[data-part=tab-strip]")!
    expect(strip.className.split(/\s+/)).toContain("h-8")
    expect(screen.getByRole("switch", { name: "Beta" })).toBeChecked()
    expect(screen.queryByRole("switch", { name: "Alpha" })).toBeNull()
    expect(bound.container.querySelector(".h-24")).toBeNull()
    bound.unmount()
    // a bound value no tab has: the strip shows with no tab's content, as the strip itself does
    const unmatched = renderNode({ type: "column", children: [mode, tabs("mode")] }, { mode: "Z", alpha: false, beta: true })
    expect(unmatched.container.querySelector("[data-part=tab-strip]")).not.toBeNull()
    expect(screen.queryByRole("switch")).toBeNull()
    unmatched.unmount()
    // unbound: the first tab until another is picked
    const { container } = renderNode(tabs(), { alpha: false, beta: true })
    expect(screen.getByRole("switch", { name: "Alpha" })).toBeInTheDocument()
    expect(screen.queryByRole("switch", { name: "Beta" })).toBeNull()
    expect(await screen.findByRole("tablist", {}, slow)).toBeInTheDocument()
    expect(container.querySelector("[data-part=tab-strip]")).toBeNull()
    expect(screen.getByRole("switch", { name: "Alpha" })).toBeInTheDocument()
  })

  it("keeps the eagerly rendered kinds synchronous", () => {
    renderNode({ type: "toggle", name: "on", label: "Enable thing" }, { on: false })
    expect(screen.getByRole("switch", { name: "Enable thing" })).toBeInTheDocument()
  })
})
