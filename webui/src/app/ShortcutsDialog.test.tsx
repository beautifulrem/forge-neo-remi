import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import { VIEWER_KEYS } from "@/features/lightbox/keys"
import { IDLE, type RunEvent, reduce, type RunState } from "@/features/progress/run-machine"
import zhCN from "@/locales/zh-CN.json"
import ShortcutsDialog from "./ShortcutsDialog"
import { useShortcutsHelp } from "./shortcuts-help"

function openWith(ctrlEnterInterrupt: boolean) {
  const app = structuredClone(bootstrap)
  app.options = { ...app.options, ctrl_enter_interrupt: ctrlEnterInterrupt }
  queryClient.setQueryData(keys.app, app)
  vi.spyOn(api, "app").mockResolvedValue(app)
  useShortcutsHelp.setState({ open: true })
  render(
    <QueryClientProvider client={queryClient}>
      <ShortcutsDialog />
    </QueryClientProvider>,
  )
}

/** The keys a group of the dialog lists, as shown. */
function keysOf(group: string): string[] {
  const section = screen.getByRole("region", { name: group })
  // one Kbd per key, inside a KbdGroup (a kbd of its own)
  return Array.from(section.querySelectorAll("kbd"))
    .filter((kbd) => kbd.querySelector("kbd") === null)
    .map((kbd) => kbd.textContent ?? "")
}

/** What Ctrl/Cmd+Enter does to a running job (the run machine the shortcut feeds). */
function shortcutWhileRunning(interruptOnly: boolean) {
  const events: RunEvent[] = [{ type: "generate" }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }]
  const running = events.reduce<RunState>((state, event) => reduce(state, event, { interruptOnly })[0], IDLE)
  const [after, effects] = reduce(running, { type: "shortcut" }, { interruptOnly })
  return { restart: after.restart, effects: effects.map((effect) => effect.type) }
}

describe("keyboard shortcuts dialog", () => {
  afterEach(() => {
    useShortcutsHelp.setState({ open: false })
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("describes Ctrl/Cmd+Enter during a run as the run machine handles it", () => {
    expect(shortcutWhileRunning(false)).toEqual({ restart: true, effects: ["interrupt"] })
    openWith(false)
    expect(screen.getByText("Generate (while running: interrupt and generate again)")).toBeTruthy()
  })

  it("follows the setting that makes Ctrl/Cmd+Enter only interrupt", () => {
    expect(shortcutWhileRunning(true)).toEqual({ restart: false, effects: ["interrupt"] })
    openWith(true)
    expect(screen.getByText("Generate (while running: interrupt)")).toBeTruthy()
    expect(screen.queryByText(/queue another/)).toBeNull()
  })

  // B (the RGBA backdrop) worked in the viewer but was listed nowhere
  it("lists every key the image viewer handles", () => {
    openWith(false)
    const shown = new Set(keysOf("Image viewer"))
    const label: Record<string, string> = { ArrowLeft: "←", ArrowRight: "→", "=": "+" }
    for (const key of Object.keys(VIEWER_KEYS)) expect(shown, key).toContain(label[key] ?? (key.length === 1 ? key.toUpperCase() : key))
    expect(within(screen.getByRole("region", { name: "Image viewer" })).getByText("Cycle RGBA backdrop")).toBeInTheDocument()
  })

  // the dialog's labels are constants translated at render: the locale test does not see them
  it.each([false, true])("has a zh-CN translation for every group and entry (interrupt only: %s)", (interruptOnly) => {
    openWith(interruptOnly)
    const zh = zhCN as Record<string, string>
    const dialog = screen.getByRole("dialog")
    const texts = [...Array.from(dialog.querySelectorAll("section"), (section) => section.getAttribute("aria-label") ?? ""), ...Array.from(dialog.querySelectorAll("dt"), (dt) => dt.textContent ?? "")]
    expect(texts.filter((text) => !zh[text])).toEqual([])
  })

  // Extras' Ctrl/Cmd+Shift+Enter (add to queue, as on the Generate page) was listed nowhere
  it("lists the Extras page's keys", () => {
    openWith(false)
    const extras = within(screen.getByRole("region", { name: "Extras" }))
    for (const [label, key] of [
      ["Process", "Ctrl+Enter"],
      ["Add to queue", "Ctrl+Shift+Enter"],
      ["Interrupt", "Esc"],
    ]) expect(extras.getByText(label).nextElementSibling?.textContent).toBe(key)
  })

  // the gallery's select all, clear and remove keys (pages/gallery) were listed nowhere in the app
  it("lists the gallery's selection keys", () => {
    openWith(false)
    const gallery = within(screen.getByRole("region", { name: "Gallery" }))
    for (const [label, key] of [
      ["Select all images", "Ctrl+A"],
      ["Clear selection", "Esc"],
      ["Remove the selected images", "Delete"],
    ]) expect(gallery.getByText(label).nextElementSibling?.textContent).toBe(key)
  })
})
