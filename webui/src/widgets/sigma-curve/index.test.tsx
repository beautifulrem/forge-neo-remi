import { act, fireEvent, render, screen } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, describe, expect, it, vi } from "vitest"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore, type FormStore } from "@/ui/form-store"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import { curvePath, daemonState } from "./curve"
import SigmaCurve from "./index"
import zh from "./locales/zh-CN.json"

const OWNER = "detail_daemon"
const DAEMONS = [
  { prefix: "d1_", title: "I", value: "1" },
  { prefix: "d2_", title: "II", value: "2" },
]
const PROPS = { daemons: DAEMONS, tab: "daemon", enable: "enabled" }
const NODE: CustomNode = { type: "custom", name: "curves", label: "", widget: "sigma-curve", props: PROPS, default: null }

function setup(values: Record<string, unknown>, { lockTab = false } = {}): FormStore {
  const store = createFormStore(`sigma-curve-${Math.random()}`, { persistDrafts: false })
  store.getState().load([
    { owner: OWNER, layout: { type: "choice", name: "daemon", label: "Daemon", choices: DAEMONS.map(({ value }) => ({ value, label: value })) }, defaults: { enabled: true, daemon: "1", d1_active: true, d1_amount: 0.1, d2_amount: -0.4, ...values } },
    ...(lockTab ? [{ owner: "pin", layout: { type: "toggle" as const, name: "daemon", label: "Keep daemon", locks: `${OWNER}.daemon` }, defaults: { daemon: true } }] : []),
  ])
  render(
    <FormProvider formId="txt2img" store={store}>
      <OwnerProvider owner={OWNER}>
        <SigmaCurve value={null} onChange={() => {}} props={PROPS} node={NODE} disabled={false} id="curves" label="" />
      </OwnerProvider>
    </FormProvider>,
  )
  return store
}

describe("sigma-curve widget", () => {
  afterEach(() => vi.restoreAllMocks())

  it("draws the selected daemon large and every daemon as a thumbnail", () => {
    setup({})
    const main = screen.getByRole("img", { name: /Daemon I: active, amount 0.1/ })
    expect(main.querySelector("path")?.getAttribute("d")).toMatch(/^M0\.00,50\.00 /)
    expect(main.getAttribute("class")).toContain("text-success")
    const thumbs = screen.getAllByRole("button")
    expect(thumbs).toHaveLength(2)
    expect(thumbs[0]).toHaveAttribute("aria-pressed", "true")
  })

  it("selects a daemon's tab from its thumbnail and follows its params", () => {
    const store = setup({ d2_active: true })
    fireEvent.click(screen.getByRole("button", { name: /Daemon II/ }))
    expect(store.getState().values[OWNER]?.daemon).toBe("2")
    // Daemon II smooths (negative amount): drawn in the destructive colour
    expect(screen.getByRole("img", { name: /Daemon II: active, amount -0.4/ }).getAttribute("class")).toContain("text-destructive")
  })

  it("leaves a locked tab and names the lock", async () => {
    const info = vi.spyOn(toast, "info").mockImplementation((() => "") as never)
    const store = setup({ d2_active: true }, { lockTab: true })
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Daemon II/ }))
    })
    expect(store.getState().values[OWNER]?.daemon).toBe("1")
    expect(info).toHaveBeenCalledExactlyOnceWith("Overridden by a lock: Daemon", expect.anything())
  })

  it("greys out inactive daemons and the whole preview when the script is off", () => {
    setup({ enabled: false })
    expect(screen.getByRole("img", { name: /Daemon I/ }).getAttribute("class")).toContain("text-muted-foreground")
  })

  it("reads params with defaults for anything missing", () => {
    const state = daemonState({ d1_start: 0.1, d1_smooth: false, d1_mode: "cond" }, "d1_")
    expect(state).toMatchObject({ start: 0.1, end: 0.8, smooth: false, mode: "cond", active: false })
    expect(curvePath([1, -1], 10, 4)).toBe("M0.00,0.00 L10.00,4.00")
  })

  // The mode was written in capitals ("COND"): sentence case everywhere, so it shows as the choice
  // does, translated where the extension's catalog has it.
  it("names a daemon's mode as its choice does, never in capitals", () => {
    setup({ d1_mode: "cond", d2_mode: "uncond" })
    expect(screen.getByText(/· cond$/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Daemon II/ })).toHaveTextContent(/II uncond/)
    expect(screen.queryByText(/COND/)).toBeNull()
  })

  it("has a zh-CN translation for every chrome string", () => {
    const sources = import.meta.glob<string>(["./*.tsx", "./*.ts", "!./*.test.*"], { query: "?raw", import: "default", eager: true })
    const keys = Object.values(sources).flatMap(widgetKeys)
    expect(keys.length).toBeGreaterThan(0)
    expect(keys.filter((key) => !(zh as Record<string, string>)[key])).toEqual([])
  })
})
