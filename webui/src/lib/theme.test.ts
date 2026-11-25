import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

/** theme.ts reads the URL, the server's --theme and the stored pick when it loads. */
async function load({ stored, server, search = "" }: { stored?: object; server?: string | null; search?: string }) {
  vi.resetModules()
  localStorage.clear()
  if (stored) localStorage.setItem("remi-appearance", JSON.stringify(stored))
  window.__REMI_THEME__ = server
  window.history.replaceState(null, "", `/${search}`)
  return import("./theme")
}

describe("theme sources", () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => {
    delete window.__REMI_THEME__
    window.history.replaceState(null, "", "/")
  })

  it("defaults to dark, or the server's --theme while the user has not picked one", async () => {
    expect((await load({})).useAppearance.getState().theme).toBe("dark")
    const { useAppearance } = await load({ server: "light" })
    expect(useAppearance.getState()).toMatchObject({ theme: "light", themeChosen: false })
    // changing the density stores the state, but the theme still follows the server
    useAppearance.getState().setDensity("compact")
    const again = await load({ server: "dark", stored: JSON.parse(localStorage.getItem("remi-appearance")!) })
    expect(again.useAppearance.getState()).toMatchObject({ theme: "dark", density: "compact", themeChosen: false })
  })

  it("keeps the user's pick over --theme (also a pick stored by version 1)", async () => {
    const picked = await load({ server: "light", stored: { state: { theme: "dark", themeChosen: true, density: "comfortable", accent: "iris" }, version: 2 } })
    expect(picked.useAppearance.getState().theme).toBe("dark")
    const legacy = await load({ server: "dark", stored: { state: { theme: "light", density: "comfortable", accent: "iris" }, version: 1 } })
    expect(legacy.useAppearance.getState()).toMatchObject({ theme: "light", themeChosen: true })
  })

  it("moves version 2's default iris accent to blue, and keeps other accents", async () => {
    const old = await load({ server: "dark", stored: { state: { theme: "dark", themeChosen: false, density: "comfortable", accent: "iris" }, version: 2 } })
    expect(old.useAppearance.getState().accent).toBe("blue")
    const teal = await load({ server: "dark", stored: { state: { theme: "dark", themeChosen: false, density: "comfortable", accent: "teal" }, version: 2 } })
    expect(teal.useAppearance.getState().accent).toBe("teal")
    const picked = await load({ server: "dark", stored: { state: { theme: "dark", themeChosen: false, density: "comfortable", accent: "iris" }, version: 3 } })
    expect(picked.useAppearance.getState().accent).toBe("iris")
  })

  it("?__theme= picks the theme and remembers it; unknown values are ignored", async () => {
    const { useAppearance, urlTheme } = await load({ server: "dark", search: "?__theme=light" })
    expect(useAppearance.getState()).toMatchObject({ theme: "light", themeChosen: true })
    expect(JSON.parse(localStorage.getItem("remi-appearance")!).state.theme).toBe("light")
    expect(urlTheme("?__theme=system")).toBe("system")
    expect(urlTheme("?__theme=purple")).toBeNull()
    expect((await load({ search: "?__theme=purple" })).useAppearance.getState().theme).toBe("dark")
  })
})

describe("theme in effect", () => {
  /** A `prefers-color-scheme: dark` query whose answer the test changes. */
  function systemScheme(dark: boolean) {
    const listeners = new Set<() => void>()
    const media = {
      matches: dark,
      addEventListener: (_type: string, listener: () => void) => listeners.add(listener),
      removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener),
    }
    vi.spyOn(window, "matchMedia").mockImplementation((query) => {
      expect(query).toBe("(prefers-color-scheme: dark)")
      return media as unknown as MediaQueryList
    })
    return {
      listeners,
      set(value: boolean) {
        media.matches = value
        for (const listener of listeners) listener()
      },
    }
  }

  afterEach(() => {
    vi.restoreAllMocks()
    document.documentElement.className = ""
    delete document.documentElement.dataset.density
    delete document.documentElement.dataset.accent
  })

  it("resolves 'system' from the OS colour scheme and follows its changes", async () => {
    const { renderHook, act } = await import("@testing-library/react")
    const scheme = systemScheme(false)
    const { useAppearance, useTheme } = await load({})
    act(() => useAppearance.getState().setTheme("system"))
    const { result, unmount } = renderHook(() => useTheme())
    expect(result.current.theme).toBe("light")
    act(() => scheme.set(true))
    expect(result.current.theme).toBe("dark")
    // a picked theme wins over the OS
    act(() => result.current.setTheme("light"))
    expect(result.current.theme).toBe("light")
    expect(useAppearance.getState().themeChosen).toBe(true)
    unmount()
    expect(scheme.listeners.size).toBe(0)
  })

  it("mirrors theme, density and accent onto <html>", async () => {
    const { renderHook, act } = await import("@testing-library/react")
    systemScheme(true)
    const { useAppearance, useApplyAppearance } = await load({})
    const root = document.documentElement
    renderHook(() => useApplyAppearance())
    expect(root.classList.contains("dark")).toBe(true)
    expect(root.dataset).toMatchObject({ density: "comfortable", accent: "blue" })
    act(() => {
      useAppearance.getState().setTheme("light")
      useAppearance.getState().setDensity("compact")
      useAppearance.getState().setAccent("teal")
    })
    expect(root.classList.contains("dark")).toBe(false)
    expect(root.dataset).toMatchObject({ density: "compact", accent: "teal" })
    act(() => useAppearance.getState().setTheme("system"))
    expect(root.classList.contains("dark")).toBe(true)
  })

  it("paints the browser's theme-color with the header's background of the theme in effect", async () => {
    const { renderHook, act } = await import("@testing-library/react")
    systemScheme(false)
    const meta = document.head.appendChild(Object.assign(document.createElement("meta"), { name: "theme-color", content: "#0f1113" }))
    try {
      const { useAppearance, useApplyAppearance, THEME_COLORS } = await load({})
      renderHook(() => useApplyAppearance())
      expect(meta.content).toBe(THEME_COLORS.dark)
      act(() => useAppearance.getState().setTheme("light"))
      expect(meta.content).toBe(THEME_COLORS.light)
    } finally {
      meta.remove()
    }
  })

  it("keeps THEME_COLORS and index.html equal to each theme's --background, and the select chevron to its --muted-foreground", async () => {
    const { readFileSync } = await import("node:fs")
    const { THEME_COLORS } = await load({})
    const css = readFileSync("src/index.css", "utf8")
    const html = readFileSync("index.html", "utf8")
    const toHex = (L: number, C: number, h: number) => {
      const a = C * Math.cos((h * Math.PI) / 180)
      const b = C * Math.sin((h * Math.PI) / 180)
      const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
      const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
      const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
      const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s]
      const channel = (x: number) => Math.round(255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055))
      return `#${rgb.map((x) => channel(Math.min(1, Math.max(0, x))).toString(16).padStart(2, "0")).join("")}`
    }
    const backgrounds = [...css.matchAll(/^ {2}--background: oklch\(([\d.]+) ([\d.]+) ([\d.]+)\);/gm)].map((m) => toHex(Number(m[1]), Number(m[2]), Number(m[3])))
    expect(backgrounds).toEqual([THEME_COLORS.light, THEME_COLORS.dark])
    for (const color of backgrounds) expect(html).toContain(color)
    // the native select chevron is a data URI, so it carries each theme's --muted-foreground as hex
    const muted = [...css.matchAll(/^ {2}--muted-foreground: oklch\(([\d.]+) ([\d.]+) ([\d.]+)\);/gm)].map((m) => toHex(Number(m[1]), Number(m[2]), Number(m[3])))
    const chevrons = [...css.matchAll(/^ {2}--select-chevron: .*fill='%23([0-9a-f]{6})'/gm)].map((m) => `#${m[1]}`)
    expect(chevrons).toEqual(muted)
  })

  // One try covered the storage and the theme: storage that throws (blocked site data) or a broken
  // stored state dropped ?__theme and --theme as well, and the page painted dark, then turned light.
  it("the first-paint script keeps the URL's and the server's theme when the stored state cannot be read", async () => {
    const { readFileSync } = await import("node:fs")
    const script = /<script>([\s\S]*?)<\/script>/.exec(readFileSync("index.html", "utf8"))![1]!
    const root = document.documentElement
    const meta = document.head.appendChild(Object.assign(document.createElement("meta"), { name: "theme-color" }))
    const paint = () => {
      root.classList.add("dark")
      new Function(script)()
      return root.classList.contains("dark") ? "dark" : "light"
    }
    try {
      window.history.replaceState(null, "", "/?__theme=light")
      const blocked = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new DOMException("The operation is insecure.", "SecurityError")
      })
      expect(paint(), "storage that throws").toBe("light")
      blocked.mockRestore()

      window.history.replaceState(null, "", "/")
      window.__REMI_THEME__ = "light"
      for (const stored of ["{", "null", '{"state":5}']) {
        localStorage.setItem("remi-appearance", stored)
        expect(paint(), stored).toBe("light")
      }
    } finally {
      meta.remove()
      root.classList.remove("dark")
    }
  })

  // The themes were looked up on a plain object: "constructor" (any Object.prototype name) in the
  // URL, the stored state or the server's --theme counted as a theme and the sources after it were
  // skipped (the page painted dark).
  it("the first-paint script takes from each source only a theme it knows", async () => {
    const { readFileSync } = await import("node:fs")
    const script = /<script>([\s\S]*?)<\/script>/.exec(readFileSync("index.html", "utf8"))![1]!
    expect(JSON.parse(/var themes = (\[[^\]]*\])/.exec(script)?.[1] ?? "null")).toEqual(["light", "dark", "system"])
    // every source goes through the list (the server's own "constructor" painted dark either way)
    expect(script).not.toMatch(/themes\[/)

    const root = document.documentElement
    const meta = document.head.appendChild(Object.assign(document.createElement("meta"), { name: "theme-color" }))
    const paint = (search: string, stored: object | null, server: string | null) => {
      window.history.replaceState(null, "", `/${search}`)
      if (stored) localStorage.setItem("remi-appearance", JSON.stringify({ state: stored, version: 3 }))
      else localStorage.removeItem("remi-appearance")
      window.__REMI_THEME__ = server
      root.classList.add("dark")
      new Function(script)()
      return root.classList.contains("dark") ? "dark" : "light"
    }
    try {
      expect(paint("?__theme=constructor", { theme: "light", themeChosen: true }, null), "the URL's").toBe("light")
      expect(paint("", { theme: "constructor", themeChosen: true }, "light"), "the stored one").toBe("light")
      expect(paint("?__theme=toString", null, "light"), "the URL's, then the server's").toBe("light")
      expect(paint("?__theme=light", null, "constructor"), "a known one").toBe("light")
    } finally {
      meta.remove()
      root.classList.remove("dark")
      localStorage.removeItem("remi-appearance")
      window.__REMI_THEME__ = undefined
    }
  })

  // A stored density or accent went into the page's data attributes as it was: "x;y" matched no
  // accent rule and the page painted without one until the app replaced it.
  it("the first-paint script applies only a known stored density and accent", async () => {
    const { readFileSync } = await import("node:fs")
    const { ACCENTS, DENSITIES } = await import("./theme")
    const script = /<script>([\s\S]*?)<\/script>/.exec(readFileSync("index.html", "utf8"))![1]!
    const list = (name: string) => JSON.parse(new RegExp(`var ${name} = (\\[[^\\]]*\\])`).exec(script)?.[1] ?? "null") as unknown
    expect(list("densities")).toEqual([...DENSITIES])
    expect(list("accents")).toEqual([...ACCENTS])

    const root = document.documentElement
    const meta = document.head.appendChild(Object.assign(document.createElement("meta"), { name: "theme-color" }))
    const paint = (state: object, version = 3) => {
      root.dataset.density = "comfortable"
      root.dataset.accent = "blue"
      localStorage.setItem("remi-appearance", JSON.stringify({ state, version }))
      new Function(script)()
      return [root.dataset.density, root.dataset.accent]
    }
    try {
      expect(paint({ density: "huge", accent: "x;y" })).toEqual(["comfortable", "blue"])
      expect(paint({ density: "constructor", accent: "toString" })).toEqual(["comfortable", "blue"])
      expect(paint({ density: "compact", accent: "teal" })).toEqual(["compact", "teal"])
      // version 2's default iris became blue
      expect(paint({ accent: "iris" }, 2)).toEqual(["comfortable", "blue"])
      expect(paint({ accent: "iris" })).toEqual(["comfortable", "iris"])
    } finally {
      meta.remove()
      localStorage.removeItem("remi-appearance")
    }
  })

  it("ignores an unknown server theme and an unreadable stored state", async () => {
    expect((await load({ server: "sepia" })).serverTheme()).toBeNull()
    localStorage.setItem("remi-appearance", "{not json")
    vi.resetModules()
    const { useAppearance } = await import("./theme")
    expect(useAppearance.getState()).toMatchObject({ theme: "dark", themeChosen: false, density: "comfortable", accent: "blue" })
  })
})
