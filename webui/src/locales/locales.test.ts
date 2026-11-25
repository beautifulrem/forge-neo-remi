/**
 * Every chrome string must have a zh-CN translation. Keys are found the way
 * docs/remi-webui/frontend.md asks them to be written: literal `t("…")`
 * calls, the values of constant maps translated at render (`t(NAMES[key])`),
 * plus `label:` / `group:` / `title=` / `description=` literals in the files
 * that pass English strings to `t()` indirectly.
 */
import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import en from "./en.json"
import zh from "./zh-CN.json"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*", "!../api/mock/**"], {
  query: "?raw",
  import: "default",
  eager: true,
})

/** Files whose English label constants are translated at render time. */
const INDIRECT = /(app\/(routes|App|AppearanceMenu|Header)\.tsx|pages\/.*\.tsx?|canvas\/.*\.tsx?|components\/PagePlaceholder\.tsx)$/

/**
 * Constant maps whose values are translated at render, wherever they are defined: `t(NAMES[key])`,
 * or one field of their entries, `t(STAGES[kind].label)` (name → the field, "" for the values).
 */
const translatedMaps = new Map<string, Set<string>>()
for (const source of Object.values(sources)) {
  for (const m of source.matchAll(/\bt\(\s*([A-Z][A-Z0-9_]*)\[[^\]]*\]!?(?:\.(\w+))?/g)) {
    const fields = translatedMaps.get(m[1]!) ?? new Set<string>()
    fields.add(m[2] ?? "")
    translatedMaps.set(m[1]!, fields)
  }
}

/** The literal after `=` at `start`: up to its matching bracket. */
function literalAt(source: string, start: number): string {
  let depth = 0
  for (let i = start; i < source.length; i++) {
    const c = source[i]
    if (c === '"') {
      i = source.indexOf('"', i + 1) // labels hold no escaped quotes
      continue
    }
    if (c === "{" || c === "[") depth++
    else if (c === "}" || c === "]") depth--
    if (depth === 0) return source.slice(start, i + 1)
  }
  return ""
}

/** The English labels of the translated maps a file defines (`const NAMES: Record<…, string> = { … }`). */
function mapLabels(source: string): string[] {
  const labels: string[] = []
  for (const m of source.matchAll(/\bconst ([A-Z][A-Z0-9_]*)\b[^=\n]*=\s*(?=[{[])/g)) {
    const fields = translatedMaps.get(m[1]!)
    if (!fields) continue
    const literal = literalAt(source, m.index + m[0].length)
    for (const field of fields) {
      const value = field ? new RegExp(`\\b${field}:\\s*"((?:[^"\\\\]|\\\\.)+)"`, "g") : /:\s*"((?:[^"\\]|\\.)+)"/g
      for (const label of literal.matchAll(value)) labels.push(label[1]!)
    }
  }
  return labels
}

function chromeKeys(): Set<string> {
  const keys = new Set<string>()
  for (const [path, source] of Object.entries(sources)) {
    // `t("…", { ns: … })` names a widget's own catalog (checked by the widget's tests), not chrome;
    // `t("…", { context: "x" })` is the key "…_x" (one English word, two meanings in Chinese).
    for (const m of source.matchAll(/\bt\(\s*"((?:[^"\\]|\\.)+)"(\s*,\s*\{\s*(?:ns:|context:\s*"(\w+)"))?/g)) {
      if (m[3]) keys.add(`${m[1]}_${m[3]}`)
      else if (!m[2]) keys.add(m[1]!)
    }
    for (const label of mapLabels(source)) keys.add(label)
    if (INDIRECT.test(path)) {
      // `label: "text" as const` is a typed data tag (a JSON Builder object kind), not a string shown as is
      for (const m of source.matchAll(/\b(?:label|title|description): "([^"]+)"(?! as const)/g)) keys.add(m[1]!)
      for (const m of source.matchAll(/\b(?:group): "([A-Z][^"]+)"/g)) keys.add(m[1]!)
      for (const m of source.matchAll(/\b(?:title|description)="([^"]+)"/g)) keys.add(m[1]!)
    }
  }
  return keys
}

describe("zh-CN chrome catalog", () => {
  it("translates every chrome string", () => {
    const catalog = zh as Record<string, string>
    const keys = chromeKeys()
    expect(keys.size).toBeGreaterThan(100) // the scan itself works
    const missing = [...keys].filter((key) => !catalog[key])
    expect(missing).toEqual([])
  })

  it("keeps interpolation placeholders intact", () => {
    for (const [key, value] of Object.entries(zh as Record<string, string>)) {
      const vars = (s: string) => [...s.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]).sort()
      expect(vars(value), key).toEqual(vars(key))
    }
  })
})

describe("shortcut hints", () => {
  // "Undo with Ctrl+Z" was wrong on macOS, where the editors undo with ⌘Z: a sentence that names
  // keys takes them from formatShortcut through a placeholder.
  it("never spell a modifier key in a chrome string", () => {
    const spelt = [...chromeKeys()].filter((key) => /\b(?:Ctrl|Cmd|Alt)\+|[⌘⌥]/.test(key))
    expect(spelt).toEqual([])
  })
})

describe("en chrome catalog", () => {
  it("holds only plural forms of chrome strings, with the key's placeholders", () => {
    const keys = chromeKeys()
    const vars = (s: string) => [...s.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]).sort()
    for (const [key, value] of Object.entries(en as Record<string, string>)) {
      const base = key.replace(/_(one|other)$/, "")
      expect(base, key).not.toBe(key)
      expect(keys.has(base) || base in zh, key).toBe(true) // also strings sent through a renamed t()
      expect(vars(value), key).toEqual(vars(base))
    }
  })

  // frontend.md: the key is the plural, en.json holds the singular (`_one`); an `_other` entry
  // turns that around for one key and leaves the reader of the source with the singular
  it("keys English plurals by the plural, with only `_one` forms in en.json", () => {
    expect(Object.keys(en).filter((key) => !key.endsWith("_one"))).toEqual([])
  })

  it("gives a count followed by a plural noun its singular form", () => {
    // Without "{{count}} steps_one" a one-step recipe reads "1 steps". These counts are never 1.
    const neverOne = new Set(["A zip holds at most {{count}} images: the first {{count}} selected are packed.", "Show all {{count}} tags"])
    const singular = new Set(Object.keys(en).flatMap((key) => (key.endsWith("_one") ? [key.slice(0, -"_one".length)] : [])))
    const missing = [...chromeKeys()].filter((key) => /\{\{count\}\} [a-z]{2,}s\b/.test(key) && !singular.has(key) && !neverOne.has(key))
    expect(missing).toEqual([])
  })
})

/** The server's zh_CN catalog as `/api/v2/i18n/zh_CN` merges it: core, then every built-in extension's. */
function serverCatalog(): Record<string, string> {
  const root = join(process.cwd(), "..")
  const files = [join(root, "localizations", "zh_CN.json")]
  for (const extension of readdirSync(join(root, "extensions-builtin")).sort()) {
    const path = join(root, "extensions-builtin", extension, "localizations", "zh_CN.json")
    try {
      readFileSync(path)
      files.push(path)
    } catch {
      // no catalog
    }
  }
  const merged: Record<string, string> = {}
  for (const path of files) Object.assign(merged, JSON.parse(readFileSync(path, "utf8")))
  return merged
}

describe("zh-CN catalog file", () => {
  // JSON.parse keeps the last of a key written twice: "Changed" read "变更" where the filter next to
  // "已固定" and "未保存" meant "已更改", and an edit to the first entry did nothing.
  it("holds each key once", () => {
    const raw = readFileSync(join(process.cwd(), "src", "locales", "zh-CN.json"), "utf8")
    const keys = [...raw.matchAll(/^\s*("(?:[^"\\]|\\.)*")\s*:/gm)].map((m) => JSON.parse(m[1]!) as string)
    expect(new Set(keys).size).toBe(Object.keys(zh).length) // the scan found every key
    const seen = new Set<string>()
    const twice = keys.filter((key) => seen.size === seen.add(key).size)
    expect(twice).toEqual([])
  })
})

/**
 * Keys no source names literally, each with what shows it. The rest are found by `unusedKeys`.
 */
const SHOWN_UNNAMED: Record<string, string> = {
  "EXIF Interop": "a metadata group the server names `EXIF {label}` (remi/image_meta.py)",
}

/** The text of the server's sources: a sentence it sends as it is takes the chrome translation first. */
function serverSources(): string {
  const root = join(process.cwd(), "..")
  const parts: string[] = []
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith(".") || entry.name === "node_modules" || entry.name === "__pycache__") continue
      const path = join(dir, entry.name)
      if (entry.isDirectory()) walk(path)
      else if (/\.(py|html)$/.test(entry.name)) parts.push(readFileSync(path, "utf8"))
    }
  }
  for (const dir of ["remi", "modules", "modules_forge", "backend", "extensions-builtin", "scripts", "html"]) walk(join(root, dir))
  return parts.join("\n")
}

/** zh-CN keys no chrome string, string literal of the SPA or server sentence names. */
function unusedKeys(): string[] {
  const literals = new Set<string>()
  for (const source of Object.values(sources)) {
    for (const m of source.matchAll(/"((?:[^"\\\n]|\\.)*)"/g)) {
      try {
        literals.add(JSON.parse(`"${m[1]}"`) as string)
      } catch {
        // not a JSON-compatible literal
      }
    }
  }
  const chrome = chromeKeys()
  const named = (key: string) => {
    // a plural form or a context key ("…_one", "…_theme") is named by its base
    const forms = new Set([key, key.replace(/_(?:one|other)$/, ""), key.replace(/_[a-z]+$/, "")])
    return [...forms].some((form) => chrome.has(form) || literals.has(form))
  }
  const left = Object.keys(zh).filter((key) => !named(key))
  const server = serverSources()
  expect(server.length).toBeGreaterThan(1_000_000) // the server's sources were found
  return left.filter((key) => !server.includes(key) && !server.includes(key.replace(/_[a-z]+$/, "")))
}

describe("zh-CN catalog keys", () => {
  // About thirty translations outlived their strings (old page placeholders, sentences since
  // reworded, a queue banner that went): dead weight for translators, and a reworded string
  // looked translated where only its old wording was.
  it("are all shown somewhere", () => {
    const unused = unusedKeys()
    expect(unused.filter((key) => !(key in SHOWN_UNNAMED))).toEqual([])
    // an entry that no longer applies: the key is gone or a source now names it
    expect(Object.keys(SHOWN_UNNAMED).filter((key) => !(key in zh) || !unused.includes(key))).toEqual([])
  })
})

describe("zh-CN chrome vs server catalogs", () => {
  it("never overrides a server translation with a different one", () => {
    // t() looks in the chrome catalog first: a chrome key that the server translates
    // differently changes the server's labels too (e.g. a theme's "System" renaming the
    // "System" settings category). Same meaning → same wording; another meaning → a context key.
    const server = serverCatalog()
    expect(Object.keys(server).length).toBeGreaterThan(1000) // the catalogs were found
    const clashes = Object.entries(zh as Record<string, string>)
      .filter(([key, value]) => key in server && server[key] !== value)
      .map(([key, value]) => `${key}: chrome "${value}" vs server "${server[key]}"`)
    expect(clashes).toEqual([])
  })
})

