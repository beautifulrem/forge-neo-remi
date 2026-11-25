/**
 * Source checks for rules of docs/remi-webui/frontend.md ("Visual rules") that a class list can break
 * without any test of the component noticing. Each rule is its own test.
 */
import { describe, expect, it } from "vitest"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*", "!../api/gen/**"], { query: "?raw", import: "default", eager: true })

const files = Object.entries(sources).map(([file, source]) => ({ file: file.replace(/^\.\.\//, ""), source }))

/** Every match of `re` in the sources, with the file it is in. */
function occurrences(re: RegExp): { file: string; match: string }[] {
  return files.flatMap(({ file, source }) => [...source.matchAll(re)].map((m) => ({ file, match: m[0] })))
}

/**
 * The opening JSX tags of a source, scanned to the `>` outside braces and strings, and the source
 * they were read from: comments are removed first (an apostrophe in one reads as a string).
 */
function openingTags(raw: string): { source: string; tags: { name: string; tag: string; index: number }[] } {
  const source = raw.replace(/^\s*\/\/.*$/gm, "").replace(/\{\/\*[\s\S]*?\*\/\}/g, "")
  const tags: { name: string; tag: string; index: number }[] = []
  for (const m of source.matchAll(/<([A-Za-z][\w.]*)\b/g)) {
    let depth = 0
    let quote = ""
    let end = -1
    for (let i = m.index + m[0].length; i < Math.min(source.length, m.index + 8000); i++) {
      const c = source[i]
      if (quote) {
        if (c === "\\") i++
        else if (c === quote) quote = ""
      } else if (c === '"' || c === "'" || c === "`") quote = c
      else if (c === "{") depth++
      else if (c === "}") depth--
      else if (c === ">" && depth === 0) {
        end = i
        break
      }
    }
    if (end > 0) tags.push({ name: m[1] ?? "", tag: source.slice(m.index, end + 1), index: m.index })
  }
  return { source, tags }
}

describe("type scale", () => {
  // 0.625 rem, 0.6875 rem, text-xs, 0.8 rem, text-sm, 0.9375 rem; text-xl is a theme name.
  const SCALE = new Set(["0.625rem", "0.6875rem", "0.75rem", "0.8rem", "0.875rem", "0.9375rem"])
  // A size off the scale, with the reason frontend.md gives for it.
  const ALLOWED = [
    // Inline code in prose: mono drawn a step smaller than the prose around it, whatever its size.
    { file: "ui/rich-text.tsx", size: "0.85em" },
  ]

  it("keeps arbitrary text sizes on the scale", () => {
    const sizes = occurrences(/\btext-\[(?:length:)?[^\]\s]+\]/g).flatMap(({ file, match }) => {
      const value = match.slice("text-[".length, -1).replace(/^length:/, "")
      // A size scaled by a variable (the JSON Builder canvas's zoom) starts from a size on the scale.
      const size = /^calc\((\d*\.?\d+(?:rem|em|px))\*/.exec(value)?.[1] ?? value
      return /^\d*\.?\d+(?:rem|em|px)$/.test(size) ? [{ file, size }] : []
    })
    expect(sizes.length).toBeGreaterThan(100)
    const off = sizes.filter(({ file, size }) => !SCALE.has(size) && !ALLOWED.some((allowed) => allowed.file === file && allowed.size === size))
    expect(off).toEqual([])
  })
})

describe("pressed toggles", () => {
  const TINT = "PRESSED_TINT_CLASS"
  const toggles = files.filter(({ file, source }) => file !== "components/ui/toggle.tsx" && /\baria-pressed=/.test(source))
  // A toggle drawing its pressed state another way, with the reason (frontend.md): exempt only
  // while its tag draws exactly this look.
  const OWN_LOOK = [
    // White on the viewer's always-black backdrop, over the picture: the accent tint does not read
    // on black with a dark accent. Its pressed wash sits well above its 12 % hover.
    { file: "features/lightbox/Lightbox.tsx", look: "aria-pressed:bg-white/25" },
    // Unpressed tag chips already wear a primary wash (up to 32 %) for how often the tag occurs:
    // a pressed one is a solid fill, above that scale, or it would read as a frequent tag.
    { file: "features/networks/TagChips.tsx", look: '"border-primary bg-primary text-primary-foreground"' },
    // A swatch picker (the RGBA backdrops): the chosen swatch is ringed; a wash would tint the swatch.
    { file: "features/compare/BackdropPicker.tsx", look: '"bg-white/22 ring-1 ring-white/80"' },
  ]
  const pressedTags = toggles.flatMap(({ file, source }) =>
    openingTags(source)
      .tags.filter(({ tag }) => /\s(?:aria-)?pressed=\{/.test(tag))
      .map(({ tag }) => ({ file, tag })),
  )

  it("wear the selection tint instead of a hand-written one", () => {
    expect(pressedTags.length).toBeGreaterThan(20)
    // A pressed look written by hand: aria-pressed: classes without the shared tint, or a fill
    // (primary, white, foreground) picked by a condition.
    const handWritten = pressedTags.filter(
      ({ file, tag }) =>
        ((/\baria-pressed:/.test(tag) && !tag.includes(TINT)) || /\?\s*"[^"]*\bbg-(?:primary|white|foreground)\b/.test(tag)) &&
        !OWN_LOOK.some((own) => own.file === file && tag.includes(own.look)),
    )
    expect(handWritten.map(({ file, tag }) => `${file}: ${tag.slice(0, 60)}`)).toEqual([])
    // Toggles that drew their pressed state with a fill, a variant or a colour of their own.
    for (const file of [
      "features/networks/NetworkBrowser.tsx",
      "pages/generate/SizeTools.tsx",
      "features/prompt-tools/PromptToolbar.tsx",
      "pages/settings/SettingRow.tsx",
      "features/compare/ParamDiffPanel.tsx",
      "features/recipes/RecipeManager.tsx",
      "pages/extensions/DiagnosticsView.tsx",
    ]) {
      expect(sources[`../${file}`], file).toContain(TINT)
    }
  })

  it("keep one variant in both states", () => {
    // A variant swapped on the pressed value draws a filled button once pressed, not the tint.
    const swapped = toggles.flatMap(({ file, source }) =>
      [...source.matchAll(/aria-pressed=\{([^}]+)\}/g)].flatMap((m) => {
        const value = (m[1] ?? "").trim()
        const around = source.slice(Math.max(0, m.index - 300), m.index + 400)
        const escaped = value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
        return new RegExp(`variant=\\{${escaped}\\s*\\?`).test(around) ? [file] : []
      }),
    )
    expect(swapped).toEqual([])
  })
})

describe("letter spacing", () => {
  // Wide tracking is the eyebrow-label look the type rules leave out. A menu's shortcut keys keep
  // shadcn's spacing: a run of key glyphs (⌘⇧K), not words.
  const SHORTCUTS = new Set(["components/ui/command.tsx", "components/ui/dropdown-menu-impl.tsx", "components/ui/dropdown-menu.tsx", "components/ui/context-menu.tsx"])

  it("never widens text", () => {
    const wide = occurrences(/\btracking-wide(?:r|st)?\b/g).filter(({ file }) => !SHORTCUTS.has(file))
    expect(wide).toEqual([])
  })
})

describe("floating lists", () => {
  const CONTENT = /<(?:PopoverContent|DropdownMenuContent|DropdownMenuSubContent|SelectContent|ComboboxContent|ContextMenuContent|HoverCardContent)\b(?:[^>"'{}]|"[^"]*"|'[^']*'|\{(?:[^{}]|\{[^{}]*\})*\})*>/g

  it("cap their own height inside the room beside the trigger", () => {
    // A max-h of their own replaces the base --available-height (cn keeps the caller's): it must
    // still take that room into account, or the popup runs off a short window instead of scrolling.
    const tags = occurrences(CONTENT)
    expect(tags.length).toBeGreaterThan(50)
    const capped = tags.filter(({ match }) => /\bmax-h-[[(]/.test(match))
    expect(capped.filter(({ match }) => !match.includes("--available-height"))).toEqual([])
  })
})

describe("tab strips", () => {
  // A strip of fixed tabs that may not fit a phone is a FitTabsList (a select once it does not fit).
  // Written by hand, with the reason:
  const OWN_STRIP = new Set([
    // a varying number of boards / tag groups: the strip scrolls sideways, its edges fading
    "features/library/BoardsBar.tsx",
    "features/prompt-tools/panels/GroupsPanel.tsx",
    // two short views on the canvas's own toolbar
    "canvas/img2img/Img2imgStage.tsx",
  ])

  it("are FitTabsList, not hand-written tab lists", () => {
    const handWritten = files.filter(({ file, source }) => /role="tablist"/.test(source) && !OWN_STRIP.has(file)).map(({ file }) => file)
    expect(handWritten).toEqual([])
  })
})

describe("toggle names", () => {
  /** The expression of `attr={…}` in a tag, or undefined. */
  function attribute(tag: string, attr: string): string | undefined {
    const start = new RegExp(`\\s${attr}=\\{`).exec(tag)
    if (!start) return undefined
    let depth = 1
    let i = start.index + start[0].length
    for (const from = i; i < tag.length; i++) {
      if (tag[i] === "{") depth++
      if (tag[i] === "}" && --depth === 0) return tag.slice(from, i)
    }
    return undefined
  }

  /** Whether a name switches on a condition: a ternary, or a constant in the file that is one. */
  function switches(expression: string, source: string): boolean {
    if (/\?[^.?]/.test(expression.replace(/\?\./g, ""))) return true
    const constant = /^\s*(\w+)\s*$/.exec(expression)?.[1]
    const declared = constant ? new RegExp(`const ${constant} = ([^\\n]+)`).exec(source)?.[1] : undefined
    return declared !== undefined && /\?[^.?]/.test(declared.replace(/\?\./g, ""))
  }

  it("keep one name while pressed", () => {
    // frontend.md: a toggle (aria-pressed) keeps one name and its pressed state says on or off; a
    // button whose name says what the next click does ("Show all", "Unpin") is an action, not a toggle.
    const offenders = files.flatMap(({ file, source: raw }) => {
      const { source, tags } = openingTags(raw)
      return tags.flatMap(({ name, tag, index }) => {
        if (attribute(tag, "aria-pressed") === undefined && attribute(tag, "pressed") === undefined) return []
        // the name (a title with text beside it only describes)
        const names = ["aria-label", "label"].map((attr) => attribute(tag, attr)).filter((value) => value !== undefined)
        // a visible name between the tag and its closing tag: text picked by a condition
        const body = tag.endsWith("/>") ? "" : source.slice(index + tag.length, source.indexOf(`</${name}>`, index))
        const switched = names.some((name) => switches(name, source)) || /\{[^{}]*\?\s*t\(/.test(body)
        return switched ? [`${file}: <${name}>`] : []
      })
    })
    expect(offenders).toEqual([])
  })
})

describe("focus rings", () => {
  // A variant of a focus state: focus, focus-visible, focus-within, group- or peer-focus…,
  // has-[…:focus…], or a focus a widget marks itself (data-focused, data-[focus-visible]).
  const FOCUS_VARIANT = /(?:^|[:-])(?:focus|focus-visible|focus-within)(?:\/[\w-]+)?:|:focus(?:-visible|-within)?\]|(?:^|[:-])data-\[?focus[\w-]*(?:=[^\]]*)?\]?(?:\/[\w-]+)?:/
  /** The variants of a class, outermost first: its parts before the colons outside brackets. */
  function variants(cls: string): string[] {
    const parts: string[] = []
    let depth = 0
    let start = 0
    for (let i = 0; i < cls.length; i++) {
      if (cls[i] === "[" || cls[i] === "(") depth++
      else if (cls[i] === "]" || cls[i] === ")") depth--
      else if (cls[i] === ":" && depth === 0) {
        parts.push(cls.slice(start, i))
        start = i + 1
      }
    }
    return parts
  }
  /** Whether a variant is a focus state (`not-focus-visible` is its absence, not one). */
  const isFocus = (variant: string) => !variant.startsWith("not-") && FOCUS_VARIANT.test(`${variant}:`)
  /** Whether a class applies under a focus state. */
  const underFocus = (cls: string) => variants(cls).some(isFocus)
  /** The variants, kind and width of a ring or outline width class, or undefined for any other utility. */
  function width(cls: string): { prefix: string; kind: string; width: string } | undefined {
    const m = /^(.*?)(ring|outline)-(\d+|\[[^\]]+\])$/.exec(cls)
    return m ? { prefix: m[1]!, kind: m[2]!, width: m[3]!.replace(/^\[(\d+)px\]$/, "$1") } : undefined
  }

  it("are 3 px wide", () => {
    // frontend.md: a focus ring is `ring-3` in --focus-ring (or a 3 px outline over an image). A 2 px
    // ring read thinner than every other focus ring beside it. `ring-0` takes the ring off a control
    // whose group draws it; shadcn's 1 px outline sits on the tabs' edge inside a 3 px ring.
    // every class in the sources (a run outside quotes, braces and spaces), brackets included
    const rings = files.flatMap(({ file, source }) =>
      (source.match(/[^\s"'`{}]+/g) ?? [])
        .filter(underFocus)
        .flatMap((cls) => {
          const drawn = width(cls)
          return drawn ? [{ file, cls, ...drawn }] : []
        }),
    )
    // A keyboard cursor or highlight (the networks grid's current item) is drawn in the focus ring's
    // colour under a state of its own: the width beside that colour, under the same variants.
    const cursors = files.flatMap(({ file, source }) =>
      (source.match(/"[^"\n]*"|'[^'\n]*'|`[^`]*`/g) ?? []).flatMap((literal) => {
        const classes = literal.slice(1, -1).split(/\s+/)
        return classes.flatMap((cls) => {
          const colour = /^(.*?)(ring|outline)-focus-ring$/.exec(cls)
          if (!colour || underFocus(cls)) return []
          return classes.flatMap((other) => {
            const drawn = width(other)
            return drawn && drawn.prefix === colour[1] && drawn.kind === colour[2] ? [{ file, cls: other, ...drawn }] : []
          })
        })
      }),
    )
    expect(rings.filter(({ width }) => width === "3").length).toBeGreaterThan(40)
    expect(cursors.length).toBeGreaterThan(1)
    const SHADCN_EDGE = new Set(["components/ui/tabs.tsx"])
    const off = [...rings, ...cursors].filter(({ file, kind, width }) => width !== "3" && width !== "0" && !(kind === "outline" && width === "1" && SHADCN_EDGE.has(file)))
    expect(off.map(({ file, cls }) => `${file}: ${cls}`)).toEqual([])
  })

  /** The utility of a class: what follows its last variant (a colon outside brackets). */
  function utility(cls: string): string {
    let depth = 0
    let start = 0
    for (let i = 0; i < cls.length; i++) {
      if (cls[i] === "[" || cls[i] === "(") depth++
      else if (cls[i] === "]" || cls[i] === ")") depth--
      else if (cls[i] === ":" && depth === 0) start = i + 1
    }
    return cls.slice(start).replace(/^!|!$/g, "")
  }

  it("take the focus ring colour", () => {
    // frontend.md: every focus ring is drawn in --focus-ring (`ring-focus-ring`, `outline-focus-ring`).
    // `ring-ring` / `outline-ring` drew it in the opaque brand colour, the nav rail in a grey of its
    // own and a destructive button in a 20 % red (about 1.3:1). A border in `border-ring` beside the
    // ring (a field) is not the ring.
    const colours = files.flatMap(({ file, source }) =>
      (source.match(/[^\s"'`{}]+/g) ?? [])
        .filter(underFocus)
        .flatMap((cls) => {
          const m = /^(?:inset-)?(ring|outline)-(.+)$/.exec(utility(cls))
          if (!m) return []
          const value = m[2]!
          // a width, the offset, the inset switch or the style: not a colour
          if (/^(?:\d+|\[\d*\.?\d+px\]|\[length:[^\]]+\]|offset-.+|inset|none|hidden|solid|dashed|dotted|double)$/.test(value)) return []
          return [{ file, cls, value }]
        }),
    )
    expect(colours.filter(({ value }) => value === "focus-ring").length).toBeGreaterThan(70)
    expect(colours.filter(({ value }) => value !== "focus-ring").map(({ file, cls }) => `${file}: ${cls}`)).toEqual([])
  })

  /** The end of the call opened at `open` (its `(`), skipping strings, or -1. */
  function callEnd(source: string, open: number): number {
    let depth = 0
    let quote = ""
    for (let i = open; i < source.length; i++) {
      const c = source[i]
      if (quote) {
        if (c === "\\") i++
        else if (c === quote) quote = ""
      } else if (c === '"' || c === "'" || c === "`") quote = c
      else if (c === "(") depth++
      else if (c === ")" && --depth === 0) return i
    }
    return -1
  }

  it("take the outline off where they draw a ring", () => {
    // frontend.md: a focus nothing else marks draws a 3 px outline (the index.css base layer); a
    // control that draws its own ring takes it off (outline-none or outline-hidden), or it shows both.
    // shadcn's 1 px outline inside the tabs' ring is an edge, not a second ring. Read per element:
    // a JSX tag, or the class call (cn, cva) a class string outside one is in.
    const RING = /(?:^|[\s"'`])(?:dark:)?(?:focus|focus-visible):ring-(?:[1-9]\d*|\[\d+px\])(?=[\s"'`]|$)/
    const OFF = /(?:^|[\s"'`:])outline-(?:none|hidden)(?=[\s"'`]|$)|(?:^|[\s"'`])focus-visible:outline-1(?=[\s"'`]|$)/
    let drawn = 0
    const both = files.flatMap(({ file, source: raw }) => {
      const { source, tags } = openingTags(raw)
      const fromTags = tags.filter(({ tag }) => RING.test(tag)).map(({ name, tag }) => ({ where: `<${name}>`, scope: tag }))
      const calls = [...source.matchAll(/\b(?:cn|cva|clsx)\(/g)].map((m) => ({ start: m.index, end: callEnd(source, m.index + m[0].length - 1) }))
      const fromCalls = [...source.matchAll(/"[^"\n]*"|'[^'\n]*'|`[^`]*`/g)]
        .filter((m) => RING.test(m[0]) && !tags.some(({ index, tag }) => m.index >= index && m.index < index + tag.length))
        .map((m) => {
          const call = calls.filter(({ start, end }) => start < m.index && end > m.index).at(-1)
          return { where: m[0].slice(0, 50), scope: call ? source.slice(call.start, call.end + 1) : m[0] }
        })
      const ringed = [...fromTags, ...fromCalls]
      drawn += ringed.length
      return ringed.filter(({ scope }) => !OFF.test(scope)).map(({ where }) => `${file}: ${where}`)
    })
    expect(drawn).toBeGreaterThan(60)
    expect(both).toEqual([])
  })

  /**
   * The class lists of a source, each with the classes that share its element: a JSX tag, or the
   * class call (cn, cva, clsx) a class string outside a tag is in, or that string alone.
   */
  function classScopes(raw: string): string[] {
    const { source, tags } = openingTags(raw)
    const calls = [...source.matchAll(/\b(?:cn|cva|clsx)\(/g)].map((m) => ({ start: m.index, end: callEnd(source, m.index + m[0].length - 1) }))
    const outside = [...source.matchAll(/"[^"\n]*"|'[^'\n]*'|`[^`]*`/g)].filter((m) => !tags.some(({ index, tag }) => m.index >= index && m.index < index + tag.length))
    const scopes = outside.map((m) => {
      const call = calls.filter(({ start, end }) => start < m.index && end > m.index).at(-1)
      return call ? source.slice(call.start, call.end + 1) : m[0]
    })
    return [...new Set([...tags.map(({ tag }) => tag), ...scopes])]
  }

  it("give way to the focus ring on an invalid field", () => {
    // frontend.md: on an invalid field the focus wins. Its halo (`aria-invalid:ring-3
    // aria-invalid:ring-destructive/20`) came after `focus-visible:ring-focus-ring` in Tailwind's CSS,
    // with the same specificity, and took the focus ring's colour: a focused invalid field looked
    // unfocused. A halo applies only without the focus, `not-` the variant the element's focus ring is
    // drawn under (not-focus-visible, not-focus-within, not-has-[…:focus-visible]), so no order of the
    // CSS can bring it back. `ring-0` (a control inside a group that draws the halo) is no ring, and
    // an inset ring is an edge, like the red border that stays.
    const halos = files.flatMap(({ file, source }) =>
      classScopes(source).flatMap((scope) => {
        const classes = scope.match(/[^\s"'`{}]+/g) ?? []
        const focus = new Set(classes.filter((cls) => /^(?:ring|outline)-focus-ring$/.test(utility(cls))).flatMap((cls) => variants(cls).filter(isFocus)))
        return classes.flatMap((cls) => {
          const chain = variants(cls)
          if (!chain.some((variant) => !variant.startsWith("not-") && /invalid/.test(variant)) || !/^ring-(?!0$|inset$|offset-)/.test(utility(cls))) return []
          const negated = chain.filter((variant) => variant.startsWith("not-")).map((variant) => variant.slice("not-".length))
          const scoped = focus.size > 0 ? [...focus].every((variant) => negated.includes(variant)) : negated.some(isFocus)
          return [{ where: `${file}: ${cls}`, scoped }]
        })
      }),
    )
    expect(halos.length).toBeGreaterThan(25)
    expect([...new Set(halos.filter(({ scoped }) => !scoped).map(({ where }) => where))]).toEqual([])
  })

  it("draw no outline width that outline-none leaves without a style", () => {
    // `outline-none` and `outline-hidden` set --tw-outline-style to none, and a width class
    // (`focus-visible:outline-1`) takes its style from it: beside them, a width with no style class of
    // its own draws nothing (the scroll area's 1 px outline never drew). An outline on a pseudo-element
    // (`after:`) or a child keeps its own style: the variable does not inherit.
    const ELSEWHERE = /^(?:before|after|placeholder|file|marker|selection|first-line|first-letter|backdrop|\*|\*\*)$|^\[.*[\s_>~+].*\]$/
    let checked = 0
    const dead = files.flatMap(({ file, source }) =>
      classScopes(source).flatMap((scope) => {
        const classes = scope.match(/[^\s"'`{}]+/g) ?? []
        if (!classes.some((cls) => cls === "outline-none" || cls === "outline-hidden")) return []
        return classes.flatMap((cls) => {
          const chain = variants(cls)
          if (!/^-?outline-(?:[1-9]\d*|\[\d*\.?\d+px\])$/.test(utility(cls))) return []
          checked++
          if (chain.some((variant) => ELSEWHERE.test(variant))) return []
          const styled = classes.some((other) => /^outline-(?:solid|dashed|dotted|double)$/.test(utility(other)) && variants(other).every((variant) => chain.includes(variant)))
          return styled ? [] : [`${file}: ${cls}`]
        })
      }),
    )
    expect(checked).toBeGreaterThan(0)
    expect([...new Set(dead)]).toEqual([])
  })

  it("ring a handle instead of filling it with --ring", () => {
    // frontend.md: a focus is a 3 px ring in --focus-ring. The stage timeline's start and end lines
    // widened and turned --ring (the opaque brand colour) on focus, beside the peak handle's focus
    // ring, and the networks drawer's edge became a 1 px --ring line: looks of their own, louder or
    // thinner than every focus ring beside them, which no drag or hover state of the handle shares.
    const fills = occurrences(/[^\s"'`{}]+/g).filter(({ match }) => underFocus(match) && /^bg-ring(?:\/\d+)?$/.test(utility(match)))
    expect(fills.map(({ file, match }) => `${file}: ${match}`)).toEqual([])
  })
})

describe("selected segments", () => {
  it("wear an edge besides their tone", () => {
    // frontend.md: a segmented control's or tab strip's selected segment is its tone (--card, --accent
    // in the dark theme) and an inset 1 px edge in --muted-foreground. The tone alone is 1.18:1
    // against the --muted track in the light theme and 1.10:1 in the dark one; a state indicator
    // needs 3:1 (WCAG 1.4.11). So is the folder tree's selected row (--accent, 1.09:1 to 1.29:1).
    const segments = files.flatMap(({ file, source }) =>
      (source.match(/"[^"\n]*"|'[^'\n]*'|`[^`]*`/g) ?? []).flatMap((literal) => {
        const state = /(?:^|\s)(?:dark:)?(data-checked|data-active|aria-selected):bg-accent(?:\s|$)/.exec(literal.slice(1, -1))?.[1]
        return state ? [{ file, state, classes: literal.slice(1, -1).split(/\s+/) }] : []
      }),
    )
    expect(segments.map(({ file }) => file).sort()).toEqual(
      expect.arrayContaining(["canvas/img2img/Img2imgStage.tsx", "canvas/img2img/SourcePanel.tsx", "components/Segmented.tsx", "components/ui/tabs.tsx", "features/networks/FolderTree.tsx"]),
    )
    const bare = segments.filter(({ state, classes }) => !classes.includes(`${state}:inset-ring`) || !classes.includes(`${state}:inset-ring-muted-foreground`))
    expect(bare.map(({ file }) => file)).toEqual([])
  })
})

describe("colours", () => {
  it("come from the tokens, not Tailwind's palette", () => {
    // frontend.md: colours are tokens of index.css (white and black stay for the viewer's
    // always-black backdrop); a palette shade ignores the theme and the accent.
    const palette = occurrences(
      /\b(?:text|bg|border|ring|outline|fill|stroke|from|via|to|decoration|caret|accent|divide|shadow)-(?:red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|slate|gray|zinc|neutral|stone)-\d{2,3}\b/g,
    )
    expect(palette).toEqual([])
  })
})
