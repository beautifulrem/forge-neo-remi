/**
 * The opening JSX tags of a source, scanned to the `>` outside braces and strings, and the source
 * they were read from: comments are removed first (an apostrophe in one reads as a string). The
 * source guards (lib/design-guards.test.ts, styles/focus-ring.test.ts) read class lists and
 * attributes tag by tag with it.
 */
export function openingTags(raw: string): { source: string; tags: { name: string; tag: string; index: number }[] } {
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

/**
 * The class-string constants of `sources` (`const FIELD_BUTTON_CLASS = "…"`), by name: a tag that names one
 * wears its classes, which a check reading the tag's own text would miss.
 */
export function classConstants(sources: string[]): Map<string, string> {
  return new Map(sources.flatMap((source) => [...source.matchAll(/\bconst ([A-Z][A-Z0-9_]*)\s*=\s*"([^"\n]*\b(?:bg|shadow|border|ring|outline)-[^"\n]*)"/g)].map((m): [string, string] => [m[1] ?? "", m[2] ?? ""])))
}

/**
 * `tag` with the classes of each constant it names added at its end. With the `source` it is in, a name that
 * file gives one (`const tone = ON_BACKDROP`, `const ring = "…"`) counts too; such a name is only the file's.
 */
export function withClassConstants(tag: string, constants: Map<string, string>, source = ""): string {
  const local = new Map(
    [...source.matchAll(/\bconst ([A-Za-z_]\w*)\s*=\s*(?:([A-Z][A-Z0-9_]+)\b(?!\s*[.([])|"([^"\n]*\b(?:bg|shadow|border|ring|outline)-[^"\n]*)")/g)].map((m): [string, string] => [
      m[1] ?? "",
      m[3] ?? constants.get(m[2] ?? "") ?? "",
    ]),
  )
  const named = [...tag.matchAll(/\b[A-Za-z_]\w*\b/g)].flatMap(([name]) => (local.get(name) || constants.get(name)) ?? [])
  return named.length ? `${tag} "${named.join(" ")}"` : tag
}
