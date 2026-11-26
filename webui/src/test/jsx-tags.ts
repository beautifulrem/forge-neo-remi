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
