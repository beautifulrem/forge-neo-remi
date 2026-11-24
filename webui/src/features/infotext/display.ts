/**
 * Splits an infotext for display: prompt, negative prompt and the
 * "Key: value, …" parameter line (`infotext_utils.parse_generation_parameters`
 * layout). Display only; the server parses for pasting.
 */

/**
 * The display label of an infotext key: its translation where the catalogs
 * have one ("Steps" → "步数"), else the key itself. The raw key stays what is
 * copied and pasted; show it as a tooltip next to a translated label.
 */
export function infotextKeyLabel(t: (key: string) => string, key: string): string {
  return t(key)
}

export type InfotextParts = { prompt: string; negative: string; params: [key: string, value: string][] }

/**
 * upstream's `re_param_code`: a key starts where no key character is before it, and is read in one
 * go (an atomic group, written as a lookahead and its backreference). It runs on every gallery
 * card, on text an image carried: tried at every character, a long run without a colon took
 * seconds per card.
 */
const PARAM = /(?<![\w\s\-/])(?=(\s*([\w\s\-/]+)))\1:\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)/g

export function parseParamLine(line: string): [string, string][] {
  const out: [string, string][] = []
  for (const match of line.matchAll(PARAM)) {
    let value = match[3]!.trim()
    if (value.startsWith('"') && value.endsWith('"') && value.length >= 2) {
      try {
        value = JSON.parse(value) as string
      } catch {
        value = value.slice(1, -1)
      }
    }
    out.push([match[2]!.trim(), value])
  }
  return out
}

export function splitInfotext(text: string): InfotextParts {
  const lines = text.trim().split("\n")
  let params: [string, string][] = []
  const last = lines.at(-1) ?? ""
  const parsed = parseParamLine(last)
  if (parsed.length >= 3) {
    params = parsed
    lines.pop()
  }
  const prompt: string[] = []
  const negative: string[] = []
  let inNegative = false
  for (const line of lines) {
    if (!inNegative && line.startsWith("Negative prompt:")) {
      inNegative = true
      negative.push(line.slice("Negative prompt:".length).trimStart())
    } else if (inNegative) {
      negative.push(line)
    } else {
      prompt.push(line)
    }
  }
  return { prompt: prompt.join("\n").trim(), negative: negative.join("\n").trim(), params }
}
