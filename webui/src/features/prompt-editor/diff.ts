/**
 * Lightweight word/token diff for before/after prompt comparisons (zero deps).
 * Splits text into word, punctuation, and whitespace tokens so edits stay
 * readable for natural-language captions, tag lists, and structured JSON.
 */

export type DiffSegment = {
  kind: "equal" | "add" | "del"
  text: string
}

/** Tokenises preserving whitespace so joining segments reproduces the original string verbatim. */
function tokenize(text: string): string[] {
  if (!text) return []
  return text.match(/\s+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) ?? [text]
}

function mergeSegments(segments: DiffSegment[]): DiffSegment[] {
  const out: DiffSegment[] = []
  for (const seg of segments) {
    if (!seg.text) continue
    const last = out.at(-1)
    if (last && last.kind === seg.kind) last.text += seg.text
    else out.push({ ...seg })
  }
  return out
}

const MAX_DP_CELLS = 250_000

/**
 * Computes a minimal token-level diff between `before` and `after`.
 * Falls back cleanly to a prefix/suffix diff on very large prompts.
 */
export function diffPromptTokens(before: string, after: string): DiffSegment[] {
  if (before === after) return before ? [{ kind: "equal", text: before }] : []
  if (!before) return [{ kind: "add", text: after }]
  if (!after) return [{ kind: "del", text: before }]

  const a = tokenize(before)
  const b = tokenize(after)

  // Strip common prefix and suffix first to keep the LCS table minimal.
  let start = 0
  while (start < a.length && start < b.length && a[start] === b[start]) start++
  let endA = a.length
  let endB = b.length
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA--
    endB--
  }

  const prefix: DiffSegment[] = start > 0 ? [{ kind: "equal", text: a.slice(0, start).join("") }] : []
  const suffix: DiffSegment[] = endA < a.length ? [{ kind: "equal", text: a.slice(endA).join("") }] : []
  const midA = a.slice(start, endA)
  const midB = b.slice(start, endB)

  if (midA.length === 0) return mergeSegments([...prefix, { kind: "add", text: midB.join("") }, ...suffix])
  if (midB.length === 0) return mergeSegments([...prefix, { kind: "del", text: midA.join("") }, ...suffix])

  if (midA.length * midB.length > MAX_DP_CELLS) {
    return mergeSegments([...prefix, { kind: "del", text: midA.join("") }, { kind: "add", text: midB.join("") }, ...suffix])
  }

  const n = midA.length
  const m = midB.length
  const dp = new Uint16Array((n + 1) * (m + 1))
  const stride = m + 1
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i * stride + j] =
        midA[i] === midB[j]
          ? (dp[(i + 1) * stride + (j + 1)] ?? 0) + 1
          : Math.max(dp[(i + 1) * stride + j] ?? 0, dp[i * stride + (j + 1)] ?? 0)
    }
  }

  const middle: DiffSegment[] = []
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (midA[i] === midB[j]) {
      middle.push({ kind: "equal", text: midA[i]! })
      i++
      j++
    } else if ((dp[(i + 1) * stride + j] ?? 0) >= (dp[i * stride + (j + 1)] ?? 0)) {
      middle.push({ kind: "del", text: midA[i]! })
      i++
    } else {
      middle.push({ kind: "add", text: midB[j]! })
      j++
    }
  }
  while (i < n) middle.push({ kind: "del", text: midA[i++]! })
  while (j < m) middle.push({ kind: "add", text: midB[j++]! })

  return mergeSegments([...prefix, ...middle, ...suffix])
}
