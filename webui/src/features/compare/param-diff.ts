/**
 * Structured Parameter Diff engine for CompareDialog:
 * - Parses two infotexts locally (deterministic fallback + instant mock/unit support)
 * - Calls `POST /api/v2/gallery/compare` when backed by a live server
 * - Diffs Prompt & Negative Prompt tokens, Core sampling params, LoRAs, and
 *   Active Extensions, and builds a one-click patch to adopt A's or B's deltas.
 */
import { api } from "@/api/endpoints"
import type { FormTab, ParamValues } from "@/api/types"
import { splitInfotext } from "@/features/infotext/display"
import { type PasteEntry, type PasteResult } from "@/features/infotext/paste"
import { workbenchRequest } from "@/features/pipelines/workbench-http"
import { deliver } from "@/features/send-to/send-to"

export type DiffKind = "added" | "removed" | "changed"

export type TokenDiff = {
  a: string
  b: string
  added: string[]
  removed: string[]
  unchanged: string[]
}

export type FieldDiff = {
  key: string
  field?: string
  a: string | null
  b: string | null
  kind: DiffKind
}

export type LoraDiff = {
  name: string
  a: string | null
  b: string | null
  kind: DiffKind
}

export type ExtensionDiff = {
  group: string
  key: string
  a: string | null
  b: string | null
  kind: DiffKind
}

export type AdoptPatch = {
  core: ParamValues
  scripts: Record<string, ParamValues>
  raw_params: Record<string, string>
}

export type CompareDiffResult = {
  id_a?: string | null
  id_b?: string | null
  infotext_a?: string | null
  infotext_b?: string | null
  identical: boolean
  prompt_diff: TokenDiff
  negative_diff: TokenDiff
  core_diffs: FieldDiff[]
  lora_diffs: LoraDiff[]
  extension_diffs: ExtensionDiff[]
  /** Every parsed param from both sides so "Show all" can render unchanged rows too. */
  all_core?: { key: string; a: string | null; b: string | null; changed: boolean }[]
  adopt_a_patch?: AdoptPatch
  adopt_b_patch?: AdoptPatch
}

const CORE_PARAM_KEYS = new Set([
  "Steps",
  "Sampler",
  "Schedule type",
  "CFG scale",
  "Distilled CFG Scale",
  "Seed",
  "Variation seed",
  "Variation seed strength",
  "Size",
  "Model",
  "Model hash",
  "VAE",
  "VAE hash",
  "Denoising strength",
  "Clip skip",
  "Hires upscale",
  "Hires resize",
  "Hires steps",
  "Hires upscaler",
  "Hires checkpoint",
  "Hires sampler",
  "Hires schedule type",
  "Refiner",
  "Refiner switch at",
  "ENSD",
  "RNG",
  "Schedule max sigma",
  "Schedule min sigma",
  "Schedule rho",
  "Shift",
])

const EXTENSION_PREFIXES: [prefix: string, group: string][] = [
  ["ADetailer ", "ADetailer"],
  ["ControlNet ", "ControlNet"],
  ["Anima ", "Anima"],
  ["FreeFuse ", "FreeFuse"],
  ["NAG ", "NAG"],
  ["DAVE ", "DAVE"],
  ["Regional ", "Regional"],
  ["Spectrum ", "Spectrum"],
  ["Dynamic Thresholding ", "Dynamic Thresholding"],
  ["MultiDiffusion ", "MultiDiffusion"],
  ["Tiled Diffusion ", "Tiled Diffusion"],
  ["Tiled VAE ", "Tiled VAE"],
  ["SAG ", "SAG"],
  ["PAG ", "PAG"],
  ["APG ", "APG"],
  ["TCFG ", "TCFG"],
  ["SPEED ", "SPEED"],
  ["ImageStitch ", "ImageStitch"],
  ["Forge Couple ", "Forge Couple"],
]

// a tag never spans another "<": "<lora:a<lora:a…" tried every "<" to the end of the text
const LORA_TAG = /<(?:lora|lyco):([^:<>]+):([^<>]+)>/gi

/** Splits a prompt string into normalized comma/newline tokens (excluding `<lora:...>` tags handled separately). */
export function tokenizePrompt(prompt: string): string[] {
  return prompt
    .replace(LORA_TAG, "")
    .split(/[,\n]+/)
    .map((part) => part.trim())
    .filter(Boolean)
}

/** Computes added, removed and unchanged tokens preserving original order (multiset-aware). */
export function diffPromptTokens(aText: string, bText: string): TokenDiff {
  const aTokens = tokenizePrompt(aText)
  const bTokens = tokenizePrompt(bText)

  const bCounts = new Map<string, number>()
  for (const tok of bTokens) bCounts.set(tok, (bCounts.get(tok) ?? 0) + 1)

  const unchanged: string[] = []
  const removed: string[] = []
  const matchedInB = new Map<string, number>()

  for (const tok of aTokens) {
    const remaining = (bCounts.get(tok) ?? 0) - (matchedInB.get(tok) ?? 0)
    if (remaining > 0) {
      unchanged.push(tok)
      matchedInB.set(tok, (matchedInB.get(tok) ?? 0) + 1)
    } else {
      removed.push(tok)
    }
  }

  const aUsed = new Map(matchedInB)
  const added: string[] = []
  for (const tok of bTokens) {
    const used = aUsed.get(tok) ?? 0
    if (used > 0) aUsed.set(tok, used - 1)
    else added.push(tok)
  }

  return { a: aText, b: bText, added, removed, unchanged }
}

/** Extracts `<lora:name:weight>` / `<lyco:name:weight>` weights from prompt text plus `Lora hashes` if present. */
export function extractLoras(prompt: string, params: ReadonlyMap<string, string>): Map<string, string> {
  const map = new Map<string, string>()
  for (const match of prompt.matchAll(LORA_TAG)) {
    const name = match[1]!.trim()
    const weight = match[2]!.trim()
    if (name) map.set(name, weight)
  }
  const hashes = params.get("Lora hashes")
  if (hashes) {
    for (const pair of hashes.split(",")) {
      const idx = pair.indexOf(":")
      if (idx === -1) continue
      const name = pair.slice(0, idx).trim()
      if (name && !map.has(name)) map.set(name, "loaded")
    }
  }
  return map
}

function extensionGroupOf(key: string): string | null {
  for (const [prefix, group] of EXTENSION_PREFIXES) {
    if (key.startsWith(prefix)) return group
  }
  return null
}

function diffKind(a: string | null, b: string | null): DiffKind {
  if (a === null) return "added"
  if (b === null) return "removed"
  return "changed"
}

/** Pure client-side structured parameter diff from two infotext strings. */
export function diffInfotext(infotextA: string | null | undefined, infotextB: string | null | undefined): CompareDiffResult {
  const aRaw = infotextA?.trim() ?? ""
  const bRaw = infotextB?.trim() ?? ""
  const aParts = aRaw ? splitInfotext(aRaw) : { prompt: "", negative: "", params: [] }
  const bParts = bRaw ? splitInfotext(bRaw) : { prompt: "", negative: "", params: [] }

  const prompt_diff = diffPromptTokens(aParts.prompt, bParts.prompt)
  const negative_diff = diffPromptTokens(aParts.negative, bParts.negative)

  const mapA = new Map<string, string>(aParts.params)
  const mapB = new Map<string, string>(bParts.params)

  const loraA = extractLoras(aParts.prompt, mapA)
  const loraB = extractLoras(bParts.prompt, mapB)
  const loraNames = new Set([...loraA.keys(), ...loraB.keys()])
  const lora_diffs: LoraDiff[] = []
  for (const name of loraNames) {
    const va = loraA.get(name) ?? null
    const vb = loraB.get(name) ?? null
    if (va !== vb) lora_diffs.push({ name, a: va, b: vb, kind: diffKind(va, vb) })
  }

  const allKeys = new Set([...mapA.keys(), ...mapB.keys()])
  allKeys.delete("Lora hashes")

  const core_diffs: FieldDiff[] = []
  const extension_diffs: ExtensionDiff[] = []
  const all_core: { key: string; a: string | null; b: string | null; changed: boolean }[] = []

  const adoptARaw: Record<string, string> = {}
  const adoptBRaw: Record<string, string> = {}

  for (const key of allKeys) {
    const va = mapA.get(key) ?? null
    const vb = mapB.get(key) ?? null
    const extGroup = extensionGroupOf(key)
    const changed = va !== vb
    if (extGroup) {
      if (changed) {
        extension_diffs.push({ group: extGroup, key, a: va, b: vb, kind: diffKind(va, vb) })
        if (va !== null) adoptARaw[key] = va
        if (vb !== null) adoptBRaw[key] = vb
      }
    } else if (CORE_PARAM_KEYS.has(key) || !key.includes(" ")) {
      all_core.push({ key, a: va, b: vb, changed })
      if (changed) {
        core_diffs.push({ key, a: va, b: vb, kind: diffKind(va, vb) })
        if (va !== null) adoptARaw[key] = va
        if (vb !== null) adoptBRaw[key] = vb
      }
    } else {
      // Multi-word non-core parameter: group under its first word as an extension parameter.
      const group = key.split(" ")[0] ?? "Extension"
      if (changed) {
        extension_diffs.push({ group, key, a: va, b: vb, kind: diffKind(va, vb) })
        if (va !== null) adoptARaw[key] = va
        if (vb !== null) adoptBRaw[key] = vb
      }
    }
  }

  const identical =
    prompt_diff.added.length === 0 &&
    prompt_diff.removed.length === 0 &&
    negative_diff.added.length === 0 &&
    negative_diff.removed.length === 0 &&
    core_diffs.length === 0 &&
    lora_diffs.length === 0 &&
    extension_diffs.length === 0

  return {
    infotext_a: aRaw || null,
    infotext_b: bRaw || null,
    identical,
    prompt_diff,
    negative_diff,
    core_diffs,
    lora_diffs,
    extension_diffs,
    all_core,
    adopt_a_patch: { core: {}, scripts: {}, raw_params: adoptARaw },
    adopt_b_patch: { core: {}, scripts: {}, raw_params: adoptBRaw },
  }
}

/** Total number of structured differences (prompts + core + LoRAs + extensions). */
export function totalDiffCount(diff: CompareDiffResult): number {
  const promptDelta = diff.prompt_diff.added.length > 0 || diff.prompt_diff.removed.length > 0 ? 1 : 0
  const negDelta = diff.negative_diff.added.length > 0 || diff.negative_diff.removed.length > 0 ? 1 : 0
  return promptDelta + negDelta + diff.core_diffs.length + diff.lora_diffs.length + diff.extension_diffs.length
}

/** Calls `POST /api/v2/gallery/compare`; falls back cleanly to client-side `diffInfotext`. */
export async function fetchCompareDiff(
  a: { id?: string; infotext?: string | null },
  b: { id?: string; infotext?: string | null },
  signal?: AbortSignal,
): Promise<CompareDiffResult> {
  const local = diffInfotext(a.infotext, b.infotext)
  try {
    const server = await workbenchRequest<CompareDiffResult>("/gallery/compare", {
      method: "POST",
      body: {
        ...(a.id ? { id_a: a.id } : {}),
        ...(b.id ? { id_b: b.id } : {}),
        ...(a.infotext ? { infotext_a: a.infotext } : {}),
        ...(b.infotext ? { infotext_b: b.infotext } : {}),
      },
      signal,
    })
    if (server && typeof server === "object" && "prompt_diff" in server && "core_diffs" in server) {
      // Preserve client-side `all_core` table for "Show all" toggle while trusting server diff & patches.
      const fallbackAll = diffInfotext(server.infotext_a ?? a.infotext, server.infotext_b ?? b.infotext).all_core
      return { ...server, all_core: server.all_core ?? fallbackAll }
    }
    return local
  } catch {
    return local
  }
}

/** Converts an `AdoptPatch` (`{core, scripts}`) into `PasteEntry[]` for `deliver()`. */
export function patchToEntries(patch: AdoptPatch | undefined): PasteEntry[] {
  if (!patch) return []
  const entries: PasteEntry[] = []
  if (patch.core && Object.keys(patch.core).length > 0) entries.push({ owner: "core", values: patch.core })
  for (const [owner, values] of Object.entries(patch.scripts ?? {})) {
    if (values && Object.keys(values).length > 0) entries.push({ owner, values })
  }
  return entries
}

/**
 * Applies ONLY the differing parameters of side `"a"` or `"b"` into `tab`:
 * uses the server's `adopt_*_patch` when populated, else parses the chosen
 * side's infotext and keeps only parameters whose keys differ from the other side.
 * Resolves with the params set and the ones a lock kept.
 */
export async function applyCompareDiffToForm(diff: CompareDiffResult, side: "a" | "b", tab: FormTab = "txt2img"): Promise<PasteResult> {
  const targetTab: FormTab = tab === "extras" ? "txt2img" : tab
  const serverPatch = side === "a" ? diff.adopt_a_patch : diff.adopt_b_patch
  const direct = patchToEntries(serverPatch)
  if (direct.length > 0) return deliver(targetTab, direct, {})

  const chosenInfotext = side === "a" ? diff.infotext_a : diff.infotext_b
  const otherInfotext = side === "a" ? diff.infotext_b : diff.infotext_a
  if (!chosenInfotext) return { count: 0, locked: [] }

  const [parsedChosen, parsedOther] = await Promise.all([
    api.parseInfotext(chosenInfotext, targetTab),
    otherInfotext ? api.parseInfotext(otherInfotext, targetTab).catch(() => null) : Promise.resolve(null),
  ])

  const entries: PasteEntry[] = []
  const diffValues = (chosen: ParamValues | undefined, other: ParamValues | undefined): ParamValues => {
    const out: ParamValues = {}
    for (const [k, v] of Object.entries(chosen ?? {})) {
      if (!other || JSON.stringify(other[k]) !== JSON.stringify(v)) out[k] = v
    }
    return out
  }

  const coreDiff = diffValues(parsedChosen.core, parsedOther?.core)
  if (Object.keys(coreDiff).length > 0) entries.push({ owner: "core", values: coreDiff })
  for (const [owner, values] of Object.entries(parsedChosen.scripts ?? {})) {
    const scriptDiff = diffValues(values, parsedOther?.scripts?.[owner])
    if (Object.keys(scriptDiff).length > 0) entries.push({ owner, values: scriptDiff })
  }
  if (entries.length === 0) return { count: 0, locked: [] }
  return deliver(targetTab, entries, {})
}
