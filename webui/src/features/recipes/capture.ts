/**
 * Pure form override capture & recipe application helpers: extracts only
 * non-default, persistable parameters from a form store so Recipes stay clean,
 * minimal and safe to apply across models.
 */
import { queryClient, schemaQuery } from "@/api/queries"
import type { FormTab, ParamValues } from "@/api/types"
import { t } from "@/i18n"
import type { PasteEntry } from "@/features/infotext/paste"
import { deliver } from "@/features/send-to/send-to"
import { ensureSchemaLoaded } from "@/pages/generate/use-form-loader"
import { jsonEqual } from "@/ui/conditions"
import { getFormStore } from "@/ui/form-store"
import { isPersisted } from "@/ui/values"
import type { Recipe, RecipeApplyResponse } from "./types"

export type CapturedOverrides = {
  core_patch: ParamValues
  scripts_patch: Record<string, ParamValues>
  count: number
}

/**
 * Captures every modified, persistable parameter from `tab`'s form store. A value counts as
 * modified only against a default (as `isModified`): a form that was never loaded holds its drafts
 * and no defaults, so read it through `captureTabOverrides`.
 */
export function captureFormOverrides(tab: FormTab): CapturedOverrides {
  const state = getFormStore(tab).getState()
  const core_patch: ParamValues = {}
  const scripts_patch: Record<string, ParamValues> = {}
  let count = 0

  for (const [owner, ownerValues] of Object.entries(state.values)) {
    const defaults = state.defaults[owner] ?? {}
    const nodes = state.index.inputs.get(owner)
    const diff: ParamValues = {}
    for (const [name, value] of Object.entries(ownerValues)) {
      const node = nodes?.get(name)
      // Never capture transient uploads or `persist: false` fields into portable recipes.
      if (node && (!isPersisted(node) || node.persist === false)) continue
      if (!Object.hasOwn(defaults, name) || jsonEqual(value, defaults[name])) continue
      diff[name] = value
      count++
    }
    if (Object.keys(diff).length === 0) continue
    if (owner === "core") Object.assign(core_patch, diff)
    else scripts_patch[owner] = diff
  }

  return { core_patch, scripts_patch, count }
}

/** `captureFormOverrides` for a form that may not be open: loads its schema into the store first. */
export async function captureTabOverrides(tab: FormTab): Promise<CapturedOverrides> {
  const store = getFormStore(tab)
  if (store.getState().revision === 0) ensureSchemaLoaded(store, await queryClient.ensureQueryData(schemaQuery(tab)))
  return captureFormOverrides(tab)
}

/** Total parameter count inside a recipe (`core_patch` + every script's overrides). */
export function countRecipeParams(recipe: Pick<Recipe, "core_patch" | "scripts_patch">): number {
  let total = Object.keys(recipe.core_patch ?? {}).length
  for (const patch of Object.values(recipe.scripts_patch ?? {})) {
    total += Object.keys(patch ?? {}).length
  }
  return total
}

/** Builds a concise summary of key parameter chips (`steps`, `cfg_scale`, `sampler_name`, etc.), in the UI language. */
export function recipeHighlights(recipe: Pick<Recipe, "core_patch" | "scripts_patch">): string[] {
  const chips: string[] = []
  const core = recipe.core_patch ?? {}
  const sampler = recipe.scripts_patch?.sampler ?? {}
  if (sampler.sampler_name) chips.push(String(sampler.sampler_name))
  if (sampler.steps !== undefined) chips.push(t("{{count}} steps", { count: Number(sampler.steps) }))
  if (core.cfg_scale !== undefined) chips.push(`CFG ${String(core.cfg_scale)}`)
  if (core.width && core.height) chips.push(`${String(core.width)}×${String(core.height)}`)
  if (core.denoising_strength !== undefined) chips.push(t("denoise {{value}}", { value: String(core.denoising_strength) }))
  const extraScripts = Object.keys(recipe.scripts_patch ?? {}).filter((key) => key !== "sampler" && key !== "seed")
  for (const name of extraScripts.slice(0, 2)) chips.push(name)
  return chips
}

/**
 * What applying a recipe changes: the targets the server lists in `patch` ("owner.param", the
 * recipe's own values plus the ones its conflicts derive). The response's `core` / `scripts` are
 * the whole form merged with the recipe, so applying those would rewrite every field (undoing edits
 * made meanwhile) and count them all. Without a `patch` the recipe's own values are applied.
 */
export function appliedRecipeValues(
  response: Partial<Pick<RecipeApplyResponse, "patch">>,
  recipe: Pick<Recipe, "core_patch" | "scripts_patch">,
): { core: ParamValues; scripts: Record<string, ParamValues> } {
  const core: ParamValues = {}
  const scripts: Record<string, ParamValues> = {}
  for (const [target, entry] of Object.entries(response.patch ?? {})) {
    const dot = target.indexOf(".")
    if (target.startsWith("__") || dot <= 0 || !entry || typeof entry !== "object" || !("value" in entry)) continue
    const owner = target.slice(0, dot)
    const values = owner === "core" ? core : (scripts[owner] ??= {})
    values[target.slice(dot + 1)] = (entry as { value: unknown }).value
  }
  if (Object.keys(core).length === 0 && Object.keys(scripts).length === 0) return { core: { ...recipe.core_patch }, scripts: structuredClone(recipe.scripts_patch) }
  return { core, scripts }
}

/** Applies a recipe's `core_patch` and `scripts_patch` into its target form via `deliver()`. */
export function applyRecipeEntries(
  tab: FormTab,
  core: ParamValues,
  scripts: Record<string, ParamValues>,
): Promise<number> {
  const entries: PasteEntry[] = []
  if (core && Object.keys(core).length > 0) entries.push({ owner: "core", values: core })
  for (const [owner, values] of Object.entries(scripts ?? {})) {
    if (values && Object.keys(values).length > 0) entries.push({ owner, values })
  }
  if (entries.length === 0) return Promise.resolve(0)
  return deliver(tab, entries, {})
}
