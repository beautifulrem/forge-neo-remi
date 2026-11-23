/** Cross-model parameter Recipes types (matching backend contract #12). */
import type { FormTab, ParamValues } from "@/api/types"

export type Recipe = {
  id: string
  name: string
  description: string
  model_family: string
  capabilities: string[]
  tab: FormTab
  core_patch: ParamValues
  scripts_patch: Record<string, ParamValues>
  tags: string[]
  created_at: string
  updated_at: string
}

export type RecipeCreate = {
  name: string
  description?: string
  model_family?: string
  capabilities?: string[]
  tab?: FormTab
  core_patch: ParamValues
  scripts_patch: Record<string, ParamValues>
  tags?: string[]
}

export type RecipeListResponse = {
  items: Recipe[]
  total: number
}

export type RecipeExportBundle = {
  version: number
  recipes: Recipe[]
}

export type RecipeImportResponse = {
  added: number
  updated: number
  skipped: number
  items: Recipe[]
}

export type RecipeApplyResponse = {
  recipe: Recipe
  tab: FormTab
  patch: Record<string, unknown>
  core: ParamValues
  scripts: Record<string, ParamValues>
  conflicts: Record<string, unknown>
  warnings: string[]
}
