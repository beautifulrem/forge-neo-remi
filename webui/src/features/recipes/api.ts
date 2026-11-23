/**
 * Thin typed API client for Cross-Model Parameter Recipes (`/api/v2/recipes`),
 * backed by `workbenchRequest` (which routes `VITE_MOCK=1` through `@/api/mock/workbench`).
 */
import { queryOptions } from "@tanstack/react-query"
import { queryClient } from "@/api/queries"
import type { FormTab, ParamValues } from "@/api/types"
import { workbenchRequest } from "@/features/pipelines/workbench-http"
import type { Recipe, RecipeApplyResponse, RecipeCreate, RecipeExportBundle, RecipeImportResponse, RecipeListResponse } from "./types"

export const recipeKeys = {
  all: ["recipes"] as const,
  list: (filters: RecipeQuery = {}) => ["recipes", "list", filters] as const,
}

export type RecipeQuery = {
  q?: string
  model_family?: string
  tab?: FormTab
  tag?: string
  capability?: string
}

export function refreshRecipes() {
  void queryClient.invalidateQueries({ queryKey: recipeKeys.all })
}

export async function listRecipes(query: RecipeQuery = {}, signal?: AbortSignal): Promise<RecipeListResponse> {
  const data = await workbenchRequest<RecipeListResponse | Recipe[]>("/recipes", { query, signal })
  if (Array.isArray(data)) return { items: data, total: data.length }
  return { items: Array.isArray(data?.items) ? data.items : [], total: typeof data?.total === "number" ? data.total : 0 }
}

export const recipesQuery = (query: RecipeQuery = {}) =>
  queryOptions({
    queryKey: recipeKeys.list(query),
    queryFn: ({ signal }) => listRecipes(query, signal),
    staleTime: 15_000,
  })

export async function createRecipeApi(body: RecipeCreate): Promise<Recipe> {
  return workbenchRequest<Recipe>("/recipes", { method: "POST", body })
}

export async function deleteRecipeApi(id: string): Promise<void> {
  await workbenchRequest<void>(`/recipes/${encodeURIComponent(id)}`, { method: "DELETE" })
}

export async function exportRecipesApi(): Promise<RecipeExportBundle> {
  return workbenchRequest<RecipeExportBundle>("/recipes/export")
}

export async function importRecipesApi(bundle: { version?: number; recipes: Partial<Recipe>[]; overwrite?: boolean }): Promise<RecipeImportResponse> {
  return workbenchRequest<RecipeImportResponse>("/recipes/import", { method: "POST", body: bundle })
}

export async function applyRecipeApi(
  id: string,
  body: { tab?: FormTab; core?: ParamValues; scripts?: Record<string, ParamValues>; check_conflicts?: boolean } = {},
): Promise<RecipeApplyResponse> {
  return workbenchRequest<RecipeApplyResponse>(`/recipes/${encodeURIComponent(id)}/apply`, {
    method: "POST",
    body: { check_conflicts: true, ...body },
  })
}
