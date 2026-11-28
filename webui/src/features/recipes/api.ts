/**
 * Thin typed API client for Cross-Model Parameter Recipes (`/api/v2/recipes`),
 * backed by `@/api/endpoints`.
 */
import { queryOptions } from "@tanstack/react-query"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type {
  Recipe,
  RecipeApplyRequest,
  RecipeApplyResponse,
  RecipeCreate,
  RecipeExportBundle,
  RecipeImportResponse,
  RecipeListResponse,
  RecipeQuery,
  RecipeUpdate,
} from "./types"

export type { RecipeQuery }

export const recipeKeys = {
  all: ["recipes"] as const,
  list: (filters: RecipeQuery = {}) => ["recipes", "list", filters] as const,
}

export function refreshRecipes() {
  void queryClient.invalidateQueries({ queryKey: recipeKeys.all })
}

export async function listRecipes(query: RecipeQuery = {}, signal?: AbortSignal): Promise<RecipeListResponse> {
  const data = (await api.recipes.list(query, signal)) as RecipeListResponse | Recipe[] | undefined
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
  return api.recipes.create(body)
}

export async function updateRecipeApi(id: string, body: RecipeUpdate): Promise<Recipe> {
  return api.recipes.update(id, body)
}

export async function deleteRecipeApi(id: string): Promise<void> {
  await api.recipes.delete(id)
}

export async function exportRecipesApi(): Promise<RecipeExportBundle> {
  return api.recipes.exportRecipes()
}

export async function importRecipesApi(bundle: { version?: number; recipes: Partial<Recipe>[]; overwrite?: boolean }): Promise<RecipeImportResponse> {
  return api.recipes.importRecipes({
    version: bundle.version ?? 1,
    overwrite: bundle.overwrite ?? false,
    recipes: bundle.recipes as Record<string, unknown>[],
  })
}

export async function applyRecipeApi(id: string, body: RecipeApplyRequest = {}): Promise<RecipeApplyResponse> {
  return api.recipes.apply(id, body)
}
