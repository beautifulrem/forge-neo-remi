import type { Recipe } from "./types"

/** A recipe's export file name: its name with what file systems reject replaced (its id when nothing is left). */
export function recipeFileName(recipe: Pick<Recipe, "id" | "name">): string {
  const name = recipe.name
    // oxlint-disable-next-line no-control-regex -- control characters are what file systems reject
    .replace(/[\\/:*?"<>|\u0000-\u001f]+/g, "-")
    .replace(/^[\s.-]+|[\s.-]+$/g, "")
  return `${name || recipe.id}.json`
}
