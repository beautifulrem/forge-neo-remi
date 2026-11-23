/**
 * Cross-Model Parameter Recipes Manager: dense Linear / Capture One preset
 * matrix without pill clutter — clean typographic hierarchy, inline mono
 * capability/parameter readouts, JSON import/export, and one-click Apply.
 * Lazily loaded.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { BookOpen, ClipboardPaste, Download, Plus, Search, Trash2, Upload } from "@/components/icons"
import { useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import type { FormTab } from "@/api/types"
import { Segmented } from "@/components/Segmented"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { errorText } from "@/features/errors/error-text"
import { cn } from "@/lib/utils"
import { getFormStore } from "@/ui/form-store"
import { applyRecipeApi, createRecipeApi, deleteRecipeApi, exportRecipesApi, importRecipesApi, recipesQuery, refreshRecipes } from "./api"
import { applyRecipeEntries, captureFormOverrides, countRecipeParams, recipeHighlights } from "./capture"
import type { Recipe } from "./types"

const FAMILIES = ["", "anima", "qwen21", "flux3", "moss", "zimage", "sdxl"] as const

function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function RecipeManager() {
  const { t } = useTranslation()
  const [search, setSearch] = useState("")
  const [family, setFamily] = useState("")
  const [captureOpen, setCaptureOpen] = useState(false)
  const [captureTab, setCaptureTab] = useState<FormTab>("txt2img")
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [modelFamily, setModelFamily] = useState("")
  const [tagsText, setTagsText] = useState("")
  const [deleting, setDeleting] = useState<Recipe | null>(null)
  const importInputRef = useRef<HTMLInputElement>(null)

  const list = useQuery(recipesQuery({ ...(search.trim() ? { q: search.trim() } : {}), ...(family ? { model_family: family } : {}) }))
  const recipes = list.data?.items ?? []

  const captured = captureOpen ? captureFormOverrides(captureTab) : { core_patch: {}, scripts_patch: {}, count: 0 }

  const save = useMutation({
    mutationFn: () =>
      createRecipeApi({
        name: name.trim(),
        description: description.trim(),
        model_family: modelFamily.trim(),
        tab: captureTab,
        core_patch: captured.core_patch,
        scripts_patch: captured.scripts_patch,
        tags: tagsText
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean),
      }),
    onSuccess: () => {
      refreshRecipes()
      setCaptureOpen(false)
      setName("")
      setDescription("")
      setTagsText("")
      toast.success(t("Recipe saved"))
    },
    onError: (error) => toast.error(t("Could not save recipe"), { description: errorText(error) }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => deleteRecipeApi(id),
    onSuccess: () => {
      refreshRecipes()
      setDeleting(null)
      toast.success(t("Recipe deleted"))
    },
    onError: (error) => toast.error(t("Could not delete recipe"), { description: errorText(error) }),
  })

  const apply = useMutation({
    mutationFn: async (recipe: Recipe) => {
      const store = getFormStore(recipe.tab).getState()
      let core = recipe.core_patch
      let scripts = recipe.scripts_patch
      let tab = recipe.tab
      let warnings: string[] = []
      try {
        const res = await applyRecipeApi(recipe.id, {
          tab: recipe.tab,
          core: store.values.core ?? {},
          scripts: Object.fromEntries(Object.entries(store.values).filter(([k]) => k !== "core")),
        })
        core = res.core ?? core
        scripts = res.scripts ?? scripts
        tab = res.tab ?? tab
        warnings = res.warnings ?? []
      } catch {
        // Fallback to applying local recipe patches directly if apply endpoint is unavailable.
      }
      const count = await applyRecipeEntries(tab, core, scripts)
      return { recipe, count, warnings }
    },
    onSuccess: ({ recipe, count, warnings }) => {
      toast.success(t("Recipe “{{name}}” applied ({{count}} params)", { name: recipe.name, count }), {
        description: warnings.length > 0 ? warnings.join("; ") : undefined,
      })
    },
    onError: (error) => toast.error(t("Could not apply recipe"), { description: errorText(error) }),
  })

  const handleExportAll = async () => {
    try {
      const bundle = await exportRecipesApi()
      downloadJson("remi-recipes.json", bundle)
      toast.success(t("Exported {{count}} recipes", { count: bundle.recipes.length }))
    } catch (error) {
      toast.error(t("Could not save recipe"), { description: errorText(error) })
    }
  }

  const handleImportFile = async (file: File) => {
    try {
      const text = await file.text()
      const parsed = JSON.parse(text) as { version?: number; recipes?: Partial<Recipe>[] } | Partial<Recipe>[] | Partial<Recipe>
      const listToImport: Partial<Recipe>[] = Array.isArray(parsed)
        ? parsed
        : "recipes" in parsed && Array.isArray(parsed.recipes)
          ? parsed.recipes
          : [parsed as Partial<Recipe>]
      const res = await importRecipesApi({ version: 1, recipes: listToImport, overwrite: true })
      refreshRecipes()
      toast.success(t("Imported {{added}} recipes ({{updated}} updated)", { added: res.added, updated: res.updated }))
    } catch (error) {
      toast.error(t("Could not import recipes"), { description: errorText(error) })
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2 border-b border-border/70 pb-3">
        <div className="relative min-w-44 flex-1 sm:max-w-xs">
          <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={t("Search recipes…")}
            aria-label={t("Search recipes…")}
            className="h-7 pl-7 text-xs"
          />
        </div>

        <div className="-mx-1 flex flex-wrap items-center gap-0.5 px-1" role="group" aria-label={t("Model family")}>
          {FAMILIES.map((item) => (
            <button
              key={item || "all"}
              type="button"
              aria-pressed={family === item}
              onClick={() => setFamily(item)}
              className={cn(
                "h-7 rounded-sm px-2 text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                item && "font-mono",
                family === item ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
              )}
            >
              {item ? item : t("All families")}
            </button>
          ))}
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-1.5">
          <Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" onClick={() => importInputRef.current?.click()}>
            <Upload className="size-3.5" aria-hidden />
            {t("Import JSON")}
          </Button>
          <input
            ref={importInputRef}
            type="file"
            accept="application/json,.json"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0]
              event.target.value = ""
              if (file) void handleImportFile(file)
            }}
          />
          <Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" disabled={recipes.length === 0} onClick={() => void handleExportAll()}>
            <Download className="size-3.5" aria-hidden />
            {t("Export JSON")}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            className="h-7 gap-1.5 px-3 text-xs font-medium"
            onClick={() => setCaptureOpen(true)}
          >
            <Plus className="size-3.5" aria-hidden />
            {t("Capture from form")}
          </Button>
        </div>
      </div>

      <section className="overflow-hidden rounded-lg border border-border bg-card" aria-label={t("Cross-model parameter recipes")}>
        <div className="flex items-center justify-between border-b border-border/70 px-4 py-2.5">
          <h2 className="text-xs font-semibold tracking-wide text-foreground">
            {t("Cross-model parameter recipes")}
            {!list.isPending ? <span className="ml-2 font-mono font-normal text-muted-foreground tabular-nums">{recipes.length}</span> : null}
          </h2>
        </div>

        {list.isPending ? (
          <div className="flex flex-col gap-2 p-4" aria-busy="true">
            {Array.from({ length: 3 }, (_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : list.isError ? (
          <p role="alert" className="px-4 py-4 text-sm text-destructive">
            {errorText(list.error)}
          </p>
        ) : recipes.length === 0 ? (
          <div className="flex flex-col items-center gap-2 px-4 py-10 text-center">
            <BookOpen className="size-7 text-muted-foreground/45" aria-hidden />
            <p className="text-sm font-medium">{t("No recipes yet")}</p>
            <p className="max-w-md text-xs text-balance text-muted-foreground">
              {t("Capture your current form tweaks as a reusable cross-model recipe, or import one from JSON.")}
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-border/70">
            {recipes.map((recipe) => {
              const highlights = recipeHighlights(recipe)
              const paramCount = countRecipeParams(recipe)
              const metaParts = [recipe.model_family, recipe.tab, t("{{count}} params", { count: paramCount })].filter(Boolean)
              const detailParts = [...highlights, ...recipe.capabilities, ...recipe.tags.map((tag) => `#${tag}`)]

              return (
                <li key={recipe.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 transition-colors hover:bg-muted/25">
                  <div className="min-w-0 flex-1 basis-64">
                    <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
                      <span className="truncate text-[0.8rem] font-semibold text-foreground" title={recipe.name}>
                        {recipe.name}
                      </span>
                      <span className="font-mono text-xs text-muted-foreground tabular-nums">{metaParts.join(" · ")}</span>
                    </div>
                    {recipe.description ? (
                      <p className="mt-0.5 truncate text-xs text-muted-foreground" title={recipe.description}>
                        {recipe.description}
                      </p>
                    ) : null}
                    {detailParts.length > 0 ? (
                      <p className="mt-1 truncate font-mono text-[0.6875rem] text-muted-foreground" title={detailParts.join(" · ")}>
                        {detailParts.join(" · ")}
                      </p>
                    ) : null}
                  </div>

                  <div className="flex shrink-0 items-center gap-1">
                    <Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" disabled={apply.isPending} onClick={() => apply.mutate(recipe)}>
                      <ClipboardPaste className="size-3.5" aria-hidden />
                      {t("Apply recipe")}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="size-7 text-muted-foreground hover:text-foreground"
                      aria-label={t("Export JSON")}
                      title={t("Export JSON")}
                      onClick={() => downloadJson(`${recipe.name.replace(/[^a-zA-Z0-9_-]+/g, "-") || recipe.id}.json`, { version: 1, recipes: [recipe] })}
                    >
                      <Download className="size-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="size-7 text-muted-foreground hover:text-destructive"
                      aria-label={t("Delete recipe")}
                      title={t("Delete recipe")}
                      onClick={() => setDeleting(recipe)}
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </section>

      <Dialog open={captureOpen} onOpenChange={setCaptureOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{t("Capture recipe")}</DialogTitle>
            <DialogDescription>{t("Captured overrides: {{count}}", { count: captured.count })}</DialogDescription>
          </DialogHeader>
          <form
            className="flex flex-col gap-3"
            onSubmit={(event) => {
              event.preventDefault()
              if (name.trim() && !save.isPending) save.mutate()
            }}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium">{t("Source form")}</span>
              <Segmented
                size="sm"
                aria-label={t("Source form")}
                value={captureTab}
                onValueChange={(value) => setCaptureTab(value)}
                options={[
                  { value: "txt2img", label: "txt2img" },
                  { value: "img2img", label: "img2img" },
                  { value: "extras", label: "Extras" },
                ]}
              />
            </div>

            {captured.count === 0 ? (
              <p className="rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
                {t("No modified parameters found on {{tab}}.", { tab: captureTab })}
              </p>
            ) : (
              <div className="-mx-1 max-h-28 overflow-y-auto px-1">
                <pre className="rounded-md bg-muted p-2 font-mono text-[0.6875rem] text-muted-foreground">
                  {JSON.stringify({ core: captured.core_patch, scripts: captured.scripts_patch }, null, 2)}
                </pre>
              </div>
            )}

            <label className="flex flex-col gap-1 text-xs font-medium">
              {t("Recipe name")}
              <Input value={name} onChange={(event) => setName(event.target.value)} placeholder={t("Recipe name")} className="h-8" autoFocus required />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium">
              {t("Description")}
              <Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder={t("Description")} className="h-8" />
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-1 text-xs font-medium">
                {t("Model family")}
                <Input value={modelFamily} onChange={(event) => setModelFamily(event.target.value)} placeholder="anima / qwen21 / flux3" className="h-8" />
              </label>
              <label className="flex flex-col gap-1 text-xs font-medium">
                {t("Tags (comma-separated)")}
                <Input value={tagsText} onChange={(event) => setTagsText(event.target.value)} placeholder="turbo, portrait" className="h-8" />
              </label>
            </div>

            <DialogFooter>
              <Button type="button" variant="ghost" size="sm" onClick={() => setCaptureOpen(false)}>
                {t("Cancel")}
              </Button>
              <Button type="submit" size="sm" disabled={!name.trim() || save.isPending}>
                {t("Save recipe")}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleting !== null} onOpenChange={(open) => !open && setDeleting(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Delete recipe “{{name}}”?", { name: deleting?.name ?? "" })}</AlertDialogTitle>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" disabled={remove.isPending} onClick={() => deleting && remove.mutate(deleting.id)}>
              {t("Delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
