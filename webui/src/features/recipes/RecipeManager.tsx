/**
 * Cross-Model Parameter Recipes Manager: dense Linear / Capture One preset
 * matrix without pill clutter — clean typographic hierarchy, inline mono
 * capability/parameter readouts, JSON import/export, and one-click Apply.
 * Lazily loaded.
 */
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query"
import { BookOpen, ClipboardPaste, Download, Plus, Search, Trash2, Upload } from "@/components/icons"
import { useDeferredValue, useRef, useState } from "react"
import { useRouteParams, useRouteSearch } from "@/app/route-params"
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
import { PRESSED_TINT_CLASS } from "@/components/ui/toggle"
import { errorText } from "@/features/errors/error-text"
import { cn } from "@/lib/utils"
import { getFormStore } from "@/ui/form-store"
import { panelClass, panelTitleClass } from "@/ui/layout/panel"
import { applyRecipeApi, createRecipeApi, deleteRecipeApi, exportRecipesApi, importRecipesApi, recipesQuery, refreshRecipes } from "./api"
import { CATEGORY_LABELS } from "@/features/library/labels"
import { appliedRecipeValues, applyRecipeEntries, type CapturedOverrides, captureTabOverrides, countRecipeParams, recipeHighlights } from "./capture"
import type { Recipe } from "./types"
import { LoadFailed } from "@/components/LoadFailed"
import { recipeFileName } from "./file-name"

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

/** The filters in the address, beside the queue's (`#/queue?view=recipes&recipes=turbo&family=Qwen`). */
const RECIPE_PARAMS = { recipes: "", family: "" }

export function RecipeManager() {
  const { t, i18n } = useTranslation()
  const [params, setParams] = useRouteParams(RECIPE_PARAMS)
  const [search, setSearch] = useRouteSearch(params.recipes, (recipes) => setParams({ recipes }))
  const family = params.family
  const setFamily = (next: string) => setParams({ family: next })
  const [captureOpen, setCaptureOpen] = useState(false)
  // focused through the dialog (`initialFocus`), not `autoFocus`: see the JSON Builder's Load prompt
  const nameRef = useRef<HTMLInputElement>(null)
  const [captureTab, setCaptureTab] = useState<FormTab>("txt2img")
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [modelFamily, setModelFamily] = useState("")
  const [tagsText, setTagsText] = useState("")
  const [deleting, setDeleting] = useState<Recipe | null>(null)
  const importInputRef = useRef<HTMLInputElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)

  const q = useDeferredValue(search.trim())
  const filtered = q !== "" || family !== ""
  // the previous list stays while a new search loads, so typing does not flash the skeleton
  const list = useQuery({ ...recipesQuery({ ...(q ? { q } : {}), ...(family ? { model_family: family } : {}) }), placeholderData: keepPreviousData })
  const recipes = list.data?.items ?? []

  // what the capture dialog read from the source form (null while it reads: a form never opened loads its schema first)
  const [capture, setCapture] = useState<{ tab: FormTab; overrides?: CapturedOverrides; error?: unknown } | null>(null)
  const captureRequest = useRef(0)
  const readForm = (tab: FormTab) => {
    setCaptureTab(tab)
    setCapture(null)
    const request = ++captureRequest.current
    captureTabOverrides(tab).then(
      (overrides) => request === captureRequest.current && setCapture({ tab, overrides }),
      (error: unknown) => request === captureRequest.current && setCapture({ tab, error }),
    )
  }
  const captured = capture?.overrides ?? { core_patch: {}, scripts_patch: {}, count: 0 }

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
      setModelFamily("")
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
      // the server checks conflicts and capabilities: a failed apply is reported, not replaced by a local one
      const res = await applyRecipeApi(recipe.id, {
        tab: recipe.tab,
        core: store.values.core ?? {},
        scripts: Object.fromEntries(Object.entries(store.values).filter(([k]) => k !== "core")),
      })
      const { core, scripts } = appliedRecipeValues(res, recipe)
      const tab = res.tab ?? recipe.tab
      const warnings = res.warnings ?? []
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
      toast.error(t("Could not export the recipes"), { description: errorText(error) })
    }
  }

  const handleImportFile = async (file: File) => {
    let parsed: unknown
    try {
      parsed = JSON.parse(await file.text())
    } catch {
      parsed = null
    }
    if (!parsed || typeof parsed !== "object") {
      toast.error(t("Could not import recipes"), { description: t("This file is not a recipes export.") })
      return
    }
    try {
      const bundle = parsed as { version?: number; recipes?: Partial<Recipe>[] } | Partial<Recipe>[] | Partial<Recipe>
      const listToImport: Partial<Recipe>[] = Array.isArray(bundle)
        ? bundle
        : "recipes" in bundle && Array.isArray(bundle.recipes)
          ? bundle.recipes
          : [bundle as Partial<Recipe>]
      const res = await importRecipesApi({ version: 1, recipes: listToImport, overwrite: true })
      refreshRecipes()
      toast.success(t("Imported {{added}} recipes ({{updated}} updated)", { added: res.added, updated: res.updated }))
    } catch (error) {
      toast.error(t("Could not import recipes"), { description: errorText(error) })
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {/* the toolbar sits above the panel; the panel's hairline separates them */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-44 flex-1 sm:max-w-xs">
          <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input
            ref={searchRef}
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
                "h-7 rounded-sm px-2 text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-focus-ring",
                item && "font-mono",
                "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                PRESSED_TINT_CLASS,
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
          <Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" disabled={!filtered && recipes.length === 0} onClick={() => void handleExportAll()}>
            <Download className="size-3.5" aria-hidden />
            {t("Export JSON")}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            className="h-7 gap-1.5 px-3 text-xs font-medium"
            onClick={() => {
              setCaptureOpen(true)
              readForm(captureTab)
            }}
          >
            <Plus className="size-3.5" aria-hidden />
            {t("Capture from form")}
          </Button>
        </div>
      </div>

      {/* an inspector panel (`ui/layout/panel.ts`), not a card */}
      <section className={cn("flex flex-col gap-1", panelClass(0))} aria-label={t("Cross-model parameter recipes")}>
        <div className="flex min-h-8 items-center justify-between">
          <h2 className={panelTitleClass(0)}>
            {t("Cross-model parameter recipes")}
            {/* not after a failed load: "0" would say the library is empty */}
            {list.data ? <span className="ml-2 font-mono text-xs font-normal text-muted-foreground tabular-nums">{recipes.length}</span> : null}
          </h2>
        </div>

        {list.isPending ? (
          <div className="flex flex-col gap-2 py-2" aria-busy="true">
            {Array.from({ length: 3 }, (_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : list.isError && !list.data ? (
          <LoadFailed error={list.error} onRetry={() => void list.refetch()} />
        ) : recipes.length === 0 && filtered ? (
          // a placeholder: a dimmed glyph over one muted line, and the way out
          <div className="flex flex-col items-center gap-2 py-10 text-center">
            <BookOpen className="size-7 text-muted-foreground/45" aria-hidden />
            <p className="text-sm text-muted-foreground">{t("No recipes match")}</p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setSearch("")
                setParams({ recipes: "", family: "" })
                // this button leaves with the state: the focus goes to the search, not to the page
                searchRef.current?.focus()
              }}
            >
              {t("Clear filters")}
            </Button>
          </div>
        ) : recipes.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-10 text-center">
            <BookOpen className="size-7 text-muted-foreground/45" aria-hidden />
            <p className="text-sm font-medium">{t("No recipes yet")}</p>
            <p className="max-w-md text-xs text-balance text-muted-foreground">
              {t("Capture your current form tweaks as a reusable cross-model recipe, or import one from JSON.")}
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-border/70">
            {recipes.map((recipe) => {
              const highlights = recipeHighlights(recipe, i18n.language)
              const paramCount = countRecipeParams(recipe)
              const metaParts = [recipe.model_family, t(CATEGORY_LABELS[recipe.tab]), t("{{count}} params", { count: paramCount })].filter(Boolean)
              const detailParts = [...highlights, ...recipe.capabilities, ...recipe.tags.map((tag) => `#${tag}`)]

              return (
                <li key={recipe.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3">
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
                      aria-label={t("Export this recipe")}
                      title={t("Export this recipe")}
                      onClick={() => downloadJson(recipeFileName(recipe), { version: 1, recipes: [recipe] })}
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
        <DialogContent className="sm:max-w-lg" initialFocus={nameRef}>
          <DialogHeader>
            <DialogTitle>{t("Capture recipe")}</DialogTitle>
            <DialogDescription>{capture?.overrides ? t("Captured overrides: {{count}}", { count: captured.count }) : capture?.error ? t("Could not read the form") : t("Reading…")}</DialogDescription>
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
                onValueChange={(value) => readForm(value)}
                options={[
                  { value: "txt2img", label: t(CATEGORY_LABELS.txt2img) },
                  { value: "img2img", label: t(CATEGORY_LABELS.img2img) },
                  { value: "extras", label: t(CATEGORY_LABELS.extras) },
                ]}
              />
            </div>

            {capture === null ? null : capture.error !== undefined ? (
              <LoadFailed error={capture.error} onRetry={() => readForm(captureTab)} className="py-0" />
            ) : captured.count === 0 ? (
              <p className="rounded-md bg-background px-3 py-2 text-xs text-muted-foreground">
                {t("No modified parameters found on {{tab}}.", { tab: t(CATEGORY_LABELS[captureTab]) })}
              </p>
            ) : (
              <div className="-mx-1 max-h-28 overflow-y-auto px-1">
                <pre className="rounded-md bg-background p-2 font-mono text-[0.6875rem] text-muted-foreground">
                  {JSON.stringify({ core: captured.core_patch, scripts: captured.scripts_patch }, null, 2)}
                </pre>
              </div>
            )}

            <label className="flex flex-col gap-1 text-xs font-medium">
              {t("Recipe name")}
              <Input ref={nameRef} value={name} onChange={(event) => setName(event.target.value)} placeholder={t("Recipe name")} className="h-8" required />
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
              <Button type="submit" size="sm" disabled={!name.trim() || save.isPending || !capture?.overrides}>
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
