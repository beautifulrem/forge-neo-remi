/**
 * Header model bar: UI preset, checkpoint (searchable, with architecture and
 * size), VAE / text encoders (`forge_additional_modules`, applied when the
 * picker closes so one change loads the model once) and "Diffusion in Low
 * Bits". While the server loads a model, `model.loading` shows its stage.
 */
import { useQuery } from "@tanstack/react-query"
import { Box, CircleHelp, Hourglass, Layers, Loader2, SlidersHorizontal, TriangleAlert } from "@/components/icons"
import { useEffect, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { useProgress } from "@/api/progress"
import { modelsQuery } from "@/api/queries"
import { LoadFailed } from "@/components/LoadFailed"
import { SearchSelect } from "@/components/SearchSelect"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { errorText, serverText } from "@/features/errors/error-text"
import { FIELD_BUTTON_CLASS } from "@/components/ui/select-styles"
import { cn } from "@/lib/utils"
import { currentRoute } from "@/app/router"
import { getFormStore } from "@/ui/form-store"
import { useStore } from "zustand"
import { deriveModelBadges } from "./model-badges"
import { checkpointLabel, fileName, formatBytes, MODEL_OPTION_KEYS, presetLabel, useApplyPreset, useOptionWrite, writeModules } from "./model-options"
import { WorkerHealthPill } from "./WorkerHealthPill"

const DEFAULT_PRESETS = ["sd", "xl", "flux", "anima"]
const DEFAULT_DTYPES = ["Automatic", "Automatic (fp16 LoRA)", "float8-e4m3fn", "float8-e5m2"]
/** Where to download VAEs and text encoders (the Gradio UI's "?" link next to the modules dropdown). */
export const MODULES_HELP_URL = "https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models"

const useModelBusy = () => useProgress((s) => s.modelLoading)

function PresetSelect({ className }: { className?: string }) {
  const { t } = useTranslation()
  const models = useQuery(modelsQuery)
  const current = useClientOption("forge_preset")
  const apply = useApplyPreset()
  const presets = models.data?.presets ?? DEFAULT_PRESETS
  const value = apply.isPending ? (apply.variables ?? current) : current
  const items = presets.map((id) => ({ value: id, label: presetLabel(id) }))
  return (
    <Select<string> items={items} value={value} onValueChange={(next) => next && next !== current && apply.mutate(next)} disabled={apply.isPending}>
      <SelectTrigger size="sm" aria-label={t("UI preset")} className={cn("min-w-24", className)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent alignItemWithTrigger={false} align="start">
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function CheckpointPicker({ className }: { className?: string }) {
  const { t } = useTranslation()
  const models = useQuery(modelsQuery)
  const current = useClientOption("sd_model_checkpoint")
  const write = useOptionWrite<string>(MODEL_OPTION_KEYS.checkpoint)
  const loading = useModelBusy()
  const options = (models.data?.checkpoints ?? []).map((m) => ({
    value: m.title ?? m.name,
    label: checkpointLabel(m.name),
    hint: [m.arch ? presetLabel(m.arch) : null, formatBytes(m.size)].filter(Boolean).join(" · ") || undefined,
  }))
  const busy = write.isPending || loading !== null
  if (models.isError && !models.data) {
    // an empty list would read as "no checkpoints installed"; say it failed and offer to load it again
    return (
      <Button
        variant="ghost"
        size="sm"
        className={cn(FIELD_BUTTON_CLASS, "min-w-0 justify-start gap-1.5 px-2.5 font-normal text-destructive", className)}
        onClick={() => void models.refetch()}
        title={[t("Could not load the models"), errorText(models.error), t("Retry")].join(" · ")}
      >
        <TriangleAlert className="size-3.5 shrink-0" aria-hidden />
        {/* the model in use stays in view: only the list of the others failed */}
        {current ? (
          <>
            <span className="sr-only">{t("Could not load the models")}</span>
            <span className="truncate text-foreground">{checkpointLabel(current)}</span>
          </>
        ) : (
          <span className="truncate">{t("Could not load the models")}</span>
        )}
        <span className="ms-auto shrink-0 text-muted-foreground">{t("Retry")}</span>
      </Button>
    )
  }
  return (
    <SearchSelect
      // the loading stage sits inside the pill, so a crowded header never cuts it off
      suffix={loading ? loadingText(t, loading) : null}
      aria-label={t("Checkpoint")}
      size="sm"
      className={className}
      // checkpoint file names are long and each row adds its architecture and size: room for both
      popupMinWidth="28rem"
      options={options}
      value={write.isPending ? (write.variables ?? null) : current}
      placeholder={models.isPending ? t("Loading…") : t("Select a checkpoint")}
      onValueChange={(name) => name !== current && write.mutate(name)}
      disabled={write.isPending}
      prefix={
        // loading a model can take minutes: a still mark then, a spinner only for the short write
        loading !== null ? (
          <Hourglass className="size-3.5 shrink-0 text-primary" aria-hidden />
        ) : busy ? (
          <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" aria-hidden />
        ) : (
          <Box className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        )
      }
    />
  )
}

function ModulesPicker({ className, segmented = false }: { className?: string; segmented?: boolean }) {
  const { t } = useTranslation()
  const id = useId()
  const models = useQuery(modelsQuery)
  // The setting holds paths; the server takes (and lists) file names.
  const current = useClientOption("forge_additional_modules").map(fileName)
  const [draft, setDraft] = useState<string[] | null>(null)
  /** The uncommitted selection (a ref too: the unmount cleanup must see the latest). */
  const pending = useRef<string[] | null>(null)
  const [query, setQuery] = useState("")
  const selected = draft ?? current
  const modules = (models.data?.modules ?? [...(models.data?.vae ?? []), ...(models.data?.text_encoders ?? [])]).map((m) => ({
    value: m.name,
    label: m.title ?? m.name,
    size: formatBytes(m.size),
  }))
  const q = query.trim().toLowerCase()
  const visible = q ? modules.filter((m) => m.label.toLowerCase().includes(q)) : modules
  // Selected files that are no longer listed stay visible so they can be removed.
  const missing = selected.filter((value) => !modules.some((m) => m.value === value))
  const rows = [...missing.map((value) => ({ value, label: value, size: null, missing: true })), ...visible.map((m) => ({ ...m, missing: false }))]

  const edit = (next: string[]) => {
    pending.current = next
    setDraft(next)
  }
  const commit = () => {
    const next = pending.current
    pending.current = null
    setDraft(null)
    setQuery("")
    if (next) writeModules(next)
  }
  const toggle = (value: string, on: boolean) => {
    const base = draft ?? current
    edit(on ? [...base, value] : base.filter((v) => v !== value))
  }
  // Nested in the narrow header's popover, the picker can unmount without closing itself.
  useEffect(
    () => () => {
      const next = pending.current
      pending.current = null
      if (next) writeModules(next)
    },
    [],
  )

  return (
    <Popover onOpenChange={(open) => !open && commit()}>
      <PopoverTrigger
        render={
          <Button
            variant="ghost"
            size="sm"
            className={cn(segmented ? "rounded-none bg-transparent hover:bg-field-hover focus-visible:z-10" : FIELD_BUTTON_CLASS, "gap-1.5 px-2 font-normal", className)}
            aria-label={t("VAE / Text encoders")}
          />
        }
      >
        <Layers className="text-muted-foreground" aria-hidden />
        <span className="hidden 2xl:inline">{t("VAE / TE")}</span>
        <span className={cn("font-mono text-xs tabular-nums", selected.length > 0 ? "font-semibold text-primary" : "text-muted-foreground")}>
          {selected.length}
        </span>
      </PopoverTrigger>
      <PopoverContent aria-labelledby={`${id}-title`} align="end" className="w-[min(26rem,calc(100vw-2rem))] gap-2 p-2">
        <div className="flex items-center justify-between gap-2 px-1">
          <p id={`${id}-title`} className="text-sm font-medium">
            {t("VAE / Text encoders")}
          </p>
          {selected.length > 0 ? (
            <Button variant="ghost" size="xs" onClick={() => edit([])}>
              {t("Clear")}
            </Button>
          ) : null}
        </div>
        <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("Search…")} aria-label={t("Search modules")} className="h-8" />
        <div
          role="group"
          aria-labelledby={`${id}-title`}
          className={cn(
            "flex max-h-72 flex-col gap-0.5 overflow-y-auto overscroll-contain pb-0.5",
            // in a short window (a phone held sideways) the list keeps three rows or so and the help
            // below scrolls with the popover; a list of fewer rows than that stays its own height
            rows.length > 3 && "min-h-24",
          )}
        >
          {rows.map((m, index) => {
            const checkId = `${id}-module-${index}` // paths contain spaces and slashes
            return (
              <div key={m.value} className="flex items-center gap-2 rounded-md px-1.5 py-1 hover:bg-accent">
                <Checkbox id={checkId} checked={selected.includes(m.value)} onCheckedChange={(on) => toggle(m.value, on)} />
                <label htmlFor={checkId} className={cn("min-w-0 flex-1 truncate text-sm", m.missing && "text-muted-foreground line-through")} title={m.label}>
                  {m.label}
                </label>
                {m.size ? <span className="shrink-0 text-xs text-muted-foreground tabular-nums">{m.size}</span> : null}
              </div>
            )
          })}
          {models.isError && !models.data ? (
            <LoadFailed error={models.error} message={t("Could not load the models")} onRetry={() => void models.refetch()} className="px-2" />
          ) : models.isPending ? (
            // a first load or a retry after a failure: the list is not empty yet, it is on its way
            <p role="status" className="px-2 py-4 text-center text-sm text-muted-foreground">{t("Loading…")}</p>
          ) : modules.length === 0 ? (
            <p className="px-2 py-4 text-center text-sm text-muted-foreground">{t("No modules found")}</p>
          ) : null}
        </div>
        <div className="flex flex-col gap-1 px-1 text-xs text-muted-foreground">
          <p>{t("Applied when this panel closes.")}</p>
          <p>
            {t("Checkpoints without a built-in VAE or text encoder (Flux, Anima, …) need them selected here.")}{" "}
            <a
              href={MODULES_HELP_URL}
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex items-center gap-1 rounded-sm font-medium text-foreground/80 underline underline-offset-2 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-focus-ring"
            >
              <CircleHelp className="size-3.5" aria-hidden />
              {t("Where to get them")}
            </a>
          </p>
        </div>
      </PopoverContent>
    </Popover>
  )
}

function DtypeSelect({ className }: { className?: string }) {
  const { t } = useTranslation()
  const models = useQuery(modelsQuery)
  const current = useClientOption("forge_unet_storage_dtype")
  const write = useOptionWrite<string>(MODEL_OPTION_KEYS.dtype)
  const dtypes = models.data?.unet_dtypes ?? DEFAULT_DTYPES
  // "Automatic" is the one word among the dtype names
  const items = dtypes.map((d) => ({ value: d, label: d === "Automatic" ? t("Automatic") : d }))
  const value = write.isPending ? (write.variables ?? current) : current
  const title = value === "Automatic" ? t("Automatic") : value
  return (
    <Select<string> items={items} value={value} onValueChange={(next) => next && next !== current && write.mutate(next)} disabled={write.isPending}>
      <SelectTrigger size="sm" aria-label={t("Diffusion in Low Bits")} title={title} className={cn("max-w-48", className)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent alignItemWithTrigger={false} align="end">
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** The stages the server sends (`model.loading`), as chrome keys; any other stage is server text. */
const MODEL_STAGES: Record<string, string> = { loading: "Loading" }

function loadingText(t: (key: string) => string, loading: NonNullable<ReturnType<typeof useModelBusy>>): string {
  const key = MODEL_STAGES[loading.stage.toLowerCase()]
  return key ? t(key) : serverText(loading.stage)
}

/** The model loading stage as a live region (visible in the stacked bar, announced only in the header). */
function LoadingStatus({ className }: { className?: string }) {
  const { t } = useTranslation()
  const loading = useModelBusy()
  return (
    <span role="status" className={cn("text-xs text-muted-foreground empty:hidden", className)}>
      {loading ? loadingText(t, loading) : null}
    </span>
  )
}

/** Quiet inline mono capability tags for the active checkpoint/preset (`Diffusers`, `FP8`, `Turbo`, `RGBA`, …). */
function ModelBadges({ className, limit }: { className?: string; limit?: number }) {
  const { t } = useTranslation()
  const checkpoint = useClientOption("sd_model_checkpoint")
  const dtype = useClientOption("forge_unet_storage_dtype")
  const preset = useClientOption("forge_preset")
  const route = currentRoute()
  const tab = route.name === "generate" ? route.tab : "txt2img"
  const capabilities = useStore(getFormStore(tab), (s) => s.capabilities)
  const all = deriveModelBadges({ checkpoint, dtype, preset, capabilities })
  if (all.length === 0) return null
  const shown = limit ? all.slice(0, limit) : all
  return (
    <div
      aria-label={t("Model capabilities")}
      data-slot="model-badges"
      className={cn("flex shrink-0 flex-wrap items-center gap-1.5 font-mono text-[0.6875rem] tracking-tight", className)}
    >
      {shown.map((badge, index) => (
        <span key={badge.label} className="inline-flex items-center gap-1.5">
          {index > 0 ? <span aria-hidden className="size-0.5 rounded-full bg-muted-foreground/50" /> : null}
          <span className={badge.kind === "variant" ? "font-medium text-primary" : "text-muted-foreground"}>{badge.label}</span>
        </span>
      ))}
    </div>
  )
}

/** The compact controls in one popover (narrow headers, phones). */
function ModelSettingsPopover({ className }: { className?: string }) {
  const { t } = useTranslation()
  return (
    <Popover>
      <PopoverTrigger render={<Button variant="ghost" size="icon-sm" className={className} aria-label={t("Model settings")} />}>
        <SlidersHorizontal />
      </PopoverTrigger>
      <PopoverContent aria-label={t("Model settings")} align="end" className="w-[min(22rem,calc(100vw-2rem))] gap-3">
        <ModelBadges />
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">{t("UI preset")}</span>
          <PresetSelect className="w-full" />
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">{t("Diffusion in Low Bits")}</span>
          <DtypeSelect className="w-full max-w-none" />
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">{t("VAE / Text encoders")}</span>
          <ModulesPicker className="w-full justify-start" />
        </div>
        <WorkerHealthPill className="w-full justify-start" />
      </PopoverContent>
    </Popover>
  )
}

export default function ModelBar() {
  const { t } = useTranslation()
  return (
    <div role="group" aria-label={t("Model")} className="flex min-w-0 items-center gap-2">
      {/* Unified recessed studio instrument strip: one surface, crisp hairlines, unclipped focus rings */}
      <div className="flex h-7 min-w-0 items-center rounded-sm bg-field">
        <PresetSelect className="hidden min-w-20 rounded-none rounded-l-sm bg-transparent hover:bg-field-hover focus-visible:z-10 xl:flex" />
        <span aria-hidden className="hidden h-3.5 w-px shrink-0 bg-border xl:block" />
        <CheckpointPicker className="w-[min(22rem,34vw)] rounded-sm bg-transparent hover:bg-field-hover focus-visible:z-10 xl:rounded-none" />
        <span aria-hidden className="hidden h-3.5 w-px shrink-0 bg-border xl:block" />
        <ModulesPicker segmented className="hidden xl:inline-flex" />
        <span aria-hidden className="hidden h-3.5 w-px shrink-0 bg-border xl:block" />
        <DtypeSelect className="hidden rounded-none rounded-r-sm bg-transparent hover:bg-field-hover focus-visible:z-10 xl:flex" />
      </div>
      <ModelBadges className="hidden 2xl:flex" limit={3} />
      <WorkerHealthPill className="hidden xl:inline-flex" />
      <ModelSettingsPopover className="xl:hidden" />
      <LoadingStatus className="sr-only" />
    </div>
  )
}

/** Stacked variant for the generate page on phones (the header hides the bar there). */
export function ModelBarStacked() {
  const { t } = useTranslation()
  return (
    <div role="group" aria-label={t("Model")} className="flex flex-col gap-2">
      <CheckpointPicker className="w-full" />
      <ModelBadges />
      <div className="flex flex-wrap items-center gap-1.5">
        <PresetSelect />
        <ModulesPicker />
        <DtypeSelect className="min-w-0 flex-1" />
        <WorkerHealthPill />
      </div>
      <LoadingStatus />
    </div>
  )
}

