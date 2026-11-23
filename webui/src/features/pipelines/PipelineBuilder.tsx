/**
 * Non-Node Multi-Stage Linear Pipeline Builder & Runner:
 * compose 2–4 linear stages (`Prompt Upsample` -> `Base Generate` ->
 * `Refine / Inpaint` -> `Finish / Upscale / PiD`), inspect per-stage cache-hit
 * badges ("Reused from cached output" vs "Will execute"), run or re-run from a
 * modified stage, and save/load reusable pipeline templates. Lazily loaded.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  Ban,
  CheckCircle2,
  ClipboardPaste,
  DatabaseZap,
  History,
  Play,
  Plus,
  RotateCw,
  Save,
  Trash2,
} from "@/components/icons"
import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { onWsEvent } from "@/api/ws"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { errorText } from "@/features/errors/error-text"
import { captureFormOverrides } from "@/features/recipes/capture"
import { cn } from "@/lib/utils"
import { panelClass, panelTitleClass } from "@/ui/layout/panel"
import {
  cancelChainApi,
  chainsQuery,
  chainTemplatesQuery,
  createChainApi,
  deleteChainTemplateApi,
  previewChainApi,
  refreshChains,
  rerunChainApi,
  saveChainTemplateApi,
} from "./api"
import { defaultStage, firstExecuteStage, planPipelineCache, STAGE_KINDS, STAGE_META, STARTER_PIPELINES } from "./cache-hash"
import type { ChainPreviewStage, ChainTemplate, JobChain, StageKind, StageSpec } from "./types"
import { useTranslation } from "react-i18next"

const MAX_STAGES = 4
const MIN_STAGES = 1

function numOr(value: unknown, fallback: number): number {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

export function PipelineBuilder() {
  const { t } = useTranslation()
  const [name, setName] = useState("Generate → Refine → Upscale")
  const [stages, setStages] = useState<StageSpec[]>(() => STARTER_PIPELINES[0]!.stages.map((s) => structuredClone(s)))
  const [activeRun, setActiveRun] = useState<JobChain | null>(null)
  const [serverPreview, setServerPreview] = useState<{ sig: string; stages: ChainPreviewStage[] } | null>(null)
  const [saveOpen, setSaveOpen] = useState(false)
  const [templateName, setTemplateName] = useState("")

  const chains = useQuery(chainsQuery)
  const templates = useQuery(chainTemplatesQuery)

  // Live updates when backend emits `chain.updated`, `chain.stage_completed` or `job.completed` over WS.
  useEffect(() => {
    return onWsEvent("*", (event) => {
      const eventType = (event as { type: string }).type
      if (eventType === "chain.updated" || eventType === "chain.stage_completed" || eventType === "job.completed") {
        refreshChains()
      }
    })
  }, [])

  // Debounced server cache preview whenever stages or the latest completed run change.
  const stagesSignature = useMemo(() => JSON.stringify(stages), [stages])
  const previewKey = `${stagesSignature}:${activeRun?.id ?? ""}:${activeRun?.updated_at ?? ""}`
  useEffect(() => {
    const controller = new AbortController()
    const timer = setTimeout(() => {
      void previewChainApi(stages, undefined, controller.signal).then((res) => {
        if (!controller.signal.aborted) {
          setServerPreview(res?.stages ? { sig: stagesSignature, stages: res.stages } : null)
        }
      })
    }, 250)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey])

  const activeServerPreview = serverPreview?.sig === stagesSignature ? serverPreview.stages : null
  const plan = useMemo(
    () => planPipelineCache(stages, { previousRun: activeRun, serverPreview: activeServerPreview }),
    [stages, activeRun, activeServerPreview],
  )
  const cachedCount = plan.filter((s) => s.decision === "cached").length
  const executeCount = plan.filter((s) => s.decision === "execute").length
  const firstMod = firstExecuteStage(plan)

  const run = useMutation({
    mutationFn: async (fromStage?: number) => {
      if (activeRun && fromStage !== undefined) {
        return rerunChainApi(activeRun.id, { stages, from_stage: fromStage })
      }
      return createChainApi({ name: name.trim() || t("Linear stage pipeline"), stages, run_now: true })
    },
    onSuccess: (chain) => {
      setActiveRun(chain)
      refreshChains()
      toast.success(t("Pipeline started"))
    },
    onError: (error) => toast.error(t("Could not start pipeline"), { description: errorText(error) }),
  })

  const cancel = useMutation({
    mutationFn: (id: string) => cancelChainApi(id),
    onSuccess: (chain) => {
      setActiveRun(chain)
      refreshChains()
      toast.info(t("Pipeline cancelled"))
    },
  })

  const saveTemplate = useMutation({
    mutationFn: (title: string) => saveChainTemplateApi({ name: title, stages }),
    onSuccess: () => {
      refreshChains()
      setTemplateName("")
      setSaveOpen(false)
      toast.success(t("Template saved"))
    },
    onError: (error) => toast.error(t("Could not save template"), { description: errorText(error) }),
  })

  const deleteTemplate = useMutation({
    mutationFn: (id: string) => deleteChainTemplateApi(id),
    onSuccess: () => refreshChains(),
  })

  const updateStage = (index: number, updater: (prev: StageSpec) => StageSpec) => {
    setStages((prev) => prev.map((stage, i) => (i === index ? updater(stage) : stage)))
  }

  const moveStage = (from: number, to: number) => {
    if (to < 0 || to >= stages.length) return
    setStages((prev) => {
      const next = [...prev]
      const [moved] = next.splice(from, 1)
      next.splice(to, 0, moved!)
      return next
    })
  }

  const removeStage = (index: number) => {
    if (stages.length <= MIN_STAGES) return
    setStages((prev) => prev.filter((_, i) => i !== index))
  }

  const addStage = (kind: StageKind) => {
    if (stages.length >= MAX_STAGES) return
    setStages((prev) => [...prev, defaultStage(kind)])
  }

  const pullFromForm = (index: number) => {
    const stage = stages[index]
    if (!stage) return
    const tab = stage.tab ?? STAGE_META[stage.kind].defaultTab ?? "txt2img"
    const captured = captureFormOverrides(tab)
    updateStage(index, (prev) => ({
      ...prev,
      tab,
      core: { ...prev.core, ...captured.core_patch },
      scripts: { ...prev.scripts, ...captured.scripts_patch },
    }))
    toast.success(t("Pulled overrides from {{tab}}", { tab }))
  }

  const loadTemplate = (template: Pick<ChainTemplate, "name" | "stages">) => {
    setName(template.name)
    setStages(template.stages.map((s) => structuredClone(s)))
    setActiveRun(null)
  }

  const loadRun = (chain: JobChain) => {
    setName(chain.name)
    setStages(
      chain.stages.map(({ kind, name: stageName, enabled, tab, core, scripts, config }) => ({
        kind,
        name: stageName,
        enabled,
        tab,
        core: core ? { ...core } : undefined,
        scripts: scripts ? structuredClone(scripts) : undefined,
        config: config ? { ...config } : undefined,
      })),
    )
    setActiveRun(chain)
  }

  const templateItems = templates.data && templates.data.length > 0 ? templates.data : STARTER_PIPELINES

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          value={name}
          onChange={(event) => setName(event.target.value)}
          aria-label={t("Pipeline name")}
          placeholder={t("Pipeline name")}
          className="h-7 min-w-48 flex-1 text-[0.8rem] sm:max-w-xs"
        />

        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs" />}>
            <History className="size-3.5" aria-hidden />
            {t("Templates")}
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-64">
            {templateItems.map((tpl) => (
              <DropdownMenuItem key={tpl.id} onClick={() => loadTemplate(tpl)} className="justify-between gap-2">
                <span className="truncate">{tpl.name}</span>
                {"created_at" in tpl ? (
                  <button
                    type="button"
                    aria-label={t("Delete template")}
                    onClick={(event) => {
                      event.stopPropagation()
                      deleteTemplate.mutate(tpl.id)
                    }}
                    className="rounded p-0.5 text-muted-foreground hover:bg-destructive/15 hover:text-destructive"
                  >
                    <Trash2 className="size-3" />
                  </button>
                ) : null}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <Popover open={saveOpen} onOpenChange={setSaveOpen}>
          <PopoverTrigger render={<Button variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs" />}>
            <Save className="size-3.5" aria-hidden />
            {t("Save template")}
          </PopoverTrigger>
          <PopoverContent aria-label={t("Save template")} align="start" className="w-64 p-3">
            <form
              className="flex flex-col gap-2"
              onSubmit={(event) => {
                event.preventDefault()
                const title = (templateName || name).trim()
                if (title && !saveTemplate.isPending) saveTemplate.mutate(title)
              }}
            >
              <Input
                value={templateName}
                onChange={(event) => setTemplateName(event.target.value)}
                placeholder={name || t("Pipeline name")}
                aria-label={t("Pipeline name")}
                className="h-7 text-xs"
                autoFocus
              />
              <div className="flex justify-end gap-1.5">
                <Button type="button" variant="ghost" size="sm" className="h-7" onClick={() => setSaveOpen(false)}>
                  {t("Cancel")}
                </Button>
                <Button type="submit" size="sm" className="h-7" disabled={saveTemplate.isPending}>
                  {t("Save")}
                </Button>
              </div>
            </form>
          </PopoverContent>
        </Popover>

        {stages.length < MAX_STAGES ? (
          <DropdownMenu>
            <DropdownMenuTrigger render={<Button variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs" />}>
              <Plus className="size-3.5" aria-hidden />
              {t("Add stage")}
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="min-w-48">
              {STAGE_KINDS.map((kind) => (
                <DropdownMenuItem key={kind} onClick={() => addStage(kind)}>
                  {t(STAGE_META[kind].label)}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}

        <div className="ml-auto flex items-center gap-2">
          <span className="hidden text-xs text-muted-foreground tabular-nums sm:inline">
            {t("{{cached}} cached · {{execute}} to run", { cached: cachedCount, execute: executeCount })}
          </span>
          {activeRun && activeRun.status === "running" ? (
            <Button variant="destructive" size="sm" className="h-7 gap-1 px-2.5 text-xs" onClick={() => cancel.mutate(activeRun.id)}>
              <Ban className="size-3.5" aria-hidden />
              {t("Cancel pipeline")}
            </Button>
          ) : (
            <Button
              size="sm"
              className="h-7 gap-1 px-3 text-xs"
              disabled={run.isPending || stages.every((s) => s.enabled === false)}
              onClick={() => run.mutate(activeRun && firstMod !== null ? firstMod : undefined)}
            >
              <Play className="size-3.5" aria-hidden />
              {activeRun && cachedCount > 0 && executeCount > 0 ? t("Re-run modified") : t("Run pipeline")}
            </Button>
          )}
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        {t("Compose 2–4 sequential stages without nodes. Unchanged upstream stages reuse cached outputs automatically.")}
      </p>

      {/* Linear stage track */}
      <ol className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-4">
        {stages.map((stage, index) => {
          const planned = plan[index]!
          const runStage = activeRun?.stages[index]
          const enabled = stage.enabled !== false
          const core = stage.core ?? {}
          const sampler = stage.scripts?.sampler ?? {}

          return (
            <li
              key={`${stage.kind}-${index}`}
              data-stage-index={index}
              data-cache-decision={planned.decision}
              className={cn(
                "relative flex flex-col gap-2.5 border-t border-border pt-3 transition-opacity",
                !enabled && "opacity-55",
              )}
            >
              <div className="flex items-center gap-1.5">
                <span className="inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[0.6875rem] font-semibold tabular-nums">
                  {index + 1}
                </span>
                <span className="min-w-0 flex-1 truncate text-sm font-semibold" title={t(STAGE_META[stage.kind].label)}>
                  {t(STAGE_META[stage.kind].label)}
                </span>
                {index < stages.length - 1 ? <ArrowRight className="hidden size-3.5 shrink-0 text-muted-foreground/50 xl:block" aria-hidden /> : null}
                <Switch
                  checked={enabled}
                  aria-label={`${t("Stage {{n}}", { n: index + 1 })} ${t(STAGE_META[stage.kind].label)}`}
                  onCheckedChange={(checked) => updateStage(index, (prev) => ({ ...prev, enabled: checked }))}
                />
              </div>

              {/* Cache-hit / execution status badge */}
              <div className="flex flex-wrap items-center gap-1.5">
                {planned.decision === "cached" ? (
                  <Badge variant="secondary" className="gap-1 bg-diff-add-bg text-[0.625rem] text-diff-add">
                    <DatabaseZap className="size-3" aria-hidden />
                    {t("Reused from cached output")}
                  </Badge>
                ) : planned.decision === "execute" ? (
                  <Badge variant="outline" className="gap-1 text-[0.625rem]">
                    <Play className="size-2.5" aria-hidden />
                    {t("Will execute")}
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-[0.625rem] text-muted-foreground">
                    {t("Stage skipped")}
                  </Badge>
                )}

                {runStage?.status === "completed" || runStage?.status === "cached" ? (
                  <CheckCircle2 className="size-3.5 text-diff-add" aria-label={t("Stage completed")} />
                ) : null}
              </div>

              {/* Cached / latest output preview thumb */}
              {planned.cached_outputs.length > 0 || (runStage?.output_file_ids?.length ?? 0) > 0 ? (
                <div className="flex items-center gap-1.5">
                  {(planned.cached_outputs.length > 0 ? planned.cached_outputs : (runStage?.output_file_ids ?? []))
                    .slice(0, 3)
                    .map((fileId) => (
                      <img
                        key={fileId}
                        src={api.thumbUrl(fileId, 96)}
                        alt=""
                        className="size-10 rounded-sm bg-checker object-cover ring-1 ring-border"
                      />
                    ))}
                </div>
              ) : null}

              {/* Stage-specific lightweight controls */}
              <div className="flex flex-col gap-2 text-xs">
                {stage.kind === "prompt_upsample" ? (
                  <>
                    <label className="flex flex-col gap-1">
                      <span className="text-muted-foreground">{t("Prompt")}</span>
                      <Input
                        value={String(core.prompt ?? "")}
                        disabled={!enabled}
                        onChange={(event) =>
                          updateStage(index, (prev) => ({ ...prev, core: { ...prev.core, prompt: event.target.value } }))
                        }
                        className="h-7 text-xs"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      <span className="text-muted-foreground">{t("PUP mode")}</span>
                      <Select
                        value={String(stage.config?.mode ?? "auto")}
                        disabled={!enabled}
                        onValueChange={(value) =>
                          value && updateStage(index, (prev) => ({ ...prev, config: { ...prev.config, mode: value } }))
                        }
                      >
                        <SelectTrigger size="sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">auto</SelectItem>
                          <SelectItem value="t2i">t2i</SelectItem>
                          <SelectItem value="i2i">i2i</SelectItem>
                          <SelectItem value="t2i_bbox">t2i_bbox</SelectItem>
                          <SelectItem value="vgl_json">vgl_json</SelectItem>
                        </SelectContent>
                      </Select>
                    </label>
                  </>
                ) : null}

                {stage.kind === "generate" ? (
                  <>
                    <label className="flex flex-col gap-1">
                      <span className="text-muted-foreground">{t("Prompt")}</span>
                      <Input
                        value={String(core.prompt ?? "")}
                        disabled={!enabled}
                        onChange={(event) =>
                          updateStage(index, (prev) => ({ ...prev, core: { ...prev.core, prompt: event.target.value } }))
                        }
                        className="h-7 text-xs"
                      />
                    </label>
                    <div className="grid grid-cols-2 gap-1.5">
                      <label className="flex flex-col gap-1">
                        <span className="text-muted-foreground">{t("Steps")}</span>
                        <Input
                          type="number"
                          min={1}
                          max={150}
                          value={numOr(sampler.steps, 28)}
                          disabled={!enabled}
                          onChange={(event) =>
                            updateStage(index, (prev) => ({
                              ...prev,
                              scripts: { ...prev.scripts, sampler: { ...prev.scripts?.sampler, steps: numOr(event.target.value, 28) } },
                            }))
                          }
                          className="h-7 text-xs tabular-nums"
                        />
                      </label>
                      <label className="flex flex-col gap-1">
                        <span className="text-muted-foreground">{t("CFG scale")}</span>
                        <Input
                          type="number"
                          min={1}
                          max={30}
                          step={0.5}
                          value={numOr(core.cfg_scale, 5)}
                          disabled={!enabled}
                          onChange={(event) =>
                            updateStage(index, (prev) => ({
                              ...prev,
                              core: { ...prev.core, cfg_scale: numOr(event.target.value, 5) },
                            }))
                          }
                          className="h-7 text-xs tabular-nums"
                        />
                      </label>
                    </div>
                  </>
                ) : null}

                {stage.kind === "refine" ? (
                  <div className="grid grid-cols-2 gap-1.5">
                    <label className="flex flex-col gap-1">
                      <span className="text-muted-foreground">{t("Denoising strength")}</span>
                      <Input
                        type="number"
                        min={0.05}
                        max={1}
                        step={0.05}
                        value={numOr(core.denoising_strength, 0.35)}
                        disabled={!enabled}
                        onChange={(event) =>
                          updateStage(index, (prev) => ({
                            ...prev,
                            core: { ...prev.core, denoising_strength: numOr(event.target.value, 0.35) },
                          }))
                        }
                        className="h-7 text-xs tabular-nums"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      <span className="text-muted-foreground">{t("Steps")}</span>
                      <Input
                        type="number"
                        min={1}
                        max={150}
                        value={numOr(sampler.steps, 20)}
                        disabled={!enabled}
                        onChange={(event) =>
                          updateStage(index, (prev) => ({
                            ...prev,
                            scripts: { ...prev.scripts, sampler: { ...prev.scripts?.sampler, steps: numOr(event.target.value, 20) } },
                          }))
                        }
                        className="h-7 text-xs tabular-nums"
                      />
                    </label>
                  </div>
                ) : null}

                {stage.kind === "finish" ? (
                  <div className="grid grid-cols-2 gap-1.5">
                    <label className="flex flex-col gap-1">
                      <span className="text-muted-foreground">{t("Upscale factor")}</span>
                      <Input
                        type="number"
                        min={1}
                        max={8}
                        step={0.5}
                        value={numOr(core.upscaling_resize, 2)}
                        disabled={!enabled}
                        onChange={(event) =>
                          updateStage(index, (prev) => ({
                            ...prev,
                            core: { ...prev.core, upscaling_resize: numOr(event.target.value, 2) },
                          }))
                        }
                        className="h-7 text-xs tabular-nums"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      <span className="text-muted-foreground">{t("Upscaler")}</span>
                      <Input
                        value={String(core.upscaler_1 ?? "R-ESRGAN 4x+")}
                        disabled={!enabled}
                        onChange={(event) =>
                          updateStage(index, (prev) => ({
                            ...prev,
                            core: { ...prev.core, upscaler_1: event.target.value },
                          }))
                        }
                        className="h-7 text-xs"
                      />
                    </label>
                  </div>
                ) : null}
              </div>

              {/* Per-stage actions */}
              <div className="mt-auto flex flex-wrap items-center gap-1 border-t border-border/50 pt-1.5">
                {stage.kind !== "prompt_upsample" ? (
                  <Button variant="ghost" size="sm" className="h-7 shrink-0 gap-1 px-1.5 text-[0.6875rem]" disabled={!enabled} onClick={() => pullFromForm(index)}>
                    <ClipboardPaste className="size-3" aria-hidden />
                    {t("Pull from form")}
                  </Button>
                ) : null}
                {activeRun ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 shrink-0 gap-1 px-1.5 text-[0.6875rem]"
                    disabled={!enabled || run.isPending}
                    onClick={() => run.mutate(index)}
                  >
                    <RotateCw className="size-3" aria-hidden />
                    {t("Re-run from stage {{n}}", { n: index + 1 })}
                  </Button>
                ) : null}
                <div className="ml-auto flex shrink-0 items-center gap-0.5">
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="size-7 shrink-0"
                    disabled={index === 0}
                    aria-label={t("Move stage left")}
                    onClick={() => moveStage(index, index - 1)}
                  >
                    <ArrowUp className="size-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="size-7 shrink-0"
                    disabled={index === stages.length - 1}
                    aria-label={t("Move stage right")}
                    onClick={() => moveStage(index, index + 1)}
                  >
                    <ArrowDown className="size-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    disabled={stages.length <= MIN_STAGES}
                    aria-label={t("Remove stage")}
                    className="size-7 shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={() => removeStage(index)}
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
              </div>
            </li>
          )
        })}
      </ol>

      {/* Recent pipeline runs */}
      <section className={panelClass(0)} aria-label={t("Recent pipeline runs")}>
        <div className="mb-2 flex items-center justify-between">
          <h3 className={panelTitleClass(0)}>
            {t("Recent pipeline runs")}
            <span className="ml-2 text-xs font-normal text-muted-foreground tabular-nums">{chains.data?.items.length ?? 0}</span>
          </h3>
        </div>
        {(chains.data?.items.length ?? 0) === 0 ? (
          <p className="py-4 text-xs text-muted-foreground">{t("No pipeline runs yet")}</p>
        ) : (
          <ul className="divide-y divide-border/70">
            {chains.data!.items.map((chain) => {
              const reused = chain.stages.filter((s) => s.cached).length
              const outputs = chain.stages.flatMap((s) => s.output_file_ids ?? []).slice(-4)
              return (
                <li key={chain.id} className="flex flex-wrap items-center gap-3 py-2">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium">{chain.name}</span>
                      <Badge variant="outline" className="text-[0.625rem]">
                        {chain.status}
                      </Badge>
                      {reused > 0 ? (
                        <Badge variant="secondary" className="bg-diff-add-bg text-[0.625rem] text-diff-add">
                          {reused}/{chain.total_stages} cached
                        </Badge>
                      ) : null}
                    </div>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {chain.stages.map((s) => t(STAGE_META[s.kind]?.label ?? s.kind)).join(" → ")}
                    </p>
                  </div>
                  {outputs.length > 0 ? (
                    <div className="flex items-center gap-1">
                      {outputs.map((id) => (
                        <img key={id} src={api.thumbUrl(id, 64)} alt="" className="size-9 rounded-sm bg-checker object-cover ring-1 ring-border" />
                      ))}
                    </div>
                  ) : null}
                  <Button variant="outline" size="sm" className="h-7 px-2 text-xs" onClick={() => loadRun(chain)}>
                    {t("Load into builder")}
                  </Button>
                </li>
              )
            })}
          </ul>
        )}
      </section>
    </div>
  )
}
