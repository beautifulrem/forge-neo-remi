/**
 * Non-Node Multi-Stage Linear Pipeline Builder & Runner:
 * connected stages (`Prompt Upsample` -> `Base Generate` -> `Refine / Inpaint`
 * -> `Finish / Upscale / PiD`), each an inspector panel (`ui/layout/panel.ts`),
 * whisper-quiet inline cache status, `--glow` lamp CTA, and recent runs.
 * Lazily loaded.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  ArrowDown,
  ArrowUp,
  Ban,
  CheckCircle2,
  ChevronRight,
  ClipboardPaste,
  History,
  Play,
  Plus,
  RotateCw,
  Save,
  Trash2,
} from "@/components/icons"
import { type ComponentProps, Fragment, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import { onWsEvent } from "@/api/ws"
import { LoadFailed } from "@/components/LoadFailed"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { CATEGORY_LABELS } from "@/features/library/labels"
import { errorText, failureText } from "@/features/errors/error-text"
import { captureTabOverrides } from "@/features/recipes/capture"
import { cn } from "@/lib/utils"
import { panelClass, panelTitleClass } from "@/ui/layout/panel"
import {
  cancelChainApi,
  chainKeys,
  chainRunQuery,
  chainsQuery,
  chainTemplatesQuery,
  createChainApi,
  deleteChainTemplateApi,
  previewChainApi,
  refreshChains,
  rerunChainApi,
  saveChainTemplateApi,
  storeChainRun,
} from "./api"
import { defaultStage, firstExecuteStage, planPipelineCache, STAGE_KINDS, STAGE_META, STARTER_PIPELINES } from "./cache-hash"
import { keepStageKey, stageKey, usePipelineDraft } from "./draft-store"
import type { ChainPreviewStage, ChainStatus, ChainTemplate, JobChain, StageKind, StageSpec } from "./types"

/** Recent runs' status (English keys, translated at render). */
const CHAIN_STATUS_LABEL: Record<ChainStatus, string> = {
  queued: "Queued",
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
}

const MAX_STAGES = 4
const MIN_STAGES = 1

function numOr(value: unknown, fallback: number): number {
  if (value === null || value === undefined || (typeof value === "string" && value.trim() === "")) return fallback
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

/**
 * A number field of a stage card: keeps what is typed (an emptied field is not
 * the default yet) and commits each valid number; leaving it invalid restores
 * the stage's value.
 */
function StageNumber({ value, onCommit, ...props }: Omit<ComponentProps<typeof Input>, "value" | "onChange" | "type"> & { value: number; onCommit: (value: number) => void }) {
  const [text, setText] = useState(String(value))
  const [shown, setShown] = useState(value)
  if (value !== shown) {
    // changed elsewhere (a template, a moved stage): show it
    setShown(value)
    setText(String(value))
  }
  return (
    <Input
      {...props}
      type="number"
      value={text}
      onChange={(event) => {
        const next = event.target.value
        setText(next)
        const n = next.trim() === "" ? Number.NaN : Number(next)
        if (Number.isFinite(n)) {
          setShown(n)
          onCommit(n)
        }
      }}
      onBlur={(event) => {
        const n = text.trim() === "" ? Number.NaN : Number(text)
        if (!Number.isFinite(n)) setText(String(value))
        props.onBlur?.(event)
      }}
    />
  )
}

/** A run still moving: it can be cancelled, not started again. */
const ACTIVE_STATUSES = new Set(["queued", "running"])

function stepNumber(index: number): string {
  return String(index + 1).padStart(2, "0")
}

const STARTER_IDS = new Set(STARTER_PIPELINES.map((starter) => starter.id))

export function PipelineBuilder() {
  const { t } = useTranslation()
  const { name: draftName, setName, stages, setStages, activeRunId, setActiveRunId } = usePipelineDraft()
  const name = draftName ?? t(STARTER_PIPELINES[0]!.name)
  const activeRunQuery = useQuery({ ...chainRunQuery(activeRunId ?? ""), enabled: activeRunId !== null })
  const activeRun = activeRunId !== null ? (activeRunQuery.data ?? null) : null
  const followRun = (chain: JobChain) => {
    storeChainRun(chain)
    setActiveRunId(chain.id)
  }
  const [serverPreview, setServerPreview] = useState<{ sig: string; stages: ChainPreviewStage[] } | null>(null)
  const [saveOpen, setSaveOpen] = useState(false)
  // focused through the popover (`initialFocus`), not `autoFocus`, which made it the place to give the focus back to
  const templateRef = useRef<HTMLInputElement>(null)
  const [templateName, setTemplateName] = useState("")

  const chains = useQuery(chainsQuery)
  const templates = useQuery(chainTemplatesQuery)

  // `chain.updated` carries the whole chain: the run in the builder follows it, the recent runs list refetches.
  useEffect(() => {
    const offChain = onWsEvent("chain.updated", (event) => {
      const { type: _type, chain_id: chainId, ...rest } = event
      const chain = rest as Partial<JobChain>
      if (typeof chain.id === "string" && Array.isArray(chain.stages)) storeChainRun(chain as JobChain)
      else void queryClient.invalidateQueries({ queryKey: chainKeys.run(chainId) })
      void queryClient.invalidateQueries({ queryKey: chainKeys.list })
    })
    const offJob = onWsEvent("job.completed", () => void queryClient.invalidateQueries({ queryKey: chainKeys.list }))
    return () => {
      offChain()
      offJob()
    }
  }, [])

  // Debounced server cache preview whenever stages or the latest completed run change.
  const stagesSignature = useMemo(() => JSON.stringify(stages), [stages])
  const previewKey = `${stagesSignature}:${activeRunId ?? ""}:${activeRun?.updated_at ?? ""}`
  useEffect(() => {
    const controller = new AbortController()
    const timer = setTimeout(() => {
      void previewChainApi(stages, activeRunId ?? undefined, controller.signal).then((res) => {
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
      followRun(chain)
      refreshChains()
      toast.success(t("Pipeline started"))
    },
    onError: (error) => toast.error(t("Could not start pipeline"), { description: errorText(error) }),
  })

  const cancel = useMutation({
    mutationFn: (id: string) => cancelChainApi(id),
    onSuccess: (chain) => {
      followRun(chain)
      refreshChains()
      toast.info(t("Pipeline cancelled"))
    },
    onError: (error) => toast.error(t("Could not cancel pipeline"), { description: errorText(error) }),
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
    onSuccess: () => {
      refreshChains()
      toast.success(t("Template deleted"))
    },
    onError: (error) => toast.error(t("Could not delete template"), { description: errorText(error) }),
  })

  const updateStage = (index: number, updater: (prev: StageSpec) => StageSpec) => {
    setStages((prev) => prev.map((stage, i) => (i === index ? keepStageKey(stage, updater(stage)) : stage)))
  }

  // The move button just used, to keep the focus on once its card has moved (the other one when
  // the card reached the end and this one is disabled).
  const trackRef = useRef<HTMLOListElement>(null)
  const movedRef = useRef<{ key: number; direction: "earlier" | "later" } | null>(null)
  useEffect(() => {
    const moved = movedRef.current
    movedRef.current = null
    if (!moved) return
    const card = trackRef.current?.querySelector(`[data-stage-key="${moved.key}"]`)
    const buttons = [...(card?.querySelectorAll<HTMLButtonElement>("[data-move]") ?? [])]
    const same = buttons.find((button) => button.dataset.move === moved.direction)
    const target = same && !same.disabled ? same : buttons.find((button) => !button.disabled)
    target?.focus()
  }, [stages])

  const moveStage = (from: number, to: number) => {
    if (to < 0 || to >= stages.length) return
    const stage = stages[from]
    if (stage) movedRef.current = { key: stageKey(stage), direction: to < from ? "earlier" : "later" }
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
    // the form is read after an await: by then the stage may sit elsewhere (moved) or be gone
    const key = stageKey(stage)
    captureTabOverrides(tab).then(
      (captured) => {
        if (!usePipelineDraft.getState().stages.some((s) => stageKey(s) === key)) return
        setStages((prev) =>
          prev.map((s) =>
            stageKey(s) === key
              ? keepStageKey(s, { ...s, tab, core: { ...s.core, ...captured.core_patch }, scripts: { ...s.scripts, ...captured.scripts_patch } })
              : s,
          ),
        )
        toast.success(t("Pulled overrides from {{tab}}", { tab: t(CATEGORY_LABELS[tab]) }))
      },
      (error: unknown) => toast.error(t("Could not read the form"), { description: failureText(error) }),
    )
  }

  // the built-in starters are named in the current language; saved templates keep the name they were given
  const templateLabel = (template: Pick<ChainTemplate, "id" | "name">) => (STARTER_IDS.has(template.id) ? t(template.name) : template.name)
  const loadTemplate = (template: Pick<ChainTemplate, "id" | "name" | "stages">) => {
    setName(templateLabel(template))
    setStages(template.stages.map((s) => structuredClone(s)))
    setActiveRunId(null)
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
    followRun(chain)
  }

  // the built-in starters always, then the templates the server keeps (only saved ones)
  const savedTemplates = (templates.data ?? []).filter((tpl) => !STARTER_IDS.has(tpl.id))

  return (
    <div className="flex flex-col gap-5">
      {/* Command bar: the stages' hairlines separate it from them */}
      <div className="flex flex-wrap items-center gap-2">
        <Input
          value={name}
          onChange={(event) => setName(event.target.value)}
          aria-label={t("Pipeline name")}
          placeholder={t("Pipeline name")}
          className="h-7 min-w-48 flex-1 text-[0.8rem] font-medium sm:max-w-xs"
        />

        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" />}>
            <History className="size-3.5" aria-hidden />
            {t("Templates")}
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-64">
            {[...STARTER_PIPELINES, ...(savedTemplates.length > 0 ? [null] : []), ...savedTemplates].map((tpl) =>
              tpl === null ? (
                <DropdownMenuSeparator key="saved" />
              ) : (
                <DropdownMenuItem key={tpl.id} onClick={() => loadTemplate(tpl)} title={templateLabel(tpl)}>
                  <span title={templateLabel(tpl)} className="truncate">{templateLabel(tpl)}</span>
                </DropdownMenuItem>
              ),
            )}
            {savedTemplates.length > 0 ? (
              <>
                <DropdownMenuSeparator />
                {/* its own submenu: a delete button inside a load item could not be reached from the keyboard */}
                <DropdownMenuSub>
                  {/* text only, like the templates above it: one menu is all text or all iconed */}
                  <DropdownMenuSubTrigger>
                    {t("Delete template")}
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent className="min-w-48">
                    {savedTemplates.map((tpl) => (
                      <DropdownMenuItem key={tpl.id} variant="destructive" disabled={deleteTemplate.isPending} onClick={() => deleteTemplate.mutate(tpl.id)} title={tpl.name}>
                        <span title={tpl.name} className="truncate">{tpl.name}</span>
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              </>
            ) : null}
          </DropdownMenuContent>
        </DropdownMenu>

        <Popover open={saveOpen} onOpenChange={setSaveOpen}>
          <PopoverTrigger render={<Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" />}>
            <Save className="size-3.5" aria-hidden />
            {t("Save template")}
          </PopoverTrigger>
          <PopoverContent aria-label={t("Save template")} align="start" className="w-64 p-3" initialFocus={templateRef}>
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
                ref={templateRef}
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
            <DropdownMenuTrigger render={<Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" />}>
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

        <div className="ml-auto flex items-center gap-2.5">
          <span className="hidden font-mono text-xs text-muted-foreground tabular-nums sm:inline">
            {t("{{cached}} cached · {{execute}} to run", { cached: cachedCount, execute: executeCount })}
          </span>
          {activeRun && ACTIVE_STATUSES.has(activeRun.status) ? (
            <Button variant="destructive" size="sm" className="h-7 gap-1.5 px-3 text-xs" disabled={cancel.isPending} onClick={() => cancel.mutate(activeRun.id)}>
              <Ban className="size-3.5" aria-hidden />
              {t("Cancel pipeline")}
            </Button>
          ) : (
            <Button
              size="sm"
              variant="glow"
              className="h-7 gap-1.5 px-3.5 text-xs font-medium"
              disabled={run.isPending || stages.every((s) => s.enabled === false)}
              onClick={() => run.mutate(activeRun && firstMod !== null ? firstMod : undefined)}
            >
              <Play className="size-3.5" aria-hidden />
              {activeRun && cachedCount > 0 && executeCount > 0 ? t("Re-run modified") : t("Run pipeline")}
            </Button>
          )}
        </div>
      </div>

      {/* Connected linear stage track */}
      <ol ref={trackRef} className="grid grid-cols-1 gap-x-4 gap-y-5 lg:grid-cols-2 xl:flex xl:items-stretch xl:gap-0">
        {stages.map((stage, index) => {
          const planned = plan[index]!
          // the run's stage at this place only while this card still is that stage (not after a move or an edit)
          const atIndex = activeRun?.stages[index]
          const runStage = atIndex && atIndex.kind === stage.kind && planned.decision !== "execute" ? atIndex : undefined
          const enabled = stage.enabled !== false
          const core = stage.core ?? {}
          const sampler = stage.scripts?.sampler ?? {}

          return (
            <Fragment key={stageKey(stage)}>
              <li
                data-stage-index={index}
                data-stage-key={stageKey(stage)}
                data-cache-decision={planned.decision}
                className={cn(
                  // an inspector panel, not a card: a hairline above, no box or fill
                  "flex min-w-0 flex-1 flex-col gap-2 transition-opacity",
                  panelClass(0),
                  !enabled && "opacity-55",
                )}
              >
                {/* Stage header */}
                <div className="flex min-h-7 items-center gap-2">
                  <span className="shrink-0 font-mono text-[0.6875rem] font-semibold text-muted-foreground tabular-nums">
                    {stepNumber(index)}
                  </span>
                  <span className={cn("min-w-0 flex-1 truncate", panelTitleClass(0))} title={t(STAGE_META[stage.kind].label)}>
                    {t(STAGE_META[stage.kind].label)}
                  </span>
                  <Switch
                    checked={enabled}
                    aria-label={t("Stage {{n}}: {{name}}", { n: index + 1, name: t(STAGE_META[stage.kind].label) })}
                    onCheckedChange={(checked) => updateStage(index, (prev) => ({ ...prev, enabled: checked }))}
                  />
                </div>

                {/* Whisper-quiet inline cache/execution status */}
                <div className="flex items-center justify-between gap-2">
                  {planned.decision === "cached" ? (
                    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-success">
                      <span className="size-1.5 shrink-0 rounded-full bg-success" aria-hidden />
                      {t("Reused from cached output")}
                    </span>
                  ) : planned.decision === "execute" ? (
                    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                      <span className="size-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
                      {t("Will execute")}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                      <span className="size-1.5 shrink-0 rounded-full bg-muted-foreground/45" aria-hidden />
                      {t("Stage skipped")}
                    </span>
                  )}

                  {runStage?.status === "completed" || runStage?.status === "cached" ? (
                    <CheckCircle2 className="size-3.5 shrink-0 text-success" aria-label={t("Stage completed")} />
                  ) : null}
                </div>

                {/* Stage parameters & cached preview */}
                <div className="flex flex-1 flex-col gap-2.5 text-xs">
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

                  {stage.kind === "prompt_upsample" ? (
                    <>
                      <label className="flex flex-col gap-1">
                        <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("Prompt")}</span>
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
                        <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("PUP mode")}</span>
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
                        <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("Prompt")}</span>
                        <Input
                          value={String(core.prompt ?? "")}
                          disabled={!enabled}
                          onChange={(event) =>
                            updateStage(index, (prev) => ({ ...prev, core: { ...prev.core, prompt: event.target.value } }))
                          }
                          className="h-7 text-xs"
                        />
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        <label className="flex flex-col gap-1">
                          <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("Steps")}</span>
                          <StageNumber
                            min={1}
                            max={150}
                            value={numOr(sampler.steps, 28)}
                            disabled={!enabled}
                            onCommit={(value) =>
                              updateStage(index, (prev) => ({
                                ...prev,
                                scripts: { ...prev.scripts, sampler: { ...prev.scripts?.sampler, steps: value } },
                              }))
                            }
                            className="h-7 font-mono text-xs tabular-nums"
                          />
                        </label>
                        <label className="flex flex-col gap-1">
                          <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("CFG scale")}</span>
                          <StageNumber
                            min={1}
                            max={30}
                            step={0.5}
                            value={numOr(core.cfg_scale, 5)}
                            disabled={!enabled}
                            onCommit={(value) =>
                              updateStage(index, (prev) => ({
                                ...prev,
                                core: { ...prev.core, cfg_scale: value },
                              }))
                            }
                            className="h-7 font-mono text-xs tabular-nums"
                          />
                        </label>
                      </div>
                    </>
                  ) : null}

                  {stage.kind === "refine" ? (
                    <div className="grid grid-cols-2 gap-2">
                      <label className="flex flex-col gap-1">
                        <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("Denoising strength")}</span>
                        <StageNumber
                          min={0.05}
                          max={1}
                          step={0.05}
                          value={numOr(core.denoising_strength, 0.35)}
                          disabled={!enabled}
                          onCommit={(value) =>
                            updateStage(index, (prev) => ({
                              ...prev,
                              core: { ...prev.core, denoising_strength: value },
                            }))
                          }
                          className="h-7 font-mono text-xs tabular-nums"
                        />
                      </label>
                      <label className="flex flex-col gap-1">
                        <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("Steps")}</span>
                        <StageNumber
                          min={1}
                          max={150}
                          value={numOr(sampler.steps, 20)}
                          disabled={!enabled}
                          onCommit={(value) =>
                            updateStage(index, (prev) => ({
                              ...prev,
                              scripts: { ...prev.scripts, sampler: { ...prev.scripts?.sampler, steps: value } },
                            }))
                          }
                          className="h-7 font-mono text-xs tabular-nums"
                        />
                      </label>
                    </div>
                  ) : null}

                  {stage.kind === "finish" ? (
                    <div className="grid grid-cols-2 gap-2">
                      <label className="flex flex-col gap-1">
                        <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("Upscale factor")}</span>
                        <StageNumber
                          min={1}
                          max={8}
                          step={0.5}
                          value={numOr(core.upscaling_resize, 2)}
                          disabled={!enabled}
                          onCommit={(value) =>
                            updateStage(index, (prev) => ({
                              ...prev,
                              core: { ...prev.core, upscaling_resize: value },
                            }))
                          }
                          className="h-7 font-mono text-xs tabular-nums"
                        />
                      </label>
                      <label className="flex flex-col gap-1">
                        <span className="text-[0.6875rem] font-medium text-muted-foreground">{t("Upscaler")}</span>
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

                {/* Stage actions */}
                <div className="mt-auto flex flex-wrap items-center gap-1">
                  {stage.kind !== "prompt_upsample" ? (
                    <Button variant="ghost" size="sm" className="h-7 shrink-0 gap-1 px-1.5 text-xs text-muted-foreground hover:text-foreground" disabled={!enabled} onClick={() => pullFromForm(index)}>
                      <ClipboardPaste className="size-3" aria-hidden />
                      {t("Pull from form")}
                    </Button>
                  ) : null}
                  {activeRun ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 shrink-0 gap-1 px-1.5 text-xs text-muted-foreground hover:text-foreground"
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
                      className="size-7 shrink-0 text-muted-foreground hover:text-foreground"
                      disabled={index === 0}
                      data-move="earlier"
                      aria-label={t("Move {{stage}} earlier", { stage: t(STAGE_META[stage.kind].label) })}
                      onClick={() => moveStage(index, index - 1)}
                    >
                      <ArrowUp className="size-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="size-7 shrink-0 text-muted-foreground hover:text-foreground"
                      disabled={index === stages.length - 1}
                      data-move="later"
                      aria-label={t("Move {{stage}} later", { stage: t(STAGE_META[stage.kind].label) })}
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

              {/* Horizontal flow connector on desktop, level with the stage headers */}
              {index < stages.length - 1 ? (
                <li aria-hidden className="hidden w-7 shrink-0 justify-center pt-3 xl:flex">
                  <div className="flex h-7 w-full items-center">
                    <span className="h-px flex-1 bg-border" />
                    <ChevronRight className="-ml-1.5 size-3.5 shrink-0 text-muted-foreground/60" />
                  </div>
                </li>
              ) : null}
            </Fragment>
          )
        })}
      </ol>

      {/* Recent pipeline runs */}
      <section className={cn("flex flex-col gap-1", panelClass(0))} aria-label={t("Recent pipeline runs")}>
        <div className="flex min-h-8 items-center justify-between">
          <h3 className={panelTitleClass(0)}>
            {t("Recent pipeline runs")}
            {chains.data ? <span className="ml-2 font-mono text-xs font-normal text-muted-foreground tabular-nums">{chains.data.items.length}</span> : null}
          </h3>
        </div>
        {chains.isError && !chains.data ? (
          <LoadFailed error={chains.error} onRetry={() => void chains.refetch()} className="py-2" />
        ) : chains.isPending ? (
          // the list is on its way, not empty
          <p role="status" className="py-3 text-xs text-muted-foreground">{t("Loading…")}</p>
        ) : (chains.data?.items.length ?? 0) === 0 ? (
          <p className="py-3 text-xs text-muted-foreground">{t("No pipeline runs yet")}</p>
        ) : (
          <ul className="divide-y divide-border/70">
            {chains.data!.items.map((chain) => {
              const reused = chain.stages.filter((s) => s.cached).length
              const outputs = chain.stages.flatMap((s) => s.output_file_ids ?? []).slice(-4)
              return (
                <li key={chain.id} className="flex flex-wrap items-center gap-3 py-2.5">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                      <span title={chain.name} className="truncate text-[0.8rem] font-medium text-foreground">{chain.name}</span>
                      <span className="inline-flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
                        <span
                          className={cn(
                            "size-1.5 rounded-full",
                            chain.status === "completed" ? "bg-success" : chain.status === "failed" ? "bg-destructive" : "bg-warning",
                          )}
                          aria-hidden
                        />
                        {t(CHAIN_STATUS_LABEL[chain.status])}
                      </span>
                      {reused > 0 ? (
                        <span className="font-mono text-xs text-success tabular-nums">
                          · {t("{{reused}}/{{total}} cached", { reused, total: chain.total_stages })}
                        </span>
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
                  <Button variant="outline" size="sm" className="h-7 px-2.5 text-xs" onClick={() => loadRun(chain)}>
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
