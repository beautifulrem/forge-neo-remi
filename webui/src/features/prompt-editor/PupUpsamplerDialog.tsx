/**
 * Lazy-loaded Multimodal Prompt Upsampler (PUP) dialog: detects the active model
 * family, attaches active form reference images and aspect ratio, previews a
 * side-by-side Before/After token diff, and offers one-click Apply / Revert.
 */
import { Check, Loader2, RotateCcw, Sparkles, TriangleAlert } from "@/components/icons"
import { useEffect, useId, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { serviceFailureText } from "@/features/errors/error-text"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import { useForm, useFormState } from "@/ui/hooks"
import { useUploadInfo } from "@/ui/uploads"
import { diffPromptTokens } from "./diff"
import {
  collectFormReferenceIds,
  detectPupFamily,
  formatAspectRatio,
  PUP_FAMILIES,
  PUP_FAMILY_LABELS,
  PUP_MODE_LABELS,
  PUP_MODES,
  type PupFamily,
  type PupMode,
  type PupResponse,
  upsamplePromptRequest,
} from "./pup"

type PupUpsamplerDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  value: string
  previousValue: string | null
  onApply: (nextPrompt: string, previousPrompt: string) => void
  onRevert: () => void
  disabled?: boolean
}

function ReferencePreview({ id, index }: { id: string; index: number }) {
  const info = useUploadInfo(id)
  const src = info?.url ?? api.thumbUrl(id, 96)
  return (
    <span className="relative size-8 shrink-0 overflow-hidden rounded-sm border border-border bg-checker">
      <img src={src} alt="" className="size-full object-cover" />
      <span aria-hidden className="absolute right-0 bottom-0 rounded-tl-xs bg-background/85 px-1 text-[0.625rem] leading-tight font-medium tabular-nums">
        {index + 1}
      </span>
    </span>
  )
}

export default function PupUpsamplerDialog({ open, onOpenChange, value, previousValue, onApply, onRevert, disabled }: PupUpsamplerDialogProps) {
  const { t } = useTranslation()
  const id = useId()
  const { store } = useForm()
  const preset = useClientOption("forge_preset")
  const checkpoint = useClientOption("sd_model_checkpoint")
  const capabilityFamily = useFormState((s) => s.capabilities?.dit_name ?? s.capabilities?.preset ?? null)
  const width = useFormState((s) => s.values.core?.width)
  const height = useFormState((s) => s.values.core?.height)
  // Primitive key keeps Zustand's useSyncExternalStore snapshot stable across renders.
  const referenceIdsKey = useFormState((s) => collectFormReferenceIds(s).join("\0"))
  const referenceIds = useMemo(() => (referenceIdsKey ? referenceIdsKey.split("\0") : []), [referenceIdsKey])

  const detectedFamily = detectPupFamily(preset, checkpoint, capabilityFamily)
  const aspectRatio = formatAspectRatio(width, height)

  const [family, setFamily] = useState<PupFamily>(detectedFamily)
  const [mode, setMode] = useState<PupMode>("auto")
  const [attachRefs, setAttachRefs] = useState(true)
  const [includeRatio, setIncludeRatio] = useState(true)
  const [instructions, setInstructions] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PupResponse | null>(null)
  const [edited, setEdited] = useState("")
  const [editing, setEditing] = useState(false)

  // Sync detected family whenever dialog opens fresh with no result yet
  const [lastOpen, setLastOpen] = useState(open)
  if (open !== lastOpen) {
    setLastOpen(open)
    if (open) {
      setFamily(detectedFamily)
      setError(null)
      // a result for another prompt (edited since) must not be applied over this one
      if (result && result.original_prompt !== value.trim()) {
        setResult(null)
        setEdited("")
        setEditing(false)
      }
    }
  }

  // closing drops a request still running: its answer would belong to a prompt that may change
  const request = useRef<AbortController | null>(null)
  useEffect(() => {
    if (!open) request.current?.abort()
  }, [open])
  useEffect(() => () => request.current?.abort(), [])

  const diff = useMemo(() => (result ? diffPromptTokens(result.original_prompt, edited) : []), [result, edited])

  const familyItems = PUP_FAMILIES.map((item) => ({
    value: item,
    label: item === "auto" ? t("Auto") : PUP_FAMILY_LABELS[item],
  }))
  const modeItems = PUP_MODES.map((item) => ({ value: item, label: t(PUP_MODE_LABELS[item]) }))

  const run = async () => {
    const prompt = value.trim()
    if (!prompt || busy) return
    setBusy(true)
    setError(null)
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    try {
      const refs = attachRefs ? collectFormReferenceIds(store.getState()) : []
      const answer = await upsamplePromptRequest({
        prompt,
        model_family: family,
        mode,
        reference_file_ids: refs.length > 0 ? refs : undefined,
        aspect_ratio: includeRatio ? aspectRatio : undefined,
        instructions: instructions.trim() || undefined,
      }, controller.signal)
      if (controller.signal.aborted) return
      setResult(answer)
      setEdited(answer.prompt)
      setEditing(false)
    } catch (failure) {
      if (controller.signal.aborted) return
      setError(serviceFailureText(failure))
    } finally {
      if (request.current === controller) request.current = null
      setBusy(false)
    }
  }

  const applyResult = () => {
    const next = edited.trim()
    if (!next) return
    onApply(next, result?.original_prompt ?? value)
    toast.success(t("Upsampled prompt applied"))
    onOpenChange(false)
  }

  const revertPrompt = () => {
    onRevert()
    toast.info(t("Reverted to original prompt"))
    onOpenChange(false)
  }

  const effectiveFamily = family === "auto" ? detectedFamily : family

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <div className="flex flex-wrap items-baseline gap-2.5 pr-6">
            <DialogTitle>{t("Upsample prompt (PUP)")}</DialogTitle>
            {detectedFamily !== "auto" ? (
              <span className="font-mono text-xs text-muted-foreground">{PUP_FAMILY_LABELS[detectedFamily]}</span>
            ) : null}
            {effectiveFamily === "fibo" ? (
              <span className="font-mono text-xs font-medium text-advice-warn">CC BY-NC 4.0</span>
            ) : null}
          </div>
          <DialogDescription>
            {t("Expand a brief prompt into a model-tailored studio description with optional reference images and aspect ratio.")}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          {/* Model family + mode row */}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="flex flex-col gap-1">
              <Label htmlFor={`${id}-family`} className="text-[0.8rem] text-foreground/85">
                {t("Model family")}
              </Label>
              <Select<PupFamily> items={familyItems} value={family} onValueChange={(next) => next && setFamily(next)}>
                <SelectTrigger id={`${id}-family`} size="sm" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {familyItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor={`${id}-mode`} className="text-[0.8rem] text-foreground/85">
                {t("Upsample mode")}
              </Label>
              <Select<PupMode> items={modeItems} value={mode} onValueChange={(next) => next && setMode(next)}>
                <SelectTrigger id={`${id}-mode`} size="sm" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {modeItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Context options: references + aspect ratio */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-md bg-background px-3 py-2">
            <Label className="font-normal text-foreground/85">
              <Checkbox checked={attachRefs && referenceIds.length > 0} disabled={referenceIds.length === 0} onCheckedChange={(checked) => setAttachRefs(checked === true)} />
              <span>{t("Attach reference images ({{count}})", { count: referenceIds.length })}</span>
            </Label>
            {aspectRatio ? (
              <Label className="font-normal text-foreground/85">
                <Checkbox checked={includeRatio} onCheckedChange={(checked) => setIncludeRatio(checked === true)} />
                <span>{t("Include aspect ratio ({{ratio}})", { ratio: aspectRatio })}</span>
              </Label>
            ) : null}
            {attachRefs && referenceIds.length > 0 ? (
              <div className="ml-auto flex items-center gap-1" aria-label={t("Attached references")}>
                {referenceIds.slice(0, 6).map((fileId, index) => (
                  <ReferencePreview key={fileId} id={fileId} index={index} />
                ))}
                {referenceIds.length > 6 ? <span className="text-xs text-muted-foreground tabular-nums">+{referenceIds.length - 6}</span> : null}
              </div>
            ) : null}
          </div>

          {/* Optional instructions + Upsample button */}
          <div className="flex flex-wrap items-center gap-2">
            <Input
              id={`${id}-instructions`}
              value={instructions}
              onChange={(event) => setInstructions(event.target.value)}
              placeholder={t("Optional steering note (e.g. dramatic rim lighting, keep exact pose)")}
              aria-label={t("Upsample instructions")}
              className="h-8 min-w-0 flex-1"
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault()
                  void run()
                }
              }}
            />
            <Button type="button" size="sm" variant={result ? "secondary" : "default"} onClick={() => void run()} disabled={disabled || busy || !value.trim()}>
              {busy ? <Loader2 className="animate-spin" aria-hidden /> : <Sparkles aria-hidden />}
              {busy ? t("Upsampling…") : t("Upsample")}
            </Button>
          </div>

          {error ? (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          ) : null}

          {result ? (
            <div className="flex flex-col gap-2 border-t border-border pt-3">
              <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-muted-foreground">
                <span className="font-medium text-foreground">
                  {/* the server answers "generic" for a model without a template of its own */}
                  {result.model_family === "generic" ? t("Generic") : (PUP_FAMILY_LABELS[result.model_family as keyof typeof PUP_FAMILY_LABELS] ?? result.model_family)}
                </span>
                <span aria-hidden className="size-0.5 rounded-full bg-muted-foreground/50" />
                <span>{t(PUP_MODE_LABELS[result.mode as PupMode] ?? result.mode)}</span>
                <span aria-hidden className="size-0.5 rounded-full bg-muted-foreground/50" />
                <span>{result.backend}</span>
                {result.used_reference_count > 0 ? (
                  <>
                    <span aria-hidden className="size-0.5 rounded-full bg-muted-foreground/50" />
                    <span className="tabular-nums">{t("{{count}} reference images attached", { count: result.used_reference_count })}</span>
                  </>
                ) : null}
                <Button type="button" variant="ghost" size="xs" className="ml-auto font-sans" onClick={() => setEditing((v) => !v)}>
                  {editing ? t("Show diff") : t("Edit result")}
                </Button>
              </div>

              {result.warnings.length > 0 ? (
                <div className="flex flex-col gap-1 rounded-r-sm border-l-2 border-l-advice-warn bg-advice-warn-bg px-2.5 py-1.5 text-xs text-advice-warn">
                  {result.warnings.map((warn, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                      <span>{warn}</span>
                    </div>
                  ))}
                </div>
              ) : null}

              {editing ? (
                <Textarea
                  aria-label={t("Upsampled prompt")}
                  value={edited}
                  rows={6}
                  onChange={(event) => setEdited(event.target.value)}
                  className="max-h-64 font-sans text-sm"
                />
              ) : (
                <div
                  className="grid grid-cols-1 divide-y divide-border overflow-hidden rounded-md bg-background md:grid-cols-2 md:divide-x md:divide-y-0"
                  aria-label={t("Before and after comparison")}
                >
                  <div className="flex min-w-0 flex-col gap-1.5 p-3">
                    <span className="font-mono text-[0.6875rem] font-medium tracking-tight text-muted-foreground">{t("Before")}</span>
                    <div className="-mx-1 max-h-56 overflow-y-auto px-1 text-sm leading-relaxed break-words whitespace-pre-wrap">
                      {diff.map((seg, index) =>
                        seg.kind === "add" ? null : (
                          <span key={index} className={cn(seg.kind === "del" && "rounded-xs bg-diff-del-bg px-0.5 text-diff-del line-through")}>
                            {seg.text}
                          </span>
                        ),
                      )}
                    </div>
                  </div>
                  <div className="flex min-w-0 flex-col gap-1.5 p-3">
                    <span className="font-mono text-[0.6875rem] font-medium tracking-tight text-muted-foreground">{t("After")}</span>
                    <div className="-mx-1 max-h-56 overflow-y-auto px-1 text-sm leading-relaxed break-words whitespace-pre-wrap">
                      {diff.map((seg, index) =>
                        seg.kind === "del" ? null : (
                          <span key={index} className={cn(seg.kind === "add" && "rounded-xs bg-diff-add-bg px-0.5 font-medium text-diff-add")}>
                            {seg.text}
                          </span>
                        ),
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </div>

        <DialogFooter className="gap-2 sm:justify-between">
          <div>
            {previousValue !== null ? (
              <Button type="button" variant="outline" size="sm" onClick={revertPrompt} disabled={disabled}>
                <RotateCcw aria-hidden />
                {t("Revert")}
              </Button>
            ) : null}
          </div>
          <div className="flex items-center gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
              {t("Close")}
            </Button>
            {result ? (
              <Button type="button" size="sm" onClick={applyResult} disabled={disabled || !edited.trim()}>
                <Check aria-hidden />
                {t("Apply")}
              </Button>
            ) : null}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
