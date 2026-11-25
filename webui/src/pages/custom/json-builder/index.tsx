/**
 * JSON Builder (upstream `sd_forge_json_builder`, a JavaScript-only tab):
 * a visual editor for MOSS-Image's structured ("Layout") prompts. Set the
 * canvas size, a summary and a list of items (object, character or text,
 * each with a caption and a box drawn on the canvas), then send the compiled
 * prompt with its width and height to txt2img or img2img. "Load prompt"
 * turns a compiled prompt back into a draft; drafts save to and load from
 * JSON files. The draft is kept in the browser (`./store`).
 */
import { Braces, ClipboardCopy, Expand, FileDown, FileInput, FileUp, Images, Pencil, Plus, Ratio, Trash2, Wand, X } from "@/components/icons"
import { Suspense, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import type { FormTab } from "@/api/types"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { NumberField } from "@/components/ui/number-field"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Segmented } from "@/components/Segmented"
import { copyText } from "@/features/gallery/files"
import { deliver } from "@/features/send-to/send-to"
import { cn } from "@/lib/utils"
import type { CustomPageProps } from "@/pages/ext"
import { getFormStore } from "@/ui/form-store"
import { CANVAS_TYPES, LABELS } from "./compile"
import { type Box, type CanvasType, decompilePrompt, type Label, splitPrompt } from "./decompile"
import { ASPECT_RATIOS, type AspectRatio, buildDraft, finalPrompt, type Item, LABEL_NAMES, type Missing, type EditableField, missingFields, resolutionFor, round16 } from "./draft"
import { decompileFlux3Prompt, isFlux3Prompt, PROMPT_FORMAT_NAMES, PROMPT_FORMATS, type PromptFormat } from "./flux3"
import { LayoutCanvas } from "./LayoutCanvas"
import { sizeLimit } from "./size-limit"
import { useBuilder } from "./store"
import { decompileVglPrompt, isVglPrompt, VGL_LICENSE, VGL_STYLE_MEDIUMS } from "./vgl"
import { itemColor } from "./view"
import { panelClass, panelTitleClass } from "@/ui/layout/panel"
import { retryableLazy } from "@/lib/retryable-lazy"

/** The field editor, a lazy chunk (it carries CodeMirror): mounted only while a field is open in it. */
const FieldEditor = retryableLazy(() => import("./FieldEditor").then((m) => ({ default: m.FieldEditor })))

const CANVAS_NAMES: Record<CanvasType, string> = { single: "Single", multi_panel: "Multi-panel", collage: "Collage" }

const message = (error: unknown) => (error instanceof Error ? error.message : String(error))

function Section({ title, children, className, actions }: { title: string; children: React.ReactNode; className?: string; actions?: React.ReactNode }) {
  return (
    // an inspector panel, not a card (`ui/layout/panel.ts`)
    <section className={cn("flex flex-col gap-2.5", panelClass(0), className)}>
      <div className="flex min-h-7 items-center justify-between gap-2">
        <h2 className={panelTitleClass(0)}>{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  )
}

function FieldLabel({ htmlFor, children }: { htmlFor: string; children: React.ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="text-[0.8rem] font-medium text-foreground/85">
      {children}
    </label>
  )
}

function SizeField({ id, label, value, onChange }: { id: string; label: string; value: number; onChange: (value: number, commit: boolean) => void }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1">
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <NumberField
        id={id}
        value={value}
        min={16}
        step={16}
        format={{ maximumFractionDigits: 0, useGrouping: false }}
        onValueChange={(next) => next !== null && next > 0 && onChange(next, false)}
        onValueCommitted={(next) => next !== null && next > 0 && onChange(next, true)}
      />
    </div>
  )
}

function ResolutionSection() {
  const { t } = useTranslation()
  const id = useId()
  const width = useBuilder((s) => s.width)
  const height = useBuilder((s) => s.height)
  const mp = useBuilder((s) => s.mp)
  const ar = useBuilder((s) => s.ar)
  const set = useBuilder((s) => s.set)
  const setSize = useBuilder((s) => s.setSize)
  const applyMp = () => {
    const size = resolutionFor(mp, ar)
    if (size) setSize(size.width, size.height, true)
  }
  return (
    <Section title={t("Resolution")}>
      <div className="flex gap-2">
        <SizeField id={`${id}-w`} label={t("Width")} value={width} onChange={(v, commit) => (commit ? setSize(round16(v), height, true) : setSize(v, height))} />
        <SizeField id={`${id}-h`} label={t("Height")} value={height} onChange={(v, commit) => (commit ? setSize(width, round16(v), true) : setSize(width, v))} />
      </div>
      {/* the button drops under the fields where the card is too narrow for all three */}
      <div className="flex flex-wrap items-end gap-2">
        <div className="flex w-24 shrink-0 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-mp`}>{t("Megapixels")}</FieldLabel>
          <NumberField id={`${id}-mp`} value={mp} min={0.1} step={0.1} format={{ maximumFractionDigits: 2 }} onValueChange={(next) => next !== null && next > 0 && set({ mp: next })} />
        </div>
        <div className="flex w-24 shrink-0 flex-col gap-1">
          <span id={`${id}-ar`} className="text-[0.8rem] font-medium text-foreground/85">
            {t("Aspect ratio")}
          </span>
          <Select<AspectRatio> items={ASPECT_RATIOS.map((value) => ({ value, label: value }))} value={ar} onValueChange={(next) => next !== null && set({ ar: next })}>
            <SelectTrigger aria-labelledby={`${id}-ar`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ASPECT_RATIOS.map((value) => (
                <SelectItem key={value} value={value}>
                  {value}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button variant="secondary" className="flex-1 basis-auto" onClick={applyMp}>
          <Ratio aria-hidden />
          {t("Set resolution")}
        </Button>
      </div>
    </Section>
  )
}

function LoadPromptDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation()
  const id = useId()
  const load = useBuilder((s) => s.load)
  const set = useBuilder((s) => s.set)
  const setVgl = useBuilder((s) => s.setVgl)
  const width = useBuilder((s) => s.width)
  const height = useBuilder((s) => s.height)
  const [text, setText] = useState("")
  const [size, setSize] = useState({ width: 1024, height: 1024 })
  const [error, setError] = useState<string | null>(null)
  const [wasOpen, setWasOpen] = useState(open)
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) {
      // Each opening starts empty, sized like the current canvas (upstream).
      setText("")
      setSize({ width, height })
      setError(null)
    }
  }
  const submit = () => {
    if (text.trim() === "") {
      onOpenChange(false)
      return
    }
    try {
      if (isVglPrompt(text)) {
        const { draft, vgl, extra } = decompileVglPrompt(text, size.width, size.height)
        load(draft)
        setVgl(vgl)
        set(extra ? { format: "vgl", extra } : { format: "vgl" })
      } else if (isFlux3Prompt(text)) {
        const { draft, extra } = decompileFlux3Prompt(text, size.width, size.height)
        load(draft)
        set(extra ? { format: "flux3", extra } : { format: "flux3" })
      } else {
        const { json, extra } = splitPrompt(text)
        load(decompilePrompt(json, size.width, size.height))
        set(extra ? { format: "moss", extra } : { format: "moss" })
      }
      onOpenChange(false)
      toast.success(t("Prompt loaded"))
    } catch (err) {
      setError(message(err))
    }
  }
  // the text box takes the focus through the dialog (`initialFocus`): `autoFocus` made it the place
  // to give the focus back to on close, and the focus was left on the page
  const textRef = useRef<HTMLTextAreaElement>(null)
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl" initialFocus={textRef}>
        <DialogHeader>
          <DialogTitle>{t("Load prompt")}</DialogTitle>
          <DialogDescription>{t("Paste a compiled MOSS-Image, FLUX.3, or Bria Fibo 2.0 VGL prompt (with the size it was made for) or a saved layout draft.")}</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault()
            submit()
          }}
        >
          <div className="flex flex-col gap-3 sm:flex-row">
            <Textarea
              aria-label={t("Compiled prompt or layout draft")}
              placeholder={t("Compiled prompt or layout draft")}
              spellCheck={false}
              ref={textRef}
              value={text}
              aria-invalid={error !== null || undefined}
              aria-describedby={error ? `${id}-error` : undefined}
              onChange={(event) => {
                setText(event.target.value)
                setError(null)
              }}
              className="field-sizing-fixed h-64 min-w-0 flex-1 resize-none font-mono text-xs md:text-xs"
            />
            <div className="flex gap-2 sm:w-32 sm:flex-col">
              <SizeField id={`${id}-w`} label={t("Width")} value={size.width} onChange={(width) => setSize((s) => ({ ...s, width }))} />
              <SizeField id={`${id}-h`} label={t("Height")} value={size.height} onChange={(height) => setSize((s) => ({ ...s, height }))} />
            </div>
          </div>
          {error ? (
            <p id={`${id}-error`} role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm break-words text-destructive">
              {error}
            </p>
          ) : null}
          <DialogFooter>
            <Button type="submit">{t("Load")}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function FileSection() {
  const { t } = useTranslation()
  const load = useBuilder((s) => s.load)
  const set = useBuilder((s) => s.set)
  const setVgl = useBuilder((s) => s.setVgl)
  const fileInput = useRef<HTMLInputElement>(null)
  const [dialog, setDialog] = useState(false)

  const save = () => {
    const blob = new Blob([`${JSON.stringify(buildDraft(useBuilder.getState()), null, 2)}\n`], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = "layout.json"
    document.body.append(link)
    link.click()
    link.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const open = async (file: File) => {
    try {
      load(decompilePrompt(await file.text(), Number.NaN, Number.NaN))
      toast.success(t("Layout loaded"))
    } catch (err) {
      toast.error(t("Could not load {{name}}", { name: file.name }), { description: message(err) })
    }
  }

  /** Upstream's "Load from Prompt": the compiled prompt and the size of a generation form, back into the draft. */
  const loadFrom = (tab: "txt2img" | "img2img") => {
    const target = t(TARGET_NAMES[tab])
    const core = getFormStore(tab).getState().values.core
    const prompt = typeof core?.prompt === "string" ? core.prompt.trim() : ""
    const w = Number(core?.width)
    const h = Number(core?.height)
    if (prompt === "") return void toast.error(t("No prompt in {{target}} to load", { target }))
    if (!Number.isFinite(w) || w <= 0 || !Number.isFinite(h) || h <= 0) return void toast.error(t("{{target}} has no usable size", { target }))
    try {
      if (isVglPrompt(prompt)) {
        const { draft, vgl, extra } = decompileVglPrompt(prompt, w, h)
        load(draft)
        setVgl(vgl)
        set(extra ? { format: "vgl", extra } : { format: "vgl" })
      } else if (isFlux3Prompt(prompt)) {
        const { draft, extra } = decompileFlux3Prompt(prompt, w, h)
        load(draft)
        set(extra ? { format: "flux3", extra } : { format: "flux3" })
      } else {
        const { json, extra } = splitPrompt(prompt)
        load(decompilePrompt(json, w, h))
        set(extra ? { format: "moss", extra } : { format: "moss" })
      }
      toast.success(t("Prompt loaded from {{target}}", { target }))
    } catch (err) {
      toast.error(t("Could not load the prompt of {{target}}", { target }), { description: message(err) })
    }
  }

  return (
    <Section title={t("Save / load")}>
      <Button variant="secondary" onClick={() => setDialog(true)}>
        <Braces aria-hidden />
        {t("Load prompt…")}
      </Button>
      <div className="flex gap-2">
        <Button variant="outline" className="flex-1" onClick={() => loadFrom("txt2img")}>
          <FileInput aria-hidden />
          {t("Load from txt2img")}
        </Button>
        <Button variant="outline" className="flex-1" onClick={() => loadFrom("img2img")}>
          <FileInput aria-hidden />
          {t("Load from img2img")}
        </Button>
      </div>
      <div className="flex gap-2">
        <Button variant="outline" className="flex-1" onClick={save}>
          <FileDown aria-hidden />
          {t("Save JSON")}
        </Button>
        <Button variant="outline" className="flex-1" onClick={() => fileInput.current?.click()}>
          <FileUp aria-hidden />
          {t("Load JSON")}
        </Button>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept=".json,application/json"
        hidden
        data-testid="json-builder-file"
        onChange={(event) => {
          const file = event.target.files?.[0]
          event.target.value = ""
          if (file) void open(file)
        }}
      />
      <LoadPromptDialog open={dialog} onOpenChange={setDialog} />
    </Section>
  )
}

function VglSettings({ id }: { id: string }) {
  const { t } = useTranslation()
  const vgl = useBuilder((s) => s.vgl)
  const setVgl = useBuilder((s) => s.setVgl)
  const mediumItems = VGL_STYLE_MEDIUMS.map((value) => ({ value, label: value }))

  return (
    <div data-slot="vgl-schema-controls" className="flex flex-col gap-2.5">
      <div className="flex items-baseline gap-2 rounded-r-sm border-l-2 border-l-advice-warn bg-muted/45 px-2.5 py-1.5 text-xs leading-snug">
        <span className="shrink-0 font-mono text-[0.6875rem] font-semibold tracking-tight text-advice-warn">{VGL_LICENSE}</span>
        <span className="min-w-0 flex-1 text-muted-foreground">
          {t("Bria Fibo 2.0 VGL 2.0 JSON: non-commercial weights (CC BY-NC 4.0) unless commercially licensed.")}
        </span>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-1">
          <span id={`${id}-vgl-medium`} className="text-[0.8rem] font-medium text-foreground/85">
            {t("Style medium")}
          </span>
          <Select<string> items={mediumItems} value={vgl.styleMedium} onValueChange={(next) => next && setVgl({ styleMedium: next })}>
            <SelectTrigger aria-labelledby={`${id}-vgl-medium`} size="sm" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {mediumItems.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-vgl-light`}>{t("Lighting")}</FieldLabel>
          <Input id={`${id}-vgl-light`} title={vgl.lightingConditions} value={vgl.lightingConditions} onChange={(e) => setVgl({ lightingConditions: e.target.value })} />
        </div>
        <div className="flex min-w-0 flex-col gap-1 sm:col-span-2">
          <FieldLabel htmlFor={`${id}-vgl-bg`}>{t("Background setting")}</FieldLabel>
          <Input id={`${id}-vgl-bg`} title={vgl.backgroundSetting} value={vgl.backgroundSetting} onChange={(e) => setVgl({ backgroundSetting: e.target.value })} />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-vgl-mood`}>{t("Mood and atmosphere")}</FieldLabel>
          <Input id={`${id}-vgl-mood`} title={vgl.moodAtmosphere} value={vgl.moodAtmosphere} onChange={(e) => setVgl({ moodAtmosphere: e.target.value })} />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-vgl-angle`}>{t("Camera angle")}</FieldLabel>
          <Input id={`${id}-vgl-angle`} title={vgl.cameraAngle} value={vgl.cameraAngle} onChange={(e) => setVgl({ cameraAngle: e.target.value })} />
        </div>
        <div className="flex min-w-0 flex-col gap-1 sm:col-span-2">
          <FieldLabel htmlFor={`${id}-vgl-lens`}>{t("Lens focal length")}</FieldLabel>
          <Input id={`${id}-vgl-lens`} title={vgl.lensFocalLength} value={vgl.lensFocalLength} onChange={(e) => setVgl({ lensFocalLength: e.target.value })} />
        </div>
      </div>
    </div>
  )
}

function SummarySection({ missing, onEditField }: { missing: ReadonlySet<Missing>; onEditField: (field: EditableField) => void }) {
  const { t } = useTranslation()
  const id = useId()
  const caption = useBuilder((s) => s.caption)
  const meta = useBuilder((s) => s.meta)
  const extra = useBuilder((s) => s.extra)
  const canvas = useBuilder((s) => s.canvas)
  const format = useBuilder((s) => s.format)
  const set = useBuilder((s) => s.set)
  return (
    <Section title={t("Summary")}>
      <Segmented<PromptFormat>
        aria-label={t("Prompt format")}
        className="w-full"
        options={PROMPT_FORMATS.map((value) => ({ value, label: PROMPT_FORMAT_NAMES[value] }))}
        value={format}
        onValueChange={(value) => set({ format: value })}
      />
      {format === "flux3" ? (
        <p className="text-xs text-muted-foreground">{t("FLUX.3 layout: a caption citing every item as <id>, then the boxes on a 0-1000 grid (any output size).")}</p>
      ) : format === "vgl" ? (
        <VglSettings id={id} />
      ) : (
        <Segmented<CanvasType>
          aria-label={t("Canvas type")}
          className="w-full"
          options={CANVAS_TYPES.map((value) => ({ value, label: t(CANVAS_NAMES[value]) }))}
          value={canvas}
          onValueChange={(value) => set({ canvas: value })}
        />
      )}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-1">
          <FieldLabel htmlFor={`${id}-caption`}>{t("Caption")}</FieldLabel>
          <Button size="icon-xs" variant="ghost" aria-label={t("Edit the summary caption in the field editor")} title={t("Open in the field editor")} onClick={() => onEditField({ kind: "summary" })}>
            <Pencil aria-hidden />
          </Button>
        </div>
        <Textarea
          id={`${id}-caption`}
          rows={2}
          className="min-h-14"
          placeholder={t("The whole picture in a sentence or two")}
          aria-invalid={missing.has("caption") || undefined}
          value={caption}
          onChange={(event) => set({ caption: event.target.value })}
          onDoubleClick={() => onEditField({ kind: "summary" })}
        />
      </div>
      <div className="flex gap-2">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-meta`}>{t("Meta")}</FieldLabel>
          <Input id={`${id}-meta`} placeholder={t("Tags, comma separated")} value={meta} onChange={(event) => set({ meta: event.target.value })} />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-extra`}>{t("Extra")}</FieldLabel>
          <Input id={`${id}-extra`} placeholder={t("e.g. a LoRA")} title={t("Added after the compiled prompt")} value={extra} onChange={(event) => set({ extra: event.target.value })} />
        </div>
      </div>
    </Section>
  )
}

const CORNERS = ["x1", "y1", "x2", "y2"] as const

function ItemRow({
  item,
  index,
  active,
  missing,
  onActivate,
  onEditField,
}: {
  item: Item
  index: number
  active: boolean
  missing: ReadonlySet<Missing>
  onActivate: (id: number) => void
  onEditField: (field: EditableField) => void
}) {
  const { t } = useTranslation()
  const id = useId()
  const updateItem = useBuilder((s) => s.updateItem)
  const removeItem = useBuilder((s) => s.removeItem)
  const edit = (field: "caption" | "text") => onEditField({ kind: "item", id: item.id, field })
  return (
    <li
      data-item={item.id}
      onFocusCapture={() => onActivate(item.id)}
      onPointerDown={() => onActivate(item.id)}
      className={cn("flex flex-col gap-1.5 rounded-md border border-transparent p-2 transition-colors hover:bg-muted/50", active && "border-primary/40 bg-primary/5 hover:bg-primary/5")}
    >
      <div className="flex items-center gap-1.5">
        <span className="w-5 shrink-0 text-center text-xs font-medium tabular-nums" style={{ color: `color-mix(in oklch, ${itemColor(index)} 45%, var(--foreground))` }} aria-hidden>
          {index + 1}
        </span>
        <Segmented<Label>
          size="sm"
          aria-label={t("Item type")}
          className="flex-1"
          options={LABELS.map((value) => ({ value, label: t(LABEL_NAMES[value]) }))}
          value={item.label}
          onValueChange={(label) => updateItem(item.id, { label })}
        />
        <Button size="icon-xs" variant="ghost" aria-label={t("Remove item {{n}}", { n: index + 1 })} title={t("Remove")} onClick={() => removeItem(item.id)}>
          <X aria-hidden />
        </Button>
      </div>
      <div className="flex items-start gap-1">
        <Textarea
          aria-label={t("Caption of item {{n}}", { n: index + 1 })}
          placeholder={t("Caption")}
          rows={1}
          className="min-h-8 min-w-0 flex-1 py-1.5"
          aria-invalid={missing.has(`item:${item.id}:caption`) || undefined}
          value={item.caption}
          onChange={(event) => updateItem(item.id, { caption: event.target.value })}
          onDoubleClick={() => edit("caption")}
        />
        <Button size="icon-xs" variant="ghost" className="self-center" aria-label={t("Edit the caption of item {{n}} in the field editor", { n: index + 1 })} title={t("Open in the field editor")} onClick={() => edit("caption")}>
          <Pencil aria-hidden />
        </Button>
      </div>
      {item.label === "text" ? (
        <div className="flex items-center gap-1">
          <Input
            aria-label={t("Text to render of item {{n}}", { n: index + 1 })}
            className="min-w-0 flex-1"
            placeholder={t("Text to render")}
            aria-invalid={missing.has(`item:${item.id}:text`) || undefined}
            value={item.text}
            onChange={(event) => updateItem(item.id, { text: event.target.value })}
            onDoubleClick={() => edit("text")}
          />
          <Button size="icon-xs" variant="ghost" aria-label={t("Edit the text of item {{n}} in the field editor", { n: index + 1 })} title={t("Open in the field editor")} onClick={() => edit("text")}>
            <Pencil aria-hidden />
          </Button>
        </div>
      ) : null}
      <div className="grid grid-cols-4 gap-1.5" role="group" aria-label={t("Box of item {{n}}", { n: index + 1 })}>
        {CORNERS.map((corner, i) => (
          <div key={corner} className="flex min-w-0 items-center gap-1">
            <label htmlFor={`${id}-${corner}`} className="text-[0.6875rem] text-muted-foreground">
              {corner}
            </label>
            <NumberField
              id={`${id}-${corner}`}
              size="sm"
              steppers={false}
              className="flex-1"
              value={item.bbox[i]!}
              format={{ maximumFractionDigits: 2, useGrouping: false }}
              onValueChange={(next) => {
                if (next === null) return
                const bbox = item.bbox.slice() as Box
                bbox[i] = next
                updateItem(item.id, { bbox })
              }}
            />
          </div>
        ))}
      </div>
    </li>
  )
}

function ItemsSection({
  activeId,
  onActivate,
  onEditField,
  missing,
  expanded,
  onToggleExpand,
}: {
  activeId: number | null
  onActivate: (id: number) => void
  onEditField: (field: EditableField) => void
  missing: ReadonlySet<Missing>
  expanded: boolean
  onToggleExpand: () => void
}) {
  const { t } = useTranslation()
  const items = useBuilder((s) => s.items)
  const addItem = useBuilder((s) => s.addItem)
  const clearItems = useBuilder((s) => s.clearItems)
  return (
    <Section
      title={t("Items")}
      className={cn("min-h-0", expanded ? "flex-1" : "lg:min-h-48 lg:flex-1")}
      actions={
        <div className="flex items-center gap-1">
          <Button size="sm" variant="secondary" onClick={addItem}>
            <Plus aria-hidden />
            {t("Add item")}
          </Button>
          <Button size="sm" variant="ghost" onClick={clearItems}>
            <Trash2 aria-hidden />
            {t("Clear all")}
          </Button>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-expanded={expanded}
            aria-label={expanded ? t("Collapse the item list") : t("Expand the item list")}
            title={expanded ? t("Collapse the item list") : t("Expand the item list")}
            onClick={onToggleExpand}
          >
            <Expand aria-hidden />
          </Button>
        </div>
      }
    >
      <ul className="-mr-1 flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-1" aria-label={t("Items")}>
        {items.map((item, index) => (
          <ItemRow key={item.id} item={item} index={index} active={item.id === activeId} missing={missing} onActivate={onActivate} onEditField={onEditField} />
        ))}
      </ul>
    </Section>
  )
}

const TARGET_NAMES: Record<Extract<FormTab, "txt2img" | "img2img">, string> = { txt2img: "txt2img", img2img: "img2img" }

export default function JsonBuilderPage(_props: CustomPageProps) {
  const { t } = useTranslation()
  const [activeId, setActiveId] = useState<number | null>(null)
  const [showMissing, setShowMissing] = useState(false)
  const [itemsExpanded, setItemsExpanded] = useState(false)
  const [editField, setEditField] = useState<EditableField | null>(null)
  const state = useBuilder()
  const missing = showMissing ? missingFields(state) : new Set<Missing>()

  /** The prompt, or null after telling what is wrong. */
  const compile = (): string | null => {
    const empty = missingFields(state)
    if (empty.size > 0) {
      // a missing summary is hidden while the item list is expanded: reveal it
      if (empty.has("caption")) setItemsExpanded(false)
      setShowMissing(true)
      toast.error(t("Fill in the highlighted fields"))
      return null
    }
    setShowMissing(false)
    try {
      return finalPrompt(state)
    } catch (err) {
      toast.error(t("Could not build the prompt"), { description: message(err) })
      return null
    }
  }

  const send = (tab: "txt2img" | "img2img") => {
    const prompt = compile()
    if (prompt === null) return
    const target = t(TARGET_NAMES[tab])
    const { width, height } = state
    deliver(tab, [{ owner: "core", values: { prompt, width, height } }])
      .then((count) => {
        if (count === 0) return void toast.error(t("Could not send to {{target}}", { target }))
        const limit = sizeLimit(tab, width, height)
        if (limit === null) toast.success(t("Sent to {{target}}", { target }))
        else toast.warning(t("Sent to {{target}}, but it accepts at most {{max}} px per side: change the size before generating", { target, max: limit }))
      })
      .catch((err: unknown) => toast.error(t("Could not send to {{target}}", { target }), { description: message(err) }))
  }

  const copy = () => {
    const prompt = compile()
    if (prompt !== null) void copyText(prompt).then((ok) => (ok ? toast.success(t("Copied")) : toast.error(t("Could not copy"))))
  }

  return (
    <div className="h-full overflow-y-auto lg:overflow-hidden" data-scroll-root>
      <div className="flex min-h-full flex-col gap-3 p-3 sm:p-4 lg:h-full lg:flex-row">
        <LayoutCanvas className="h-[60vh] max-h-[75vh] min-h-64 shrink-0 lg:h-auto lg:max-h-none lg:min-h-0 lg:flex-1 lg:shrink" activeId={activeId} onActivate={setActiveId} onEditField={setEditField} missing={missing} />
        {/* lg: a scroller; 4 px of room either side (-mx-1 / px-1) keeps the fields' focus rings */}
        <div className="flex min-h-0 flex-col gap-3 lg:-mx-1 lg:w-[calc(26rem+0.5rem)] lg:shrink-0 lg:overflow-y-auto lg:px-1 xl:w-[calc(30rem+0.5rem)]">
          {itemsExpanded ? null : (
            <>
              {/* side by side only where the panel spans the page: the side column (lg+) is too narrow for two */}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
                <ResolutionSection />
                <FileSection />
              </div>
              <SummarySection missing={missing} onEditField={setEditField} />
            </>
          )}
          <ItemsSection activeId={activeId} onActivate={setActiveId} onEditField={setEditField} missing={missing} expanded={itemsExpanded} onToggleExpand={() => setItemsExpanded((v) => !v)} />
          {/* pinned to the bottom of the page (phone) or of the scrolling side column (lg+): the send buttons stay in reach */}
          <div className="sticky bottom-0 z-10 flex gap-2 border-t border-border/70 bg-background py-2 lg:border-t-0 lg:pt-2 lg:pb-0" data-toast-obstacle>
            <Button variant="glow" className="flex-1 font-semibold" onClick={() => send("txt2img")}>
              <Wand aria-hidden />
              {t("Send to txt2img")}
            </Button>
            <Button variant="secondary" className="flex-1" onClick={() => send("img2img")}>
              <Images aria-hidden />
              {t("Send to img2img")}
            </Button>
            <Button size="icon" variant="outline" aria-label={t("Copy prompt")} title={t("Copy prompt")} onClick={copy}>
              <ClipboardCopy aria-hidden />
            </Button>
          </div>
        </div>
      </div>
      {editField ? (
        <Suspense fallback={null}>
          <FieldEditor key={editField.kind === "summary" ? "summary" : `${editField.id}:${editField.field}`} field={editField} onClose={() => setEditField(null)} />
        </Suspense>
      ) : null}
    </div>
  )
}
