/**
 * The user-metadata editor (former sd_forge_lora / core editor popups): the
 * fields the kind offers (description, architecture, a checkpoint's
 * preferred VAE / text encoders, activation text with the training tags to
 * pick from, preferred weight, negative text, tags, notes), a random prompt from the tags, saved to the `<model>.json`
 * sidecar with `PATCH /networks/{kind}/{name}/meta`.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { clientOption } from "@/api/app-options"
import { Copy, Dices, Save } from "@/components/icons"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { modelsQuery, queryClient } from "@/api/queries"
import type { NetworkItem, NetworkKind, NetworkMeta } from "@/api/types"
import { SearchSelect } from "@/components/SearchSelect"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { NumberField } from "@/components/ui/number-field"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { fileName, PRESET_LABELS } from "@/features/model-bar/model-options"
import { useArchLabel } from "./labels"
import { metaQuery, networkKeys } from "./queries"
import { TagChips } from "./TagChips"
import { hasTag, randomPrompt, toggleTag } from "./tags"
import { type Draft, draftFrom, PREFERRED_WEIGHT_LIMIT, patchFrom } from "./metadata-draft"
import { LoadFailed } from "@/components/LoadFailed"
import { failureText } from "@/features/errors/error-text"

function Field({ id, label, info, children }: { id: string; label: string; info?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-[0.8rem] font-medium text-foreground/85">
        {label}
      </label>
      {children}
      {info ? (
        <p id={`${id}-info`} className="text-xs text-muted-foreground">
          {info}
        </p>
      ) : null}
    </div>
  )
}

/** A checkpoint's preferred VAE / text encoders: checked modules replace the selection when the card loads the model. Exported for its tests. */
export function PreferredModules({ id, value, onChange }: { id: string; value: string[]; onChange: (next: string[]) => void }) {
  const { t } = useTranslation()
  const models = useQuery(modelsQuery)
  const listed = (models.data?.modules ?? [...(models.data?.vae ?? []), ...(models.data?.text_encoders ?? [])]).map((m) => ({ value: m.name, label: m.title ?? m.name }))
  // chosen files that are no longer listed stay visible so they can be removed
  const rows = [...value.filter((name) => !listed.some((m) => m.value === name)).map((name) => ({ value: name, label: name, missing: true })), ...listed.map((m) => ({ ...m, missing: false }))]
  const current = clientOption("forge_additional_modules").map(fileName)
  return (
    <div className="flex flex-col gap-1.5 sm:col-span-2">
      <div className="flex items-center gap-2">
        <span id={id} className="text-[0.8rem] font-medium text-foreground/85">
          {t("Preferred VAE / Text encoders")}
        </span>
        <Button type="button" variant="ghost" size="xs" className="ml-auto" disabled={current.length === 0} onClick={() => onChange(current)}>
          {t("Use the current selection")}
        </Button>
        <Button type="button" variant="ghost" size="xs" disabled={value.length === 0} onClick={() => onChange([])}>
          {t("Clear")}
        </Button>
      </div>
      <div role="group" aria-labelledby={id} aria-describedby={`${id}-info`} className="flex max-h-40 flex-col gap-0.5 overflow-y-auto rounded-lg bg-background p-1">
        {rows.map((m, index) => (
          <div key={m.value} className="flex items-center gap-2 rounded-md px-1.5 py-1 hover:bg-accent">
            <Checkbox id={`${id}-${index}`} checked={value.includes(m.value)} onCheckedChange={(on) => onChange(on ? [...value, m.value] : value.filter((v) => v !== m.value))} />
            <label htmlFor={`${id}-${index}`} className={m.missing ? "min-w-0 flex-1 truncate text-sm text-muted-foreground line-through" : "min-w-0 flex-1 truncate text-sm"} title={m.label}>
              {m.label}
            </label>
          </div>
        ))}
        {models.isError && !models.data ? (
          <LoadFailed error={models.error} message={t("Could not load the models")} onRetry={() => void models.refetch()} className="px-2" />
        ) : models.isPending ? (
          // the list is on its way, not empty (the model bar's picker says the same)
          <p role="status" className="px-2 py-3 text-center text-sm text-muted-foreground">{t("Loading…")}</p>
        ) : rows.length === 0 ? (
          <p className="px-2 py-3 text-center text-sm text-muted-foreground">{t("No modules found")}</p>
        ) : null}
      </div>
      <p id={`${id}-info`} className="text-xs text-muted-foreground">
        {t("Loading the checkpoint from its card also selects these (none: the selection is kept).")}
      </p>
    </div>
  )
}

function Editor({ meta, kind, onDone }: { meta: NetworkMeta; kind: NetworkKind; onDone: () => void }) {
  const { t } = useTranslation()
  const id = useId()
  const archLabel = useArchLabel()
  const [draft, setDraft] = useState<Draft>(() => draftFrom(meta))
  const [random, setRandom] = useState("")
  const fields = kind.fields
  const has = (field: string) => fields.includes(field)
  const update = (patch: Partial<Draft>) => setDraft((old) => ({ ...old, ...patch }))
  const tags = meta.dataset_tags

  const save = useMutation({
    mutationFn: () => api.networks.editMeta(meta.kind, meta.name, patchFrom(draft, fields)),
    onSuccess: (data) => {
      queryClient.setQueryData(networkKeys.meta(meta.kind, meta.name), data)
      void queryClient.invalidateQueries({ queryKey: ["networks", "list", meta.kind] })
      void queryClient.invalidateQueries({ queryKey: ["networks", "facets", meta.kind] })
      toast.success(t("Saved"))
      onDone()
    },
    onError: (error) => toast.error(t("Could not save the metadata"), { description: failureText(error) }),
  })

  const versions = ["Unknown", ...Object.keys(PRESET_LABELS)].map((value) => ({ value, label: value === "Unknown" ? t("Unknown") : archLabel(value) }))

  return (
    <form
      className="flex min-h-0 flex-1 flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault()
        save.mutate()
      }}
    >
      <div className="-mx-4 grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-y-auto px-4 pb-1 sm:grid-cols-2">
        {has("description") ? (
          <div className="sm:col-span-2">
            <Field id={`${id}-description`} label={t("Description")}>
              <Textarea id={`${id}-description`} value={draft.description} onChange={(event) => update({ description: event.target.value })} rows={3} />
            </Field>
          </div>
        ) : null}
        {has("sd version") ? (
          <Field id={`${id}-arch`} label={t("Architecture")} info={t("Which models it is made for; used by the compatibility filter.")}>
            <SearchSelect
              id={`${id}-arch`}
              options={versions}
              value={draft.sdVersion}
              onValueChange={(value) => update({ sdVersion: value })}
              aria-describedby={`${id}-arch-info`}
            />
          </Field>
        ) : null}
        {has("vae_te") ? <PreferredModules id={`${id}-modules`} value={draft.modules} onChange={(modules) => update({ modules })} /> : null}
        {has("preferred weight") ? (
          <Field id={`${id}-weight`} label={t("Preferred weight")} info={t("0 uses the default weight for extra networks (Settings).")}>
            <NumberField
              id={`${id}-weight`}
              value={draft.weight}
              onValueChange={(value) => update({ weight: value ?? 0 })}
              min={-PREFERRED_WEIGHT_LIMIT}
              max={PREFERRED_WEIGHT_LIMIT}
              step={0.05}
              smallStep={0.01}
              format={{ maximumFractionDigits: 3 }}
              aria-describedby={`${id}-weight-info`}
            />
          </Field>
        ) : null}
        {has("activation text") ? (
          <div className="flex flex-col gap-2 sm:col-span-2">
            <Field id={`${id}-activation`} label={t("Activation text")} info={t("Added to the prompt after the LoRA.")}>
              <Input id={`${id}-activation`} value={draft.activation} onChange={(event) => update({ activation: event.target.value })} aria-describedby={`${id}-activation-info`} />
            </Field>
            {tags.length ? (
              <TagChips
                tags={tags}
                pickLabel={t("Training tags: click to add to or remove from the activation text")}
                isOn={(tag) => hasTag(draft.activation, tag)}
                onPick={(tag) => update({ activation: toggleTag(draft.activation, tag) })}
                className="max-h-40 overflow-y-auto rounded-lg bg-background p-2"
              />
            ) : null}
          </div>
        ) : null}
        {has("negative text") ? (
          <div className="sm:col-span-2">
            <Field id={`${id}-negative`} label={t("Negative text")} info={t("Added to the negative prompt with the card.")}>
              <Input id={`${id}-negative`} value={draft.negative} onChange={(event) => update({ negative: event.target.value })} aria-describedby={`${id}-negative-info`} />
            </Field>
          </div>
        ) : null}
        {has("tags") ? (
          <div className="sm:col-span-2">
            <Field id={`${id}-tags`} label={t("Tags")} info={t("Comma-separated; searchable in the browser.")}>
              <Input id={`${id}-tags`} value={draft.tags} onChange={(event) => update({ tags: event.target.value })} aria-describedby={`${id}-tags-info`} />
            </Field>
          </div>
        ) : null}
        {has("notes") ? (
          <div className="sm:col-span-2">
            <Field id={`${id}-notes`} label={t("Notes")}>
              <Textarea id={`${id}-notes`} value={draft.notes} onChange={(event) => update({ notes: event.target.value })} rows={2} />
            </Field>
          </div>
        ) : null}
        {tags.length ? (
          <div className="flex flex-col gap-1.5 sm:col-span-2">
            <div className="flex items-center gap-2">
              <span className="text-[0.8rem] font-medium text-foreground/85">{t("Random prompt")}</span>
              <Button type="button" variant="outline" size="xs" className="ml-auto" onClick={() => setRandom(randomPrompt(tags))}>
                <Dices aria-hidden />
                {t("Generate")}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                disabled={!random}
                onClick={() => void navigator.clipboard.writeText(random).then(() => toast.success(t("Copied")))}
              >
                <Copy aria-hidden />
                {t("Copy")}
              </Button>
            </div>
            <p aria-live="polite" className="min-h-10 rounded-lg bg-background px-2.5 py-2 text-xs text-foreground/85">
              {random || <span className="text-muted-foreground">{t("A prompt drawn from the training tags, by frequency.")}</span>}
            </p>
          </div>
        ) : null}
      </div>
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onDone}>
          {t("Cancel")}
        </Button>
        <Button type="submit" disabled={save.isPending}>
          <Save aria-hidden />
          {t("Save")}
        </Button>
      </DialogFooter>
    </form>
  )
}

export function EditMetadataDialog({ item, kind, open, onOpenChange }: { item: NetworkItem | null; kind: NetworkKind | undefined; open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation()
  const meta = useQuery({ ...metaQuery(item?.kind ?? "", item?.name ?? ""), enabled: Boolean(item && open) })
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[min(calc(90dvh-env(safe-area-inset-top)-env(safe-area-inset-bottom)),52rem)] flex-col gap-4 sm:max-w-2xl">
        <DialogHeader className="pr-8">
          <DialogTitle className="truncate" title={item?.title}>{t("Edit {{name}}", { name: item?.title ?? "" })}</DialogTitle>
          <DialogDescription>{item?.path}</DialogDescription>
        </DialogHeader>
        {meta.data && kind ? (
          <Editor key={`${meta.data.kind}:${meta.data.name}`} meta={meta.data} kind={kind} onDone={() => onOpenChange(false)} />
        ) : meta.isError ? (
          <LoadFailed error={meta.error} onRetry={() => void meta.refetch()} />
        ) : (
          <div className="flex flex-col gap-3" aria-busy="true">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
