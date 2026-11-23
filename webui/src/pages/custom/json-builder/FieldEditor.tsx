/**
 * The field editor dialog (upstream's "prompt link", which synced a caption
 * or text field to the main prompt for its completions and styles): a real
 * prompt editor — same completions and attention editing as the generation
 * fields — for one draft field, plus a styles picker that applies the
 * selected styles into that field only. The generation prompt, negative
 * prompt and styles are never touched. Loaded lazily: it mounts only while
 * a field is being edited.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { Suspense, useEffect, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { stylesQuery } from "@/api/queries"
import { Wand } from "@/components/icons"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { LazyPromptEditor } from "@/features/prompt-editor"
import { ChipsSelect } from "@/ui/inputs/ChipsSelect"
import type { EditableField } from "./draft"
import { useBuilder } from "./store"
import { failureText } from "@/features/errors/error-text"

/** The draft value of `field`, or null when its item is gone. */
function fieldValue(field: EditableField): string | null {
  const s = useBuilder.getState()
  if (field.kind === "summary") return s.caption
  const item = s.items.find((item) => item.id === field.id)
  // a "text" field exists only while the item is still a text item
  return item && (field.field !== "text" || item.label === "text") ? item[field.field] : null
}

function setFieldValue(field: EditableField, value: string) {
  const s = useBuilder.getState()
  if (field.kind === "summary") s.set({ caption: value })
  else s.updateItem(field.id, { [field.field]: value })
}

export function FieldEditor({ field, onClose }: { field: EditableField; onClose: () => void }) {
  const { t } = useTranslation()
  const id = useId()
  const styles = useQuery(stylesQuery)
  const [selected, setSelected] = useState<string[]>([])
  const [query, setQuery] = useState("")
  // Responses that arrive after the dialog closed (or for a field whose value
  // or draft moved on meanwhile — an external update or a load) must not
  // overwrite it. StrictMode replays effects, so `alive` resets each mount.
  const alive = useRef(true)
  // The draft's revision when this dialog opened: a `load` invalidates it even
  // when the field's value happens to be identical.
  const [openedRevision] = useState(() => useBuilder.getState().draftRevision)

  const value = useBuilder((s) => {
    if (field.kind === "summary") return s.caption
    const item = s.items.find((item) => item.id === field.id)
    return item && (field.field !== "text" || item.label === "text") ? item[field.field] : null
  })
  const itemIndex = useBuilder((s) => (field.kind === "item" ? s.items.findIndex((item) => item.id === field.id) : -1))
  const revision = useBuilder((s) => s.draftRevision)

  const gone = (field.kind === "item" && (itemIndex < 0 || value === null)) || revision !== openedRevision
  useEffect(() => {
    // The item was deleted or a load replaced the draft: nothing to edit.
    if (gone) onClose()
  }, [gone, onClose])
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  const title =
    field.kind === "summary"
      ? t("Edit the summary caption")
      : field.field === "caption"
        ? t("Edit the caption of item {{n}}", { n: itemIndex + 1 })
        : t("Edit the text of item {{n}}", { n: itemIndex + 1 })

  const apply = useMutation({
    mutationFn: (vars: { value: string; revision: number; styles: string[] }) => api.styles.apply({ prompt: vars.value, negative_prompt: "", styles: vars.styles }),
    onSuccess: (applied, vars) => {
      if (!alive.current || useBuilder.getState().draftRevision !== vars.revision || fieldValue(field) !== vars.value) return
      setFieldValue(field, applied.prompt)
      setSelected([])
      toast.success(t("Styles applied"))
    },
    onError: (error) => {
      if (alive.current) toast.error(t("Could not apply the styles"), { description: failureText(error) })
    },
  })

  const busy = apply.isPending
  const options = (Array.isArray(styles.data) ? styles.data : []).map((style) => ({ value: style.name, label: style.name }))

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) {
          alive.current = false
          onClose()
        }
      }}
    >
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{t("Edit with prompt completions and styles; only this field of the draft changes")}</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          {value !== null ? (
            <Suspense fallback={<div aria-hidden className="rounded-sm bg-field" style={{ minHeight: "5.9rem" }} />}>
              <LazyPromptEditor
                value={value}
                onChange={(next) => setFieldValue(field, next)}
                disabled={busy}
                kind="positive"
                scope={{ form: null, extension: true }}
                lint={false}
                minLines={4}
                maxLines={12}
                ariaLabel={title}
              />
            </Suspense>
          ) : null}
          <div className="flex flex-col gap-2">
            <span id={`${id}-styles`} className="text-[0.8rem] font-medium text-foreground/85">
              {t("Prompt styles")}
            </span>
            <div className="flex items-start gap-2">
              <div className="min-w-0 flex-1">
                <ChipsSelect
                  id={`${id}-styles-select`}
                  options={options}
                  selected={selected}
                  onChange={(values) => setSelected(values.map(String))}
                  allowCustom={false}
                  disabled={busy}
                  ariaLabel={t("Prompt styles")}
                  ariaDescribedBy={`${id}-styles`}
                  invalid={false}
                  query={query}
                  setQuery={setQuery}
                />
              </div>
              <Button
                variant="secondary"
                disabled={busy || selected.length === 0 || value === null}
                onClick={() => value !== null && apply.mutate({ value, revision: useBuilder.getState().draftRevision, styles: [...selected] })}
              >
                <Wand aria-hidden />
                {t("Apply styles")}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
