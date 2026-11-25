/**
 * Style editor (former styles popup): pick or name a style, edit its prompt
 * and negative prompt, copy them from the form's prompts, save or delete.
 * Saving refreshes the style list and the forms' `styles` choices.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { ClipboardCopy, Save, Trash2 } from "@/components/icons"
import { Suspense, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { keys, queryClient, stylesQuery } from "@/api/queries"
import type { PromptStyle } from "@/api/types"
import { SearchSelect } from "@/components/SearchSelect"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { LoadFailed } from "@/components/LoadFailed"
import { useLoadFailure } from "@/api/load-failure"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { LazyPromptEditor } from "../index"
import { recountStyledTokens } from "../use-token-count"
import { failureText } from "@/features/errors/error-text"

type StyleEditorDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** The form's current prompts, for "Copy from prompts". */
  currentPrompts: () => { prompt: string; negative: string }
  /** Style to show first. */
  initial?: string | null
}

function refreshStyles() {
  void queryClient.invalidateQueries({ queryKey: keys.styles })
  // The forms' `styles` choices are resolved into the schema.
  void queryClient.invalidateQueries({ queryKey: ["schema"] })
  // Token counters that include styles count the new style text.
  recountStyledTokens()
}

export function StyleEditorDialog({ open, onOpenChange, currentPrompts, initial }: StyleEditorDialogProps) {
  const { t } = useTranslation()
  const id = useId()
  // Every opening asks for the list again (stale at once): another client, or an edit of
  // styles.csv, may have changed a style since it was fetched, and no event says so.
  const styles = useQuery({ ...stylesQuery, enabled: open, staleTime: 0 })
  const failure = useLoadFailure(styles)
  // Until the user edits, the editor shows the form's style as the list has it: the list may
  // arrive after the dialog opens, and an editor left blank would save a blank style over it.
  const [edited, setEdited] = useState<PromptStyle | null>(null)
  const initialStyle = styles.data?.find((style) => style.name === initial)
  const draft = edited ?? (initialStyle ? { ...initialStyle } : { name: initial ?? "", prompt: "", negative_prompt: "" })
  const setDraft = (update: (draft: PromptStyle) => PromptStyle) => setEdited((current) => update(current ?? draft))
  const [confirmDelete, setConfirmDelete] = useState(false)
  const existing = styles.data?.find((style) => style.name === draft.name)
  const dirty = !existing || existing.prompt !== draft.prompt || existing.negative_prompt !== draft.negative_prompt

  // The dialog stays mounted once opened (StylesField): closing it (Esc) keeps an unsaved edit for
  // the next opening, while an untouched or saved one starts again on the form's style.
  const [shownOpen, setShownOpen] = useState(open)
  if (open !== shownOpen) {
    setShownOpen(open)
    if (!open && !dirty) setEdited(null)
  }

  // After a retry, the focus goes to the list it brought, not to the dialog the Retry button left.
  const focusAfterRetry = useRef(false)
  const fieldsShown = (fields: HTMLDivElement | null) => {
    if (!fields || !focusAfterRetry.current) return
    focusAfterRetry.current = false
    fields.querySelector<HTMLElement>("[role=combobox]")?.focus()
  }

  const select = (name: string) => {
    const style = styles.data?.find((s) => s.name === name)
    setEdited(style ? { ...style } : { name, prompt: draft.prompt, negative_prompt: draft.negative_prompt })
  }

  const save = useMutation({
    mutationFn: () => api.styles.save({ ...draft, name: draft.name.trim() }),
    onSuccess: (saved) => {
      refreshStyles()
      toast.success(t("Style “{{name}}” saved", { name: saved?.name ?? draft.name }))
    },
    onError: (error) => toast.error(t("Could not save the style"), { description: failureText(error) }),
  })
  const remove = useMutation({
    mutationFn: () => api.styles.remove(draft.name),
    onSuccess: () => {
      refreshStyles()
      toast.success(t("Style “{{name}}” deleted", { name: draft.name }))
      setEdited({ name: "", prompt: "", negative_prompt: "" })
    },
    onError: (error) => toast.error(t("Could not delete the style"), { description: failureText(error) }),
  })

  const options = (styles.data ?? []).map((style) => ({ value: style.name, label: style.name }))

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="flex max-h-[min(44rem,calc(100dvh-2rem-env(safe-area-inset-top)-env(safe-area-inset-bottom)))] flex-col gap-4 sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("Styles")}</DialogTitle>
            <DialogDescription>
              {t("Use {prompt} in a style to place your prompt there; otherwise the style is appended to the prompt.")}
            </DialogDescription>
          </DialogHeader>
          <div className="-mx-1 flex min-h-0 flex-col gap-3 overflow-y-auto px-1 py-1">
            {/* No fields before the list: an edit made then would start from a blank style and save
                that blank over the real one. */}
            {failure ? (
              <LoadFailed
                error={failure.error}
                retrying={failure.retrying}
                message={t("Could not load the styles")}
                onRetry={() => {
                  focusAfterRetry.current = true
                  void styles.refetch()
                }}
                className="py-0"
              />
            ) : !styles.data ? (
              // the fields' outline, so the dialog does not grow when the list arrives
              <div role="status" className="flex flex-col gap-3">
                <span className="sr-only">{t("Loading…")}</span>
                {["h-8", "h-20", "h-20"].map((field, index) => (
                  <div key={index} aria-hidden className="flex flex-col gap-1.5">
                    <Skeleton className="h-5 w-24" />
                    <Skeleton className={field} />
                  </div>
                ))}
              </div>
            ) : (
              <div ref={fieldsShown} className="flex flex-col gap-3">
                <div className="flex flex-col gap-1.5">
                  <label id={`${id}-name`} className="text-[0.8rem] font-medium">
                    {t("Style", { context: "prompt" })}
                  </label>
                  <SearchSelect
                    aria-labelledby={`${id}-name`}
                    options={options}
                    value={draft.name || null}
                    onValueChange={select}
                    allowCustom
                    placeholder={t("Choose a style or type a new name")}
                  />
                </div>
                {(["prompt", "negative_prompt"] as const).map((field) => (
                  <div key={field} className="flex flex-col gap-1.5">
                    <span id={`${id}-${field}`} className="text-[0.8rem] font-medium">
                      {field === "prompt" ? t("Prompt") : t("Negative prompt")}
                    </span>
                    <Suspense fallback={<div aria-hidden className="h-20 rounded-sm bg-field" />}>
                      <LazyPromptEditor
                        value={draft[field]}
                        onChange={(value) => setDraft((d) => ({ ...d, [field]: value }))}
                        ariaLabel={field === "prompt" ? t("Style prompt") : t("Style negative prompt")}
                        kind={field === "prompt" ? "positive" : "negative"}
                        minLines={3}
                        maxLines={10}
                      />
                    </Suspense>
                  </div>
                ))}
              </div>
            )}
          </div>
          <DialogFooter className="flex-wrap gap-2 sm:justify-between">
            <Button
              variant="ghost"
              disabled={!styles.data}
              onClick={() => {
                const { prompt, negative } = currentPrompts()
                setDraft((d) => ({ ...d, prompt, negative_prompt: negative }))
              }}
            >
              <ClipboardCopy />
              {t("Copy from prompts")}
            </Button>
            <div className="flex flex-col-reverse gap-2 sm:flex-row">
              <Button variant="outline" disabled={!existing || remove.isPending} onClick={() => setConfirmDelete(true)}>
                <Trash2 />
                {t("Delete")}
              </Button>
              {/* Without the list, or before a fresh one answers, saving could replace a style this
                  editor does not show as it is. */}
              <Button disabled={!styles.data || styles.isFetching || !draft.name.trim() || !dirty || save.isPending} onClick={() => save.mutate()}>
                <Save />
                {t("Save")}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Delete style “{{name}}”?", { name: draft.name })}</AlertDialogTitle>
            <AlertDialogDescription>{t("The style is removed from styles.csv.")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmDelete(false)
                remove.mutate()
              }}
            >
              {t("Delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
