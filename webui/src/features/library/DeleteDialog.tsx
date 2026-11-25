/** Confirms removing images from the gallery, optionally deleting the files. */
import { useRef, useState } from "react"
import { useTranslation } from "react-i18next"
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
import { Checkbox } from "@/components/ui/checkbox"

type DeleteDialogProps = {
  /** Ids to remove; null closes the dialog. */
  ids: readonly string[] | null
  onCancel: () => void
  onConfirm: (ids: readonly string[], deleteFiles: boolean) => void
}

export function DeleteDialog({ ids, onCancel, onConfirm }: DeleteDialogProps) {
  const { t } = useTranslation()
  const [deleteFiles, setDeleteFiles] = useState(false)
  // the dialog opens on Cancel: its first control is the box that makes the removal irreversible
  const cancel = useRef<HTMLButtonElement>(null)
  const count = ids?.length ?? 0
  return (
    <AlertDialog open={ids !== null} onOpenChange={(open) => !open && onCancel()}>
      <AlertDialogContent initialFocus={cancel}>
        <AlertDialogHeader>
          <AlertDialogTitle>{count === 1 ? t("Remove this image?") : t("Remove {{count}} images?", { count })}</AlertDialogTitle>
          <AlertDialogDescription>
            {deleteFiles
              ? t("The files are deleted from the output folder. This cannot be undone.")
              : t("The images leave the gallery; their files stay in the output folder.")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <label className="flex items-center gap-2 text-sm">
          <Checkbox checked={deleteFiles} onCheckedChange={(checked) => setDeleteFiles(checked === true)} />
          {t("Also delete the files from disk")}
        </label>
        <AlertDialogFooter>
          <AlertDialogCancel ref={cancel}>{t("Cancel")}</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={() => ids && onConfirm(ids, deleteFiles)}>
            {deleteFiles ? t("Delete files") : t("Remove")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
