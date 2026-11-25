/**
 * Gallery Boards architectural tab-strip & assignment popover:
 * sleek 32px hairline tab-strip with 2px `--primary` active underline,
 * right-folded image count readout (reclaiming a full vertical row above
 * the contact sheet), and ImageStitch bulk-send.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { Check, Folder, FolderPlus, Images, MoreHorizontal, Pencil, Plus, Trash2, X } from "@/components/icons"
import { type ReactNode, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { FadeStrip } from "@/components/FadeStrip"
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
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { LoadFailed } from "@/components/LoadFailed"
import { errorText } from "@/features/errors/error-text"
import { sendToReferenceWithFeedback } from "@/features/send-to/result-actions"
import type { SendItem } from "@/features/send-to/send-to"
import { cn } from "@/lib/utils"
import { addBoardItemsApi, type Board, boardsQuery, createBoardApi, deleteBoardApi, refreshBoards, removeBoardItemsApi, updateBoardApi } from "./boards-api"
import { refreshGallery } from "./cache"
import { activeFilters, useGalleryView } from "./filters"
import { galleryFacetsQuery } from "./queries"

type BoardsBarProps = {
  /** Currently visible items (for bulk-sending the active board to ImageStitch). */
  activeItems: readonly SendItem[]
  /** Loaded item count for the inline gallery status readout. */
  count?: number
  /** Total matching images for the inline gallery status readout (`null` while loading). */
  total?: number | null
  /** Suppresses the count readout when the gallery failed to load. */
  failed?: boolean
}

export function BoardsBar({ activeItems, count, total, failed = false }: BoardsBarProps) {
  const { t } = useTranslation()
  const view = useGalleryView()
  const boardId = view.boardId
  const setView = view.set
  const boardsList = useQuery(boardsQuery)
  const boards = boardsList.data ?? []
  const facets = useQuery({ ...galleryFacetsQuery, enabled: count !== undefined })

  const [createOpen, setCreateOpen] = useState(false)
  const [newName, setNewName] = useState("")
  const [renaming, setRenaming] = useState<Board | null>(null)
  const [renameText, setRenameText] = useState("")
  // the name fields are focused through their popup (`initialFocus`), not `autoFocus`, which made the
  // field the place to give the focus back to on close
  const newNameRef = useRef<HTMLInputElement>(null)
  const renameRef = useRef<HTMLInputElement>(null)
  const [deleting, setDeleting] = useState<Board | null>(null)

  const activeBoard = boardId ? boards.find((board) => board.id === boardId) ?? null : null
  // A remembered board that is gone (deleted in another tab or on another device) selects All
  // boards, not nothing. Only a settled list decides: while it refetches (a board just created)
  // it may not have the new board yet.
  const boardGone = Boolean(boardId) && boardsList.isSuccess && !boardsList.isFetching && activeBoard === null
  useEffect(() => {
    if (boardGone) setView({ boardId: null })
  }, [boardGone, setView])
  const filtered = activeFilters(view)
  const showCount = count !== undefined && !failed

  const create = useMutation({
    mutationFn: (name: string) => createBoardApi({ name }),
    onSuccess: (board) => {
      refreshBoards()
      setNewName("")
      setCreateOpen(false)
      setView({ boardId: board.id })
      toast.success(t("Board created"))
    },
    onError: (error) => toast.error(t("Could not create board"), { description: errorText(error) }),
  })

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => updateBoardApi(id, { name }),
    onSuccess: () => {
      refreshBoards()
      setRenaming(null)
      toast.success(t("Board renamed"))
    },
    onError: (error) => toast.error(t("Could not rename board"), { description: errorText(error) }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => deleteBoardApi(id),
    onSuccess: (_data, id) => {
      if (boardId === id) setView({ boardId: null })
      refreshBoards()
      refreshGallery()
      setDeleting(null)
      toast.success(t("Board deleted"))
    },
    onError: (error) => toast.error(t("Could not delete board"), { description: errorText(error) }),
  })

  const submitCreate = () => {
    const trimmed = newName.trim()
    if (!trimmed || create.isPending) return
    create.mutate(trimmed)
  }

  const submitRename = () => {
    const trimmed = renameText.trim()
    if (!renaming || !trimmed || rename.isPending) return
    rename.mutate({ id: renaming.id, name: trimmed })
  }

  return (
    <div className="flex h-8 items-center gap-2 border-b border-border bg-background px-3">
      <FadeStrip
        role="tablist"
        aria-label={t("Boards")}
        className="-mx-1 flex min-w-0 flex-1 items-stretch gap-1 px-1 *:shrink-0"
      >
        <button
          type="button"
          role="tab"
          aria-selected={boardId === null}
          onClick={() => setView({ boardId: null })}
          className={cn(
            "relative inline-flex h-8 items-center gap-1.5 px-2 text-xs transition-colors outline-none focus-visible:ring-2 focus-visible:ring-focus-ring",
            boardId === null
              ? "font-medium text-foreground after:absolute after:inset-x-0.5 after:bottom-0 after:h-0.5 after:bg-primary"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {t("All boards")}
        </button>

        {boards.map((board) => {
          const selected = boardId === board.id
          return (
            <button
              key={board.id}
              type="button"
              role="tab"
              aria-selected={selected}
              title={board.description || board.name}
              onClick={() => setView({ boardId: selected ? null : board.id })}
              className={cn(
                "relative inline-flex h-8 max-w-48 items-center gap-1.5 px-2 text-xs transition-colors outline-none focus-visible:ring-2 focus-visible:ring-focus-ring",
                selected
                  ? "font-medium text-foreground after:absolute after:inset-x-0.5 after:bottom-0 after:h-0.5 after:bg-primary"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <span title={board.name} className="truncate">{board.name}</span>
              <span className="font-mono text-[0.6875rem] text-muted-foreground tabular-nums">{board.item_count}</span>
            </button>
          )
        })}
        {boardsList.isError && !boardsList.data ? (
          <button
            type="button"
            title={errorText(boardsList.error)}
            onClick={() => void boardsList.refetch()}
            className="inline-flex h-8 items-center gap-1 px-2 text-xs text-destructive outline-none hover:underline focus-visible:ring-2 focus-visible:ring-focus-ring"
          >
            {t("Could not load boards. Retry")}
          </button>
        ) : null}
      </FadeStrip>

      <div className="flex shrink-0 items-center gap-1.5">
        {showCount || view.q ? (
          <p className="hidden items-center gap-1.5 font-mono text-xs text-muted-foreground tabular-nums sm:flex" aria-live="polite">
            {showCount ? (
              <>
                {total === null || total === undefined
                  ? t("Loading…")
                  : filtered > 0
                    ? t("{{count}} of {{total}} images match", { count: total, total: facets.data?.total ?? total })
                    : t("Images: {{count}}", { count: total })}
                {total != null && count < total ? <span className="hidden lg:inline">· {t("{{count}} loaded", { count })}</span> : null}
              </>
            ) : null}
            {view.q ? (
              <button type="button" className="inline-flex items-center gap-0.5 rounded-sm px-1 font-sans hover:bg-muted" onClick={() => setView({ q: "" })}>
                “{view.q}” <X className="size-3" aria-label={t("Clear search")} />
              </button>
            ) : null}
          </p>
        ) : null}

        {showCount || view.q ? <span className="hidden h-3.5 w-px bg-border sm:block" aria-hidden /> : null}

        <Popover open={createOpen} onOpenChange={setCreateOpen}>
          <PopoverTrigger
            render={
              <Button
                variant="ghost"
                size="sm"
                aria-label={t("New board")}
                title={t("New board")}
                className="h-7 gap-1 px-2 text-xs text-muted-foreground hover:text-foreground"
              >
                <Plus className="size-3.5" aria-hidden />
                <span className="hidden sm:inline">{t("New board")}</span>
              </Button>
            }
          />
          <PopoverContent aria-label={t("New board")} align="end" className="w-64 p-3" initialFocus={newNameRef}>
            <form
              className="flex flex-col gap-2"
              onSubmit={(event) => {
                event.preventDefault()
                submitCreate()
              }}
            >
              <label className="flex flex-col gap-1 text-xs font-medium">
                {t("Board name")}
                <Input
                  value={newName}
                  onChange={(event) => setNewName(event.target.value)}
                  placeholder={t("Board name")}
                  className="h-7 text-[0.8rem]"
                  ref={newNameRef}
                />
              </label>
              <div className="flex justify-end gap-1.5">
                <Button type="button" variant="ghost" size="sm" className="h-7" onClick={() => setCreateOpen(false)}>
                  {t("Cancel")}
                </Button>
                <Button type="submit" size="sm" className="h-7" disabled={!newName.trim() || create.isPending}>
                  {t("Create board")}
                </Button>
              </div>
            </form>
          </PopoverContent>
        </Popover>

        {activeBoard ? (
          <>
            <Button
              variant="outline"
              size="sm"
              className="h-7 gap-1.5 px-2 text-xs"
              disabled={activeItems.length === 0}
              aria-label={t("Send board to Reference Images")}
              title={t("Send board to Reference Images")}
              onClick={() => sendToReferenceWithFeedback(activeItems)}
            >
              <Images className="size-3.5" aria-hidden />
              <span className="hidden md:inline">{t("Send board to Reference Images")}</span>
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger
                render={<Button variant="ghost" size="icon-sm" className="size-7" aria-label={t("Board actions")} />}
              >
                <MoreHorizontal />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-44">
                <DropdownMenuItem
                  onClick={() => {
                    setRenaming(activeBoard)
                    setRenameText(activeBoard.name)
                  }}
                >
                  <Pencil />
                  {t("Rename board")}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem variant="destructive" onClick={() => setDeleting(activeBoard)}>
                  <Trash2 />
                  {t("Delete board")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        ) : null}
      </div>

      <AlertDialog open={renaming !== null} onOpenChange={(open) => !open && setRenaming(null)}>
        <AlertDialogContent initialFocus={renameRef}>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Rename board")}</AlertDialogTitle>
          </AlertDialogHeader>
          <form
            onSubmit={(event) => {
              event.preventDefault()
              submitRename()
            }}
            className="flex flex-col gap-3"
          >
            <Input
              value={renameText}
              onChange={(event) => setRenameText(event.target.value)}
              aria-label={t("Board name")}
              ref={renameRef}
            />
            <AlertDialogFooter>
              <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
              <AlertDialogAction type="submit" disabled={!renameText.trim() || rename.isPending}>
                {t("Save")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </form>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={deleting !== null} onOpenChange={(open) => !open && setDeleting(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Delete board “{{name}}”?", { name: deleting?.name ?? "" })}</AlertDialogTitle>
            <AlertDialogDescription>{t("Images stay in the gallery; only the board grouping is removed.")}</AlertDialogDescription>
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

type AssignBoardPopoverProps = {
  fileIds: readonly string[]
  /** If currently filtered to a board, also offer removing from it. */
  activeBoardId?: string | null
  trigger?: ReactNode
  tone?: "light" | "default"
}

/** Popover to assign one or more gallery images to an existing or newly created Board. */
export function AssignBoardPopover({ fileIds, activeBoardId, trigger, tone = "default" }: AssignBoardPopoverProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [quickName, setQuickName] = useState("")
  const boardsList = useQuery(boardsQuery)
  const boards = boardsList.data ?? []

  const assign = useMutation({
    mutationFn: async (board: Board) => {
      await addBoardItemsApi(board.id, fileIds)
      return board
    },
    onSuccess: (board) => {
      refreshBoards()
      refreshGallery()
      setOpen(false)
      toast.success(t("Added {{count}} images to “{{name}}”", { count: fileIds.length, name: board.name }))
    },
    onError: (error) => toast.error(t("Could not update board items"), { description: errorText(error) }),
  })

  const unassign = useMutation({
    mutationFn: async (board: Board) => {
      await removeBoardItemsApi(board.id, fileIds)
      return board
    },
    onSuccess: (board) => {
      refreshBoards()
      refreshGallery()
      setOpen(false)
      toast.success(t("Removed from “{{name}}”", { name: board.name }))
    },
    onError: (error) => toast.error(t("Could not update board items"), { description: errorText(error) }),
  })

  const createAndAssign = useMutation({
    mutationFn: async (name: string) => {
      const board = await createBoardApi({ name })
      await addBoardItemsApi(board.id, fileIds)
      return board
    },
    onSuccess: (board) => {
      refreshBoards()
      refreshGallery()
      setQuickName("")
      setOpen(false)
      toast.success(t("Added {{count}} images to “{{name}}”", { count: fileIds.length, name: board.name }))
    },
    onError: (error) => toast.error(t("Could not create board"), { description: errorText(error) }),
  })

  const activeBoard = activeBoardId ? boards.find((b) => b.id === activeBoardId) ?? null : null

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          (trigger as React.ReactElement | undefined) ?? (
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={t("Add to board")}
              title={t("Add to board")}
              className={cn(tone === "light" && "text-white/85 hover:bg-white/12 hover:text-white")}
            >
              <FolderPlus />
            </Button>
          )
        }
      />
      <PopoverContent aria-label={t("Add to board")} align="center" className="w-64 p-2.5">
        <div className="flex flex-col gap-2">
          <p className="px-1 text-xs font-semibold">{t("Add to board")}</p>
          {boards.length > 0 ? (
            <ul className="-mx-1 flex max-h-48 flex-col gap-0.5 overflow-y-auto px-1">
              {boards.map((board) => (
                <li key={board.id}>
                  <button
                    type="button"
                    disabled={assign.isPending}
                    onClick={() => assign.mutate(board)}
                    className="flex h-7 w-full items-center gap-2 rounded-sm px-2 text-left text-xs transition-colors outline-none hover:bg-muted focus-visible:ring-2 focus-visible:ring-focus-ring"
                  >
                    <Folder className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                    <span title={board.name} className="min-w-0 flex-1 truncate">{board.name}</span>
                    <span className="font-mono text-[0.6875rem] text-muted-foreground tabular-nums">{board.item_count}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : boardsList.isError && !boardsList.data ? (
            <LoadFailed error={boardsList.error} onRetry={() => void boardsList.refetch()} className="px-1 py-1 text-xs" />
          ) : boardsList.isPending ? (
            // the list is on its way, not empty
            <p role="status" className="px-1 text-xs text-muted-foreground">{t("Loading…")}</p>
          ) : (
            <p className="px-1 text-xs text-muted-foreground">{t("No boards yet")}</p>
          )}
          {activeBoard ? (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 justify-start px-2 text-xs text-destructive hover:text-destructive"
              disabled={unassign.isPending}
              onClick={() => unassign.mutate(activeBoard)}
            >
              <Trash2 className="size-3.5" aria-hidden />
              {t("Remove from “{{name}}”", { name: activeBoard.name })}
            </Button>
          ) : null}
          <form
            className="flex items-center gap-1 border-t border-border/70 pt-2"
            onSubmit={(event) => {
              event.preventDefault()
              const name = quickName.trim()
              if (name && !createAndAssign.isPending) createAndAssign.mutate(name)
            }}
          >
            <Input
              value={quickName}
              onChange={(event) => setQuickName(event.target.value)}
              placeholder={t("New board")}
              aria-label={t("New board")}
              className="h-7 flex-1 text-xs"
            />
            <Button type="submit" size="icon-sm" className="h-7 w-7 shrink-0" disabled={!quickName.trim() || createAndAssign.isPending} aria-label={t("Create board")}>
              <Check className="size-3.5" aria-hidden />
            </Button>
          </form>
        </div>
      </PopoverContent>
    </Popover>
  )
}
