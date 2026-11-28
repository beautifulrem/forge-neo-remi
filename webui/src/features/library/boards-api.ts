/**
 * Thin typed API client for Gallery Boards (`/api/v2/boards`) and Asset
 * Lineage (`/api/v2/gallery/{file_id}/lineage`), backed by `@/api/endpoints`.
 */
import { queryOptions } from "@tanstack/react-query"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import type { Board, BoardCreate, BoardItemsResult, BoardList, BoardUpdate, LineageNode, LineageResponse } from "@/api/types"

export type { Board, LineageNode, LineageResponse }

export const boardKeys = {
  all: ["gallery", "boards"] as const,
  lineage: (fileId: string) => ["gallery", "lineage", fileId] as const,
}

export function refreshBoards() {
  void queryClient.invalidateQueries({ queryKey: boardKeys.all })
}

// ---------------------------------------------------------------- boards API

/** A server without this feature (404): no boards, no lineage. Anything else is an error. */
function absent(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}

export async function listBoards(signal?: AbortSignal): Promise<Board[]> {
  try {
    const data = (await api.boards.list(signal)) as BoardList | Board[] | undefined
    if (Array.isArray(data)) return data
    if (data && typeof data === "object" && Array.isArray(data.items)) return data.items
    return []
  } catch (error) {
    if (absent(error)) return []
    throw error // a failure is shown (and retried), not cached as "no boards"
  }
}

export async function createBoardApi(body: BoardCreate): Promise<Board> {
  return api.boards.create(body)
}

export async function updateBoardApi(boardId: string, body: BoardUpdate): Promise<Board> {
  return api.boards.update(boardId, body)
}

export async function deleteBoardApi(boardId: string): Promise<void> {
  await api.boards.delete(boardId)
}

export async function addBoardItemsApi(boardId: string, fileIds: readonly string[]): Promise<BoardItemsResult> {
  return api.boards.addItems(boardId, fileIds)
}

export async function removeBoardItemsApi(boardId: string, fileIds: readonly string[]): Promise<BoardItemsResult> {
  return api.boards.removeItems(boardId, fileIds)
}

// ---------------------------------------------------------------- lineage API

export async function fetchLineage(fileId: string, signal?: AbortSignal): Promise<LineageResponse | null> {
  try {
    const data = await api.gallery.lineage(fileId, signal)
    if (!data || typeof data !== "object" || !Array.isArray(data.ancestors) || !Array.isArray(data.descendants)) {
      return null
    }
    return {
      file_id: data.file_id ?? fileId,
      item: data.item ?? null,
      ancestors: data.ancestors,
      siblings: Array.isArray(data.siblings) ? data.siblings : [],
      descendants: data.descendants,
    }
  } catch (error) {
    if (absent(error)) return null
    throw error
  }
}

export const boardsQuery = queryOptions({
  queryKey: boardKeys.all,
  queryFn: ({ signal }) => listBoards(signal),
  staleTime: 30_000,
})

export const lineageQuery = (fileId: string) =>
  queryOptions({
    queryKey: boardKeys.lineage(fileId),
    queryFn: ({ signal }) => fetchLineage(fileId, signal),
    staleTime: 30_000,
  })
