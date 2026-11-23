/**
 * Thin typed API client for Gallery Boards (`/api/v2/boards`) and Asset
 * Lineage (`/api/v2/gallery/{file_id}/lineage`), backed by `workbenchRequest`
 * (which automatically routes `VITE_MOCK=1` through `@/api/mock/workbench`).
 */
import { queryOptions } from "@tanstack/react-query"
import { queryClient } from "@/api/queries"
import type { GalleryItem, Page } from "@/api/types"
import { workbenchRequest } from "@/features/pipelines/workbench-http"

export const boardKeys = {
  all: ["gallery", "boards"] as const,
  lineage: (fileId: string) => ["gallery", "lineage", fileId] as const,
}

export function refreshBoards() {
  void queryClient.invalidateQueries({ queryKey: boardKeys.all })
}

export type Board = {
  id: string
  name: string
  description: string
  color?: string | null
  item_count: number
  cover_file_id?: string | null
  created_at: string
  updated_at: string
}

export type LineageNode = {
  file_id: string
  relation: string
  depth: number
  job_id?: string | null
  chain_id?: string | null
  stage_index?: number | null
  name?: string | null
  width?: number | null
  height?: number | null
  tab?: string | null
  model?: string | null
  created_at?: string | null
  exists?: boolean
}

export type LineageResponse = {
  file_id: string
  item?: LineageNode | null
  ancestors: LineageNode[]
  siblings: LineageNode[]
  descendants: LineageNode[]
}

// ---------------------------------------------------------------- boards API

export async function listBoards(signal?: AbortSignal): Promise<Board[]> {
  try {
    const data = await workbenchRequest<Board[] | { items?: Board[] }>("/boards", { signal })
    if (Array.isArray(data)) return data
    if (data && typeof data === "object" && Array.isArray(data.items)) return data.items
    return []
  } catch {
    return []
  }
}

export async function createBoardApi(body: { name: string; description?: string; color?: string | null }): Promise<Board> {
  return workbenchRequest<Board>("/boards", { method: "POST", body })
}

export async function updateBoardApi(boardId: string, body: { name?: string; description?: string; color?: string | null }): Promise<Board> {
  return workbenchRequest<Board>(`/boards/${encodeURIComponent(boardId)}`, { method: "PATCH", body })
}

export async function deleteBoardApi(boardId: string): Promise<void> {
  await workbenchRequest<void>(`/boards/${encodeURIComponent(boardId)}`, { method: "DELETE" })
}

export async function addBoardItemsApi(boardId: string, fileIds: readonly string[]): Promise<{ board_id: string; count: number }> {
  return workbenchRequest<{ board_id: string; count: number }>(`/boards/${encodeURIComponent(boardId)}/items`, {
    method: "POST",
    body: { file_ids: [...fileIds] },
  })
}

export async function removeBoardItemsApi(boardId: string, fileIds: readonly string[]): Promise<{ board_id: string; count: number }> {
  return workbenchRequest<{ board_id: string; count: number }>(`/boards/${encodeURIComponent(boardId)}/items/remove`, {
    method: "POST",
    body: { file_ids: [...fileIds] },
  })
}

/** Fetches the gallery page filtered by `board_id` (plus standard gallery query params). */
export async function listBoardGalleryPage(
  query: Record<string, string | number | boolean | null | undefined>,
  signal?: AbortSignal,
): Promise<Page<GalleryItem>> {
  const data = await workbenchRequest<Page<GalleryItem>>("/gallery", { query, signal })
  return {
    items: Array.isArray(data?.items) ? data.items : [],
    next_cursor: data?.next_cursor ?? null,
    total: data?.total ?? null,
  }
}

// ---------------------------------------------------------------- lineage API

export async function fetchLineage(fileId: string, signal?: AbortSignal): Promise<LineageResponse | null> {
  try {
    const data = await workbenchRequest<LineageResponse>(`/gallery/${encodeURIComponent(fileId)}/lineage`, { signal })
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
  } catch {
    return null
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
