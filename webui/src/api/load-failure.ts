/**
 * The failure a view shows for a query that has nothing to show (see `useLoadFailure`).
 */
import { useState } from "react"

/** The parts of a `useQuery` / `useInfiniteQuery` result a failed load is read from. */
type QueryState = {
  data: unknown
  error: unknown
  isError: boolean
  isFetching: boolean
  isPlaceholderData: boolean
  errorUpdatedAt: number
}

export type LoadFailure = {
  error: unknown
  /** A retry (the Retry button, a refetch on focus or reconnect) is on its way. */
  retrying: boolean
}

/**
 * Why a query has nothing to show, if it failed: its error, and whether a retry is on its way;
 * `null` while it loads or once it has data. A refetch of a query without data goes back to
 * "pending" and drops its error, which swapped the failure, and the Retry under the focus, for a
 * skeleton (or the previous filter's rows, a placeholder that is not this query's data) until
 * the retry failed again: the failure now stays, retrying, until the data arrives. A view that
 * mounts while such a retry runs has no failure of its own to show, and shows loading.
 */
export function useLoadFailure(query: QueryState): LoadFailure | null {
  const [seen, setSeen] = useState<{ error: unknown; at: number } | null>(null)
  if (query.data !== undefined && !query.isPlaceholderData) return null
  if (query.isError) {
    if (seen?.at !== query.errorUpdatedAt || seen.error !== query.error) setSeen({ error: query.error, at: query.errorUpdatedAt })
    return { error: query.error, retrying: false }
  }
  // the failure this view showed, while this query's retry runs (not another key's failure)
  if (query.isFetching && seen !== null && seen.at === query.errorUpdatedAt) return { error: seen.error, retrying: true }
  return null
}
