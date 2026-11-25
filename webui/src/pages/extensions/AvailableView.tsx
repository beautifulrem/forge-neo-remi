/**
 * Available extensions: load an extension index (default: the third-party
 * Forge Neo extensions list), filter by text, tags and update date ("2025+",
 * on by default, as upstream), sort, and install. The list is
 * virtualized (indexes hold well over a thousand entries) inside the page's
 * own scroll (`data-scroll-root`), so there is one scrollbar.
 */
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query"
import { useVirtualizer } from "@tanstack/react-virtual"
import { Check, Download, ExternalLink, Loader2, PackageSearch, Search, Star } from "@/components/icons"
import { useLayoutEffect, useMemo, useRef, useState } from "react"
import { useRouteParams, useRouteSearch } from "@/app/route-params"
import type { TFunction } from "i18next"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import { LoadFailed } from "@/components/LoadFailed"
import { Segmented } from "@/components/Segmented"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Toggle } from "@/components/ui/toggle"
import { requestServerCommand } from "@/features/server-control/server-control"
import { cn } from "@/lib/utils"
import { FIELD_BUTTON_CLASS } from "@/components/ui/select-styles"
import {
  adminErrorMessage,
  DEFAULT_HIDDEN_TAGS,
  entryTags,
  type ExtensionIndex,
  filterIndex,
  formatDate,
  type IndexEntry,
  type IndexSort,
  RECENT_SINCE,
  SORT_LABELS,
  SORTS,
} from "./extensions-model"
import { extensionKeys, installedQuery } from "./queries"
import { safeUrl } from "@/ui/rich-text"

/**
 * A tag's description: the index's own as it is, except "installed", which the server adds
 * (`remi.extensions_admin`) and the SPA translates.
 */
function tagTitle(tag: string, tags: Record<string, string>, t: TFunction): string | undefined {
  return tag === "installed" ? t("extensions that are already installed") : tags[tag] || undefined
}

function EntryRow({ entry, tags, access, installing, onInstall }: { entry: IndexEntry; tags: Record<string, string>; access: boolean; installing: boolean; onInstall: () => void }) {
  const { t, i18n } = useTranslation()
  const updated = formatDate(entry.commit_time, i18n.language)
  return (
    <div className="flex gap-3 border-b px-1 py-3">
      <div className="min-w-0 flex-1">
        <a href={safeUrl(entry.url) ?? undefined} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 text-sm font-medium underline-offset-2 hover:underline">
          {entry.name}
          <ExternalLink className="size-3 text-muted-foreground" aria-hidden />
        </a>
        {entry.description ? <p title={entry.description} className="mt-0.5 line-clamp-2 text-[0.8rem] text-muted-foreground">{entry.description}</p> : null}
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
          {entryTags(entry).map((tag) => (
            <Badge key={tag} variant="outline" className="h-4.5 px-1.5 text-[0.6875rem] font-normal" title={tagTitle(tag, tags, t)}>
              {tag}
            </Badge>
          ))}
          {entry.stars ? (
            <span className="inline-flex items-center gap-0.5 tabular-nums">
              <Star className="size-3" aria-hidden />
              <span className="sr-only">{t("Stars")}</span>
              {entry.stars}
            </span>
          ) : null}
          {updated ? <span>{t("Updated {{date}}", { date: updated })}</span> : null}
          {entry.added ? <span>{t("Added {{date}}", { date: entry.added })}</span> : null}
        </div>
      </div>
      <div className="flex shrink-0 items-start pt-0.5">
        {entry.installed ? (
          <Badge variant="secondary" className="gap-1">
            <Check aria-hidden />
            {t("Installed")}
          </Badge>
        ) : (
          <Button size="sm" variant="outline" disabled={!access || installing} onClick={onInstall} aria-label={t("Install {{name}}", { name: entry.name })}>
            {installing ? <Loader2 className="animate-spin" aria-hidden /> : <Download aria-hidden />}
            {t("Install")}
          </Button>
        )}
      </div>
    </div>
  )
}

/** `onClearFilters`: back to a fresh tab's filters, offered when the user's own filters match nothing. */
function IndexList({ items, tags, access, onClearFilters }: { items: IndexEntry[]; tags: Record<string, string>; access: boolean; onClearFilters?: () => void }) {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t } = useTranslation()
  const list = useRef<HTMLDivElement>(null)
  const [scrollRoot, setScrollRoot] = useState<HTMLElement | null>(null)
  /** The list's offset in its scroll root's content (the controls above it). */
  const [margin, setMargin] = useState(0)
  const [installing, setInstalling] = useState<ReadonlySet<string>>(new Set())

  const empty = items.length === 0
  useLayoutEffect(() => {
    const element = list.current
    if (!element) return
    const root = element.closest<HTMLElement>("[data-scroll-root]") ?? (document.scrollingElement as HTMLElement | null) ?? document.documentElement
    setScrollRoot(root)
    const measure = () => setMargin(Math.max(0, Math.round(element.getBoundingClientRect().top - root.getBoundingClientRect().top + root.scrollTop)))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(root)
    if (root.firstElementChild) observer.observe(root.firstElementChild)
    return () => observer.disconnect()
  }, [empty])

  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => scrollRoot,
    estimateSize: () => 92,
    overscan: 8,
    scrollMargin: margin,
    getItemKey: (index) => items[index]?.url ?? index,
  })

  const install = async (entry: IndexEntry) => {
    setInstalling((s) => new Set(s).add(entry.url))
    try {
      const info = await api.extensions.install({ url: entry.url })
      // Mark it installed in every loaded index at once.
      queryClient.setQueriesData<ExtensionIndex>({ queryKey: ["extensions", "index"] }, (data) =>
        data ? { ...data, items: data.items.map((item) => (item.url === entry.url ? { ...item, installed: true } : item)) } : data,
      )
      void queryClient.invalidateQueries({ queryKey: extensionKeys.installed })
      toast.success(t("{{name}} is installed", { name: info.name }), {
        description: t("Reload the UI to load it."),
        action: { label: t("Reload UI"), onClick: () => requestServerCommand("reload") },
      })
    } catch (error) {
      toast.error(t("Could not install {{name}}", { name: entry.name }), { description: adminErrorMessage(error, t) })
    } finally {
      setInstalling((s) => {
        const next = new Set(s)
        next.delete(entry.url)
        return next
      })
    }
  }

  if (empty) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <PackageSearch aria-hidden />
          </EmptyMedia>
          <EmptyTitle>{t("Nothing matches")}</EmptyTitle>
          <EmptyDescription>{t("Change the search or the tag filters.")}</EmptyDescription>
        </EmptyHeader>
        {onClearFilters ? (
          <EmptyContent>
            <Button variant="outline" size="sm" onClick={onClearFilters}>
              {t("Clear filters")}
            </Button>
          </EmptyContent>
        ) : null}
      </Empty>
    )
  }
  return (
    <div>
      <div ref={list} role="list" aria-label={t("Available extensions")} className="relative" style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((row) => {
          const entry = items[row.index]
          if (!entry) return null
          return (
            <div
              key={row.key}
              ref={virtualizer.measureElement}
              data-index={row.index}
              role="listitem"
              aria-posinset={row.index + 1}
              aria-setsize={items.length}
              className="absolute inset-x-0 top-0"
              style={{ transform: `translateY(${row.start - margin}px)` }}
            >
              <EntryRow entry={entry} tags={tags} access={access} installing={installing.has(entry.url)} onInstall={() => void install(entry)} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

/** The filters in the address (`#/extensions?tab=available&q=lora&sort=stars&tags=ads`). */
const AVAILABLE_PARAMS = { q: "", sort: "newest", recent: "1", tags: [...DEFAULT_HIDDEN_TAGS].sort().join(","), tagmode: "hide", match: "any" }
/** What "Clear filters" resets besides the search. */
const FILTER_KEYS = ["recent", "tags", "tagmode", "match"] as const

/**
 * The index URL last loaded. Another tab and back remounts the view: it shows
 * that index again (cached, or refetched if it went stale) without a second
 * Load, while the query cache still knows it.
 */
let loadedUrl: string | null = null
const remembered = () => (loadedUrl !== null && queryClient.getQueryState(extensionKeys.index(loadedUrl)) ? loadedUrl : null)

export function AvailableView() {
  const { t } = useTranslation()
  const installed = useQuery(installedQuery)
  const defaultUrl = installed.data?.index_url ?? ""
  const access = installed.data?.access ?? false
  const [urlDraft, setUrlDraft] = useState<string | null>(remembered)
  const [url, setUrl] = useState<string | null>(remembered)
  const [params, setParams] = useRouteParams(AVAILABLE_PARAMS)
  const [query, setQuery] = useRouteSearch(params.q, (q) => setParams({ q }))
  const tagSet = useMemo<ReadonlySet<string>>(() => new Set(params.tags.split(",").filter(Boolean)), [params.tags])
  const tagMode = params.tagmode === "show" ? "show" : "hide"
  const match = params.match === "all" ? "all" : "any"
  const sort: IndexSort = (SORTS as readonly string[]).includes(params.sort) ? (params.sort as IndexSort) : "newest"
  const recentOnly = params.recent !== "0"
  const source = url ?? defaultUrl
  const index = useQuery({
    queryKey: extensionKeys.index(source),
    queryFn: ({ signal }) => api.extensions.index({ url: source }, signal),
    enabled: Boolean(source) && access && url !== null,
    staleTime: 10 * 60_000,
    placeholderData: keepPreviousData,
  })
  const reload = useMutation({
    mutationFn: () => api.extensions.index({ url: source, refresh: true }),
    onSuccess: (data) => queryClient.setQueryData(extensionKeys.index(source), data),
    onError: (error) => toast.error(t("Could not load the index"), { description: adminErrorMessage(error, t) }),
  })
  const items = useMemo(() => (index.data ? filterIndex(index.data.items, { query, tags: tagSet, tagMode, match, sort, recentOnly }) : []), [index.data, query, tagSet, tagMode, match, sort, recentOnly])
  const allTags = index.data ? Object.keys(index.data.tags).sort() : []
  // the filters a fresh tab starts with (the sort is an order, not a filter: it stays)
  const filtered = query.trim() !== "" || FILTER_KEYS.some((key) => params[key] !== AVAILABLE_PARAMS[key])
  const clearFilters = () => {
    setQuery("")
    setParams(Object.fromEntries(["q", ...FILTER_KEYS].map((key) => [key, undefined])))
  }

  const load = () => {
    const next = (urlDraft ?? defaultUrl).trim()
    loadedUrl = next
    if (next === source && url !== null) reload.mutate()
    else setUrl(next)
  }

  return (
    <div className="flex flex-col gap-3">
      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          load()
        }}
      >
        <label htmlFor="index-url" className="sr-only">
          {t("Extension index URL")}
        </label>
        <Input id="index-url" type="url" value={urlDraft ?? defaultUrl} onChange={(event) => setUrlDraft(event.target.value)} className="min-w-64 flex-1 font-mono text-xs" disabled={!access} />
        <Button type="submit" disabled={!access || index.isFetching || reload.isPending}>
          {index.isFetching || reload.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Download aria-hidden />}
          {url === null ? t("Load index") : t("Reload index")}
        </Button>
        <p className="basis-full text-xs text-muted-foreground">{t("This index is maintained by third parties. Please verify an extension before installing it.")}</p>
      </form>

      {installed.isPending ? (
        <Skeleton className="h-40 w-full" aria-busy="true" />
      ) : installed.isError && !installed.data ? (
        // without the extension list neither the index URL nor the access is known: say why the form is off
        <LoadFailed error={installed.error} message={adminErrorMessage(installed.error, t)} onRetry={() => void installed.refetch()} />
      ) : !access ? null : url === null ? (
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <PackageSearch aria-hidden />
            </EmptyMedia>
            <EmptyTitle>{t("Browse the extension index")}</EmptyTitle>
            <EmptyDescription>{t("Load the index to search extensions by name, description and tags.")}</EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : index.isError && !index.data ? (
        <LoadFailed error={index.error} message={adminErrorMessage(index.error, t)} onRetry={() => void index.refetch()} />
      ) : !index.data ? (
        <div className="flex flex-col gap-2" aria-busy="true">
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <InputGroup className="h-7 max-w-sm min-w-48 flex-1">
              <InputGroupAddon>
                <Search aria-hidden />
              </InputGroupAddon>
              <InputGroupInput className="h-full" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("Search extensions")} aria-label={t("Search extensions")} />
            </InputGroup>
            <Select<IndexSort> items={SORTS.map((value) => ({ value, label: t(SORT_LABELS[value]) }))} value={sort} onValueChange={(value) => value && setParams({ sort: value })}>
              <SelectTrigger size="sm" aria-label={t("Sort order")} className="min-w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SORTS.map((value) => (
                  <SelectItem key={value} value={value}>
                    {t(SORT_LABELS[value])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Toggle size="sm" pressed={recentOnly} onPressedChange={(pressed) => setParams({ recent: pressed ? "1" : "0" })} title={t("Only extensions updated after {{date}}", { date: RECENT_SINCE })} className={cn(FIELD_BUTTON_CLASS, "h-7 px-2 text-xs")}>
              {t("2025+")}
            </Toggle>
            <span className="ml-auto text-xs text-muted-foreground tabular-nums" role="status" aria-live="polite">
              {t("{{shown}} of {{count}} extensions", { shown: items.length, count: index.data.items.length })}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Segmented
              size="sm"
              aria-label={t("Tag filter")}
              options={[
                { value: "hide", label: t("Hide tagged") },
                { value: "show", label: t("Only tagged") },
              ]}
              value={tagMode}
              onValueChange={(tagmode) => setParams({ tagmode })}
            />
            <Segmented
              size="sm"
              aria-label={t("Tag matching")}
              options={[
                { value: "any", label: t("Any tag") },
                { value: "all", label: t("All tags") },
              ]}
              value={match}
              onValueChange={(next) => setParams({ match: next })}
            />
            <div role="group" aria-label={t("Tags")} className="flex flex-wrap gap-1">
              {allTags.map((tag) => (
                <Toggle
                  key={tag}
                  size="sm"
                  variant="outline"
                  pressed={tagSet.has(tag)}
                  onPressedChange={(pressed) => {
                    const next = new Set(tagSet)
                    if (pressed) next.add(tag)
                    else next.delete(tag)
                    setParams({ tags: [...next].sort().join(",") })
                  }}
                  title={tagTitle(tag, index.data.tags, t)}
                  className={cn("h-6 px-2 text-xs", tagSet.has(tag) && tagMode === "hide" && "line-through")}
                >
                  {tag}
                </Toggle>
              ))}
            </div>
          </div>
          <IndexList items={items} tags={index.data.tags} access={access} onClearFilters={filtered ? clearFilters : undefined} />
        </>
      )}
    </div>
  )
}
