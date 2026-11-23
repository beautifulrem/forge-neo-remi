/** Gallery filters: search, form / folder, model, date range, stars, sort, layout and tile size. */
import { useQuery } from "@tanstack/react-query"
import { CalendarRange, FilterX, Search, SlidersHorizontal, Star } from "@/components/icons"
import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import type { GalleryCategory, GallerySort } from "@/api/types"
import { FadeStrip } from "@/components/FadeStrip"
import { SearchSelect } from "@/components/SearchSelect"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useFittingPlaceholder } from "@/lib/use-fitting-placeholder"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Toggle } from "@/components/ui/toggle"
import { FIELD_BUTTON_CLASS } from "@/components/ui/select-styles"
import { cn } from "@/lib/utils"
import { activeFilters, useGalleryView } from "./filters"
import { CATEGORY_LABELS, daysAgo } from "./labels"
import { galleryFacetsQuery } from "./queries"
import { isComposing } from "@/lib/ime"

const ALL = "__all__"

const SORT_LABELS: Record<GallerySort, string> = {
  newest: "Newest first",
  oldest: "Oldest first",
  name: "File name",
  largest: "Largest first",
}

function PlainSelect<V extends string>(props: { value: V; onChange: (value: V) => void; options: { value: V; label: string }[]; label: string; className?: string }) {
  return (
    <Select<V> items={props.options} value={props.value} onValueChange={(next) => next !== null && props.onChange(next)}>
      <SelectTrigger size="sm" aria-label={props.label} className={props.className}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {props.options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** The search box commits 300 ms after typing stops (or on Enter). */
function SearchBox() {
  const { t } = useTranslation()
  const q = useGalleryView((s) => s.q)
  const set = useGalleryView((s) => s.set)
  const [text, setText] = useState(q)
  const [seen, setSeen] = useState(q)
  if (q !== seen) {
    setSeen(q)
    setText(q)
  }
  useEffect(() => {
    if (text === q) return
    const timer = setTimeout(() => set({ q: text }), 300)
    return () => clearTimeout(timer)
  }, [text, q, set])
  const input = useRef<HTMLInputElement>(null)
  const placeholder = useFittingPlaceholder(input, [t("Search prompts, parameters, file names…"), t("Search…")])
  return (
    <div className="relative min-w-40 flex-1 @6xl/gallerybar:max-w-80">
      <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
      <Input
        type="search"
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => event.key === "Enter" && !isComposing(event) && set({ q: text })}
        ref={input}
        placeholder={placeholder}
        aria-label={t("Search the gallery")}
        className="h-7 pl-7 text-[0.8rem]"
      />
    </div>
  )
}

function DateRange() {
  const { t, i18n } = useTranslation()
  const from = useGalleryView((s) => s.from)
  const to = useGalleryView((s) => s.to)
  const set = useGalleryView((s) => s.set)
  const format = (date: string) => new Date(`${date}T00:00:00`).toLocaleDateString(i18n.language, { month: "short", day: "numeric" })
  const summary = from && to ? `${format(from)} – ${format(to)}` : from ? t("Since {{date}}", { date: format(from) }) : to ? t("Until {{date}}", { date: format(to) }) : t("Any date")
  return (
    <Popover>
      <PopoverTrigger render={<Button variant="ghost" size="sm" aria-label={t("Date range")} className={cn(FIELD_BUTTON_CLASS, "font-normal", (from || to) && "text-primary")} />}>
        <CalendarRange aria-hidden />
        <span title={summary} className="max-w-36 truncate">{summary}</span>
      </PopoverTrigger>
      <PopoverContent aria-label={t("Date range")} align="start" className="w-72">
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1 text-xs font-medium">
              {t("From")}
              <Input type="date" value={from ?? ""} max={to ?? undefined} onChange={(event) => set({ from: event.target.value || null })} className="h-8" />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium">
              {t("To")}
              <Input type="date" value={to ?? ""} min={from ?? undefined} onChange={(event) => set({ to: event.target.value || null })} className="h-8" />
            </label>
          </div>
          <div className="flex flex-wrap gap-1">
            <Button variant="secondary" size="xs" onClick={() => set({ from: daysAgo(0), to: daysAgo(0) })}>
              {t("Today")}
            </Button>
            <Button variant="secondary" size="xs" onClick={() => set({ from: daysAgo(6), to: null })}>
              {t("Last 7 days")}
            </Button>
            <Button variant="secondary" size="xs" onClick={() => set({ from: daysAgo(29), to: null })}>
              {t("Last 30 days")}
            </Button>
            <Button variant="ghost" size="xs" onClick={() => set({ from: null, to: null })} disabled={!from && !to}>
              {t("Any date")}
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

/** Sort, layout and tile size: in the toolbar's row when it is wide, stacked in the "View options" popover when not. */
function ViewControls({ stacked = false }: { stacked?: boolean }) {
  const { t } = useTranslation()
  const view = useGalleryView()
  const controls = [
    {
      label: t("Sort"),
      control: (
        <PlainSelect
          label={t("Sort")}
          value={view.sort}
          onChange={(sort) => view.set({ sort })}
          options={(Object.keys(SORT_LABELS) as GallerySort[]).map((value) => ({ value, label: t(SORT_LABELS[value]) }))}
          className={stacked ? "w-full" : undefined}
        />
      ),
    },
    {
      label: t("Layout"),
      control: (
        <Segmented
          size="sm"
          aria-label={t("Layout")}
          value={view.layout}
          onValueChange={(layout) => view.set({ layout })}
          options={[
            { value: "justified", label: t("Rows") },
            { value: "masonry", label: t("Columns") },
          ]}
        />
      ),
    },
    {
      label: t("Tile size"),
      control: (
        <Segmented
          size="sm"
          aria-label={t("Tile size")}
          value={view.size}
          onValueChange={(size) => view.set({ size })}
          options={[
            { value: "s", label: t("S") },
            { value: "m", label: t("M") },
            { value: "l", label: t("L") },
          ]}
        />
      ),
    },
  ]
  if (!stacked) return <>{controls.map((entry) => <span key={entry.label} className="contents">{entry.control}</span>)}</>
  return (
    <div className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-2">
      {controls.map((entry) => (
        <div key={entry.label} className="contents">
          <span className="text-xs text-muted-foreground">{entry.label}</span>
          <div className="flex justify-end">{entry.control}</div>
        </div>
      ))}
    </div>
  )
}

/** Single-row gallery filter & view-options toolbar (image count folds into `BoardsBar`). */
export function GalleryToolbar() {
  const { t } = useTranslation()
  const view = useGalleryView()
  const facets = useQuery(galleryFacetsQuery)
  const categories = facets.data?.categories ?? []
  const models = facets.data?.models ?? []
  const filtered = activeFilters(view)

  return (
    // Wide: one row, the view controls at its end. Narrow (a phone, a tablet, the details pane open): the
    // search and a "View options" button on the first row, the filters on a second that scrolls sideways,
    // never three ragged rows.
    <div className="@container/gallerybar flex flex-col gap-1.5 border-b border-border/70 bg-background px-3 py-1.5">
      <div className="flex flex-col gap-1.5 @6xl/gallerybar:flex-row @6xl/gallerybar:items-center">
        <div className="flex items-center gap-1.5 @6xl/gallerybar:contents">
          <SearchBox />
          <Popover>
            <PopoverTrigger
              render={<Button variant="ghost" size="icon-sm" aria-label={t("View options")} title={t("View options")} className="@6xl/gallerybar:hidden" />}
            >
              <SlidersHorizontal aria-hidden />
            </PopoverTrigger>
            <PopoverContent aria-label={t("View options")} align="end" className="w-auto">
              <ViewControls stacked />
            </PopoverContent>
          </Popover>
        </div>
        <FadeStrip
          role="group"
          aria-label={t("Filters")}
          className="-mx-3 flex items-center gap-1.5 px-3 *:shrink-0 @6xl/gallerybar:mx-0 @6xl/gallerybar:flex-1 @6xl/gallerybar:overflow-visible @6xl/gallerybar:px-0"
        >
          <PlainSelect
            label={t("Form or folder")}
            value={view.category ?? ALL}
            onChange={(value) => view.set({ category: value === ALL ? null : (value as GalleryCategory) })}
            options={[
              { value: ALL, label: t("All images") },
              ...categories.filter((entry) => entry.count > 0 || entry.name === view.category).map((entry) => ({ value: entry.name, label: `${t(CATEGORY_LABELS[entry.name as GalleryCategory] ?? entry.name)} · ${entry.count}` })),
            ]}
          />
          <SearchSelect
            size="sm"
            aria-label={t("Model")}
            className="w-44"
            value={view.model ?? ALL}
            onValueChange={(value) => view.set({ model: value === ALL ? null : value })}
            options={[{ value: ALL, label: t("All models") }, ...models.map((model) => ({ value: model.name, label: model.name, hint: String(model.count) }))]}
          />
          <DateRange />
          <Toggle
            size="sm"
            className={FIELD_BUTTON_CLASS}
            pressed={view.starred}
            onPressedChange={(starred) => view.set({ starred })}
            aria-label={t("Starred only")}
            title={t("Starred only")}
          >
            <Star className={cn(view.starred && "fill-star text-star")} aria-hidden />
          </Toggle>
          {filtered > 0 ? (
            <Button variant="ghost" size="sm" onClick={view.reset}>
              <FilterX aria-hidden />
              {t("Clear filters")}
            </Button>
          ) : null}
          <div className="ml-auto hidden items-center gap-1.5 @6xl/gallerybar:flex">
            <ViewControls />
          </div>
        </FadeStrip>
      </div>
    </div>
  )
}
