/** Settings section navigation: categories → sections (match counts, unsaved dots), then the system views. */
import { useLayoutEffect, useRef } from "react"
import { useTranslation } from "react-i18next"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import type { FilteredSection } from "./settings-model"
import { type SettingsView, SYSTEM_VIEWS } from "./views"

type Props = {
  sections: FilteredSection[]
  /** The settings schema has not arrived yet: placeholders, not "no matching sections". */
  loading?: boolean
  view: SettingsView
  activeSection: string | null
  filtering: boolean
  /** section id → unsaved edits in it */
  unsaved: ReadonlyMap<string, number>
  onSection: (id: string) => void
  onView: (view: SettingsView) => void
}

const itemClass = (active: boolean) =>
  cn(
    "flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left text-[0.8rem] outline-none transition-colors",
    "text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-3 focus-visible:ring-focus-ring",
    active && "bg-sidebar-accent font-medium text-sidebar-accent-foreground",
  )

/** Section rows per group of the loading placeholder, and the widths of their lines. */
const PLACEHOLDER_GROUPS = [4, 3, 5, 4, 2, 4, 3, 5, 4, 3]
const PLACEHOLDER_WIDTHS = ["w-28", "w-20", "w-32", "w-24", "w-36", "w-16"]

/**
 * Stands in for the sections while the schema loads: category titles over section rows, each as tall
 * as the real ones (a 0.8rem line, the row's py-1.5). Showing every setting (`fill`), there are always
 * more sections than a window holds: the placeholder is a window tall, so the System views under it
 * start out of sight, where the sections will push them, instead of being pushed off the nav as they
 * arrive. Filtered, how many sections come is not known: one group stands in.
 */
function SectionsPlaceholder({ fill }: { fill: boolean }) {
  const { t } = useTranslation()
  const groups = fill ? PLACEHOLDER_GROUPS : PLACEHOLDER_GROUPS.slice(0, 1)
  return (
    <div role="status" aria-busy="true" className={cn("flex flex-col gap-4", fill && "h-dvh overflow-hidden")}>
      <span className="sr-only">{t("Loading…")}</span>
      {groups.map((rows, group) => (
        <div key={group} aria-hidden className="flex flex-col gap-0.5">
          <div data-part="category" className="flex items-center px-2 pb-1 text-[0.8rem]">
            <Skeleton className="h-3 w-20" />
            {" "}
          </div>
          <div className="flex flex-col gap-px">
            {Array.from({ length: rows }, (_, row) => (
              <div key={row} data-part="section" className="flex items-center px-2 py-1.5 text-[0.8rem]">
                <Skeleton className={cn("h-3", PLACEHOLDER_WIDTHS[(group * 2 + row) % PLACEHOLDER_WIDTHS.length])} />
                {" "}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export function SettingsNav({ sections, loading = false, view, activeSection, filtering, unsaved, onSection, onView }: Props) {
  const { t } = useTranslation()
  const categories: { id: string; label: string; sections: FilteredSection[] }[] = []
  for (const section of sections) {
    const last = categories.at(-1)
    if (last && last.id === section.categoryId) last.sections.push(section)
    else categories.push({ id: section.categoryId, label: section.categoryLabel, sections: [section] })
  }
  // the current item in sight: a system view opened from a link sits below every section. A system
  // view's is in sight while the sections load too, and again before the frame they arrive in is
  // drawn (a layout effect): it stays where it was instead of going off the nav and coming back
  const navRef = useRef<HTMLElement>(null)
  const current = view === "settings" ? activeSection : view
  const shown = current !== null && (!loading || view !== "settings")
  useLayoutEffect(() => {
    if (shown) navRef.current?.querySelector("[aria-current]")?.scrollIntoView?.({ block: "nearest" })
  }, [current, shown, loading])
  return (
    <nav ref={navRef} aria-label={t("Settings sections")} className="flex flex-col gap-4 p-3">
      {categories.map((category) => (
        <div key={category.id} className="flex flex-col gap-0.5">
          <p className="px-2 pb-1 text-[0.8rem] font-semibold">{t(category.label)}</p>
          <ul className="flex flex-col gap-px">
            {category.sections.map((section) => {
              const active = view === "settings" && activeSection === section.id
              const edits = unsaved.get(section.id) ?? 0
              return (
                <li key={section.id}>
                  <button type="button" className={itemClass(active)} aria-current={active ? "location" : undefined} onClick={() => onSection(section.id)}>
                    <span className="min-w-0 flex-1 truncate">{t(section.label)}</span>
                    {edits > 0 ? (
                      <>
                        <span aria-hidden className="size-1.5 shrink-0 rounded-full bg-primary" />
                        <span className="sr-only">{t("{{count}} unsaved", { count: edits })}</span>
                      </>
                    ) : null}
                    {filtering ? <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{section.count}</span> : null}
                  </button>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
      {loading ? (
        <SectionsPlaceholder fill={!filtering} />
      ) : categories.length === 0 ? (
        <p className="px-2 text-sm text-muted-foreground">{t("No matching sections")}</p>
      ) : null}
      <div className="flex flex-col gap-0.5 border-t pt-3">
        <p className="px-2 pb-1 text-[0.8rem] font-semibold">{t("System")}</p>
        <ul className="flex flex-col gap-px">
          {SYSTEM_VIEWS.map(({ id, label, icon: Icon }) => (
            <li key={id}>
              <button type="button" className={itemClass(view === id)} aria-current={view === id ? "page" : undefined} onClick={() => onView(id)}>
                <Icon className="size-3.5 shrink-0" aria-hidden />
                <span className="truncate">{t(label)}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}
