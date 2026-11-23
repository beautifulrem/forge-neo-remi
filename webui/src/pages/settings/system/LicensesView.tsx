/**
 * Settings → Licenses: third-party code (from `html/licenses.html`, served
 * as data — no server HTML is rendered). A license text is fetched from its
 * upstream repository when its entry is opened, as plain text.
 */
import { useQuery } from "@tanstack/react-query"
import { ChevronRight, ExternalLink } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { ROOT } from "@/api/http"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import { panelClass, panelTitleClass } from "@/ui/layout/panel"

type Entry = { id: string; name: string; url?: string | null; description?: string; license_url?: string | null }

async function fetchLicense(url: string, signal: AbortSignal): Promise<string> {
  const response = await fetch(url, { signal, credentials: "omit", referrerPolicy: "no-referrer" })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.text()
}

function LicenseText({ url }: { url: string }) {
  const { t } = useTranslation()
  const text = useQuery({ queryKey: ["license", url], queryFn: ({ signal }) => fetchLicense(url, signal), staleTime: Infinity, retry: 1 })
  if (text.isPending) return <Skeleton className="mt-1 h-24" />
  if (text.isError) {
    return (
      <p className="pt-1 text-sm text-muted-foreground">
        {t("Could not load the license text.")}{" "}
        <a href={url} target="_blank" rel="noreferrer noopener" className="text-primary underline-offset-2 hover:underline">
          {t("Open it on the web")}
        </a>
      </p>
    )
  }
  // a flat fill, no edge: the entry above it is a panel, not a card
  return <pre className="mt-1 max-h-96 overflow-auto rounded-md bg-muted/50 px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap">{text.data}</pre>
}

function LicenseEntry({ entry, depth }: { entry: Entry; depth: number }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    // a panel (`ui/layout/panel.ts`), a level down when its group has a titled panel of its own
    <li className={cn("flex flex-col gap-1", panelClass(depth))}>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className={panelTitleClass(depth)}>
            {entry.url ? (
              <a href={entry.url} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
                {entry.name}
                <ExternalLink className="size-3 text-muted-foreground" aria-hidden />
              </a>
            ) : (
              entry.name
            )}
          </p>
          {entry.description ? <p className="text-xs text-muted-foreground">{t(entry.description)}</p> : null}
        </div>
      </div>
      {entry.license_url ? (
        <Collapsible open={open} onOpenChange={setOpen}>
          <CollapsibleTrigger className="relative flex w-fit items-center gap-1.5 rounded-sm py-0.5 text-left after:absolute after:-inset-x-2 after:-inset-y-2.5 text-xs text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50">
            <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} aria-hidden />
            {open ? t("Hide license") : t("Show license")}
          </CollapsibleTrigger>
          <CollapsibleContent>{open ? <LicenseText url={entry.license_url} /> : null}</CollapsibleContent>
        </Collapsible>
      ) : null}
    </li>
  )
}

export function LicensesView() {
  const { t } = useTranslation()
  const licenses = useQuery({ queryKey: ["licenses"], queryFn: ({ signal }) => api.system.licenses(signal), staleTime: Infinity })
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6 sm:px-6">
      <header>
        <h1 className="font-heading text-xl font-semibold tracking-tight">{t("Licenses")}</h1>
        <p className="text-sm text-muted-foreground">{t("Code from these projects is used or adapted in this WebUI.")}</p>
        {/* written by the build (webui/tools/notices.mjs): every package bundled into the web interface */}
        <p className="mt-1 text-sm">
          <a href={`${ROOT}/THIRD-PARTY-NOTICES.txt`} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
            {t("Web interface libraries and their licenses")}
            <ExternalLink className="size-3 text-muted-foreground" aria-hidden />
          </a>
        </p>
      </header>
      {licenses.isPending ? (
        <div className="flex flex-col gap-2" aria-busy="true">
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} className="h-14 w-full" />
          ))}
        </div>
      ) : licenses.isError ? (
        <p role="alert" className="text-sm text-destructive">
          {licenses.error.message}
        </p>
      ) : (
        licenses.data.groups.map((group) => (
          <section key={group.title} aria-label={t(group.title)} className={cn("flex flex-col gap-2.5", group.title && panelClass(0))}>
            {group.title ? <h2 className={panelTitleClass(0)}>{t(group.title)}</h2> : null}
            <ul className="flex flex-col gap-2.5">
              {group.items.map((entry) => (
                <LicenseEntry key={entry.id} entry={entry} depth={group.title ? 1 : 0} />
              ))}
            </ul>
          </section>
        ))
      )}
    </div>
  )
}
