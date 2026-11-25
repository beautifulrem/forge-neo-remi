/**
 * Extension pages (former `on_ui_tabs`), at `#/ext/<id>`. A page with
 * `widget` renders the dedicated view `src/pages/custom/<widget>/index.tsx`;
 * otherwise its layout comes from GET /api/v2/pages/{id} and goes through the
 * schema renderer (`PageView`). Other views send values to a page with
 * `sendToPage` (`./deliver`), e.g. PNG Info's send targets (`./send-targets`).
 */
import { useQuery, useSuspenseQuery } from "@tanstack/react-query"
import { Puzzle } from "@/components/icons"
import { type ComponentType, Suspense, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { bootstrapQuery, pageSchemaQuery } from "@/api/queries"
import type { PageInfo } from "@/api/types"
import type { Route } from "@/app/router"
import { PAGE_CONTAINER } from "@/components/page-container"
import { PagePlaceholder } from "@/components/PagePlaceholder"
import { PageSkeleton } from "@/components/PageSkeleton"
import { Button } from "@/components/ui/button"
import { errorText } from "@/features/errors/error-text"
import { cn } from "@/lib/utils"
import { dropPending } from "./deliver"
import { PageView } from "./PageView"
import { retryableLazy } from "@/lib/retryable-lazy"

export type CustomPageProps = { page: PageInfo }

const customModules = import.meta.glob<{ default: ComponentType<CustomPageProps> }>("../custom/*/index.tsx")
const customPages = new Map<string, ComponentType<CustomPageProps>>(
  Object.entries(customModules).map(([path, load]) => [path.split("/").at(-2) ?? path, retryableLazy(load)]),
)

function SchemaPage({ page }: { page: PageInfo }) {
  const { data } = useSuspenseQuery(pageSchemaQuery(page.id))
  return (
    <div className="h-full overflow-y-auto" data-scroll-root>
      <div className={cn(PAGE_CONTAINER, "pb-16")}>
        <PageView schema={data} />
      </div>
    </div>
  )
}

export default function ExtensionPage({ route }: { route: Extract<Route, { name: "page" }> }) {
  const { t, i18n } = useTranslation()
  const bootstrap = useQuery(bootstrapQuery)
  const page = bootstrap.data?.pages.find((p) => p.id === route.id)
  // known only once the app's page list arrived: a failed request is no answer
  const missing = bootstrap.data !== undefined && !page
  // A send to a page that is not there (its extension disabled, hidden) ends here.
  useEffect(() => {
    if (missing) dropPending(route.id)
  }, [missing, route.id])
  if (bootstrap.isPending) return <PageSkeleton />
  if (!bootstrap.data) {
    return (
      <PagePlaceholder icon={Puzzle} title="Page unavailable" description={errorText(bootstrap.error, {}, i18n.language)}>
        <Button variant="outline" size="sm" onClick={() => void bootstrap.refetch()}>
          {t("Retry")}
        </Button>
      </PagePlaceholder>
    )
  }
  if (!page) {
    return <PagePlaceholder icon={Puzzle} title="Page not found" description="No extension provides this page." />
  }
  const Custom = page.widget ? customPages.get(page.widget) : undefined
  if (page.widget && !Custom) {
    return <PagePlaceholder icon={Puzzle} title={page.title} description="This page's view is not part of this build." />
  }
  return (
    <Suspense fallback={<PageSkeleton />}>
      {/* oxlint-disable-next-line react/static-components -- a stable lazy component from the page registry */}
      {Custom ? <Custom page={page} /> : <SchemaPage key={page.id} page={page} />}
    </Suspense>
  )
}
