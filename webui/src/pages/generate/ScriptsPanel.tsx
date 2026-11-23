/**
 * The scripts that have no slot of their own (always-on extensions) as a
 * searchable, collapsible list with an "only modified" filter and per-script
 * reset, followed by a quiet collapsed drawer for scripts marked
 * `applicable === false` for the active model, and the selectable-script choice.
 */
import { Collapsible } from "@base-ui/react/collapsible"
import { ChevronDown, Info, ListFilter, RotateCcw, Search } from "@/components/icons"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Toggle } from "@/components/ui/toggle"
import { cn } from "@/lib/utils"
import { resetOwners, resetParams } from "@/ui/actions"
import { evaluate } from "@/ui/conditions"
import { OwnerProvider } from "@/ui/context"
import { DEFAULT_SLOT, inputsIn, scriptsForSlot } from "@/ui/form-index"
import { anyModified, type FormState } from "@/ui/form-store"
import { useForm, useFormState } from "@/ui/hooks"
import { PanelDepthContext, panelClass, panelTitleClass, usePanelDepth } from "@/ui/layout/panel"
import { RenderLayout, RenderNode } from "@/ui/Renderer"
import type { ChoiceNode, Node, ScriptSchema, TabSchema } from "@/ui/schema"

function scriptNames(script: ScriptSchema): string[] {
  return inputsIn(script.layout).map((input) => input.name)
}

/** Enabled = the section's enable toggle is on (scripts without one count as enabled when modified). */
function isActive(s: FormState, script: ScriptSchema): boolean {
  const layout = script.layout
  if (layout?.type === "section" && layout.enable) return Boolean(s.values[script.id]?.[layout.enable.name])
  return false
}

function ScriptCard({ script, reason }: { script: ScriptSchema; reason?: string | null }) {
  const { t } = useTranslation()
  const { store, formId } = useForm()
  const names = scriptNames(script)
  const modified = useFormState((s) => anyModified(s, script.id, names))
  const visible = useFormState((s) => evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]))
  const depth = usePanelDepth()
  if (!visible || !script.layout) return null
  const layout = script.layout
  const titled = layout.type === "section" || (layout.type === "group" && Boolean(layout.title))
  const body: Node = layout.type === "group" && !titled ? { ...layout, type: "column" } : layout
  // Listed extensions start closed; the chosen selectable script stays open.
  const collapsed: Node | null = !titled && script.kind !== "selectable" ? { type: "section", title: script.title, open: false, children: [body] } : null
  return (
    <OwnerProvider owner={script.id}>
      <div data-script={script.id} data-applicable={script.applicable === false ? "false" : undefined} className="min-w-0">
        {reason ? (
          <p className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Info className="size-3 shrink-0" aria-hidden />
            <span className="truncate" title={reason}>
              {reason}
            </span>
          </p>
        ) : null}
        {titled ? (
          <RenderLayout layout={layout} />
        ) : collapsed ? (
          <RenderLayout layout={collapsed} />
        ) : (
          <div role="group" aria-label={t(script.title)} className={cn("flex flex-col gap-3", panelClass(depth))}>
            <div className="flex min-h-8 items-center gap-2">
              <span className={panelTitleClass(depth)}>{t(script.title)}</span>
              {modified ? (
                <Button
                  variant="ghost"
                  size="icon-xs"
                  className="ml-auto"
                  aria-label={t("Reset {{label}} to default", { label: t(script.title) })}
                  onClick={() => resetParams({ store, tab: formId }, script.id, names)}
                >
                  <RotateCcw />
                </Button>
              ) : null}
            </div>
            <RenderLayout layout={body} />
          </div>
        )}
      </div>
    </OwnerProvider>
  )
}

function matches(script: ScriptSchema, query: string, translate: (s: string) => string): boolean {
  if (!query) return true
  const haystack = [script.title, translate(script.title), ...inputsIn(script.layout).flatMap((input) => [input.label, translate(input.label)])]
  return haystack.some((text) => text.toLowerCase().includes(query))
}

export function ScriptsPanel({ schema, scriptChoice }: { schema: TabSchema; scriptChoice: ChoiceNode | null }) {
  const { t } = useTranslation()
  const id = useId()
  const { store, formId } = useForm()
  const slots = useFormState((s) => s.index.slots)
  const [query, setQuery] = useState("")
  const [onlyModified, setOnlyModified] = useState(false)
  const [open, setOpen] = useState(true)
  const [inapplicableOpen, setInapplicableOpen] = useState(false)

  // Applicable unslotted extensions render normally; any non-selectable script marked
  // `applicable === false` (unslotted OR slotted) moves into the quiet collapsed drawer.
  const allNonSelectable = schema.scripts.filter((script) => script.kind !== "selectable" && script.layout)
  const listed = scriptsForSlot(schema.scripts, DEFAULT_SLOT, slots).filter(
    (script) => script.kind !== "selectable" && script.layout && script.applicable !== false,
  )
  const inapplicable = allNonSelectable.filter((script) => script.applicable === false)
  const selectable = schema.scripts.filter((script) => script.kind === "selectable")
  const modifiedIds = useFormState((s) => listed.filter((script) => anyModified(s, script.id, scriptNames(script))).map((script) => script.id).join(" "))
  const activeCount = useFormState((s) => listed.filter((script) => isActive(s, script)).length)
  const inapplicableActiveCount = useFormState((s) => inapplicable.filter((script) => isActive(s, script)).length)
  const modifiedSet = new Set(modifiedIds.split(" ").filter(Boolean))
  const q = query.trim().toLowerCase()
  const shown = listed.filter((script) => (!onlyModified || modifiedSet.has(script.id)) && matches(script, q, t))
  const shownInapplicable = inapplicable.filter((script) => matches(script, q, t))

  if (listed.length === 0 && inapplicable.length === 0 && !scriptChoice) return null

  return (
    <div className="flex flex-col gap-3">
      {listed.length > 0 || inapplicable.length > 0 ? (
        <Collapsible.Root open={open} onOpenChange={setOpen} render={<section aria-labelledby={`${id}-title`} />} className={cn("flex flex-col gap-2", panelClass(0))}>
          <div className="flex min-h-8 items-center gap-2">
            <Collapsible.Trigger className={cn("group/trigger flex min-w-0 flex-1 items-center gap-2 self-stretch rounded-sm text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50", panelTitleClass(0))}>
              <span id={`${id}-title`}>{t("Extensions")}</span>
              <Badge variant="secondary" className="tabular-nums">
                {listed.length}
              </Badge>
              {activeCount > 0 ? <Badge className="tabular-nums">{t("{{count}} on", { count: activeCount })}</Badge> : null}
              <ChevronDown aria-hidden className="ml-auto size-4 shrink-0 -rotate-90 text-muted-foreground transition-transform duration-150 group-data-[panel-open]/trigger:rotate-0" />
            </Collapsible.Trigger>
            {modifiedSet.size > 0 ? (
              <Button
                variant="ghost"
                size="xs"
                className="text-muted-foreground"
                onClick={() =>
                  resetOwners(
                    { store, tab: formId },
                    listed.filter((script) => modifiedSet.has(script.id)).map((script) => script.id),
                  )
                }
              >
                <RotateCcw />
                {t("Reset all")}
              </Button>
            ) : null}
          </div>
          <Collapsible.Panel className="flex flex-col gap-2">
            <div className="flex items-center gap-1.5">
              <div className="relative min-w-0 flex-1">
                <Search aria-hidden className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder={t("Filter extensions…")}
                  aria-label={t("Filter extensions")}
                  className="h-8 pl-7"
                />
              </div>
              <Toggle
                variant="outline"
                pressed={onlyModified}
                onPressedChange={setOnlyModified}
                aria-label={t("Only modified")}
                title={t("Only modified")}
                className="gap-1"
              >
                <ListFilter />
                <span className="hidden sm:inline">{t("Modified")}</span>
              </Toggle>
            </div>
            <div className="flex flex-col gap-3">
              <PanelDepthContext value={1}>
                {shown.map((script) => (
                  <ScriptCard key={script.id} script={script} />
                ))}
              </PanelDepthContext>
              {listed.length > 0 && shown.length === 0 ? (
                <p className={cn("rounded-lg border border-dashed p-4 text-center text-xs text-muted-foreground")}>{t("No extension matches")}</p>
              ) : null}
            </div>

            {shownInapplicable.length > 0 ? (
              <Collapsible.Root
                open={inapplicableOpen}
                onOpenChange={setInapplicableOpen}
                data-slot="inapplicable-scripts"
                className="mt-1 flex flex-col gap-2 border-t border-border/60 pt-2.5"
              >
                <Collapsible.Trigger className="group/inapp flex min-h-7 w-full items-center gap-2 rounded-sm text-left text-xs font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50">
                  <span className="truncate">{t("Not applicable for active model")}</span>
                  <Badge variant="outline" className="h-4 border-border px-1.5 text-[0.625rem] text-muted-foreground tabular-nums">
                    {shownInapplicable.length}
                  </Badge>
                  {inapplicableActiveCount > 0 ? (
                    <Badge variant="outline" className="h-4 border-advice-warn-border bg-advice-warn-bg px-1.5 text-[0.625rem] text-advice-warn tabular-nums">
                      {t("{{count}} on", { count: inapplicableActiveCount })}
                    </Badge>
                  ) : null}
                  <ChevronDown aria-hidden className="ml-auto size-3.5 shrink-0 -rotate-90 transition-transform duration-150 group-data-[panel-open]/inapp:rotate-0" />
                </Collapsible.Trigger>
                <Collapsible.Panel className="-mx-1 flex flex-col gap-2.5 px-1 pt-1 opacity-85">
                  <PanelDepthContext value={1}>
                    {shownInapplicable.map((script) => (
                      <ScriptCard
                        key={script.id}
                        script={script}
                        reason={
                          script.inapplicable_reason ??
                          (script.requires_capabilities?.length
                            ? t("Requires: {{capabilities}}", { capabilities: script.requires_capabilities.join(", ") })
                            : null)
                        }
                      />
                    ))}
                  </PanelDepthContext>
                </Collapsible.Panel>
              </Collapsible.Root>
            ) : null}
          </Collapsible.Panel>
        </Collapsible.Root>
      ) : null}

      {scriptChoice ? (
        <section aria-label={t("Script")} className={cn("flex flex-col gap-3", panelClass(0))}>
          <OwnerProvider owner="core">
            <RenderNode node={scriptChoice} />
          </OwnerProvider>
          <PanelDepthContext value={1}>
            {selectable.map((script) => (
              <ScriptCard key={script.id} script={script} />
            ))}
          </PanelDepthContext>
        </section>
      ) : null}
    </div>
  )
}
