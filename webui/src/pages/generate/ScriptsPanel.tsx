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
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { FIELD_BUTTON_CLASS } from "@/components/ui/select-styles"
import { Toggle } from "@/components/ui/toggle"
import { cn } from "@/lib/utils"
import { resetOwners, resetParams } from "@/ui/actions"
import { evaluate } from "@/ui/conditions"
import { EnabledProvider, OwnerProvider } from "@/ui/context"
import { DEFAULT_SLOT, inputsIn, scriptsForSlot } from "@/ui/form-index"
import { anyModified, type FormState } from "@/ui/form-store"
import { useForm, useFormState } from "@/ui/hooks"
import { PanelDepthContext, panelClass, panelTitleClass, usePanelDepth } from "@/ui/layout/panel"
import { RenderLayout, RenderNode } from "@/ui/Renderer"
import { sentenceText } from "@/ui/messages"
import type { ChoiceNode, ChoiceOption, Node, ScriptSchema, TabSchema } from "@/ui/schema"

function scriptNames(script: ScriptSchema): string[] {
  return inputsIn(script.layout).map((input) => input.name)
}

/** Enabled = the section's enable toggle is on (scripts without one count as enabled when modified). */
function isActive(s: FormState, script: ScriptSchema): boolean {
  const layout = script.layout
  if (layout?.type === "section" && layout.enable) return Boolean(s.values[script.id]?.[layout.enable.name])
  return false
}

function scriptReason(
  script: ScriptSchema,
  t: ReturnType<typeof useTranslation>["t"],
  i18n: ReturnType<typeof useTranslation>["i18n"],
): string | null {
  return (
    (script.inapplicable_reason
      ? sentenceText(i18n, script.inapplicable_reason, script.inapplicable_template, script.inapplicable_params, i18n.language)
      : null) ??
    (script.requires_capabilities?.length
      ? t("Requires: {{capabilities}}", { capabilities: script.requires_capabilities.join(", ") })
      : null)
  )
}

function matchSelectableScript(choice: ChoiceOption, scripts: readonly ScriptSchema[]): ScriptSchema | undefined {
  const key = String(choice.value)
  return scripts.find((script) => {
    if (script.title === key || script.title === choice.label) return true
    const cond = script.visible_if
    return Boolean(cond && typeof cond === "object" && "param" in cond && cond.param === "core.script" && "eq" in cond && String(cond.eq) === key)
  })
}

function ScriptCard({ script, reason }: { script: ScriptSchema; reason?: string | null }) {
  const { t } = useTranslation()
  const { store, formId } = useForm()
  const names = scriptNames(script)
  const modified = useFormState((s) => anyModified(s, script.id, names))
  const visible = useFormState((s) => evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]))
  const depth = usePanelDepth()
  if (!visible || !script.layout) return null
  const inapplicable = script.applicable === false
  const layout = script.layout
  const titled = layout.type === "section" || (layout.type === "group" && Boolean(layout.title))
  const body: Node = layout.type === "group" && !titled ? { ...layout, type: "column" } : layout
  // Listed extensions start closed; the chosen selectable script stays open.
  const collapsed: Node | null = !titled && script.kind !== "selectable" ? { type: "section", title: script.title, open: false, children: [body] } : null
  return (
    <OwnerProvider owner={script.id}>
      <div data-script={script.id} data-applicable={inapplicable ? "false" : undefined} aria-disabled={inapplicable || undefined} className="min-w-0">
        {reason ? (
          <p className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Info className="size-3 shrink-0" aria-hidden />
            <span className="truncate" title={reason}>
              {reason}
            </span>
          </p>
        ) : null}
        <EnabledProvider enabled={!inapplicable}>
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
                    disabled={inapplicable}
                    className="relative after:absolute after:-inset-2 ml-auto"
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
        </EnabledProvider>
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
  const { t, i18n } = useTranslation()
  const id = useId()
  const { store, formId } = useForm()
  const slots = useFormState((s) => s.index.slots)
  const [query, setQuery] = useState("")
  const [onlyModified, setOnlyModified] = useState(false)
  const [open, setOpen] = useState(true)
  const [inapplicableOpen, setInapplicableOpen] = useState(false)

  // Applicable unslotted extensions render normally; any non-selectable script marked
  // `applicable === false` (unslotted OR slotted) moves into the quiet collapsed drawer.
  // Scripts whose `visible_if` evaluates to false are hidden from counts, lists, and drawers.
  const allNonSelectable = schema.scripts.filter((script) => script.kind !== "selectable" && script.layout)
  const allListed = scriptsForSlot(schema.scripts, DEFAULT_SLOT, slots).filter(
    (script) => script.kind !== "selectable" && script.layout && script.applicable !== false,
  )
  const allInapplicable = allNonSelectable.filter((script) => script.applicable === false)
  const visibleListedMask = useFormState((s) =>
    allListed.map((script) => (evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]) ? "1" : "0")).join(""),
  )
  const visibleInapplicableMask = useFormState((s) =>
    allInapplicable.map((script) => (evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]) ? "1" : "0")).join(""),
  )
  const listed = allListed.filter((_, index) => visibleListedMask[index] === "1")
  const inapplicable = allInapplicable.filter((_, index) => visibleInapplicableMask[index] === "1")
  const selectable = schema.scripts.filter((script) => script.kind === "selectable")
  const allInapplicableSelectable = selectable.filter((script) => script.applicable === false && script.visible_if !== false)
  const renderedInCardMask = useFormState((s) =>
    allInapplicableSelectable
      .map((script) => (script.layout && evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]) ? "1" : "0"))
      .join(""),
  )
  const inapplicableSelectable = allInapplicableSelectable.filter((_, index) => renderedInCardMask[index] !== "1")
  const selectedScript = useFormState((s) => (scriptChoice ? s.values.core?.[scriptChoice.name] : undefined))
  const effectiveScriptChoice: ChoiceNode | null = scriptChoice
    ? {
        ...scriptChoice,
        choices: scriptChoice.choices.filter((choice) => {
          const matched = matchSelectableScript(choice, selectable)
          if (!matched || matched.applicable !== false) return true
          return selectedScript !== undefined && selectedScript !== null && String(selectedScript) === String(choice.value)
        }),
      }
    : null
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
            <Collapsible.Trigger className={cn("group/trigger flex min-w-0 flex-1 items-center gap-2 self-stretch rounded-sm text-left outline-none focus-visible:ring-3 focus-visible:ring-focus-ring", panelTitleClass(0))}>
              <span id={`${id}-title`}>{t("Extensions")}</span>
              <span className="font-mono text-xs font-normal text-muted-foreground tabular-nums">{listed.length}</span>
              {activeCount > 0 ? <span className="font-mono text-xs font-medium text-primary tabular-nums">{t("{{count}} on", { count: activeCount })}</span> : null}
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
                pressed={onlyModified}
                onPressedChange={setOnlyModified}
                aria-label={t("Only modified")}
                title={t("Only modified")}
                className={cn(FIELD_BUTTON_CLASS, "gap-1")}
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
                <p className="p-4 text-center text-xs text-muted-foreground">{t("No extension matches")}</p>
              ) : null}
            </div>

            {shownInapplicable.length > 0 ? (
              <Collapsible.Root
                open={inapplicableOpen}
                onOpenChange={setInapplicableOpen}
                data-slot="inapplicable-scripts"
                className="mt-1 flex flex-col gap-2 border-t border-border/60 pt-2.5"
              >
                <Collapsible.Trigger className="group/inapp flex min-h-7 w-full items-center gap-2 rounded-sm text-left text-xs font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-focus-ring">
                  <span className="truncate">{t("Not applicable for active model")}</span>
                  <span className="font-mono text-xs font-normal text-muted-foreground tabular-nums">{shownInapplicable.length}</span>
                  {inapplicableActiveCount > 0 ? (
                    <span className="font-mono text-xs font-medium text-advice-warn tabular-nums">
                      {t("{{count}} on", { count: inapplicableActiveCount })}
                    </span>
                  ) : null}
                  <ChevronDown aria-hidden className="ml-auto size-3.5 shrink-0 -rotate-90 transition-transform duration-150 group-data-[panel-open]/inapp:rotate-0" />
                </Collapsible.Trigger>
                <Collapsible.Panel className="-mx-1 flex flex-col gap-2.5 px-1 pt-1 opacity-85">
                  <PanelDepthContext value={1}>
                    {shownInapplicable.map((script) => (
                      <ScriptCard key={script.id} script={script} reason={scriptReason(script, t, i18n)} />
                    ))}
                  </PanelDepthContext>
                </Collapsible.Panel>
              </Collapsible.Root>
            ) : null}
          </Collapsible.Panel>
        </Collapsible.Root>
      ) : null}

      {effectiveScriptChoice ? (
        <section aria-label={t("Script")} className={cn("flex flex-col gap-3", panelClass(0))}>
          <OwnerProvider owner="core">
            <RenderNode node={effectiveScriptChoice} />
          </OwnerProvider>
          <PanelDepthContext value={1}>
            {selectable.map((script) => (
              <ScriptCard key={script.id} script={script} reason={script.applicable === false ? scriptReason(script, t, i18n) : null} />
            ))}
          </PanelDepthContext>
          {inapplicableSelectable.length > 0 ? (
            <div data-slot="inapplicable-selectable-scripts" className="flex flex-col gap-1 border-t border-border/60 pt-2 text-xs text-muted-foreground">
              <span className="font-medium">{t("Not applicable for active model")}</span>
              {inapplicableSelectable.map((script) => {
                const reason = scriptReason(script, t, i18n)
                return (
                  <p key={script.id} data-script={script.id} data-applicable="false" className="flex items-start gap-1.5">
                    <Info className="mt-0.5 size-3 shrink-0" aria-hidden />
                    <span className="min-w-0">
                      <span className="font-medium text-foreground/80">{t(script.title)}</span>
                      {reason ? <span className="text-muted-foreground"> — {reason}</span> : null}
                    </span>
                  </p>
                )
              })}
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}
