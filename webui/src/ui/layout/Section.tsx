import { Collapsible } from "@base-ui/react/collapsible"
import { ChevronDown } from "@/components/icons"
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { resetParams } from "../actions"
import { useForm, useFormState, useOwner, useParam } from "../hooks"
import { ModifiedDot } from "../Field"
import { inputsIn, sectionKey } from "../form-index"
import { anyModified } from "../form-store"
import { ContainerTitleContext } from "../fallback-name"
import { RenderChildren } from "../Renderer"
import type { SectionNode, ToggleNode } from "../schema"
import { summarizeSection } from "../section-summary"
import { PanelDepthContext, panelClass, panelTitleClass, usePanelDepth } from "./panel"

function EnableSwitch({ node, title, onEnabled }: { node: ToggleNode; title: string; onEnabled: () => void }) {
  const { t } = useTranslation()
  const param = useParam<boolean>(node)
  return (
    <Switch
      checked={Boolean(param.value)}
      onCheckedChange={(checked) => {
        param.setValue(checked)
        if (checked) onEnabled()
      }}
      disabled={!param.enabled}
      aria-label={t("Enable {{title}}", { title })}
      aria-invalid={param.error ? true : undefined}
      title={param.error}
      // the row's gap plus this margin hold the switch's 12 px after: box off the title before it
      className="ml-1"
    />
  )
}

/**
 * Collapsible panel. With `enable`, the header carries the owner's enable
 * toggle (the former InputAccordion); switching it on also opens the panel.
 * The header's reset restores every param of the section and reruns their
 * `on_change` actions and conflict locks. Widgets may add a summary of
 * their values to the header (`section-summary.ts`).
 */
export function Section({ node }: { node: SectionNode }) {
  const { t } = useTranslation()
  const id = useId()
  const { store, formId } = useForm()
  const owner = useOwner()
  const key = sectionKey(owner, node)
  const open = useFormState((s) => s.open[key] ?? node.open ?? false)
  const inputs = inputsIn(node)
  const names = inputs.map((input) => input.name)
  const summary = useFormState((s) => summarizeSection(s.values[owner], inputs))
  const modified = useFormState((s) => anyModified(s, owner, names))
  const enabled = useFormState((s) => (node.enable ? Boolean(s.values[owner]?.[node.enable.name]) : false))
  const title = t(node.title)
  const setOpen = (next: boolean) => store.getState().setOpen(key, next)
  const depth = usePanelDepth()

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      render={<section aria-labelledby={`${id}-title`} />}
      data-enabled={enabled || undefined}
      // An inspector panel, not a card (`panel.ts`): a hairline above, the title on the column's edge.
      className={cn("group/section", panelClass(depth))}
    >
      <div className="flex min-h-8 items-center gap-2">
        <Collapsible.Trigger
          className={cn(
            "group/trigger flex min-w-0 flex-1 items-center gap-1.5 self-stretch rounded-sm text-left outline-none",
            panelTitleClass(depth),
            "focus-visible:ring-3 focus-visible:ring-focus-ring",
          )}
        >
          <span id={`${id}-title`} title={title} className={cn("truncate", enabled && "text-foreground")}>
            {title}
          </span>
          {summary ? (
            <span title={summary} className="min-w-0 shrink truncate text-xs font-normal text-primary">{summary}</span>
          ) : null}
          {/* at the row's end: the title keeps the column's left edge */}
          <ChevronDown
            aria-hidden
            className="ml-auto size-4 shrink-0 -rotate-90 text-muted-foreground transition-transform duration-150 group-data-[panel-open]/trigger:rotate-0"
          />
        </Collapsible.Trigger>
        {modified ? (
          <ModifiedDot
            label={title}
            onReset={() => resetParams({ store, tab: formId }, owner, names)}
          />
        ) : null}
        {node.enable ? <EnableSwitch node={node.enable} title={title} onEnabled={() => setOpen(true)} /> : null}
      </div>
      <Collapsible.Panel
        className={cn(
          // 4 px of room either side (-mx-1 / px-1): the clip keeps the fields' focus rings
          "-mx-1 h-(--collapsible-panel-height) overflow-hidden transition-[height] duration-200 ease-out-soft",
          "data-ending-style:h-0 data-starting-style:h-0",
        )}
      >
        <div className="flex flex-col gap-3 px-1 pt-2 pb-1">
          <ContainerTitleContext value={title}>
            <PanelDepthContext value={depth + 1}>
              <RenderChildren nodes={node.children} />
            </PanelDepthContext>
          </ContainerTitleContext>
        </div>
      </Collapsible.Panel>
    </Collapsible.Root>
  )
}
