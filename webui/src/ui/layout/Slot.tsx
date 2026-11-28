import { createContext, type ReactNode, use } from "react"
import { useTranslation } from "react-i18next"
import { evaluate } from "../conditions"
import { OwnerProvider } from "../context"
import { useFormState } from "../hooks"
import { scriptsForSlot } from "../form-index"
import { RenderLayout } from "../Renderer"
import type { Node, ScriptSchema } from "../schema"
import { cn } from "@/lib/utils"
import { panelClass, panelTitleClass, usePanelDepth } from "./panel"

/** Scripts of the tab being rendered; core-form `slot` nodes place them. */
const ScriptsContext = createContext<readonly ScriptSchema[]>([])

export function ScriptsProvider({ scripts, children }: { scripts: readonly ScriptSchema[]; children: ReactNode }) {
  return <ScriptsContext value={scripts}>{children}</ScriptsContext>
}

function ScriptView({ script }: { script: ScriptSchema }) {
  const { t } = useTranslation()
  const depth = usePanelDepth()
  const visible = useFormState((s) => evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]))
  if (!visible || !script.layout) return null
  const layout = script.layout
  // A section or titled group names the script itself; anything else (e.g. an
  // untitled group) is framed with the title.
  const panel = layout.type === "section" || (layout.type === "group" && Boolean(layout.title))
  // The frame replaces an untitled group's box (keeping its conditions).
  const body: Node = layout.type === "group" && !panel ? { ...layout, type: "column" } : layout
  return (
    <OwnerProvider owner={script.id}>
      <div data-script={script.id} className="min-w-0">
        {panel ? (
          <RenderLayout layout={layout} />
        ) : (
          // A bare layout gets a light frame so scripts stay distinguishable.
          <div role="group" aria-label={t(script.title)} className={cn("flex flex-col gap-3", panelClass(depth))}>
            <h2 className={panelTitleClass(depth)}>{t(script.title)}</h2>
            <RenderLayout layout={body} />
          </div>
        )}
      </div>
    </OwnerProvider>
  )
}

export function Slot({ section }: { section: string }) {
  const scripts = use(ScriptsContext)
  const slots = useFormState((s) => s.index.slots)
  // Non-applicable scripts move into the quiet "Not applicable for active model" drawer in ScriptsPanel.
  const allPlaced = scriptsForSlot(scripts, section, slots).filter((script) => script.applicable !== false)
  const visibleMask = useFormState((s) =>
    allPlaced
      .map((script) => (script.layout && evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]) ? "1" : "0"))
      .join(""),
  )
  const placed = allPlaced.filter((_, index) => visibleMask[index] === "1")
  if (placed.length === 0) return null
  return (
    <div className="flex flex-col gap-3" data-slot-section={section}>
      {placed.map((script) => (
        <ScriptView key={script.id} script={script} />
      ))}
    </div>
  )
}
