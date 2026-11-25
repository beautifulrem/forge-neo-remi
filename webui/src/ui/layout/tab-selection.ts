/**
 * Which tabs of a `tabs` node show and which one is selected, shared by the strip (`Tabs.tsx`, deferred)
 * and the stand-in that renders the selected tab's content while the strip's code loads (`lazy-views.tsx`).
 */
import { evaluate } from "../conditions"
import { tabsKey } from "../form-index"
import { useFormState, useOwner } from "../hooks"
import type { TabNode, TabsNode } from "../schema"

/** Tab ids: the bound choice value, else the tab's index. */
export function tabId(tab: TabNode, index: number, bound: boolean): string | number {
  return bound && tab.value !== undefined && tab.value !== null ? tab.value : String(index)
}

/** Which tabs pass their `visible_if`, as a stable string of 0/1 per tab. */
export function useVisibleTabs(node: TabsNode): boolean[] {
  const owner = useOwner()
  const mask = useFormState((s) =>
    node.children.map((tab) => (evaluate(tab.visible_if, owner, (o, n) => s.values[o]?.[n]) ? "1" : "0")).join(""),
  )
  return [...mask].map((bit) => bit === "1")
}

/** The selected id of an unbound strip: the stored one, or the first visible tab when that one is hidden. */
export function unboundSelection(selected: string, visible: readonly boolean[]): string {
  return visible[Number(selected)] ? selected : String(Math.max(0, visible.indexOf(true)))
}

/** The tab whose content shows: the bound choice's tab, else the stored or first visible one; null when none shows. */
export function useSelectedTab(node: TabsNode): TabNode | null {
  const owner = useOwner()
  const visible = useVisibleTabs(node)
  const bound = useFormState((s) => Boolean(node.bind && s.index.inputs.get(owner)?.has(node.bind)))
  const selected = useFormState((s) => {
    const value = bound ? s.values[owner]?.[node.bind!] : (s.tabs[tabsKey(owner, node)] ?? "0")
    return value === undefined || value === null ? null : String(value)
  })
  if (selected === null) return null
  const id = bound ? selected : unboundSelection(selected, visible)
  const index = node.children.findIndex((tab, i) => visible[i] && String(tabId(tab, i, bound)) === id)
  return index >= 0 ? node.children[index]! : null
}
