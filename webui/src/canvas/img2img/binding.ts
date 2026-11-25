/**
 * Which image params the img2img editor binds (from the schema) and which layer the current mode
 * edits: what the Source card reads, its stand-in included, without the writes of `use-img2img`.
 */
import { useQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import { useStore } from "zustand"
import { schemaQuery } from "@/api/queries"
import { getFormStore } from "@/ui/form-store"
import { editorBinding, type EditorBinding, modeLayers, type ModeLayers } from "../core/modes"

export const IMG2IMG = "img2img"

/** The img2img editor binding from the (cached) schema; null until it loads or without image inputs. */
export function useImg2imgBinding(): EditorBinding | null {
  const { data: schema } = useQuery(schemaQuery(IMG2IMG))
  const layout = schema?.core.layout ?? null
  return useMemo(() => (layout ? editorBinding(layout) : null), [layout])
}

/** Mode layers for the current values (re-renders only when they change). */
export function useModeLayers(binding: EditorBinding | null): ModeLayers {
  const store = getFormStore(IMG2IMG)
  const signature = useStore(store, (s) => {
    if (!binding) return "none"
    const layers = modeLayers(binding, (owner, name) => s.values[owner]?.[name])
    return `${layers.canvas ? 1 : 0}|${layers.active?.param ?? ""}|${layers.mask ?? ""}`
  })
  return useMemo(() => {
    const [canvas, active, mask] = signature.split("|")
    return {
      canvas: canvas === "1",
      active: binding?.layers.find((layer) => layer.param === active) ?? null,
      mask: (mask || null) as ModeLayers["mask"],
    }
  }, [signature, binding])
}
