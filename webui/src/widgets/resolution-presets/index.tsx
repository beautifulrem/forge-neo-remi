/**
 * `resolution-presets`: Anima resolution presets for the generation form's
 * size. Picking one writes the two params `props.width` / `props.height`
 * (`"core.width"`, …) at once, as a programmatic write (a lock keeps its
 * param and the write names it), and runs their change effects (the Hires.
 * fix size line follows). Each list shows the preset matching the current
 * size, if any. The widget's own value is unused (null).
 *
 * props: { standard: [w, h][], highres: [w, h][], width: string, height: string }   value: null
 */
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { SearchSelect, type SelectOption } from "@/components/SearchSelect"
import { runBulkChangeEffects } from "@/ui/actions"
import { splitRef } from "@/ui/conditions"
import { useForm, useFormState, useOwner } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { presetOptions, type Size } from "./sizes"

const NS = registerWidgetCatalogs("resolution-presets", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { standard?: Size[]; highres?: Size[]; width?: string; height?: string }

export default function ResolutionPresets({ props, disabled, id }: WidgetProps<null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { store, formId } = useForm()
  const self = useOwner()
  const highresId = useId()
  const [widthOwner, widthName] = splitRef(props.width ?? "core.width", self)
  const [heightOwner, heightName] = splitRef(props.height ?? "core.height", self)
  const width = useFormState((s) => s.values[widthOwner]?.[widthName])
  const height = useFormState((s) => s.values[heightOwner]?.[heightName])
  const current = typeof width === "number" && typeof height === "number" ? `${width}x${height}` : null

  const lists: { key: string; label: string; options: SelectOption[] }[] = [
    { key: "standard", label: tw("Preset"), options: presetOptions(props.standard ?? []) },
    { key: "highres", label: tw("High-res preset (W+H 2560–3072, max 1856 per side)"), options: presetOptions(props.highres ?? []) },
  ]

  const pick = (value: string) => {
    const [w, h] = value.split("x").map(Number)
    if (!w || !h) return
    const state = store.getState()
    // one write per owner (their lock notes share a toast); change effects for what was written
    const writes: [owner: string, values: Record<string, number>][] =
      widthOwner === heightOwner
        ? [[widthOwner, { [widthName]: w, [heightName]: h }]]
        : [
            [widthOwner, { [widthName]: w }],
            [heightOwner, { [heightName]: h }],
          ]
    const changes = writes.map(([owner, values]) => {
      const kept = state.setValues(owner, values)
      return { owner, names: Object.keys(values).filter((name) => !kept.includes(name)) }
    })
    runBulkChangeEffects({ store, tab: formId }, changes)
  }

  return (
    <div className="@container">
      <div className="grid gap-2 @md:grid-cols-2">
        {lists.map((list, index) => {
          const controlId = index === 0 ? id : highresId
          const matching = list.options.some((option) => option.value === current) ? current : null
          return (
            <div key={list.key} className="flex min-w-0 flex-col gap-1">
              <label htmlFor={controlId} className="text-xs font-medium text-muted-foreground">
                {list.label}
              </label>
              <SearchSelect
                id={controlId}
                options={list.options}
                value={matching}
                onValueChange={pick}
                placeholder={tw("Choose a size")}
                disabled={disabled}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}
