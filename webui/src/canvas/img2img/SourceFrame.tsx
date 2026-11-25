/**
 * The Source card's shared components (`source-frame.ts` has the classes and the parts): a mode
 * tile's face, the card's layout, and the stand-in shown while the panel's code loads.
 */
import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { OwnerProvider } from "@/ui/context"
import { useFormState } from "@/ui/hooks"
import { RenderChildren, RenderNode } from "@/ui/Renderer"
import type { Node, TabNode, TabsNode } from "@/ui/schema"
import { useImg2imgBinding, useModeLayers } from "./binding"
import {
  IMAGE_LABEL,
  IMAGE_LINE,
  IMAGE_LINES,
  IMAGE_TEXT,
  IMAGE_THUMB,
  MODE_HINT,
  MODE_PICKER,
  MODE_TEXT,
  modeGrid,
  modeLabel,
  modeTile,
  PAINT_HINT,
  useModeOptions,
  useSourceParts,
} from "./source-frame"

/** A tile's icon (16 px) over its label; the tiles of a row share its height: a one-line label sits centred in the room a two-line one takes. */
export function ModeTileFace({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <>
      {icon}
      <span className="flex w-full flex-1 items-center justify-center">
        <span className="line-clamp-2 w-full text-center text-xs leading-tight font-medium break-words">{label}</span>
      </span>
    </>
  )
}

/** The card's content: the picker, the image lines, then the selected mode's options and the rest of the source blocks. */
export function SourceLayout({ picker, images, selected, rest }: { picker: ReactNode; images: ReactNode; selected: TabNode | null; rest: readonly Node[] }) {
  return (
    <OwnerProvider owner="core">
      <div className="flex flex-col gap-3">
        {picker}
        {images}
        {selected && selected.children.length > 0 ? (
          <div className="flex flex-col gap-3">
            <RenderChildren nodes={selected.children} />
          </div>
        ) : null}
        {rest.map((node, index) => (
          <RenderNode key={index} node={node} />
        ))}
      </div>
    </OwnerProvider>
  )
}

/** An image line without its controls: the frame, the label, the text, and `actions` 28 px squares for its buttons. */
function ImageLineStandIn({ label, text, actions, mask }: { label: string; text: string; actions: number; mask?: boolean }) {
  return (
    <div className={IMAGE_LINE}>
      <div className={cn(IMAGE_THUMB, mask ? "bg-checker" : "bg-muted")} />
      <div className="flex min-w-0 flex-1 flex-col">
        <span className={IMAGE_LABEL}>{label}</span>
        <span className={cn(IMAGE_TEXT, "text-muted-foreground")}>{text}</span>
      </div>
      <div className="flex shrink-0 items-center">
        {Array.from({ length: actions }, (_, index) => (
          <span key={index} className="size-7" />
        ))}
      </div>
    </div>
  )
}

/**
 * The Source card while the panel's code loads: the mode grid with its labels and the hint, the
 * image lines as they start (no thumbnail read yet), shown but not yet usable, and the mode's
 * options, rendered as they will be (as a tabs placeholder renders the selected tab).
 */
export function SourcePanelStandIn({ nodes }: { nodes: readonly Node[] }) {
  const { t } = useTranslation()
  const binding = useImg2imgBinding()
  const layers = useModeLayers(binding)
  const { tabs, modeNode, mode, selected, rest } = useSourceParts(nodes, binding)
  const image = useFormState((s) => (binding ? Boolean(s.values.core?.[binding.background]) : false))
  const active = layers.active
  const layerValue = useFormState((s) => (active ? Boolean(s.values.core?.[active.param]) : false))
  const hint = MODE_TEXT[String(mode)]?.description
  return (
    <div aria-busy className="contents">
      <SourceLayout
        selected={selected}
        rest={rest}
        picker={tabs && modeNode ? <ModesStandIn tabs={tabs} mode={mode} hint={hint ? t(hint) : null} /> : null}
        images={
          binding && layers.canvas ? (
            <div className={IMAGE_LINES} aria-hidden>
              {/* edit on the stage and upload, and remove with an image */}
              <ImageLineStandIn label={t("Source image")} text={image ? "…" : t("Drop an image here or on the stage")} actions={image ? 3 : 2} />
              {active && !active.painted ? (
                <ImageLineStandIn mask label={t("Mask")} text={layerValue ? "…" : t("White (or opaque) areas are redrawn")} actions={layerValue ? 2 : 1} />
              ) : active ? (
                <p className={cn(PAINT_HINT, "text-muted-foreground")}>{active.kind === "mask" ? t("Paint the mask on the stage.") : t("Paint on the stage.")}</p>
              ) : null}
            </div>
          ) : null
        }
      />
    </div>
  )
}

function ModesStandIn({ tabs, mode, hint }: { tabs: TabsNode; mode: unknown; hint: string | null }) {
  const { t } = useTranslation()
  const options = useModeOptions(tabs)
  const grid = modeGrid(options.length)
  return (
    <div className={MODE_PICKER} aria-hidden>
      <div style={grid.style} className={grid.className}>
        {options.map((tab, index) => (
          <div key={String(tab.value)} data-checked={tab.value === mode ? "" : undefined} className={modeTile(options.length, index)}>
            <ModeTileFace icon={<span className="size-4 shrink-0" />} label={t(modeLabel(tab))} />
          </div>
        ))}
      </div>
      {hint ? <p className={MODE_HINT}>{hint}</p> : null}
    </div>
  )
}
