/** Compact 28px machined 4-swatch RGBA backdrop picker shared by Lightbox and CompareDialog. */
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { BACKDROP_META, BACKDROP_MODES, type BackdropMode } from "./backdrop-modes"

type BackdropPickerProps = {
  value: BackdropMode
  onChange: (mode: BackdropMode) => void
  /** Light chrome over dark backdrops (Lightbox). */
  tone?: "light" | "default"
  className?: string
}

export function BackdropPicker({ value, onChange, tone = "default", className }: BackdropPickerProps) {
  const { t } = useTranslation()
  return (
    <div
      role="group"
      aria-label={t("RGBA backdrop")}
      className={cn(
        "inline-flex h-7 items-center gap-0.5 rounded-md border p-0.5",
        tone === "light" ? "border-white/15 bg-white/8" : "border-border bg-muted/70",
        className,
      )}
    >
      {BACKDROP_MODES.map((mode) => {
        const meta = BACKDROP_META[mode]
        const label = t(meta.label)
        const active = value === mode
        return (
          <button
            key={mode}
            type="button"
            aria-label={label}
            aria-pressed={active}
            title={label}
            onClick={() => onChange(mode)}
            className={cn(
              "grid size-6 place-items-center rounded-sm transition-colors outline-none focus-visible:ring-3 focus-visible:ring-focus-ring",
              active
                ? tone === "light"
                  ? "bg-white/22 ring-1 ring-white/80"
                  : "bg-card ring-1 ring-primary"
                : tone === "light"
                  ? "hover:bg-white/14"
                  : "hover:bg-background/60",
            )}
          >
            <span
              aria-hidden
              className={cn(
                "block size-3.5 rounded-xs border",
                mode === "checker" && "bg-checker",
                tone === "light" ? "border-white/35" : "border-border",
              )}
              style={meta.color ? { backgroundColor: meta.color } : undefined}
            />
          </button>
        )
      })}
    </div>
  )
}
