/**
 * Lightweight prompt-toolbar trigger for the Multimodal Prompt Upsampler (PUP).
 * The dialog itself is a lazy chunk so the Generate route + ParamsPane stay
 * comfortably under the 150 KB gzip budget.
 */
import { RotateCcw, WandSparkles } from "@/components/icons"
import { Suspense, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { retryableLazy } from "@/lib/retryable-lazy"

const PupUpsamplerDialog = retryableLazy(() => import("./PupUpsamplerDialog"))

export function PupTrigger({ value, setValue, disabled }: { value: string; setValue: (next: string) => void; disabled?: boolean }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [previous, setPrevious] = useState<string | null>(null)
  const [applied, setApplied] = useState<string | null>(null)

  const openDialog = () => {
    setLoaded(true)
    setOpen(true)
  }

  const canQuickRevert = previous !== null && applied !== null && value === applied

  return (
    <>
      {canQuickRevert ? (
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={t("Revert upsampled prompt")}
                disabled={disabled}
                onClick={() => {
                  setValue(previous)
                  setPrevious(null)
                  setApplied(null)
                }}
              />
            }
          >
            <RotateCcw />
          </TooltipTrigger>
          <TooltipContent>{t("Revert upsampled prompt")}</TooltipContent>
        </Tooltip>
      ) : null}
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={t("Upsample prompt (PUP)")}
              disabled={disabled}
              onClick={openDialog}
            />
          }
        >
          <WandSparkles />
        </TooltipTrigger>
        <TooltipContent>{t("Upsample prompt (PUP)")}</TooltipContent>
      </Tooltip>
      {loaded ? (
        <Suspense fallback={null}>
          <PupUpsamplerDialog
            open={open}
            onOpenChange={setOpen}
            value={value}
            previousValue={previous}
            disabled={disabled}
            onApply={(next, prev) => {
              setPrevious(prev)
              setApplied(next)
              setValue(next)
            }}
            onRevert={() => {
              if (previous !== null) setValue(previous)
              setPrevious(null)
              setApplied(null)
            }}
          />
        </Suspense>
      ) : null}
    </>
  )
}
