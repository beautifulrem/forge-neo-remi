import { Loader2 } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { useForm, useInheritedEnabled, useRunAction } from "../hooks"
import type { ButtonNode } from "../schema"

const VARIANT = {
  primary: { variant: "default", size: "default" },
  secondary: { variant: "secondary", size: "default" },
  ghost: { variant: "ghost", size: "default" },
  tool: { variant: "outline", size: "sm" },
  danger: { variant: "destructive", size: "default" },
} as const

/** Calls the owner's action with the owner's current values. */
export function ActionButton({ node }: { node: ButtonNode }) {
  const { t } = useTranslation()
  const run = useRunAction()
  const enabled = useInheritedEnabled()
  const [busy, setBusy] = useState(false)
  const { formId } = useForm()
  // In a generate form the Generate button is the one filled button: an extension's "primary" there is secondary.
  const generate = formId === "txt2img" || formId === "img2img"
  const variant = node.variant === "primary" && generate ? "secondary" : (node.variant ?? "secondary")
  const style = VARIANT[variant] ?? VARIANT.secondary

  return (
    <Button
      variant={style.variant}
      size={style.size}
      disabled={!enabled || busy}
      aria-busy={busy || undefined}
      onClick={async () => {
        setBusy(true)
        try {
          await run(node.action)
        } finally {
          setBusy(false)
        }
      }}
      className="self-start"
    >
      {busy ? <Loader2 className="animate-spin" aria-hidden /> : null}
      {t(node.label)}
    </Button>
  )
}
