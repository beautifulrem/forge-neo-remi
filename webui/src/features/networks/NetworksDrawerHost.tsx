/**
 * Generate's networks panel: a header button (Alt+N, also in the command
 * palette) toggling the drawer, whose chunk loads on first open, and the
 * prompt-focus tracking that decides where cards go.
 */
import { Layers } from "@/components/icons"
import { Suspense, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { renderFailureText } from "@/features/errors/error-text"
import { type Command, formatShortcut, useRegisterCommands } from "@/app/commands"
import { HeaderSlot } from "@/app/header-slot"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { PRESSED_TINT_CLASS } from "@/components/ui/toggle"
import type { GenerateTab } from "@/features/progress/run-controller"
import { useTrackPromptFocus } from "./prompt-target"
import { DRAWER_ID, isNetworksShortcut, NETWORKS_SHORTCUT } from "./shortcut"
import { useNetworksUi } from "./store"
import { retryableLazy, retryFailedLoads } from "@/lib/retryable-lazy"
import { cn } from "@/lib/utils"

const NetworksDrawer = retryableLazy(() => import("./NetworksDrawer"))

export function NetworksDrawerHost({ tab }: { tab: GenerateTab }) {
  const { t } = useTranslation()
  useTrackPromptFocus()
  const open = useNetworksUi((s) => s.drawerOpen)
  const setDrawer = useNetworksUi((s) => s.setDrawer)
  const trigger = useRef<HTMLButtonElement>(null)
  // Opened by the user (focus the search) rather than restored after a reload.
  const [userOpened, setUserOpened] = useState(false)
  const toggle = (next: boolean) => {
    setUserOpened(next)
    setDrawer(next)
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!isNetworksShortcut(event) || event.repeat || event.isComposing) return
      // Not over another dialog (the image viewer, a menu's popover): the drawer would open beneath it and take the focus.
      const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"]')
      if ([...dialogs].some((dialog) => !dialog.closest(`#${DRAWER_ID}`))) return
      event.preventDefault()
      const next = !useNetworksUi.getState().drawerOpen
      setUserOpened(next)
      useNetworksUi.getState().setDrawer(next)
      if (!next) trigger.current?.focus()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  const commands = useMemo<Command[]>(
    () => [
      {
        id: "networks.toggle",
        label: "Toggle the networks panel",
        group: "Generate",
        icon: Layers,
        keywords: ["lora", "embedding", "checkpoint", "extra networks"],
        shortcut: NETWORKS_SHORTCUT,
        run: () => {
          const next = !useNetworksUi.getState().drawerOpen
          setUserOpened(next)
          useNetworksUi.getState().setDrawer(next)
        },
      },
    ],
    [],
  )
  useRegisterCommands("networks-drawer", commands)

  const label = open ? t("Hide networks") : t("Show networks")
  return (
    <>
      <HeaderSlot>
        {/* a hairline after the txt2img / img2img tabs: this opens a drawer, it is not a third mode (on a
            phone, which has no room for it, the button's icon alone sets it apart from the tabs' words) */}
        <span aria-hidden className="mx-1 h-5 w-px shrink-0 bg-border max-sm:hidden" />
        <Button
          ref={trigger}
          variant="ghost"
          size="sm"
          aria-pressed={open}
          aria-controls={open ? DRAWER_ID : undefined}
          aria-keyshortcuts="Alt+N"
          title={`${label} (${formatShortcut(NETWORKS_SHORTCUT)})`}
          onClick={() => toggle(!open)}
          // below lg, its label hidden, a 28 px square like the header's other icon buttons
          className={cn("gap-1.5 max-lg:size-7 max-lg:px-0", PRESSED_TINT_CLASS)}
        >
          <Layers aria-hidden />
          <span className="hidden lg:inline">{t("Networks")}</span>
          {/* the hint needs room: at 1280 px the header also holds the full model bar */}
          <Kbd className="hidden 2xl:inline-flex touch-only:hidden!">{formatShortcut(NETWORKS_SHORTCUT)}</Kbd>
        </Button>
      </HeaderSlot>
      {/* The drawer stays open across reloads: one that fails to render closes (and says so)
          rather than taking Generate down with it on every load. */}
      {open ? (
        <ErrorBoundary
          fallback={null}
          onError={(error) => {
            toggle(false)
            toast.error(t("The networks panel could not be shown"), {
              description: renderFailureText(error),
              // opening it again loads a chunk that failed afresh (retryFailedLoads below)
              action: { label: t("Retry"), onClick: () => toggle(true) },
            })
            // a chunk that failed to load is fetched again when the panel is next opened (closed, it does not render)
            retryFailedLoads()
          }}
        >
          <Suspense fallback={null}>
            <NetworksDrawer tab={tab} autoFocus={userOpened} onClose={() => toggle(false)} returnFocus={() => trigger.current?.focus()} />
          </Suspense>
        </ErrorBoundary>
      ) : null}
    </>
  )
}
