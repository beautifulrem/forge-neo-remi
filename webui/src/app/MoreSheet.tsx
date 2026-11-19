/** Mobile "More" sheet: the destinations that do not fit the bottom bar. */
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { BottomTab } from "./NavRail"
import { isActive, type NavItem } from "./routes"
import type { Route } from "./router"

export default function MoreSheet({ items, route, onClose }: { items: NavItem[]; route: Route; onClose: () => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(true)
  const close = () => setOpen(false)
  return (
    <Sheet open={open} onOpenChange={(next) => !next && close()} onOpenChangeComplete={(next) => !next && onClose()}>
      <SheetContent side="bottom" className="pb-[max(1rem,env(safe-area-inset-bottom))]">
        <SheetHeader className="pb-2">
          <SheetTitle>{t("More")}</SheetTitle>
        </SheetHeader>
        {/* Four to a row, a short last row centred (no empty cell); the header's X closes the sheet. */}
        <div className="flex flex-wrap justify-center gap-y-2 px-4">
          {items.map((item) => (
            <div key={item.id} className="flex w-1/4">
              <BottomTab item={item} active={isActive(item, route)} onClick={close} />
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  )
}
