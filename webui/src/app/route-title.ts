/**
 * The page's name in the header and the window title (an English key, translated by the caller):
 * its nav label; a built-in page hidden from the nav ("Hide UI Tabs") but opened by its address keeps
 * its own; an extension page not listed (the page list loading, or a page that is gone) is an
 * extension page. Never the route's id.
 */
import { type Route, routeKey } from "./router"
import { BUILTIN_NAV, type NavItem } from "./routes"

export function routeTitle(route: Route, nav: NavItem[]): string {
  if (route.name === "generate") return "Generate"
  if (route.name === "not-found") return "Not found"
  const key = routeKey(route)
  const item = nav.find((entry) => routeKey(entry.route) === key) ?? BUILTIN_NAV.find((entry) => routeKey(entry.route) === key)
  return item?.label ?? "Extension"
}
