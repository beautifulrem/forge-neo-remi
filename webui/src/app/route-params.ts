/**
 * List filters kept in the page's address (`#/queue?status=failed&q=cat`), so a
 * reload, a shared link and Back bring them back. A write replaces the history
 * entry (a filter is not a page to go back to) and keeps the address's other
 * entries (the view, the tab); a value equal to its default is left out.
 */
import { useCallback, useEffect, useRef, useState } from "react"
import { currentRoute, navigate, routeKey, useRoute } from "./router"

/**
 * The values of `defaults`' keys in the address, and a setter. `defaults` is a
 * module constant: its identity keeps the setter stable.
 */
export function useRouteParams<K extends string>(defaults: Readonly<Record<K, string>>): [Record<K, string>, (patch: Partial<Record<K, string>>) => void] {
  const route = useRoute()
  const page = routeKey(route)
  const keys = Object.keys(defaults) as K[]
  const values = Object.fromEntries(keys.map((key) => [key, route.query?.get(key) ?? defaults[key]])) as Record<K, string>
  const set = useCallback(
    (patch: Partial<Record<K, string>>) => {
      const route = currentRoute()
      if (routeKey(route) !== page) return // the page was left meanwhile: its filters are not the next page's
      const params = new URLSearchParams(route.query)
      for (const [key, value] of Object.entries(patch) as [K, string | undefined][]) {
        if (value === undefined || value === defaults[key]) params.delete(key)
        else params.set(key, value)
      }
      navigate({ ...route, query: params.size > 0 ? params : undefined }, { replace: true })
    },
    [defaults, page],
  )
  return [values, set]
}

/**
 * A search field over a value in the address. The text is the field's own (a
 * list can filter with it at once) and is written after a pause; a value the
 * address gets elsewhere (Back, a link) replaces the text, unless the text
 * already says it (a space being typed stays).
 */
export function useRouteSearch(value: string, write: (text: string) => void, delay = 300): [string, (text: string) => void] {
  const [text, setText] = useState(value)
  const [seen, setSeen] = useState(value)
  if (value !== seen) {
    setSeen(value)
    if (value !== text.trim()) setText(value)
  }
  const writer = useRef(write)
  useEffect(() => {
    writer.current = write
  })
  useEffect(() => {
    const next = text.trim()
    if (next === value) return
    const timer = setTimeout(() => writer.current(next), delay)
    return () => clearTimeout(timer)
  }, [text, value, delay])
  return [text, setText]
}
