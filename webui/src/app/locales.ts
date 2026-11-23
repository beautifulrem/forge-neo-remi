import type { LocaleInfo } from "@/api/types"
import { DEFAULT_LOCALE, languageTag } from "@/i18n"

/** Selectable UI languages: English plus whatever catalogs the server has. */
export function availableLocales(server: readonly LocaleInfo[] | undefined): LocaleInfo[] {
  const seen = new Set<string>([DEFAULT_LOCALE])
  const list: LocaleInfo[] = [{ id: DEFAULT_LOCALE, name: "English" }]
  for (const locale of server ?? []) {
    // the tag activateLocale stores as the current locale ("pt_br" → "pt-BR"), so the menu can mark it
    const id = languageTag(locale.id)
    if (seen.has(id)) continue
    seen.add(id)
    list.push({ id, name: locale.name || nativeName(id) })
  }
  return list
}

function nativeName(id: string): string {
  try {
    return new Intl.DisplayNames([id], { type: "language" }).of(id) ?? id
  } catch {
    return id
  }
}
