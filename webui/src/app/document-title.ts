/**
 * The window title: the page's name, then the app's (`Generate · Remi`). index.html carries the
 * app's name alone until the app sets it (a test holds the two together).
 */
export const APP_TITLE = "Remi"

/** The window title of a page (`page`: its name, already translated). */
export function documentTitle(page: string): string {
  return `${page} · ${APP_TITLE}`
}
