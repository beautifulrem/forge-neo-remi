/**
 * The image viewer's controls over its near-black backdrop (frontend.md, the viewer's toggles): white
 * glyphs, a 12 % white wash on hover and a 25 % one when pressed. Each wash is repeated under `dark:`:
 * the viewer's chrome is scoped `dark` (for its focus ring), and there the kit's ghost button's
 * `dark:hover:bg-muted/50` outranked a plain `hover:` class (`:is(.dark *)` adds a class), so the hover
 * was a near-black wash and a pressed toggle looked unpressed under the pointer.
 */
export const ON_BACKDROP = "text-white/85 hover:bg-white/12 hover:text-white dark:hover:bg-white/12"
export const ON_BACKDROP_PRESSED = "aria-pressed:bg-white/25 aria-pressed:text-white dark:aria-pressed:bg-white/25"
