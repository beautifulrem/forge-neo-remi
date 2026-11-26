/** Classes shared by the select's stand-in (`select.tsx`) and its implementation (`select-impl.tsx`). */
export const SELECT_TRIGGER_CLASS =
  "flex w-fit items-center justify-between gap-1.5 rounded-sm border border-transparent bg-field py-2 pr-2 pl-2.5 text-sm whitespace-nowrap transition-[color,background-color,box-shadow] outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-focus-ring disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:not-focus-visible:ring-3 aria-invalid:not-focus-visible:ring-destructive/20 data-placeholder:text-muted-foreground data-[size=default]:h-8 data-[size=sm]:h-7 *:data-[slot=select-value]:block *:data-[slot=select-value]:min-w-0 *:data-[slot=select-value]:truncate hover:bg-field-hover dark:aria-invalid:not-focus-visible:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"

/**
 * A button that sits in a row of selects (header, toolbars) wears their flat field fill. Its hover is
 * repeated under `dark:`: on a ghost button the kit's `dark:hover:bg-muted/50` (`:is(.dark *)`) outranks a
 * plain `hover:` class, and the field went darker under the pointer in the dark theme.
 */
export const FIELD_BUTTON_CLASS = "rounded-sm border-transparent bg-field hover:bg-field-hover dark:hover:bg-field-hover"

export const SELECT_VALUE_CLASS = "flex flex-1 text-left"
