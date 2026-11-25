import { Slider as SliderPrimitive } from "@base-ui/react/slider"
import { cn } from "cn"

function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  orientation = "horizontal",
  getAriaLabel,
  getAriaValueText,
  ...props
}: SliderPrimitive.Root.Props & Pick<SliderPrimitive.Thumb.Props, "getAriaLabel" | "getAriaValueText">) {
  // One thumb per value; a scalar value is a single-thumb slider (upstream rendered two).
  const _values = Array.isArray(value)
    ? value
    : Array.isArray(defaultValue)
      ? defaultValue
      : typeof value === "number" || typeof defaultValue === "number"
        ? [value ?? defaultValue]
        : [min, max]

  return (
    <SliderPrimitive.Root
      // A plain `w-full`, so a width the caller gives (`w-18`) replaces it: a `data-horizontal:`
      // variant comes later in the stylesheet and would win over the caller's class.
      className={cn("group/slider", orientation === "vertical" ? "data-vertical:h-full" : "w-full", className)}
      data-slot="slider"
      orientation={orientation}
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      thumbAlignment="edge"
      {...props}
    >
      {/* A fader: a recessed groove filled with the blue pencil up to the value, and a squared cap with a
          centre notch (the switch's shape family), the one raised thing on a form while it is dragged. */}
      <SliderPrimitive.Control className="relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:min-h-40 data-vertical:w-auto data-vertical:flex-col">
        <SliderPrimitive.Track
          data-slot="slider-track"
          className="relative grow overflow-hidden rounded-full bg-(--slider-groove) shadow-[inset_0_1px_1px_var(--slider-groove-shade)] select-none data-horizontal:h-1 data-horizontal:w-full data-vertical:h-full data-vertical:w-1"
        >
          <SliderPrimitive.Indicator
            data-slot="slider-range"
            className="bg-primary/85 transition-colors group-hover/slider:bg-primary select-none data-horizontal:h-full data-vertical:w-full"
          />
        </SliderPrimitive.Track>
        {Array.from({ length: _values.length }, (_, index) => (
          <SliderPrimitive.Thumb
            data-slot="slider-thumb"
            key={index}
            getAriaLabel={getAriaLabel}
            getAriaValueText={getAriaValueText}
            className={cn(
              "group/cap relative block h-4.5 w-2.5 shrink-0 rounded-[2.5px] border border-(--slider-cap-edge) bg-(--slider-cap) shadow-[0_1px_2px_var(--slider-cap-shade)] ring-focus-ring transition-[border-color,box-shadow] select-none",
              "group-data-vertical/slider:h-2.5 group-data-vertical/slider:w-4.5",
              // the notch: across the cap, along the groove's normal
              "before:absolute before:top-1/2 before:left-1/2 before:h-2 before:w-px before:-translate-x-1/2 before:-translate-y-1/2 before:rounded-full before:bg-(--slider-notch) before:transition-colors",
              "group-data-vertical/slider:before:h-px group-data-vertical/slider:before:w-2",
              "after:absolute after:-inset-x-2 after:-inset-y-1.5",
              // the focus lands on the range input inside the thumb: the cap shows it (has-focus-visible)
              "hover:before:bg-primary has-focus-visible:border-primary/70 has-focus-visible:ring-3 has-focus-visible:before:bg-primary active:border-primary/70 active:before:bg-primary data-dragging:border-primary/70 data-dragging:before:bg-primary",
              "disabled:pointer-events-none disabled:opacity-50",
            )}
          />
        ))}
      </SliderPrimitive.Control>
    </SliderPrimitive.Root>
  )
}

export { Slider }
