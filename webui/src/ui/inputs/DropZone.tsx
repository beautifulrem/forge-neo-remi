/**
 * Focusable drop target: drag & drop, click/Enter to browse, paste while focused or hovered.
 *
 * Its name is its label followed by what it shows (WCAG 2.5.3, Label in Name: someone who speaks
 * to the computer says the words on the screen). `aria-label` is read out before the content
 * instead of replacing it; a label elsewhere (`aria-labelledby`) is followed by the zone itself.
 */
import { type ReactNode, useEffect, useEffectEvent, useId, useRef, useState } from "react"
import { cn } from "@/lib/utils"

type DropZoneProps = {
  onFiles: (files: File[]) => void
  accept?: string
  multiple?: boolean
  disabled?: boolean
  id?: string
  "aria-label"?: string
  "aria-labelledby"?: string
  "aria-describedby"?: string
  "aria-invalid"?: boolean
  className?: string
  children: ReactNode
}

export function DropZone(props: DropZoneProps) {
  const input = useRef<HTMLInputElement>(null)
  const zone = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState(false)
  const hovered = useRef(false)
  const ownId = useId()
  const id = props.id ?? ownId
  const { onFiles, disabled } = props

  // Paste goes to the zone under the pointer too, not only the focused one. One listener for the
  // zone's life: it reads the latest props (a caller's inline onFiles is new on every render).
  const onPaste = useEffectEvent((event: ClipboardEvent) => {
    if (disabled || !zone.current) return
    const focused = zone.current.contains(document.activeElement)
    if (!focused && !hovered.current) return
    const files = Array.from(event.clipboardData?.files ?? [])
    if (files.length === 0) return
    event.preventDefault()
    onFiles(props.multiple ? files : files.slice(0, 1))
  })
  useEffect(() => {
    const listener = (event: ClipboardEvent) => onPaste(event)
    window.addEventListener("paste", listener)
    return () => window.removeEventListener("paste", listener)
  }, [])

  const browse = () => {
    if (!disabled) input.current?.click()
  }

  return (
    <div
      ref={zone}
      id={id}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-labelledby={props["aria-labelledby"] ? `${props["aria-labelledby"]} ${id}` : undefined}
      aria-describedby={props["aria-describedby"]}
      aria-invalid={props["aria-invalid"] || undefined}
      aria-disabled={disabled || undefined}
      data-dragging={dragging || undefined}
      onClick={browse}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          browse()
        }
      }}
      onPointerEnter={() => (hovered.current = true)}
      onPointerLeave={() => (hovered.current = false)}
      onDragOver={(event) => {
        if (disabled) return
        event.preventDefault()
        event.dataTransfer.dropEffect = "copy"
        setDragging(true)
      }}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as globalThis.Node | null)) setDragging(false)
      }}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        if (disabled) return
        const files = Array.from(event.dataTransfer.files)
        if (files.length > 0) onFiles(props.multiple ? files : files.slice(0, 1))
      }}
      className={cn(
        "relative cursor-pointer rounded-lg border border-dashed border-input outline-none transition-colors",
        "hover:border-ring/60 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-focus-ring",
        "data-dragging:border-primary data-dragging:bg-primary/5",
        "aria-invalid:border-destructive aria-invalid:not-focus-visible:ring-3 aria-invalid:not-focus-visible:ring-destructive/20",
        disabled && "pointer-events-none opacity-60",
        props.className,
      )}
    >
      {props["aria-label"] && !props["aria-labelledby"] ? <span className="sr-only">{props["aria-label"]}</span> : null}
      {props.children}
      <input
        ref={input}
        type="file"
        hidden
        accept={props.accept}
        multiple={props.multiple}
        onChange={(event) => {
          const files = Array.from(event.target.files ?? [])
          event.target.value = ""
          if (files.length > 0) onFiles(files)
        }}
      />
    </div>
  )
}
