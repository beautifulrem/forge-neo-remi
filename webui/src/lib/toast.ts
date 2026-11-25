import { toast } from "sonner"

export type ToastLevel = "info" | "success" | "warning" | "error"

/** A toast description of several notes, one per line (none: undefined). */
export function noteLines(...notes: (string | null | undefined | false)[]): string | undefined {
  const lines = notes.filter((note): note is string => typeof note === "string" && note !== "")
  return lines.length > 0 ? lines.join("\n") : undefined
}

/** Shows a toast for a server-provided level (`undefined` = info). */
export function showToast(level: ToastLevel | undefined, message: string, description?: string) {
  const fn = level && level !== "info" ? toast[level] : toast.info
  fn(message, description ? { description } : undefined)
}
