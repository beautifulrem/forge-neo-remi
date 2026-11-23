/**
 * Pure aspect-ratio & resolution telemetry helpers for the empty Generate stage
 * darkroom viewfinder. Kept separate from `Stage.tsx` to satisfy
 * `react/only-export-components` and allow fast unit tests.
 */

export type ViewfinderMetrics = {
  width: number
  height: number
  ratio: string
  megapixels: string
}

function gcd(a: number, b: number): number {
  let x = Math.abs(Math.round(a))
  let y = Math.abs(Math.round(b))
  while (y !== 0) {
    const t = x % y
    x = y
    y = t
  }
  return x || 1
}

function positiveInt(value: unknown, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value)
  if (!Number.isFinite(n) || n <= 0) return fallback
  return Math.max(64, Math.min(8192, Math.round(n)))
}

export function viewfinderMetrics(rawWidth: unknown, rawHeight: unknown): ViewfinderMetrics {
  const width = positiveInt(rawWidth, 1024)
  const height = positiveInt(rawHeight, 1024)
  const d = gcd(width, height)
  const rw = width / d
  const rh = height / d
  const ratio =
    rw <= 32 && rh <= 32
      ? `${rw}:${rh}`
      : width >= height
        ? `${(width / height).toFixed(2)}:1`
        : `1:${(height / width).toFixed(2)}`
  const megapixels = `${((width * height) / 1_000_000).toFixed(2)} MP`
  return { width, height, ratio, megapixels }
}
