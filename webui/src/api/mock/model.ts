/**
 * The mock's model, loaded as the server loads it: a change of the model selection (the
 * checkpoint, its modules or dtype through `PATCH /options`, a UI preset) only selects it
 * (remi/server/options.py, `apply_forge_preset`), and the next job loads it, announcing
 * `model.loading` at its start and `model.loaded` once loaded (`_note_model_loading` and `poll` in
 * remi/jobs/queue.py). `POST /models/reload` loads it at once; `/models/unload` drops it.
 */
import { bootstrap } from "./fixtures"
import { broadcast } from "./socket"

/** What the model is loaded with; the mock has one model per preset (the server keeps one per preset). */
const SELECTION = ["forge_preset", "sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype"] as const

/** How long a load takes. */
const LOAD_MS = 1200

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function selection(): string {
  return JSON.stringify(SELECTION.map((key) => (bootstrap.options as Record<string, unknown>)[key] ?? null))
}

function selected(): string {
  return String(bootstrap.options.sd_model_checkpoint ?? "") || "model"
}

/** The mock starts with the selected model loaded. */
let loaded: { model: string; selection: string } | null = { model: selected(), selection: selection() }

/** The loaded checkpoint (`/server/status`, `/models/loaded`), or null. */
export function loadedModel(): string | null {
  return loaded?.model ?? null
}

/** The checkpoint the next job loads, if the selection changed since the load (`pending_model()`). */
export function pendingModel(): string | null {
  return loaded?.selection === selection() ? null : selected()
}

async function load(model: string): Promise<void> {
  broadcast({ type: "model.loading", stage: "loading", model })
  await delay(LOAD_MS)
  loaded = { model, selection: selection() }
  broadcast({ type: "model.loaded", model })
}

/** At a job's start: loads the selected model when it is not the loaded one. */
export async function loadPendingModel(): Promise<void> {
  const model = pendingModel()
  if (model !== null) await load(model)
}

/** `POST /models/reload`: loads the selected model now, even when it is loaded. */
export async function reloadModel(): Promise<string> {
  await load(selected())
  return selected()
}

/** `POST /models/unload`. */
export function unloadModel(): void {
  loaded = null
}
