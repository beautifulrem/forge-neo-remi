import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { DiagnosticsView } from "./DiagnosticsView"

const DIAG_PAYLOAD = {
  api_version: "1.0",
  schema_version: 1,
  summary: { total: 2, healthy: 1, degraded: 1, error: 0, disabled: 0, with_manifest: 2 },
  items: [
    {
      name: "sd_forge_image_stitch",
      enabled: true,
      builtin: true,
      health: "healthy",
      load_time_ms: 9.5,
      hook_warnings: 0,
      hook_errors: 0,
      action_errors: 0,
      action_timeouts: 0,
      manifest_status: "valid",
      manifest: { api_version: "1.0", requires_capabilities: ["multi_reference"], license: "MIT" },
    },
    {
      name: "sd_forge_spectrum",
      enabled: true,
      builtin: true,
      health: "degraded",
      load_time_ms: 23.1,
      hook_warnings: 2,
      hook_errors: 0,
      action_errors: 0,
      action_timeouts: 0,
      last_error: "Skipped incompatible schedule",
      manifest_status: "valid",
      manifest: { api_version: "1.0", requires_capabilities: ["spectrum"] },
    },
  ],
}

const WORKERS_PAYLOAD = {
  mode: "pool",
  workers: [
    {
      worker_id: "gpu-0",
      pid: 1234,
      state: "idle",
      backend: "pytorch",
      device: "cuda:0",
      gpu_name: "RTX 4090",
      vram_total_mb: 24576,
      vram_free_mb: 18000,
      loaded_model: "anima-2b",
      quant_formats: ["bf16", "fp8"],
      jobs_completed: 7,
      jobs_failed: 0,
      restarts: 1,
    },
  ],
  mlx: {
    available: true,
    platform_supported: true,
    mlx_version: "0.24.0",
    unified_memory_mb: 32768,
    supported_families: ["flux3", "qwen21"],
  },
}

describe("Extensions Diagnostics & Workers panel", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("renders REMI_API_VERSION, extension diagnostics, worker pool and restart button", async () => {
    const restarted: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url
      const method = init?.method ?? (typeof input === "string" ? "GET" : (input as Request).method)
      if (url.includes("/extensions/diagnostics")) return new Response(JSON.stringify(DIAG_PAYLOAD), { status: 200 })
      if (url.endsWith("/workers")) return new Response(JSON.stringify(WORKERS_PAYLOAD), { status: 200 })
      if (url.includes("/workers/gpu-0/restart") && method === "POST") {
        restarted.push("gpu-0")
        return new Response(JSON.stringify({ ok: true, worker: WORKERS_PAYLOAD.workers[0] }), { status: 200 })
      }
      return new Response("{}", { status: 404 })
    })

    render(
      <QueryClientProvider client={queryClient}>
        <DiagnosticsView />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/Remi API version: 1\.0/)).toBeInTheDocument()
    expect(screen.getByText("Multi-GPU pool", { exact: false })).toBeInTheDocument()
    expect(screen.getByText("gpu-0")).toBeInTheDocument()
    expect(screen.getByText("MLX available")).toBeInTheDocument()
    expect(screen.getByText("sd_forge_image_stitch")).toBeInTheDocument()
    expect(screen.getByText("9.5 ms")).toBeInTheDocument()
    expect(screen.getByText("Skipped incompatible schedule")).toBeInTheDocument()

    // Filter to issues only hides the healthy extension.
    fireEvent.click(screen.getByRole("button", { name: "Issues only" }))
    expect(screen.queryByText("sd_forge_image_stitch")).not.toBeInTheDocument()
    expect(screen.getByText("sd_forge_spectrum")).toBeInTheDocument()

    // Restart worker triggers POST /api/v2/workers/gpu-0/restart.
    fireEvent.click(screen.getByRole("button", { name: "Restart worker" }))
    await waitFor(() => expect(restarted).toEqual(["gpu-0"]))
  })
})
