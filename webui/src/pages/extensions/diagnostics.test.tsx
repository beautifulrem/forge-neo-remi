import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { keys, queryClient } from "@/api/queries"
import { i18n } from "@/i18n"
import zhCN from "@/locales/zh-CN.json"
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

  it("says why a manifest is incompatible, in the UI language when the reason is a sentence", async () => {
    const template = "requires Remi {{required}} or newer; this is Remi {{running}}"
    i18n.addResource("zh-CN", "server", template, "需要 Remi {{required}} 或更新版本；当前为 Remi {{running}}")
    const item = (name: string, extra: Record<string, unknown>) => ({
      name,
      enabled: true,
      builtin: false,
      health: "error",
      load_time_ms: null,
      hook_warnings: 0,
      hook_errors: 0,
      action_errors: 0,
      action_timeouts: 0,
      manifest_status: "incompatible",
      manifest: { api_version: "1.0" },
      ...extra,
    })
    const payload = {
      ...DIAG_PAYLOAD,
      items: [
        item("ext_new", {
          manifest: { api_version: "1.0", min_remi_version: "9.0" },
          manifest_errors: ["requires Remi 9.0 or newer; this is Remi 1.0.0"],
          manifest_error_details: [{ message: "requires Remi 9.0 or newer; this is Remi 1.0.0", template, params: { required: "9.0", running: "1.0.0" } }],
        }),
        // a server without the details: the English errors as they are
        item("ext_old_server", { manifest_errors: ["min_remi_version 'x' is not valid SemVer"] }),
      ],
    }
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/extensions/diagnostics")) return new Response(JSON.stringify(payload), { status: 200 })
      if (url.endsWith("/workers")) return new Response(JSON.stringify(WORKERS_PAYLOAD), { status: 200 })
      return new Response("{}", { status: 404 })
    })
    await i18n.changeLanguage("zh-CN")
    try {
      render(
        <QueryClientProvider client={queryClient}>
          <DiagnosticsView />
        </QueryClientProvider>,
      )
      expect(await screen.findByText("需要 Remi 9.0 或更新版本；当前为 Remi 1.0.0")).toBeInTheDocument()
      expect(screen.getByText("min_remi_version 'x' is not valid SemVer")).toBeInTheDocument()
    } finally {
      await i18n.changeLanguage("en")
    }
  })

  it("renders recent_errors, disabled_scripts, worker jobs_failed/current_job_id, note, disables inprocess restart, and translates state/mlx reason", async () => {
    i18n.addResourceBundle("zh-CN", "chrome", zhCN, true, true)
    const diagPayload = {
      api_version: "1.0",
      schema_version: 1,
      summary: { total: 2, healthy: 0, degraded: 1, error: 0, disabled: 1, with_manifest: 1 },
      items: [
        {
          name: "sd_forge_spectrum",
          enabled: true,
          builtin: true,
          health: "degraded",
          load_time_ms: 12.0,
          hook_warnings: 1,
          hook_errors: 0,
          action_errors: 0,
          action_timeouts: 0,
          last_error: "Recent hook warning",
          recent_errors: [{ kind: "hook_warning", message: "Recent hook warning", timestamp: "2026-11-27T00:00:00Z" }],
          manifest_status: "valid",
          manifest: { api_version: "1.0" },
          disabled_scripts: [],
        },
        {
          name: "legacy_ext",
          enabled: false,
          builtin: false,
          health: "disabled",
          load_time_ms: 1.0,
          hook_warnings: 0,
          hook_errors: 0,
          action_errors: 0,
          action_timeouts: 0,
          manifest_status: "missing",
          disabled_scripts: [{ kind: "script", id: "legacy", title: "legacy.py", tab: "txt2img", reason: "uses ui()" }],
        },
      ],
    }
    const workersPayload = {
      mode: "inprocess",
      dispatch_active: false,
      preview: false,
      note: "Queued jobs run in-process; restarting resets worker counters without restarting the server.",
      workers: [
        {
          worker_id: "inprocess-0",
          state: "busy",
          backend: "inprocess",
          device: "mps",
          current_job_id: "job-42",
          jobs_completed: 5,
          jobs_failed: 2,
          restarts: 0,
        },
      ],
      mlx: {
        available: false,
        platform_supported: true,
        reason: "Optional mflux sidecar not installed",
      },
    }

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url
      if (url.includes("/extensions/diagnostics")) return new Response(JSON.stringify(diagPayload), { status: 200 })
      if (url.endsWith("/workers")) return new Response(JSON.stringify(workersPayload), { status: 200 })
      return new Response("{}", { status: 404 })
    })

    i18n.addResource("zh-CN", "server", "Optional mflux sidecar not installed", "未安装可选的 mflux 协进程")
    await i18n.changeLanguage("zh-CN")
    try {
      render(
        <QueryClientProvider client={queryClient}>
          <DiagnosticsView />
        </QueryClientProvider>,
      )

      expect(await screen.findByText("inprocess-0")).toBeInTheDocument()
      expect(queryClient.getQueryData(keys.workers)).toEqual(workersPayload)
      expect(screen.getByText(zhCN["Busy"])).toBeInTheDocument()
      expect(screen.queryByText("busy")).toBeNull()
      expect(screen.getByText("未安装可选的 mflux 协进程")).toBeInTheDocument()
      expect(screen.getByText("job-42")).toBeInTheDocument()
      expect(screen.getByText("2")).toBeInTheDocument()
      expect(screen.getByText("Queued jobs run in-process; restarting resets worker counters without restarting the server.")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: zhCN["Restart worker"] })).toBeDisabled()
      expect(screen.getByText(/legacy\.py \(txt2img\).*uses ui\(\)/)).toBeInTheDocument()
      expect(screen.getByText(/Recent hook warning/)).toBeInTheDocument()
    } finally {
      await i18n.changeLanguage("en")
    }
  })
})
