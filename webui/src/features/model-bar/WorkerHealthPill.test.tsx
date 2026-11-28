import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { keys, queryClient } from "@/api/queries"
import * as session from "@/api/session"
import { WorkerHealthPill } from "./WorkerHealthPill"

describe("WorkerHealthPill", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("fetches via typed api.workers.status sharing keys.workers and renders server note", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          mode: "subprocess",
          dispatch_active: false,
          preview: true,
          note: "Preview only: queued jobs still run in-process.",
          workers: [
            {
              worker_id: "worker-0",
              state: "idle",
              backend: "forge-pytorch",
              device: "mps:0",
              gpu_name: "Apple M4",
              vram_total_mb: 24576,
              vram_free_mb: 16384,
              jobs_completed: 4,
              jobs_failed: 1,
              current_job_id: "job-77",
              restarts: 0,
            },
          ],
          mlx: { available: false, platform_supported: true, reason: "Standby" },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(
      <QueryClientProvider client={queryClient}>
        <WorkerHealthPill />
      </QueryClientProvider>,
    )

    const button = await screen.findByRole("button", { name: /^Worker status/ })
    expect(button).toHaveTextContent("MPS")
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(queryClient.getQueryData(keys.workers)).toBeTruthy()

    fireEvent.click(button)
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveTextContent("Preview only: queued jobs still run in-process.")
    expect(dialog).toHaveTextContent("Failed: 1")
    expect(dialog).toHaveTextContent("Running: job-77")
  })

  it("hides on 404 but stays visible with retry on 500 server error, and triggers sessionRefused on 401", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response("{}", { status: 404 }))
    const { unmount } = render(
      <QueryClientProvider client={queryClient}>
        <WorkerHealthPill />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(1))
    expect(screen.queryByRole("button", { name: /^Worker status/ })).toBeNull()
    unmount()
    queryClient.clear()

    // 500 surfaces degraded indicator + retry in popover.
    fetchSpy.mockResolvedValueOnce(new Response(JSON.stringify({ detail: "worker monitor crashed" }), { status: 500, headers: { "Content-Type": "application/json" } }))
    const { container, unmount: unmount500 } = render(
      <QueryClientProvider client={queryClient}>
        <WorkerHealthPill />
      </QueryClientProvider>,
    )
    const errBtn = await screen.findByRole("button", { name: /^Worker status/ })
    expect(container.querySelector('[data-slot="worker-alert-dot"]')).not.toBeNull()
    fireEvent.click(errBtn)
    expect(await screen.findByRole("button", { name: "Retry" })).toBeVisible()
    unmount500()
    queryClient.clear()

    // 401 routes through transport -> sessionRefused().
    const refused = vi.spyOn(session, "sessionRefused").mockImplementation(() => {})
    fetchSpy.mockResolvedValueOnce(new Response("{}", { status: 401 }))
    render(
      <QueryClientProvider client={queryClient}>
        <WorkerHealthPill />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(refused).toHaveBeenCalledTimes(1))
  })
})
