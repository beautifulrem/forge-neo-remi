import { onlineManager, QueryClientProvider, useQuery } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useOptionWrite } from "@/features/model-bar/model-options"
import { api } from "./endpoints"
import { bootstrap, models } from "./mock/fixtures"
import { keys, modelsQuery, queryClient, workersQuery } from "./queries"

// The server usually runs on this machine: a laptop that drops its Wi-Fi still reaches it, so the
// browser's offline flag must not hold requests back.
describe("the app's query client while the browser reports offline", () => {
  afterEach(() => {
    onlineManager.setOnline(true)
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("still asks the server for data", async () => {
    vi.spyOn(api, "models").mockResolvedValue(models)
    onlineManager.setOnline(false)
    const answer = await Promise.race([queryClient.fetchQuery(modelsQuery), new Promise((resolve) => setTimeout(() => resolve("paused"), 200))])
    expect(answer).toEqual(models)
  })

  it("sends a model switch to the server instead of only showing it", async () => {
    queryClient.setQueryData(keys.app, structuredClone(bootstrap))
    vi.spyOn(api, "app").mockResolvedValue(structuredClone(bootstrap))
    const patch = vi.spyOn(api, "patchOptions").mockResolvedValue({} as never)
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    const { result } = renderHook(() => useOptionWrite<string>("sd_model_checkpoint"), { wrapper })
    onlineManager.setOnline(false)
    act(() => result.current.mutate("new.safetensors"))
    await waitFor(() => expect(patch).toHaveBeenCalledWith({ sd_model_checkpoint: "new.safetensors" }))
    expect(result.current.isPaused).toBe(false)
  })

  // networkMode "always" turns TanStack's refetchOnReconnect off by default: after a short loss of
  // the network that the socket rode out, stale lists stayed as they were until something else
  // asked for them.
  it("asks again for stale data when the browser is back online", async () => {
    const load = vi.spyOn(api, "models").mockResolvedValue(models)
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    renderHook(() => useQuery({ ...modelsQuery, staleTime: 0 }), { wrapper })
    await waitFor(() => expect(load).toHaveBeenCalledTimes(1))
    act(() => onlineManager.setOnline(false))
    act(() => onlineManager.setOnline(true))
    await waitFor(() => expect(load).toHaveBeenCalledTimes(2))
  })

  it("shares the unified ['workers'] query key via api.workers.status", async () => {
    const payload = { mode: "inprocess", dispatch_active: false, preview: false, workers: [], mlx: { available: false, platform_supported: false } }
    const status = vi.spyOn(api.workers, "status").mockResolvedValue(payload as never)
    expect(keys.workers).toEqual(["workers"])
    expect(workersQuery.queryKey).toEqual(["workers"])
    const data = await queryClient.fetchQuery(workersQuery)
    expect(status).toHaveBeenCalledTimes(1)
    expect(data).toEqual(payload)
  })
})
