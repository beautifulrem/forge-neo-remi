import { onlineManager, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useOptionWrite } from "@/features/model-bar/model-options"
import { api } from "./endpoints"
import { bootstrap, models } from "./mock/fixtures"
import { keys, modelsQuery, queryClient } from "./queries"

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
})
