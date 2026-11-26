/**
 * The mock preview answers every route the SPA calls, through the typed client (`client.GET(…)`) or
 * the workbench transport (`workbenchRequest(…)`). One it did not (GET /outputs, POST /interrupt,
 * POST /queue/stop-forever) answered 404 "Mock: … not found", and the view reading it showed a load
 * failure that the product does not have. The prompt tools are the exception: they are off in the
 * mock (`features/prompt-tools/queries.ts`), but the upsampler, which has a mock of its own.
 */
import { expect, it } from "vitest"
import { handleMockRequest } from "./server"

const sources = import.meta.glob<string>(["../../**/*.{ts,tsx}", "!../../**/*.test.*", "!../gen/**", "!./**"], { query: "?raw", import: "default", eager: true })
/** A path parameter the mock matches by name, and a value it knows. */
const SAMPLE: Record<string, string> = { tab: "txt2img", page_id: "toolbox", source: "agent-scheduler" }

/** The calls through the workbench transport (`workbenchRequest(path, { method })`), as "METHOD /api/v2/path". */
function workbenchCalls(source: string): string[] {
  return [...source.matchAll(/\bworkbenchRequest\b[^(\n]*\(/g)].flatMap((m) => {
    let depth = 1
    let end = m.index + m[0].length
    for (; end < source.length && depth > 0; end++) {
      if (source[end] === "(") depth++
      else if (source[end] === ")") depth--
    }
    const args = source.slice(m.index + m[0].length, end)
    const path = /^\s*(["`])(\/[^"`]*)\1/.exec(args)?.[2]
    if (!path) return []
    const method = /\bmethod:\s*"([A-Z]+)"/.exec(args)?.[1] ?? "GET"
    return [`${method} /api/v2${path.replace(/\$\{[^}]*\}/g, "{id}")}`]
  })
}

it("answers every route the SPA calls", async () => {
  const calls = new Set<string>()
  for (const source of Object.values(sources)) {
    for (const m of source.matchAll(/client\.(GET|POST|PUT|PATCH|DELETE)\("(\/api\/v2\/[^"]+)"/g)) calls.add(`${m[1]} ${m[2]}`)
    for (const call of workbenchCalls(source)) calls.add(call)
  }
  expect([...calls].filter((call) => call.startsWith("PATCH /api/v2/boards/"))).toEqual(["PATCH /api/v2/boards/{id}"])
  const missing: string[] = []
  for (const call of calls) {
    const [method = "", path = ""] = call.split(" ")
    if (path.startsWith("/api/v2/prompt-tools/") && path !== "/api/v2/prompt-tools/upsample") continue
    const url = `http://localhost${path.replace(/\{([^}]+)\}/g, (_, name: string) => SAMPLE[name] ?? "x")}`
    const init: RequestInit = method === "GET" || method === "DELETE" ? { method } : { method, body: "{}", headers: { "Content-Type": "application/json" } }
    const response = await handleMockRequest(new Request(url, init))
    if (response.status === 404 && (await response.text()).includes("not found")) missing.push(call)
  }
  expect(calls.size).toBeGreaterThan(80)
  expect(missing).toEqual([])
})
