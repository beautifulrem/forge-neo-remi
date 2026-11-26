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

/**
 * The calls through the workbench transport (`workbenchRequest(path, { method })`), as "METHOD /api/v2/path"
 * (the transport's default method is GET), and the call sites whose path or method is not written out (a
 * variable, a helper's result): this test cannot check those, so it lists them instead of skipping them.
 */
function workbenchCalls(raw: string): { calls: string[]; unread: string[] } {
  const source = raw.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "")
  const calls: string[] = []
  const unread: string[] = []
  // not its definition, nor a name in an import list
  for (const m of source.matchAll(/(?<!\bfunction\s+)\bworkbenchRequest\b(?!\s*[,}])/g)) {
    let open = m.index + m[0].length
    if (source[open] === "<") {
      // past the type arguments (an arrow's `=>` closes nothing)
      for (let depth = 0; open < source.length; open++) {
        if (source[open] === "<") depth++
        else if (source[open] === ">" && source[open - 1] !== "=" && --depth === 0) {
          open++
          break
        }
      }
    }
    let end = open + 1
    for (let depth = 1; source[open] === "(" && end < source.length && depth > 0; end++) {
      if (source[end] === "(") depth++
      else if (source[end] === ")") depth--
    }
    const args = source[open] === "(" ? source.slice(open + 1, end - 1) : ""
    const path = /^\s*(["`])(\/[^"`]*)\1/.exec(args)?.[2]
    const method = /\bmethod\b\s*(?::\s*([^,}\s]+))?/.exec(args)
    if (!path || (method && !/^"[A-Z]+"$/.test(method[1] ?? ""))) {
      unread.push(source.slice(m.index, source[open] === "(" ? end : m.index + 60).replace(/\s+/g, " "))
      continue
    }
    calls.push(`${method?.[1]?.slice(1, -1) ?? "GET"} /api/v2${path.replace(/\$\{[^}]*\}/g, "{id}")}`)
  }
  return { calls, unread }
}

it("reads every workbench call site, or lists the one it cannot", () => {
  const source = [
    "/** backed by `workbenchRequest` */",
    'import { ApiError, workbenchRequest } from "./workbench-http"',
    "export async function workbenchRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {}",
    'workbenchRequest<\n  Page<Item>\n>("/boards", { method: "PATCH" })',
    "workbenchRequest<() => void>(`/chains/${id}/cancel`, { method: \"POST\" })",
    'workbenchRequest("/recipes")',
    "workbenchRequest(path)",
    'workbenchRequest("/recipes", { method })',
  ].join("\n")
  expect(workbenchCalls(source)).toEqual({
    calls: ["PATCH /api/v2/boards", "POST /api/v2/chains/{id}/cancel", "GET /api/v2/recipes"],
    unread: ["workbenchRequest(path)", 'workbenchRequest("/recipes", { method })'],
  })
})

it("answers every route the SPA calls", async () => {
  const calls = new Set<string>()
  const unread: string[] = []
  let read = 0
  for (const source of Object.values(sources)) {
    for (const m of source.matchAll(/client\.(GET|POST|PUT|PATCH|DELETE)\("(\/api\/v2\/[^"]+)"/g)) calls.add(`${m[1]} ${m[2]}`)
    const workbench = workbenchCalls(source)
    for (const call of workbench.calls) calls.add(call)
    read += workbench.calls.length
    unread.push(...workbench.unread)
  }
  expect(unread).toEqual([])
  // 28 call sites in boards, pipelines, recipes, compare, diagnostics and the upsampler
  expect(read).toBeGreaterThanOrEqual(28)
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
