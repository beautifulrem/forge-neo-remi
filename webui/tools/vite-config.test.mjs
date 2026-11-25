/**
 * What vite.config.ts gives the dev server (`pnpm dev`, `pnpm dev:mock`): the routes it answers
 * itself and the ones it proxies to the WebUI server.
 */
import { describe, expect, it } from "vitest"
import config from "../vite.config.ts"

/** The config `vite` (serve) resolves for `mode`. */
async function serveConfig(mode) {
  return await config({ command: "serve", mode, isSsrBuild: false, isPreview: false })
}

/** Runs the dev server middlewares of the app's own plugins on a GET of `url`. */
async function devGet(resolved, url) {
  const middlewares = []
  for (const plugin of resolved.plugins.flat(Infinity)) {
    if (plugin?.name?.startsWith("remi-") && plugin.apply !== "build" && typeof plugin.configureServer === "function") plugin.configureServer({ middlewares: { use: (fn) => middlewares.push(fn) } })
  }
  const headers = {}
  return await new Promise((resolve, reject) => {
    const res = {
      statusCode: 200,
      setHeader: (name, value) => (headers[name.toLowerCase()] = value),
      end: (body) => resolve({ status: res.statusCode, headers, body: String(body ?? "") }),
    }
    const run = (index) => {
      if (index >= middlewares.length) return resolve({ status: 404, headers, body: "" })
      try {
        middlewares[index]({ url, method: "GET", headers: {} }, res, (error) => (error ? reject(error) : run(index + 1)))
      } catch (error) {
        reject(error)
      }
    }
    run(0)
  })
}

describe("the dev server", () => {
  // Settings → Licenses links /THIRD-PARTY-NOTICES.txt, which only `pnpm build` writes: a 404 in dev
  it.each(["development", "mock"])("answers the third-party notices the Licenses view links to (%s)", async (mode) => {
    const response = await devGet(await serveConfig(mode), "/THIRD-PARTY-NOTICES.txt")
    expect(response.status).toBe(200)
    expect(response.headers["content-type"]).toMatch(/^text\/plain/)
    expect(response.body).toMatch(/pnpm build/)
    expect(response.body).toMatch(/^react \d/m)
    expect(response.body).toMatch(/^shadcn\/ui /m)
  })
})
