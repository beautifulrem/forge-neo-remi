/**
 * What vite.config.ts gives the dev server (`pnpm dev`, `pnpm dev:mock`): the routes it answers
 * itself and the ones it proxies to the WebUI server.
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import { loadEnv } from "vite"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import config from "../vite.config.ts"

const webui = path.resolve(import.meta.dirname, "..")
const manifest = JSON.parse(readFileSync(path.join(webui, "package.json"), "utf8"))

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
  // loadEnv merges process.env: a VITE_MOCK of the shell running the tests must not decide them
  let saved
  beforeEach(() => {
    saved = process.env.VITE_MOCK
    delete process.env.VITE_MOCK
  })
  afterEach(() => {
    if (saved === undefined) delete process.env.VITE_MOCK
    else process.env.VITE_MOCK = saved
  })

  // `VITE_MOCK=1 vite` is POSIX shell syntax: Windows' cmd.exe cannot run it
  it("runs every package script in any shell: no NAME=value prefix", () => {
    const posixEnv = Object.entries(manifest.scripts).filter(([, command]) => /^\s*[A-Z_][A-Z0-9_]*=\S*\s/.test(command))
    expect(posixEnv).toEqual([])
  })

  it("answers the API in the browser in the mock mode (`pnpm dev:mock`, `vite --mode mock`)", async () => {
    expect(manifest.scripts["dev:mock"]).toBe("vite --mode mock")
    expect(loadEnv("mock", webui, "VITE_").VITE_MOCK).toBe("1")
    expect(loadEnv("development", webui, "VITE_").VITE_MOCK).toBeUndefined()
    const mock = await serveConfig("mock")
    expect(mock.server.proxy).toBeUndefined()
    expect(mock.plugins.flat(Infinity).some((plugin) => plugin?.name === "remi-mock-files")).toBe(true)
    // `VITE_MOCK=1 vite` (the preview Playwright starts or reuses) stays a mock server too
    process.env.VITE_MOCK = "1"
    expect((await serveConfig("development")).server.proxy).toBeUndefined()
  })

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
