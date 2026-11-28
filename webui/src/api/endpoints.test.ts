/**
 * Every endpoint function sends the method, path, query and body the server's OpenAPI schema
 * (`openapi.json`, exported by tools/export_openapi.py) declares, with the client-side defaults.
 */
import { afterEach, describe, expect, it, vi } from "vitest"
import { declaredInOpenApi, recordFetch } from "@/test/fetch-recorder"
import { api } from "./endpoints"
import { ApiError, client } from "./http"

type Case = {
  call: () => Promise<unknown>
  method: string
  path: string
  body?: unknown
  query?: Record<string, string>
}


const CASES: Record<string, Case> = {
  app: { call: () => api.app(), method: "GET", path: "/api/v2/app" },
  "auth.status": { call: () => api.auth.status(), method: "GET", path: "/api/v2/auth/status" },
  "auth.login": { call: () => api.auth.login("ann", "pw"), method: "POST", path: "/api/v2/auth/login", body: { username: "ann", password: "pw" } },
  "auth.logout": { call: () => api.auth.logout(), method: "POST", path: "/api/v2/auth/logout" },
  schema: { call: () => api.schema("img2img"), method: "GET", path: "/api/v2/schema/img2img" },
  pageSchema: { call: () => api.pageSchema("prompt workshop"), method: "GET", path: "/api/v2/pages/prompt%20workshop" },
  action: {
    call: () => api.action("page:x", "refresh", { values: { a: 1 } }),
    method: "POST",
    path: "/api/v2/actions/page%3Ax/refresh",
    body: { values: { a: 1 } },
  },
  "jobs.create": {
    call: () => api.jobs.create({ tab: "txt2img", core: {}, scripts: {}, priority: 2 }),
    method: "POST",
    path: "/api/v2/jobs",
    body: { tab: "txt2img", core: {}, scripts: {}, priority: 2, repeat: 1, forever: false },
  },
  "jobs.list": { call: () => api.jobs.list({ status: "queued", limit: 5 }), method: "GET", path: "/api/v2/jobs", query: { status: "queued", limit: "5" } },
  "jobs.get": { call: () => api.jobs.get("j/1"), method: "GET", path: "/api/v2/jobs/j%2F1" },
  "jobs.remove": { call: () => api.jobs.remove("j1"), method: "DELETE", path: "/api/v2/jobs/j1" },
  "jobs.interrupt": { call: () => api.jobs.interrupt("j1"), method: "POST", path: "/api/v2/jobs/j1/interrupt" },
  "jobs.skip": { call: () => api.jobs.skip("j1"), method: "POST", path: "/api/v2/jobs/j1/skip" },
  "jobs.retry": { call: () => api.jobs.retry("j1"), method: "POST", path: "/api/v2/jobs/j1/retry" },
  "jobs.duplicate": { call: () => api.jobs.duplicate("j1"), method: "POST", path: "/api/v2/jobs/j1/duplicate" },
  "jobs.update": { call: () => api.jobs.update("j1", { forever: true }), method: "PATCH", path: "/api/v2/jobs/j1", body: { forever: true } },
  "jobs.updateParams": {
    call: () => api.jobs.updateParams("j1", { core: { steps: 3 }, scripts: { seed: { seed: 1 } } }),
    method: "PUT",
    path: "/api/v2/jobs/j1/params",
    body: { core: { steps: 3 }, scripts: { seed: { seed: 1 } } },
  },
  "jobs.star": { call: () => api.jobs.star(["a", "b"], true), method: "POST", path: "/api/v2/jobs/star", body: { ids: ["a", "b"], starred: true } },
  "jobs.clearHistory": { call: () => api.jobs.clearHistory(), method: "POST", path: "/api/v2/jobs/clear-history", body: { keep_starred: true } },
  "queue.items": { call: () => api.queue.items(), method: "GET", path: "/api/v2/queue/items" },
  "queue.status": { call: () => api.queue.status(), method: "GET", path: "/api/v2/queue" },
  "queue.pause": { call: () => api.queue.pause(), method: "POST", path: "/api/v2/queue/pause" },
  "queue.resume": { call: () => api.queue.resume(), method: "POST", path: "/api/v2/queue/resume" },
  "queue.reorder": { call: () => api.queue.reorder(["b", "a"]), method: "POST", path: "/api/v2/queue/reorder", body: { order: ["b", "a"] } },
  "queue.move": {
    call: () => api.queue.move({ id: "a", after: "b", priority: 1 }),
    method: "POST",
    path: "/api/v2/queue/move",
    body: { id: "a", after: "b", priority: 1 },
  },
  "queue.stopForever": { call: () => api.queue.stopForever(), method: "POST", path: "/api/v2/queue/stop-forever" },
  "queue.interruptAll": { call: () => api.queue.interruptAll(), method: "POST", path: "/api/v2/interrupt", body: { clear_queue: false } },
  "queue.importSources": { call: () => api.queue.importSources(), method: "GET", path: "/api/v2/queue/import" },
  "queue.importFrom": { call: () => api.queue.importFrom("agent-scheduler"), method: "POST", path: "/api/v2/queue/import/agent-scheduler" },
  "gallery.list": { call: () => api.gallery.list({ q: "cat", starred: true }), method: "GET", path: "/api/v2/gallery", query: { q: "cat", starred: "true" } },
  "gallery.facets": { call: () => api.gallery.facets(), method: "GET", path: "/api/v2/gallery/facets" },
  "gallery.get": { call: () => api.gallery.get("f1"), method: "GET", path: "/api/v2/gallery/f1" },
  "gallery.star": { call: () => api.gallery.star(["f1"], false), method: "POST", path: "/api/v2/gallery/star", body: { ids: ["f1"], starred: false } },
  "gallery.remove": { call: () => api.gallery.remove(["f1"]), method: "POST", path: "/api/v2/gallery/delete", body: { ids: ["f1"], delete_files: false } },
  "gallery.zip": { call: () => api.gallery.zip(["f1", "f2"]), method: "POST", path: "/api/v2/gallery/zip", body: { ids: ["f1", "f2"] } },
  "gallery.reveal": { call: () => api.gallery.reveal("f1"), method: "POST", path: "/api/v2/gallery/f1/reveal" },
  "gallery.lineage": { call: () => api.gallery.lineage("f/1"), method: "GET", path: "/api/v2/gallery/f%2F1/lineage" },
  "gallery.compare": {
    call: () => api.gallery.compare({ id_a: "f1", id_b: "f2" }),
    method: "POST",
    path: "/api/v2/gallery/compare",
    body: { id_a: "f1", id_b: "f2" },
  },
  "boards.list": { call: () => api.boards.list(), method: "GET", path: "/api/v2/boards" },
  "boards.get": { call: () => api.boards.get("b 1"), method: "GET", path: "/api/v2/boards/b%201" },
  "boards.create": { call: () => api.boards.create({ name: "Mood" }), method: "POST", path: "/api/v2/boards", body: { name: "Mood", description: "" } },
  "boards.update": { call: () => api.boards.update("b1", { name: "Mood 2" }), method: "PATCH", path: "/api/v2/boards/b1", body: { name: "Mood 2" } },
  "boards.delete": { call: () => api.boards.delete("b1"), method: "DELETE", path: "/api/v2/boards/b1" },
  "boards.addItems": { call: () => api.boards.addItems("b1", ["f1"]), method: "POST", path: "/api/v2/boards/b1/items", body: { file_ids: ["f1"] } },
  "boards.removeItems": { call: () => api.boards.removeItems("b1", ["f1"]), method: "POST", path: "/api/v2/boards/b1/items/remove", body: { file_ids: ["f1"] } },
  "boards.removeItem": { call: () => api.boards.removeItem("b1", "f1"), method: "DELETE", path: "/api/v2/boards/b1/items/f1" },
  "chains.list": { call: () => api.chains.list({ status: "completed", limit: 10 }), method: "GET", path: "/api/v2/chains", query: { status: "completed", limit: "10" } },
  "chains.templates": { call: () => api.chains.templates(), method: "GET", path: "/api/v2/chains/templates" },
  "chains.getTemplate": { call: () => api.chains.getTemplate("t1"), method: "GET", path: "/api/v2/chains/templates/t1" },
  "chains.createTemplate": {
    call: () => api.chains.createTemplate({ name: "Base", stages: [{ kind: "generate", tab: "txt2img" }] }),
    method: "POST",
    path: "/api/v2/chains/templates",
    body: {
      name: "Base",
      description: "",
      tags: [],
      stages: [{ kind: "generate", name: "", enabled: true, tab: "txt2img", core: {}, scripts: {}, config: {} }],
    },
  },
  "chains.deleteTemplate": { call: () => api.chains.deleteTemplate("t1"), method: "DELETE", path: "/api/v2/chains/templates/t1" },
  "chains.preview": {
    call: () => api.chains.preview({ stages: [{ kind: "generate" }], chain_id: "c1" }),
    method: "POST",
    path: "/api/v2/chains/preview",
    body: {
      chain_id: "c1",
      stages: [{ kind: "generate", name: "", enabled: true, tab: "", core: {}, scripts: {}, config: {} }],
    },
  },
  "chains.create": {
    call: () => api.chains.create({ name: "Run", stages: [{ kind: "generate" }] }),
    method: "POST",
    path: "/api/v2/chains",
    body: {
      name: "Run",
      run_now: true,
      priority: 0,
      stages: [{ kind: "generate", name: "", enabled: true, tab: "", core: {}, scripts: {}, config: {} }],
    },
  },
  "chains.get": { call: () => api.chains.get("c1"), method: "GET", path: "/api/v2/chains/c1" },
  "chains.rerun": {
    call: () => api.chains.rerun("c1", { from_stage: 1 }),
    method: "POST",
    path: "/api/v2/chains/c1/rerun",
    body: { from_stage: 1, force_stages: [] },
  },
  "chains.pause": { call: () => api.chains.pause("c1"), method: "POST", path: "/api/v2/chains/c1/pause" },
  "chains.resume": { call: () => api.chains.resume("c1"), method: "POST", path: "/api/v2/chains/c1/resume" },
  "chains.cancel": { call: () => api.chains.cancel("c1"), method: "POST", path: "/api/v2/chains/c1/cancel" },
  "chains.delete": { call: () => api.chains.delete("c1"), method: "DELETE", path: "/api/v2/chains/c1" },
  "recipes.list": { call: () => api.recipes.list({ model_family: "anima", q: "fox" }), method: "GET", path: "/api/v2/recipes", query: { model_family: "anima", q: "fox" } },
  "recipes.get": { call: () => api.recipes.get("r1"), method: "GET", path: "/api/v2/recipes/r1" },
  "recipes.create": {
    call: () => api.recipes.create({ name: "Balanced", core_patch: { cfg_scale: 4 } }),
    method: "POST",
    path: "/api/v2/recipes",
    body: { name: "Balanced", description: "", capabilities: [], tab: "txt2img", core_patch: { cfg_scale: 4 }, scripts_patch: {}, tags: [] },
  },
  "recipes.update": { call: () => api.recipes.update("r1", { name: "Balanced v2" }), method: "PATCH", path: "/api/v2/recipes/r1", body: { name: "Balanced v2" } },
  "recipes.delete": { call: () => api.recipes.delete("r1"), method: "DELETE", path: "/api/v2/recipes/r1" },
  "recipes.apply": {
    call: () => api.recipes.apply("r1", { tab: "txt2img" }),
    method: "POST",
    path: "/api/v2/recipes/r1/apply",
    body: { tab: "txt2img", core: {}, scripts: {}, check_conflicts: true },
  },
  "recipes.exportRecipes": { call: () => api.recipes.exportRecipes(), method: "GET", path: "/api/v2/recipes/export" },
  "recipes.importRecipes": {
    call: () => api.recipes.importRecipes({ recipes: [{ name: "X" }] }),
    method: "POST",
    path: "/api/v2/recipes/import",
    body: { version: 1, overwrite: false, recipes: [{ name: "X" }] },
  },
  "workers.status": { call: () => api.workers.status(), method: "GET", path: "/api/v2/workers" },
  "workers.get": { call: () => api.workers.get("worker-0"), method: "GET", path: "/api/v2/workers/worker-0" },
  "workers.restart": { call: () => api.workers.restart("worker-0"), method: "POST", path: "/api/v2/workers/worker-0/restart" },
  "promptTools.config": { call: () => api.promptTools.config(), method: "GET", path: "/api/v2/prompt-tools/config" },
  "promptTools.refresh": { call: () => api.promptTools.refresh(), method: "POST", path: "/api/v2/prompt-tools/refresh" },
  "promptTools.upsample": {
    call: () => api.promptTools.upsample({ prompt: "red fox" }),
    method: "POST",
    path: "/api/v2/prompt-tools/upsample",
    body: {
      prompt: "red fox",
      reference_file_ids: [],
      reference_images: [],
      mode: "auto",
      current: "",
      negative: false,
      instructions: "",
      temperature: 0.7,
      max_tokens: 1024,
    },
  },
  scanOutputs: { call: () => api.scanOutputs(), method: "POST", path: "/api/v2/outputs/refresh" },
  pngInfo: { call: () => api.pngInfo("f1"), method: "POST", path: "/api/v2/pnginfo", body: { file_id: "f1" } },
  outputs: { call: () => api.outputs({ q: "x", cursor: null }), method: "GET", path: "/api/v2/outputs", query: { q: "x" } },
  saveOutputs: { call: () => api.saveOutputs({ ids: ["f1"] }), method: "POST", path: "/api/v2/outputs/save", body: { ids: ["f1"], zip: false } },
  "networks.kinds": { call: () => api.networks.kinds(), method: "GET", path: "/api/v2/networks" },
  "networks.list": { call: () => api.networks.list("lora", { q: "a" }), method: "GET", path: "/api/v2/networks/lora", query: { q: "a" } },
  "networks.facets": { call: () => api.networks.facets("lora"), method: "GET", path: "/api/v2/networks/lora/facets" },
  "networks.editMeta": {
    call: () => api.networks.editMeta("lora", "dir/a", { notes: null }),
    method: "PATCH",
    path: "/api/v2/networks/lora/dir%2Fa/meta",
    body: { user_metadata: { notes: null }, replace: false },
  },
  "networks.setPreview": {
    call: () => api.networks.setPreview("lora", "a", "f1"),
    method: "PUT",
    path: "/api/v2/networks/lora/a/preview",
    body: { file_id: "f1" },
  },
  "networks.openFolder": { call: () => api.networks.openFolder("lora", "a"), method: "POST", path: "/api/v2/networks/lora/a/open-folder" },
  "networks.refresh(kind)": { call: () => api.networks.refresh("lora"), method: "POST", path: "/api/v2/networks/lora/refresh" },
  "networks.refresh()": { call: () => api.networks.refresh(), method: "POST", path: "/api/v2/networks/refresh" },
  options: { call: () => api.options(), method: "GET", path: "/api/v2/options" },
  patchOptions: { call: () => api.patchOptions({ a: 1 }), method: "PATCH", path: "/api/v2/options", body: { a: 1 } },
  "parseInfotext(tab)": { call: () => api.parseInfotext("x", "img2img"), method: "POST", path: "/api/v2/infotext/parse", body: { text: "x", tab: "img2img" } },
  "parseInfotext()": { call: () => api.parseInfotext("x", null), method: "POST", path: "/api/v2/infotext/parse", body: { text: "x" } },
  models: { call: () => api.models(), method: "GET", path: "/api/v2/models" },
  countTokens: {
    call: () => api.countTokens({ text: "a cat" }),
    method: "POST",
    path: "/api/v2/prompt/tokens",
    body: { text: "a cat", steps: 20, styles: [], negative: false },
  },
  "imageInfo(tab)": { call: () => api.imageInfo("f1", "txt2img"), method: "POST", path: "/api/v2/infotext/image", body: { file_id: "f1", tab: "txt2img" } },
  "imageInfo()": { call: () => api.imageInfo("f1"), method: "POST", path: "/api/v2/infotext/image", body: { file_id: "f1" } },
  i18n: { call: () => api.i18n("zh-CN"), method: "GET", path: "/api/v2/i18n/zh-CN" },
  "styles.list": { call: () => api.styles.list(), method: "GET", path: "/api/v2/styles" },
  "styles.apply": {
    call: () => api.styles.apply({ prompt: "p", negative_prompt: "n", styles: ["s"] }),
    method: "POST",
    path: "/api/v2/styles/apply",
    body: { prompt: "p", negative_prompt: "n", styles: ["s"] },
  },
  "styles.save": {
    call: () => api.styles.save({ name: "my style", prompt: "p", negative_prompt: "n" }),
    method: "PUT",
    path: "/api/v2/styles/my%20style",
    body: { prompt: "p", negative_prompt: "n" },
  },
  "styles.remove": { call: () => api.styles.remove("s"), method: "DELETE", path: "/api/v2/styles/s" },
  validateOptions: { call: () => api.validateOptions({ a: 1 }), method: "POST", path: "/api/v2/options/validate", body: { a: 1 } },
  "extensions.list": { call: () => api.extensions.list(), method: "GET", path: "/api/v2/extensions" },
  "extensions.diagnostics": { call: () => api.extensions.diagnostics(), method: "GET", path: "/api/v2/extensions/diagnostics" },
  "extensions.configure": {
    call: () => api.extensions.configure({ disable_all: "extra" }),
    method: "PATCH",
    path: "/api/v2/extensions",
    body: { disable_all: "extra" },
  },
  "extensions.check": { call: () => api.extensions.check("ext"), method: "POST", path: "/api/v2/extensions/ext/check" },
  "extensions.update": { call: () => api.extensions.update(["ext"]), method: "POST", path: "/api/v2/extensions/update", body: { names: ["ext"] } },
  "extensions.install": {
    call: () => api.extensions.install({ url: "https://example.com/x.git" }),
    method: "POST",
    path: "/api/v2/extensions/install",
    body: { url: "https://example.com/x.git" },
  },
  "extensions.uninstall": { call: () => api.extensions.uninstall("ext"), method: "DELETE", path: "/api/v2/extensions/ext" },
  "extensions.index": { call: () => api.extensions.index({ refresh: true }), method: "GET", path: "/api/v2/extensions/index", query: { refresh: "true" } },
  "extensions.states.list": { call: () => api.extensions.states.list(), method: "GET", path: "/api/v2/extensions/config-states" },
  "extensions.states.get": { call: () => api.extensions.states.get("s1"), method: "GET", path: "/api/v2/extensions/config-states/s1" },
  "extensions.states.save": { call: () => api.extensions.states.save("b"), method: "POST", path: "/api/v2/extensions/config-states", body: { name: "b" } },
  "extensions.states.remove": { call: () => api.extensions.states.remove("s1"), method: "DELETE", path: "/api/v2/extensions/config-states/s1" },
  "extensions.states.restore": {
    call: () => api.extensions.states.restore("s1"),
    method: "POST",
    path: "/api/v2/extensions/config-states/s1/restore",
    body: { what: "extensions" },
  },
  "system.status": { call: () => api.system.status(), method: "GET", path: "/api/v2/server/status" },
  "system.reload": { call: () => api.system.reload(), method: "POST", path: "/api/v2/server/reload" },
  "system.restart": { call: () => api.system.restart(), method: "POST", path: "/api/v2/server/restart" },
  "system.ping": { call: () => api.system.ping(), method: "GET", path: "/internal/ping" },
  "system.sysinfo": { call: () => api.system.sysinfo(), method: "GET", path: "/api/v2/sysinfo" },
  "system.checkSysinfo": { call: () => api.system.checkSysinfo("text"), method: "POST", path: "/api/v2/sysinfo/check", body: { text: "text" } },
  "system.licenses": { call: () => api.system.licenses(), method: "GET", path: "/api/v2/licenses" },
  "system.unloadModel": { call: () => api.system.unloadModel(), method: "POST", path: "/api/v2/models/unload" },
  "system.reloadModel": { call: () => api.system.reloadModel(), method: "POST", path: "/api/v2/models/reload", body: { force: false } },
  "system.loadedModels": { call: () => api.system.loadedModels(), method: "GET", path: "/api/v2/models/loaded" },
  "system.hashStatus": { call: () => api.system.hashStatus(), method: "GET", path: "/api/v2/models/hashes" },
  "system.hashCheckpoints": { call: () => api.system.hashCheckpoints(4), method: "POST", path: "/api/v2/models/hashes", body: { threads: 4 } },
}

const request = { tab: "txt2img" as const, core: {}, scripts: {} }
const created = { job_id: "j1", job_ids: ["j1"], seq: 1 }

describe("api endpoints", () => {
  afterEach(() => vi.restoreAllMocks())

  it.each(Object.entries(CASES))("%s sends the declared request", async (_name, { call, method, path, body, query }) => {
    const fetch = recordFetch()
    await call()
    expect(fetch.requests).toHaveLength(1)
    const sent = fetch.last()
    expect({ method: sent.method, path: sent.path }).toEqual({ method, path })
    expect(declaredInOpenApi(sent.method, sent.path), `${method} ${path} is not in openapi.json`).toBe(true)
    expect(sent.body).toEqual(body)
    expect(Object.fromEntries(sent.query)).toEqual(query ?? {})
  })

  it("covers every endpoint function", () => {
    const names: string[] = []
    const walk = (value: object, prefix: string) => {
      for (const [key, item] of Object.entries(value)) {
        if (typeof item === "function") names.push(prefix + key)
        else if (item && typeof item === "object") walk(item as object, `${prefix}${key}.`)
      }
    }
    walk(api, "")
    // URL builders and the multipart upload are tested below, networks.meta in http.test.ts.
    const urlBuilders = new Set(["fileUrl", "downloadUrl", "thumbUrl", "userAssetUrl", "upload", "networks.meta", "system.sysinfoDownloadUrl"])
    const tested = new Set(Object.keys(CASES).map((name) => name.replace(/\(.*\)$/, "")))
    expect(names.filter((name) => !tested.has(name) && !urlBuilders.has(name))).toEqual([])
  })

  it("uploads a file as multipart form data, named after the file", async () => {
    const fetch = recordFetch(() => ({ status: 201, body: { id: "u1" } }))
    // jsdom's File is not undici's, so the multipart body is checked before the Request encodes it
    const post = vi.spyOn(client, "POST")
    await expect(api.upload(new File(["png"], "cat.png", { type: "image/png" }))).resolves.toEqual({ id: "u1" })
    const sent = fetch.last()
    expect({ method: sent.method, path: sent.path }).toEqual({ method: "POST", path: "/api/v2/uploads" })
    expect(sent.contentType).toMatch(/^multipart\/form-data; boundary=/)
    const form = (post.mock.calls[0]![1] as { body: FormData }).body
    expect(form).toBeInstanceOf(FormData)
    expect((form.get("file") as File).name).toBe("cat.png")
    await api.upload(new Blob(["x"]))
    expect(((post.mock.calls[1]![1] as { body: FormData }).body.get("file") as File).name).toBe("upload")
  })

  it("builds file URLs without a request, with encoded ids", () => {
    const fetch = recordFetch()
    expect(api.fileUrl("a/b c")).toBe("/api/v2/files/a%2Fb%20c")
    expect(api.downloadUrl("f1")).toBe("/api/v2/files/f1?download=1")
    expect(api.thumbUrl("f1", 256)).toBe("/api/v2/thumbs/f1?w=256")
    expect(api.thumbUrl("f1")).toBe("/api/v2/thumbs/f1")
    expect(api.userAssetUrl("bg #1.png")).toBe("/api/v2/user-assets/bg%20%231.png")
    expect(api.system.sysinfoDownloadUrl()).toBe("/internal/sysinfo-download")
    expect(fetch.requests).toEqual([])
  })

  it("sends each job submission with a fresh Idempotency-Key, or the one given", async () => {
    const fetch = recordFetch(() => ({ status: 201, body: created }))
    await api.jobs.create(request)
    await api.jobs.create(request)
    await api.jobs.create(request, "press-1")
    const keys = fetch.requests.map((sent) => sent.headers.get("Idempotency-Key"))
    expect(keys[0]).toMatch(/^[\w-]{16,}$/)
    expect(keys[1]).toMatch(/^[\w-]{16,}$/)
    expect(keys[1]).not.toBe(keys[0])
    expect(keys[2]).toBe("press-1")
  })

  it("sends a job submission that got no answer once more, with the same key", async () => {
    vi.useFakeTimers()
    try {
      const fetch = recordFetch(() => ({ status: 201, body: created }))
      const answer = fetch.spy.getMockImplementation()!
      fetch.spy.mockImplementationOnce(async (input, init) => {
        await answer(input, init) // recorded: the server may have queued it, the answer was lost
        throw new TypeError("Failed to fetch")
      })
      const sent = expect(api.jobs.create(request)).resolves.toEqual(created)
      await vi.runAllTimersAsync()
      await sent
      expect(fetch.requests).toHaveLength(2)
      expect(fetch.requests[1]!.headers.get("Idempotency-Key")).toBe(fetch.requests[0]!.headers.get("Idempotency-Key"))
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not send a refused job submission again", async () => {
    const fetch = recordFetch(() => ({ status: 422, body: { errors: [{ owner: "core", param: "width", message: "too wide" }] } }))
    await expect(api.jobs.create(request)).rejects.toBeInstanceOf(ApiError)
    expect(fetch.requests).toHaveLength(1)
  })

  it("rejects with the server's status and message", async () => {
    recordFetch(() => ({ status: 409, body: { detail: "the job already started" } }))
    const error = await api.jobs.updateParams("j1", { core: {}, scripts: {} }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 409, message: "the job already started" })
  })

  it("sends chain creation with a fresh or caller-supplied Idempotency-Key", async () => {
    const fetch = recordFetch(() => ({ status: 201, body: { id: "c1" } }))
    await api.chains.create({ stages: [{ kind: "generate" }] })
    await api.chains.create({ stages: [{ kind: "generate" }] }, "chain-key-1")
    await api.chains.create({ stages: [{ kind: "generate" }] }, null)
    expect(fetch.requests[0]!.headers.get("Idempotency-Key")).toMatch(/^[\w-]{16,}$/)
    expect(fetch.requests[1]!.headers.get("Idempotency-Key")).toBe("chain-key-1")
    expect(fetch.requests[2]!.headers.get("Idempotency-Key")).toBeNull()
  })
})
