import { readFileSync } from "node:fs"
import { join } from "node:path"
import { toast } from "sonner"
import { describe, expect, it, vi } from "vitest"
import { ApiError, RequestError, validationErrors } from "@/api/http"
import { tabSchema } from "@/api/mock/fixtures"
import { buildIndex } from "@/ui/form-index"
import { tabOwners } from "@/ui/tab"
import { errorText, failureText, jobErrorText, renderFailureText, serverText, serviceFailureText, toastInvalid, validationText } from "./error-text"

const failed = (status: number) => new ApiError(status, `HTTP ${status}`, { detail: "English detail for the logs" })

describe("errorText", () => {
  it("says the server cannot be reached for anything but an API error", () => {
    expect(errorText(new TypeError("Failed to fetch"))).toBe("The server cannot be reached.")
    expect(errorText("offline")).toBe("The server cannot be reached.")
    expect(errorText(undefined)).toBe("The server cannot be reached.")
  })

  it.each([
    [401, "Sign in again to continue."],
    [403, "The server does not allow this."],
    [404, "It is no longer there."],
    [409, "It changed meanwhile; refresh and try again."],
    [413, "The file is too large."],
    [415, "This file cannot be read as an image."],
    [422, "The server rejected the request."],
    [503, "The server is busy; try again shortly."],
  ])("explains status %i in a sentence, not the server's English detail", (status, text) => {
    expect(errorText(failed(status))).toBe(text)
  })

  // The model bar, a checkpoint card and the quick settings said only "The server rejected the
  // request." when --freeze-settings (or a narrower form) refused the change.
  it("gives the reasons a 422 lists, in the UI language", async () => {
    const frozen = new ApiError(422, "HTTP 422", { errors: [{ owner: "settings", param: "sd_model_checkpoint", message: "changing settings is disabled (--freeze-settings)" }] })
    expect(errorText(frozen)).toBe("changing settings is disabled (--freeze-settings)")
    expect(failureText(frozen)).toBe("changing settings is disabled (--freeze-settings)")
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "changing settings is disabled (--freeze-settings)", "已禁止修改设置（--freeze-settings）")
    await i18n.changeLanguage("zh-CN")
    try {
      expect(errorText(frozen)).toBe("已禁止修改设置（--freeze-settings）")
    } finally {
      await i18n.changeLanguage("en")
    }
    expect(errorText(frozen, { 422: "Their own sentence." })).toBe("Their own sentence.")
  })

  // A board or collection name, or a network's user metadata, refused with no owner: the body names
  // only the field. These answered free English text, or FastAPI's own 422, shown as "The server
  // rejected the request.".
  it("gives the sentence of a refused value that has no owner", async () => {
    const template = "board name must be at most {{maximum}} characters"
    const tooLong = new ApiError(422, "HTTP 422", { errors: [{ owner: null, param: "name", message: "board name must be at most 120 characters", template, params: { maximum: "120" } }] })
    expect(errorText(tooLong)).toBe("board name must be at most 120 characters")
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", template, "看板名称不能超过 {{maximum}} 个字符")
    await i18n.changeLanguage("zh-CN")
    try {
      expect(errorText(tooLong)).toBe("看板名称不能超过 120 个字符")
      expect(failureText(tooLong)).toBe("看板名称不能超过 120 个字符")
    } finally {
      await i18n.changeLanguage("en")
    }
  })

  // A metadata edit refused because the sidecar next to the model cannot be read: the sentence
  // names the file (a template with its values), in the UI language.
  it("gives a refused value's sentence with the values it names", async () => {
    const template = "the metadata file {{file}} cannot be read; saving would replace it, so fix or remove it first"
    const sent = { owner: null, param: "user_metadata", message: template.replace("{{file}}", "watercolor.json"), template, params: { file: "watercolor.json" } }
    const refused = new ApiError(422, "HTTP 422", { errors: [sent] })
    expect(failureText(refused)).toBe("the metadata file watercolor.json cannot be read; saving would replace it, so fix or remove it first")
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", template, "无法读取元数据文件 {{file}}；保存会覆盖它，请先修复或删除该文件")
    await i18n.changeLanguage("zh-CN")
    try {
      expect(failureText(refused)).toBe("无法读取元数据文件 watercolor.json；保存会覆盖它，请先修复或删除该文件")
    } finally {
      await i18n.changeLanguage("en")
    }
  })


  // A recipe name already taken read "It changed meanwhile; refresh and try again.", and a job the
  // queue cannot run again the Queue page's own guess at what a 409 means.
  it("says the sentence a refused request carries, in the UI language", async () => {
    const sentence = (status: number, detail: string, template: string, params: Record<string, string>) => new ApiError(status, detail, { detail, template, params })
    const taken = sentence(409, "a recipe named 'Portrait' already exists", "a recipe named {{name}} already exists", { name: "'Portrait'" })
    const stage = sentence(409, "a pipeline stage cannot generate forever", "a pipeline stage cannot generate forever", {})
    const queueTexts = { 409: "The job changed meanwhile (it started or finished)." }
    expect(errorText(taken)).toBe("a recipe named 'Portrait' already exists")
    expect(errorText(stage, queueTexts)).toBe("a pipeline stage cannot generate forever")
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "a recipe named {{name}} already exists", "已存在名为 {{name}} 的配方")
    i18n.addResource("zh-CN", "server", "a pipeline stage cannot generate forever", "管线阶段不能无限生成")
    await i18n.changeLanguage("zh-CN")
    try {
      expect(errorText(taken)).toBe("已存在名为 'Portrait' 的配方")
      expect(errorText(stage, queueTexts)).toBe("管线阶段不能无限生成")
      expect(failureText(taken)).toBe("已存在名为 'Portrait' 的配方")
    } finally {
      await i18n.changeLanguage("en")
    }
    // a body without a template is said by its status
    expect(errorText(failed(409), queueTexts)).toBe("The job changed meanwhile (it started or finished).")
  })

  // A refused request's body names what is gone by its id ("no recipe 'r1'", "no such job"): the
  // caller knows what that is in the user's words, so its own 404 sentence comes first.
  it("prefers the caller's own 404 sentence to the body's", async () => {
    const gone = new ApiError(404, "no recipe 'r1'", { detail: "no recipe 'r1'", template: "no recipe {{id}}", params: { id: "'r1'" } })
    expect(errorText(gone, { 404: "The recipe was deleted meanwhile." })).toBe("The recipe was deleted meanwhile.")
    expect(failureText(gone, { 404: "The recipe was deleted meanwhile." })).toBe("The recipe was deleted meanwhile.")
    // without one, the body's sentence still says more than the status
    expect(errorText(gone)).toBe("no recipe 'r1'")
    // any other status keeps the body's sentence first
    const started = new ApiError(409, "the job started meanwhile", { detail: "the job started meanwhile", template: "the job started meanwhile", params: {} })
    expect(errorText(started, { 404: "The job was removed meanwhile.", 409: "The job changed meanwhile (it started or finished)." })).toBe("the job started meanwhile")
  })

  it("falls back on the status class for other codes", () => {
    expect(errorText(failed(500))).toBe("The server ran into an error (500).")
    expect(errorText(failed(502))).toBe("The server ran into an error (502).")
    expect(errorText(failed(400))).toBe("The request failed (400).")
    expect(errorText(failed(429))).toBe("The request failed (429).")
  })

  it("prefers the caller's own sentence for a status", () => {
    const texts = { 404: "The image was deleted.", 409: "Someone else saved first." }
    expect(errorText(failed(404), texts)).toBe("The image was deleted.")
    expect(errorText(failed(409), texts)).toBe("Someone else saved first.")
    // Statuses the caller did not name keep the generic text.
    expect(errorText(failed(403), texts)).toBe("The server does not allow this.")
    expect(errorText(new Error("network"), texts)).toBe("The server cannot be reached.")
  })
})

describe("validationText", () => {
  it("lists each distinct message once, translated like the inline errors", async () => {
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "an image is required", "需要一张图像")
    i18n.addResource("zh-CN", "server", "paint a mask over the image", "请在图像上绘制蒙版")
    const errors = [
      { owner: "core", param: "init_img", message: "an image is required" },
      { owner: "core", param: "mask", message: "paint a mask over the image" },
      { owner: "other", param: "image", message: "an image is required" },
    ]
    expect(validationText(errors)).toBe("an image is required · paint a mask over the image")
    await i18n.changeLanguage("zh-CN")
    try {
      expect(validationText(errors)).toBe("需要一张图像 · 请在图像上绘制蒙版")
    } finally {
      await i18n.changeLanguage("en")
    }
  })
})

// The toast for a refused job listed only the messages ("must be at most 150, got 200"): with the
// form's sections closed, nothing said which value they were about.
describe("toastInvalid", () => {
  it("names the field of each refused value", () => {
    const failed = vi.spyOn(toast, "error")
    const refused = new ApiError(422, "HTTP 422", {
      errors: [
        { owner: "sampler", param: "steps", message: "must be at most 150, got 200", template: "must be at most {{maximum}}, got {{value}}", params: { maximum: "150", value: "200" } },
        { owner: "gone", param: "x", message: "the script is not loaded" },
      ],
    })
    toastInvalid(validationErrors(refused)!, buildIndex(tabOwners(tabSchema("txt2img"))))
    expect(failed).toHaveBeenCalledWith("Some parameters are invalid", { description: "Sampling steps: must be at most 150, got 200 · the script is not loaded" })
  })
})

// A job that refused values when it ran listed only their messages, where the 422 of a submission
// names the field of each.
describe("jobErrorText", () => {
  const stored = [
    { owner: "sampler", param: "steps", message: "must be at most 150, got 200", template: "must be at most {{maximum}}, got {{value}}", params: { maximum: "150", value: "200" } },
    { owner: "gone", param: "x", message: "the script is not loaded" },
    { owner: null, param: null, message: "tab is one of txt2img, img2img, extras" },
  ]

  it("names the field of each refused value, in the given language", async () => {
    const index = buildIndex(tabOwners(tabSchema("txt2img")))
    expect(jobErrorText("sampler.steps: must be at most 150, got 200", stored, "en", index)).toBe("Sampling steps: must be at most 150, got 200 · the script is not loaded · tab is one of txt2img, img2img, extras")
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "must be at most {{maximum}}, got {{value}}", "不能大于 {{maximum}}，当前为 {{value}}")
    i18n.addResource("zh-CN", "server", "Sampling steps", "采样步数")
    i18n.addResource("zh-CN", "chrome", "{{field}}: {{message}}", "{{field}}：{{message}}")
    // the language asked for, not the one the app is in
    expect(jobErrorText("", stored, "zh-CN", index)).toBe("采样步数：不能大于 150，当前为 200 · the script is not loaded · tab is one of txt2img, img2img, extras")
  })

  it("says the messages alone without a form to name them by", () => {
    expect(jobErrorText("", stored, "en")).toBe("must be at most 150, got 200 · the script is not loaded · tab is one of txt2img, img2img, extras")
  })
})

describe("failureText", () => {
  it("explains a request by its status and keeps the app's own (translated) message", () => {
    expect(failureText(failed(500))).toBe("The server ran into an error (500).")
    expect(failureText(failed(404), { 404: "The image was deleted." })).toBe("The image was deleted.")
    expect(failureText(new RequestError(new TypeError("Failed to fetch")))).toBe("The server cannot be reached.")
    expect(failureText("offline")).toBe("The server cannot be reached.")
    expect(failureText(new Error("The preset was not applied"))).toBe("The preset was not applied")
  })

  // Any TypeError was read as fetch failing, so a bug in the app ("Cannot read properties of
  // undefined") told the user the server could not be reached. A request that got no answer is
  // marked as a RequestError where it is made (api/http.ts).
  it("keeps a bug's own message instead of blaming the server", () => {
    expect(failureText(new TypeError("Cannot read properties of undefined (reading 'width')"))).toBe("Cannot read properties of undefined (reading 'width')")
  })
})

// The prompt tools (translate, write a prompt, PUP) showed the server's English detail as is: a
// provider that failed is only explained by its own answer, which stays, behind a sentence in the UI
// language; every other failure is explained by its status, as elsewhere.
describe("serviceFailureText", () => {
  it("keeps what a failed service answered behind a sentence in the UI language", () => {
    expect(serviceFailureText(new ApiError(502, "translation service answered 401: invalid key"))).toBe("The service failed: translation service answered 401: invalid key")
  })

  it("explains the other failures by their status", () => {
    expect(serviceFailureText(new ApiError(409, "no translation provider is configured"))).toBe("Set up the service in Settings first.")
    expect(serviceFailureText(new ApiError(413, "too much text to translate at once"))).toBe("The text is too long.")
    expect(serviceFailureText(failed(500))).toBe("The server ran into an error (500).")
    expect(serviceFailureText(new RequestError(new TypeError("Failed to fetch")))).toBe("The server cannot be reached.")
    expect(serviceFailureText(new ApiError(409, "x"), { 409: "Choose a model first." })).toBe("Choose a model first.")
  })

  // A value a prompt tool refused (a local model path that does not exist) came as free text, which
  // read only "The server rejected the request.": it is a refused value now, with its template.
  it("says why a value was refused, in the UI language", async () => {
    const template = "local_model_path does not exist: {{path}}"
    const refused = new ApiError(422, "HTTP 422", { errors: [{ owner: null, param: "local_model_path", message: "local_model_path does not exist: /m.gguf", template, params: { path: "/m.gguf" } }] })
    expect(serviceFailureText(refused)).toBe("local_model_path does not exist: /m.gguf")
    // the shipped catalog has it (localizations/zh_CN.json)
    const catalog = JSON.parse(readFileSync(join(process.cwd(), "..", "localizations", "zh_CN.json"), "utf8")) as Record<string, string>
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", template, catalog[template]!)
    await i18n.changeLanguage("zh-CN")
    try {
      expect(serviceFailureText(refused)).toBe("本地模型路径不存在：/m.gguf")
    } finally {
      await i18n.changeLanguage("en")
    }
  })

  // A tool that cannot run or was given too much said only its status's sentence: "The text is too
  // long." for a reference image that was too large. The server says why now, as a sentence the
  // shipped catalog translates, and it comes before those.
  it("says why a tool cannot run, in the UI language", async () => {
    const sentence = (status: number, detail: string, template: string, params: Record<string, string> = {}) => new ApiError(status, detail, { detail, template, params })
    const off = sentence(409, "no translation provider is configured", "no translation provider is configured")
    const image = sentence(413, "reference_images[0] is too large (9000×9000)", "reference_images[{{index}}] is too large ({{width}}×{{height}})", { index: "0", width: "9000", height: "9000" })
    expect(serviceFailureText(off)).toBe("no translation provider is configured")
    expect(serviceFailureText(image)).toBe("reference_images[0] is too large (9000×9000)")
    const catalog = JSON.parse(readFileSync(join(process.cwd(), "..", "localizations", "zh_CN.json"), "utf8")) as Record<string, string>
    const { i18n } = await import("@/i18n")
    for (const template of ["no translation provider is configured", "reference_images[{{index}}] is too large ({{width}}×{{height}})"]) {
      expect(catalog[template]).toBeTruthy()
      i18n.addResource("zh-CN", "server", template, catalog[template]!)
    }
    await i18n.changeLanguage("zh-CN")
    try {
      expect(serviceFailureText(off)).toBe(catalog["no translation provider is configured"])
      expect(serviceFailureText(image)).toBe(catalog["reference_images[{{index}}] is too large ({{width}}×{{height}})"]!.replace("{{index}}", "0").replace("{{width}}", "9000").replace("{{height}}", "9000"))
    } finally {
      await i18n.changeLanguage("en")
    }
  })
})

// An error boundary showed a request that got no answer (the server down, a chunk that could not be
// fetched) in the browser's English: only a bug keeps its own message there.
describe("renderFailureText", () => {
  it("explains a request by its status and a bug by its own message", () => {
    expect(renderFailureText(failed(500))).toBe("The server ran into an error (500).")
    expect(renderFailureText(new RequestError(new TypeError("Failed to fetch")))).toBe("The server cannot be reached.")
    expect(renderFailureText(new TypeError("Cannot read properties of undefined (reading 'x')"))).toBe("Cannot read properties of undefined (reading 'x')")
  })
})

// The history and the failure toasts showed how a job ended ("the server stopped while the job ran",
// "Out of memory: the model was unloaded") in English in every language.
describe("serverText", () => {
  it("says a fixed sentence the server stored in the UI language, and any other text as it is", async () => {
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "the server stopped while the job ran", "任务运行时服务器停止了")
    await i18n.changeLanguage("zh-CN")
    try {
      expect(serverText("the server stopped while the job ran")).toBe("任务运行时服务器停止了")
      // a job's own exception is a diagnostic: shown as it is, never read as a template
      expect(serverText("KeyError: '{{width}}' $t(Prompt)")).toBe("KeyError: '{{width}}' $t(Prompt)")
    } finally {
      await i18n.changeLanguage("en")
    }
  })
})
