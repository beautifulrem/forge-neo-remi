import { describe, expect, it } from "vitest"
import { ApiError, RequestError } from "@/api/http"
import { errorText, failureText, renderFailureText, serverText, serviceFailureText, validationText } from "./error-text"

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
