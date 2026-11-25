import { afterEach, beforeAll, describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import { sentenceText } from "./messages"

const FIBO = "Bria Fibo 2.0 supports at most {{max}} reference images; {{count}} are attached."
const REQUIRES = "Requires {{capabilities}} (active model: {{model}})"

describe("sentenceText", () => {
  beforeAll(() => {
    i18n.addResource("zh-CN", "server", FIBO, "Bria Fibo 2.0 最多支持 {{max}} 张参考图，当前附加了 {{count}} 张。")
    i18n.addResource("zh-CN", "server", REQUIRES, "需要 {{capabilities}}（当前模型：{{model}}）")
    i18n.addResource("zh-CN", "server", "Anima engine hooks", "Anima 引擎钩子")
    i18n.addResource("zh-CN", "server", "Multi-reference", "多参考图")
    i18n.addResource("zh-CN", "server", "a choice is required", "需要选择一项")
  })
  afterEach(async () => {
    await i18n.changeLanguage("en")
  })

  it("translates the template and fills it, a value named like an i18next option included", async () => {
    await i18n.changeLanguage("zh-CN")
    expect(sentenceText(i18n, "Bria Fibo 2.0 supports at most 2 reference images; 3 are attached.", FIBO, { max: "2", count: "3" })).toBe(
      "Bria Fibo 2.0 最多支持 2 张参考图，当前附加了 3 张。",
    )
  })

  it("translates each catalog name of a list value and joins them the language's way", async () => {
    await i18n.changeLanguage("zh-CN")
    const params = { capabilities: ["Anima engine hooks", "Multi-reference", "an_unknown_token"], model: "Qwen-Image-2.1" }
    expect(sentenceText(i18n, "Requires Anima engine hooks, Multi-reference, an_unknown_token (active model: Qwen-Image-2.1)", REQUIRES, params)).toBe(
      "需要 Anima 引擎钩子、多参考图和an_unknown_token（当前模型：Qwen-Image-2.1）",
    )
  })

  it("keeps the server's English where no catalog has the template", () => {
    const text = "Requires Anima engine hooks, Multi-reference (active model: Qwen-Image-2.1)"
    expect(sentenceText(i18n, text, REQUIRES, { capabilities: ["Anima engine hooks", "Multi-reference"], model: "Qwen-Image-2.1" })).toBe(text)
    expect(sentenceText(i18n, "Unknown 7 things", "Unknown {{count}} things", { count: "7" })).toBe("Unknown 7 things")
  })

  it("translates a sentence a param holds, from its template and params under the param's name", async () => {
    await i18n.changeLanguage("zh-CN")
    i18n.addResource("zh-CN", "server", "{{field}}: {{reason}}", "{{field}}：{{reason}}")
    i18n.addResource("zh-CN", "server", "expected a number, got {{value}}", "应为数字，收到 {{value}}")
    i18n.addResource("zh-CN", "server", "a pose is an OpenPose JSON object", "姿势须为 OpenPose JSON 对象")
    const number = { field: "weight", reason: "expected a number, got 'heavy'", "reason.template": "expected a number, got {{value}}", "reason.value": "'heavy'" }
    expect(sentenceText(i18n, "weight: expected a number, got 'heavy'", "{{field}}: {{reason}}", number)).toBe("weight：应为数字，收到 'heavy'")
    const pose = { field: "pose", reason: "a pose is an OpenPose JSON object", "reason.template": "a pose is an OpenPose JSON object" }
    expect(sentenceText(i18n, "pose: a pose is an OpenPose JSON object", "{{field}}: {{reason}}", pose)).toBe("pose：姿势须为 OpenPose JSON 对象")
  })

  it("finds a sentence without values by its exact key and leaves other text alone", async () => {
    await i18n.changeLanguage("zh-CN")
    expect(sentenceText(i18n, "a choice is required")).toBe("需要选择一项")
    expect(sentenceText(i18n, "a value like {{this}} stays")).toBe("a value like {{this}} stays")
  })
})
