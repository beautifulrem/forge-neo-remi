import { describe, expect, it } from "vitest"
import html from "../../index.html?raw"
import { APP_TITLE, documentTitle } from "./document-title"

// The page said "Remi WebUI" until the app set "Generate · Remi": one name, before and after
describe("document title", () => {
  it("names the app in index.html as every page title does", () => {
    expect(/<title>([^<]*)<\/title>/.exec(html)?.[1]).toBe(APP_TITLE)
    expect(documentTitle("Generate")).toBe(`Generate · ${APP_TITLE}`)
  })
})
