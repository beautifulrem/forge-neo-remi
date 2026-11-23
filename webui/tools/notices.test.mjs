import path from "node:path"
import { describe, expect, it } from "vitest"
import { collectNotices, formatNotices, packageRoot } from "./notices.mjs"

const modules = path.resolve(import.meta.dirname, "../node_modules")

describe("third-party notices", () => {
  it("finds the package a bundled module comes from, scoped and inside pnpm's store too", () => {
    expect(packageRoot("/a/node_modules/.pnpm/react@19.2.0/node_modules/react/index.js")).toBe("/a/node_modules/.pnpm/react@19.2.0/node_modules/react")
    expect(packageRoot("\0/a/node_modules/@base-ui/react/esm/x.js?commonjs-proxy")).toBe("/a/node_modules/@base-ui/react")
    expect(packageRoot("C:\\a\\node_modules\\sonner\\dist\\index.mjs")).toBe("C:/a/node_modules/sonner")
    expect(packageRoot("/a/src/main.tsx")).toBeNull()
  })

  it("lists each package once with its license text", async () => {
    const react = path.join(modules, "react")
    const notices = await collectNotices([path.join(react, "index.js"), path.join(react, "cjs/react.production.js"), "/app/src/main.tsx"])
    expect(notices).toHaveLength(1)
    expect(notices[0]).toMatchObject({ name: "react", license: "MIT" })
    expect(notices[0].texts.join("\n")).toMatch(/Copyright \(c\) Meta Platforms/)
    const text = formatNotices(notices)
    expect(text).toContain(`react ${notices[0].version}\nLicense: MIT`)
  })
})
