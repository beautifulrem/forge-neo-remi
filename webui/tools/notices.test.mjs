import path from "node:path"
import { describe, expect, it } from "vitest"
import { readdir, readFile } from "node:fs/promises"
import { collectNotices, cssImportedPackages, formatNotices, packageRoot, vendoredNotices } from "./notices.mjs"

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

  // its npm release has no license file (its repository has one): the notice cited only package.json
  it("takes the vendored license text of a package whose release ships none", async () => {
    const store = path.join(modules, ".pnpm")
    const folder = (await readdir(store)).find((name) => name.startsWith("react-remove-scroll-bar@"))
    const root = path.join(store, folder, "node_modules/react-remove-scroll-bar")
    const [notice] = await collectNotices([path.join(root, "dist/es2015/index.js")])
    expect(notice).toMatchObject({ name: "react-remove-scroll-bar", license: "MIT" })
    expect(notice.texts.join("\n")).toMatch(/Copyright \(c\) \d{4} Anton Korzunov[\s\S]*Permission is hereby granted/)
  })

  it("finds the packages the stylesheets import, which never reach the bundle's module list", async () => {
    expect(cssImportedPackages('@import "tailwindcss";\n@import "./local.css";\n@import url("@fontsource-variable/geist/wght.css");')).toEqual(["tailwindcss", "@fontsource-variable/geist"])
    const css = await readFile(path.resolve(import.meta.dirname, "../src/index.css"), "utf8")
    expect(cssImportedPackages(css)).toEqual(expect.arrayContaining(["tailwindcss", "tw-animate-css"]))
  })

  it("lists the shadcn/ui components copied into src/components/ui, which no package brings", async () => {
    const notices = await vendoredNotices()
    const shadcn = notices.find((pkg) => pkg.name === "shadcn/ui")
    expect(shadcn).toMatchObject({ license: "MIT" })
    expect(shadcn?.texts.join("\n")).toMatch(/Copyright \(c\) 2023 shadcn/)
  })
})
