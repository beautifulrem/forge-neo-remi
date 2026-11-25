/**
 * The Node.js a contributor needs for `pnpm test`: jsdom 30, vitest 5 and @babel/core 8 each ask
 * for newer releases than the build's Vite. webui/README.md and the frontend.md command table name
 * jsdom's minimums, which satisfy the other two as well. (The launcher's NODE_REQUIREMENT, for the
 * build, is held to Vite's engines by tests/test_remi_server_runtime.py.)
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const webui = path.resolve(import.meta.dirname, "..")
const engines = (name) => JSON.parse(readFileSync(path.join(webui, "node_modules", name, "package.json"), "utf8")).engines.node

/** The lowest version of each part of a range: "^22.22.2 || >=26.0.0" → ["22.22.2", "26.0.0"]. */
const minimums = (range) => range.split("||").map((part) => /^\s*(?:\^|>=)\s*(\d+\.\d+\.\d+)\s*$/.exec(part)[1])

/** Whether `version` is in `range` (only `^x.y.z` and `>=x.y.z` parts, as these packages write them). */
function satisfies(version, range) {
  const [major, minor, patch] = version.split(".").map(Number)
  return range.split("||").some((part) => {
    const [, op, low] = /^\s*(\^|>=)\s*(\d+\.\d+\.\d+)\s*$/.exec(part)
    const [lowMajor, lowMinor, lowPatch] = low.split(".").map(Number)
    const atLeast = major !== lowMajor ? major > lowMajor : minor !== lowMinor ? minor > lowMinor : patch >= lowPatch
    return atLeast && (op === ">=" || major === lowMajor)
  })
}

/** "22.22.2", "24.15.0", "26.0.0" → "Node.js 22.22.2+, 24.15+ or 26+". */
function phrase(versions) {
  const short = versions.map((version) => `${version.replace(/(?:\.0)+$/, "")}+`)
  return `Node.js ${short.slice(0, -1).join(", ")} or ${short.at(-1)}`
}

describe("the Node.js the tests need", () => {
  it("is named where contributors read the commands, and covers every test tool", () => {
    const needed = minimums(engines("jsdom"))
    for (const tool of ["vitest", "@babel/core"]) for (const version of needed) expect(satisfies(version, engines(tool)), `${tool} ${engines(tool)} on ${version}`).toBe(true)
    const text = phrase(needed)
    expect(readFileSync(path.join(webui, "README.md"), "utf8")).toContain(text)
    expect(readFileSync(path.resolve(webui, "../docs/remi-webui/frontend.md"), "utf8")).toContain(text)
  })
})
