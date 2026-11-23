/**
 * THIRD-PARTY-NOTICES.txt for the web interface build: the bundle minifies the dependencies'
 * license comments away, so the build ships each bundled package's name, version, license and
 * license text in one file next to index.html (vite.config.ts, `thirdPartyNotices()`).
 */
import { readdir, readFile } from "node:fs/promises"
import path from "node:path"

/** The package folder a bundled module comes from, or null for the app's own code. */
export function packageRoot(id) {
  const file = id.replace(/^\0+/, "").split("?")[0].replaceAll("\\", "/")
  const match = /^(.*\/node_modules\/(?:@[^/]+\/)?[^/]+)\//.exec(file)
  return match ? match[1] : null
}

const LICENSE_FILE = /^(?:licen[cs]e|copying|notice)(?:[.-].*)?$/i

/** Name, version, license and license texts of each package the bundle uses, sorted by name. */
export async function collectNotices(ids) {
  const roots = new Set()
  for (const id of ids) {
    const root = packageRoot(id)
    if (root) roots.add(root)
  }
  const packages = new Map()
  for (const root of roots) {
    let manifest
    try {
      manifest = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"))
    } catch {
      continue // a nested folder without its own package.json belongs to the package above it
    }
    const key = `${manifest.name}@${manifest.version}`
    if (packages.has(key)) continue
    const files = (await readdir(root)).filter((name) => LICENSE_FILE.test(name)).sort()
    const texts = []
    for (const name of files) texts.push((await readFile(path.join(root, name), "utf8")).trim())
    const license = typeof manifest.license === "string" ? manifest.license : (manifest.license?.type ?? "see the text below")
    packages.set(key, { name: manifest.name, version: manifest.version, license, texts })
  }
  return [...packages.values()].sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version))
}

/** The notices as one plain-text file. */
export function formatNotices(packages) {
  const head = "Third-party software in the Remi WebUI web interface build\n\nEach package below is bundled into this build under its own license.\n"
  const blocks = packages.map((pkg) => {
    const text = pkg.texts.length > 0 ? pkg.texts.join("\n\n") : "(the package ships no license file; its package.json names the license above)"
    return `${"=".repeat(72)}\n${pkg.name} ${pkg.version}\nLicense: ${pkg.license}\n\n${text}\n`
  })
  return `${head}\n${blocks.join("\n")}`
}
