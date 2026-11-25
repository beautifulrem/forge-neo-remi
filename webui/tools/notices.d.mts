export type PackageNotice = { name: string; version: string; license: string; texts: string[] }
export function packageRoot(id: string): string | null
export function bundlerPackage(id: string): "vite" | "rolldown" | null
export function collectNotices(ids: Iterable<string>, options?: { modules?: string }): Promise<PackageNotice[]>
export function formatNotices(packages: readonly PackageNotice[], head?: string): string
export function devNotices(root: string): Promise<string>
export function cssImportedPackages(css: string): string[]
export function runtimeDependencies(manifest: { dependencies?: Record<string, string> }): string[]
export function vendoredNotices(): Promise<PackageNotice[]>
