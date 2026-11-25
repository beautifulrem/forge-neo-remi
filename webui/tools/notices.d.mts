export type PackageNotice = { name: string; version: string; license: string; texts: string[] }
export function packageRoot(id: string): string | null
export function collectNotices(ids: Iterable<string>): Promise<PackageNotice[]>
export function formatNotices(packages: readonly PackageNotice[]): string
export function cssImportedPackages(css: string): string[]
export function runtimeDependencies(manifest: { dependencies?: Record<string, string> }): string[]
export function vendoredNotices(): Promise<PackageNotice[]>
