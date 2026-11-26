/**
 * A failure is shown in the UI language (docs/remi-webui/frontend.md, "Failure text"): an error's
 * own message is the server's English detail or the browser's English reason, so only the helpers
 * in `error-text.ts` (and the transport in `api/`, which builds those errors) read it. Every other
 * read of an error's message, detail or text is listed below with the reason it may be shown.
 *
 * This guard exists because the rule was broken one call site at a time: the extensions pages, the
 * Settings > Server view and the error boundaries each showed the English text after the helpers
 * were fixed.
 */
import ts from "@typescript/typescript6"
import { describe, expect, it } from "vitest"

const sources = import.meta.glob<string>(["../../**/*.{ts,tsx}", "!../../**/*.test.*", "!../../**/*.d.ts", "!../../api/**", "!./error-text.ts"], {
  query: "?raw",
  import: "default",
  eager: true,
})

/** Names an error is held under (`catch (e)`, `onError: (error) =>`, `loadError`, a rejection's `reason`). */
const ERROR_NAMES = /^(e|err|ex|exception|thrown|caught|problem)$|(error|reason|failure|cause)$/i

/** Properties that hold one (a query's `.error` and `.failureReason`, a rejection's `.reason`, `.cause`). */
const ERROR_PROPERTIES = /^(error|cause|reason|failureReason)$|Error$/

/** Reads that may show an error's own text, by file and the read's source text, with the reason. */
const ALLOWED: { file: string; read: string; reason: string }[] = [
  { file: "components/ui/field.tsx", read: "error?.message", reason: "shadcn's FieldError lists form-library validation messages the app wrote itself" },
  { file: "components/ui/field.tsx", read: "error.message", reason: "shadcn's FieldError lists form-library validation messages the app wrote itself" },
  { file: "ui/inputs/Json.tsx", read: "e.message", reason: "the browser's reason the user's own JSON does not parse, under the field (recorded P3)" },
  { file: "ui/inputs/Json.tsx", read: "String(e)", reason: "as above" },
  { file: "widgets/text-file/index.tsx", read: "error.message", reason: "the browser's reason a local file could not be read: no request is involved (recorded P3)" },
  { file: "widgets/text-file/index.tsx", read: "String(error)", reason: "as above" },
  { file: "pages/extensions/extensions-model.ts", read: "error.detail", reason: "reads the `template` and `params` of an admin error, which are translated" },
  { file: "pages/extensions/DiagnosticsView.tsx", read: "${s.reason}", reason: "why a script was left out (\"declares ui() instead of params()\", \"params() failed: <exception>\"): a diagnostic for extension authors, shown as it is" },
]

/** The expression without parentheses, `as` casts and `!`. */
function bare(node: ts.Expression): ts.Expression {
  while (ts.isParenthesizedExpression(node) || ts.isAsExpression(node) || ts.isNonNullExpression(node) || ts.isTypeAssertionExpression(node)) node = node.expression
  return node
}

/** An expression that holds an error: one of {@link ERROR_NAMES}, or a property in {@link ERROR_PROPERTIES}. */
function isError(node: ts.Expression): boolean {
  const expression = bare(node)
  if (ts.isIdentifier(expression)) return ERROR_NAMES.test(expression.text)
  return (ts.isPropertyAccessExpression(expression) || ts.isPropertyAccessChain(expression)) && ERROR_PROPERTIES.test(expression.name.text)
}

/** Every read of an error's text in `source`: `.message`, `.detail`, `.statusText`, `.toString()`, `String(error)`, `${error}`. */
function errorTextReads(file: string, source: string): string[] {
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS)
  const reads: string[] = []
  const visit = (node: ts.Node) => {
    if (ts.isPropertyAccessExpression(node) && /^(message|detail|statusText)$/.test(node.name.text) && isError(node.expression)) reads.push(node.getText(sf))
    else if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === "toString" && isError(node.expression.expression)) reads.push(node.getText(sf))
    else if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === "String" && node.arguments[0] && isError(node.arguments[0])) reads.push(node.getText(sf))
    else if (ts.isTemplateSpan(node) && isError(node.expression)) reads.push(`\${${node.expression.getText(sf)}}`)
    ts.forEachChild(node, visit)
  }
  visit(sf)
  return reads.map((read) => read.replace(/\s+/g, " "))
}

const files = Object.entries(sources).map(([path, source]) => [path.replace(/^\.\.\/\.\.\//, ""), source] as const)

describe("failure text", () => {
  it("is read from an error only by the helpers, or where a reason is given", () => {
    const unexplained: string[] = []
    for (const [file, source] of files) {
      for (const read of errorTextReads(file, source)) {
        if (!ALLOWED.some((allowed) => allowed.file === file && allowed.read === read)) unexplained.push(`${file}: ${read}`)
      }
    }
    expect(unexplained).toEqual([])
  })

  it("lists no read that is gone", () => {
    const stale = ALLOWED.filter(({ file, read }) => !errorTextReads(file, files.find(([path]) => path === file)?.[1] ?? "").includes(read))
    expect(stale).toEqual([])
  })

  it("sees an error's text however it is read", () => {
    const reads = errorTextReads(
      "probe.tsx",
      `toast.error("x", { description: (error as Error).message })
       const text = \`failed: \${err}\`
       const shown = <p>{query.error?.message}</p>
       showToast(String(e))
       const raw = (failure as ApiError).detail
       const fine = result.message + event.detail + details.reason
       const loaded = <p>{loadError.message}</p>
       const settled = rejected.reason.message + query.failureReason?.message + String(lastError)
       const said = \`\${thrown}\` + caught.toString() + response.error.statusText`,
    )
    expect(reads).toEqual([
      "(error as Error).message",
      "${err}",
      "query.error?.message",
      "String(e)",
      "(failure as ApiError).detail",
      "loadError.message",
      "rejected.reason.message",
      "query.failureReason?.message",
      "String(lastError)",
      "${thrown}",
      "caught.toString()",
      "response.error.statusText",
    ])
  })
})
