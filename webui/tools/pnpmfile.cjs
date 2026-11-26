// TypeScript 7 is the native compiler: its package has `tsc` and no JavaScript compiler API.
// openapi-typescript (pnpm gen:api, src/api/gen/gen.test.ts) builds its output with that API and
// asks for `typescript` as a peer, which pnpm would give it from this project (7). Its
// `typescript` is TypeScript 6's instead, from the package TypeScript publishes for running both
// (the same one the devDependencies list for the tests that read the sources' syntax trees).
const TYPESCRIPT_6 = "npm:@typescript/typescript6@^6.0.2"

function readPackage(pkg) {
  if (pkg.name === "openapi-typescript") {
    delete pkg.peerDependencies?.typescript
    pkg.dependencies = { ...pkg.dependencies, typescript: TYPESCRIPT_6 }
  }
  return pkg
}

module.exports = { hooks: { readPackage } }
