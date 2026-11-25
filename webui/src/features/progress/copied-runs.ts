/**
 * A duplicated tab drops the jobs in flight it copied from its original (lib/tab-identity.ts) at
 * boot, whatever page it shows. The generate and Extras pages also drop them when they restore,
 * keeping a job queued while the tab asked; but a copy showing another page kept them in its
 * session storage, and its next reload, under its new tab id, took them over. Called by
 * `main.tsx`, which does not load the run modules for it unless the tab is a copy.
 */
import { sessionIsOwn } from "@/lib/tab-identity"

export async function forgetCopiedRuns(): Promise<void> {
  if (await sessionIsOwn()) return
  const [{ forgetStoredRun, GENERATE_FORMS }, { forgetCopiedJobs }] = await Promise.all([import("./run-controller"), import("@/features/extras/run")])
  for (const tab of GENERATE_FORMS) forgetStoredRun(tab)
  forgetCopiedJobs()
}
