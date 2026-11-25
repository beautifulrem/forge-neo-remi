/**
 * Boot-time registration of the prompt tools (imported once by main.tsx;
 * small: the completion engine, the tools bar and the panels are lazy).
 *
 * - tag autocompletion in every prompt editor (`registerCompletionSource`);
 * - the tools bar and tag-chips view around every prompt editor
 *   (`registerPromptDecorator`);
 * - fresh data when settings, models, networks or the history change.
 */
import { queryClient } from "@/api/queries"
import { onWsEvent } from "@/api/ws"
import { registerCompletionSource } from "@/features/prompt-editor/completion-registry"
import { registerPromptDecorator } from "@/features/prompt-editor/decorators"
import { lazyModule } from "@/lib/lazy-module"
import { notifyLoadFailed } from "@/lib/retryable-lazy"
import { PromptToolsDecorator } from "./PromptToolsDecorator"
import { toolKeys } from "./queries"

// the completion engine, fetched at the first completion (again after a failed fetch: offline)
const engine = lazyModule(() => import("./completion/source"))

/**
 * The engine, or null while it cannot load (no completions then). Typing asks for it, so a failure
 * is told to `chunk-reload.ts`, as any load the user asked for (`loadForUse`): the server may run
 * another build, whose chunks have other names.
 */
function loadEngine() {
  return engine.load().catch(() => {
    notifyLoadFailed()
    return null
  })
}

function forgetCompletions() {
  engine.get()?.clearCompletionCache()
}

registerCompletionSource(
  "prompt-tools",
  async (context) => {
    const module = await loadEngine()
    return module ? module.promptToolsCompletions(context) : null
  },
  { priority: 10 },
)
registerPromptDecorator("prompt-tools", PromptToolsDecorator)

onWsEvent("options.changed", () => {
  forgetCompletions()
  void queryClient.invalidateQueries({ queryKey: toolKeys.config })
  void queryClient.invalidateQueries({ queryKey: ["prompt-tools", "lookup"] })
})
onWsEvent("model.loaded", () => void queryClient.invalidateQueries({ queryKey: toolKeys.config }))
onWsEvent("networks.changed", forgetCompletions)
// the server records the prompts of a queued job a moment later
onWsEvent("job.queued", () => {
  setTimeout(() => void queryClient.invalidateQueries({ queryKey: ["prompt-tools", "history"] }), 500)
})
