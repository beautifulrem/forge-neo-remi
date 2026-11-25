import { QueryErrorResetBoundary } from "@tanstack/react-query"
import { TriangleAlert } from "@/components/icons"
import { Component, type ErrorInfo, type ReactNode } from "react"
import { ApiError, RequestError } from "@/api/http"
import { Button } from "@/components/ui/button"
import { renderFailureText } from "@/features/errors/error-text"
import { t } from "@/i18n"
import { retryFailedLoads } from "@/lib/retryable-lazy"

type Props = {
  children: ReactNode
  /** Changing this resets the boundary (e.g. the route key). */
  resetKey?: unknown
  /** Compact rendering for widgets inside a form. */
  inline?: boolean
  /** What to show instead of the default message (shell controls: their placeholder, or nothing). */
  fallback?: ReactNode
  /** Told of the error once it is caught (a panel that closes itself instead of showing it). */
  onError?: (error: Error) => void
}

type State = { error: Error | null; resetKey: unknown }

/**
 * A request that failed (answered or not): asking again may work. A render bug would only throw
 * again, so the boundary does not render it again by itself (only "Try again" does).
 */
function isRequest(error: Error | null): boolean {
  return error instanceof ApiError || error instanceof RequestError
}

/**
 * Catches a render error below it and offers to try again. A suspense query
 * that threw is fetched again on its next mount only once its error boundary
 * is reset (TanStack Query): each boundary has its own, reset on every retry
 * ("Try again", the browser back online, and for a failed request also coming
 * back to the page or the boundary mounting with it).
 */
export function ErrorBoundary(props: Props) {
  return <QueryErrorResetBoundary>{({ reset }) => <Boundary {...props} onRetry={reset} />}</QueryErrorResetBoundary>
}

class Boundary extends Component<Props & { onRetry: () => void }, State> {
  state: State = { error: null, resetKey: this.props.resetKey }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    return props.resetKey !== state.resetKey ? { error: null, resetKey: props.resetKey } : null
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack)
    this.props.onError?.(error)
  }

  componentDidMount() {
    window.addEventListener("online", this.retry)
    // mounted on a load that failed before (the More sheet opened again): fetch it once more
    if (isRequest(this.state.error)) this.retry()
  }

  componentDidUpdate(previous: Props) {
    // back on a page whose load failed (the route key changed): fetch it once more
    if (previous.resetKey !== this.props.resetKey && isRequest(this.state.error)) this.retry()
  }

  componentWillUnmount() {
    window.removeEventListener("online", this.retry)
  }

  /** Renders the children again, with chunks that failed to load fetched anew (a boundary with no message of its own, such as a shell control's, recovers when the browser is back online). */
  retry = () => {
    if (!this.state.error) return
    retryFailedLoads()
    this.props.onRetry()
    this.setState({ error: null })
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    if (this.props.fallback !== undefined) return this.props.fallback
    if (this.props.inline) {
      return (
        // in the field's place, like the field's own message: a plain line, no box
        <div role="alert" className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs leading-snug text-destructive">
          <span className="min-w-0 break-words">{t("This control failed to render: {{reason}}", { reason: renderFailureText(error) })}</span>
          <Button variant="outline" size="xs" onClick={this.retry}>
            {t("Try again")}
          </Button>
        </div>
      )
    }
    return (
      <div role="alert" className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <TriangleAlert className="size-8 text-destructive" aria-hidden />
        <div className="space-y-1">
          <p className="font-medium">{t("Something went wrong")}</p>
          <p className="max-w-md text-sm break-words text-muted-foreground">{renderFailureText(error)}</p>
        </div>
        <Button variant="outline" size="sm" onClick={this.retry}>
          {t("Try again")}
        </Button>
      </div>
    )
  }
}
