/**
 * Motion (and the icon defaults) for the whole app: `m` components render at once and start
 * animating when the feature bundle arrives (its own chunk, off the first
 * load). Reduced-motion users get no transform or layout animation.
 */
import { LazyMotion, MotionConfig } from "motion/react"
import type { ReactNode } from "react"
import { IconDefaults } from "@/components/icons"
import { notifyLoadFailed } from "@/lib/retryable-lazy"

/**
 * The feature bundle. LazyMotion asks once, so a failed fetch (a dropped connection, a build
 * replaced by a newer one) is tried again a few times before giving up, and told to
 * `chunk-reload.ts`, which reloads when the server now runs another build. Without the bundle
 * `m` components render without animating; nothing may start hidden waiting for it (the stage
 * image uses a CSS entrance).
 */
const RETRY_DELAYS = [1_000, 4_000, 15_000]

async function features() {
  for (let attempt = 0; ; attempt++) {
    try {
      return (await import("@/lib/motion-features")).default
    } catch (error) {
      notifyLoadFailed()
      const delay = RETRY_DELAYS[attempt]
      if (delay === undefined) throw error
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }
}

export function MotionRoot({ children }: { children: ReactNode }) {
  return (
    <LazyMotion features={features} strict>
      <MotionConfig reducedMotion="user" transition={{ type: "spring", visualDuration: 0.22, bounce: 0 }}>
        <IconDefaults>{children}</IconDefaults>
      </MotionConfig>
    </LazyMotion>
  )
}
