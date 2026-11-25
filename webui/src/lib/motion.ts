/**
 * The `behavior` of a scroll the app starts itself (`scrollIntoView`, `scrollTo`): smooth, or
 * `auto` when the system asks for reduced motion. index.css's reduced-motion rule sets
 * `scroll-behavior: auto`, but a scroll that passes `behavior: "smooth"` does not read it; `auto`
 * defers to it. Scrolls take this instead of writing "smooth" (lib/motion.test.ts checks).
 */
export function scrollBehavior(): ScrollBehavior {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"
}
