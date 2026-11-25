import { describe, expect, it } from "vitest"
import { routeTitle } from "./route-title"
import { buildNav } from "./routes"

// The header and the window title showed a route's id: "page" for an extension page while the page
// list loaded (or for one that is gone), and "extras" / "png-info" / "networks" for a built-in page
// hidden from the nav ("Hide UI Tabs") but opened by its address or the viewer's "PNG Info" button.
describe("route title", () => {
  it("names a built-in page hidden from the nav as the nav would", () => {
    const nav = buildNav(["txt2img", "img2img", "gallery"], [])
    expect(routeTitle({ name: "extras" }, nav)).toBe("Extras")
    expect(routeTitle({ name: "png-info" }, nav)).toBe("PNG Info")
    expect(routeTitle({ name: "networks" }, nav)).toBe("Networks")
  })

  it("names an extension page by its title, and one not listed (yet) as an extension page", () => {
    const nav = buildNav(undefined, [{ id: "prompt-lab", title: "Prompt Lab", icon: null } as never])
    expect(routeTitle({ name: "page", id: "prompt-lab" }, nav)).toBe("Prompt Lab")
    expect(routeTitle({ name: "page", id: "gone" }, nav)).toBe("Extension")
    expect(routeTitle({ name: "page", id: "prompt-lab" }, buildNav(undefined, undefined))).toBe("Extension")
  })
})
