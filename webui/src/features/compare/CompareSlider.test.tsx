import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { CompareSlider } from "./CompareSlider"

describe("compare slider", () => {
  it("moves the divider from the keyboard and clips the after image", () => {
    render(<CompareSlider before="/a.png" after="/b.png" beforeLabel="Source" afterLabel="Result" aspect={1} />)
    const slider = screen.getByRole("slider", { name: "Comparison divider" })
    expect(slider).toHaveAttribute("aria-valuenow", "50")
    fireEvent.keyDown(slider, { key: "ArrowRight", shiftKey: true })
    expect(slider).toHaveAttribute("aria-valuenow", "60")
    // read back as a browser serializes it (lengths with their unit, numbers without trailing zeros)
    expect(screen.getByRole("img", { name: "Result" }).style.clipPath).toBe("inset(0px 0px 0px 60%)")
    fireEvent.keyDown(slider, { key: "Home" })
    expect(slider).toHaveAttribute("aria-valuenow", "0")
  })

  it("follows the pointer across the frame", () => {
    render(<CompareSlider before="/a.png" after="/b.png" aspect={2} />)
    const frame = screen.getByRole("img", { name: "Before" }).parentElement!
    vi.spyOn(frame, "getBoundingClientRect").mockReturnValue({ left: 100, width: 200, top: 0, height: 100, right: 300, bottom: 100, x: 100, y: 0, toJSON: () => ({}) })
    fireEvent.pointerDown(frame, { clientX: 250, button: 0, pointerId: 1 })
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuenow", "75")
    fireEvent.pointerMove(frame, { clientX: 120, pointerId: 1 })
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuenow", "10")
    fireEvent.pointerUp(frame, { pointerId: 1 })
    fireEvent.pointerMove(frame, { clientX: 300, pointerId: 1 })
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuenow", "10")
  })
})

describe("compare slider labels", () => {
  // In a narrow frame (Extras held sideways, a portrait image 150 px tall) the two corner labels ran
  // into each other: each keeps to its half, cut with its full text as a title.
  it("keep each label to its half of the frame, with its text as a title", () => {
    render(<CompareSlider before="/a.png" after="/b.png" beforeLabel="Source image" afterLabel="Processed result" aspect={0.5} />)
    for (const text of ["Source image", "Processed result"]) {
      const label = screen.getByText(text)
      expect(label).toHaveAttribute("title", text)
      expect(label.className.split(/\s+/)).toEqual(expect.arrayContaining(["max-w-[calc(50%-0.75rem)]", "truncate"]))
    }
  })

  it("still drag the divider when pressed", () => {
    render(<CompareSlider before="/a.png" after="/b.png" beforeLabel="Source" afterLabel="Result" aspect={2} />)
    const frame = screen.getByRole("img", { name: "Source" }).parentElement!
    vi.spyOn(frame, "getBoundingClientRect").mockReturnValue({ left: 0, width: 200, top: 0, height: 100, right: 200, bottom: 100, x: 0, y: 0, toJSON: () => ({}) })
    fireEvent.pointerDown(screen.getByText("Source", { selector: "span" }), { clientX: 20, button: 0, pointerId: 1 })
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuenow", "10")
  })
})
