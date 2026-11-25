/**
 * The editor's "?" list against what its keys do: every key the list names for a command runs
 * that command (commandFor), and the keys the editor handles beside commandFor (the brush
 * adjusters, panning, the staged candidates in EditorBody) are listed too.
 */
import { describe, expect, it } from "vitest"
import { type Command, commandFor, type KeyLike, shortcutList } from "./shortcuts"

/** A key press for a key cap of the list (`Ctrl` held for a combination). */
function press(cap: string, ctrlKey = false): KeyLike {
  const named: Record<string, Pick<KeyLike, "key" | "code" | "shiftKey">> = {
    "[": { key: "[", code: "BracketLeft", shiftKey: false },
    "]": { key: "]", code: "BracketRight", shiftKey: false },
    "+": { key: "+", code: "Equal", shiftKey: true },
    "−": { key: "-", code: "Minus", shiftKey: false },
    "1": { key: "1", code: "Digit1", shiftKey: false },
    Delete: { key: "Delete", code: "Delete", shiftKey: false },
  }
  const base = named[cap] ?? { key: cap.toLowerCase(), code: `Key${cap}`, shiftKey: false }
  return { ...base, ctrlKey, metaKey: false, altKey: false }
}

/** What each listed entry runs, one command per key (alternatives) or for the combination. */
const COMMANDS: Record<string, Command[]> = {
  Brush: [{ type: "tool", tool: "brush" }],
  Eraser: [{ type: "tool", tool: "eraser" }],
  Fill: [{ type: "tool", tool: "fill" }],
  Rectangle: [{ type: "tool", tool: "rect" }],
  Lasso: [{ type: "tool", tool: "lasso" }],
  Eyedropper: [{ type: "tool", tool: "eyedropper" }],
  Hand: [{ type: "tool", tool: "hand" }],
  "Brush size": [
    { type: "size", direction: -1 },
    { type: "size", direction: 1 },
  ],
  Undo: [{ type: "undo" }],
  Redo: [{ type: "redo" }],
  "Clear layer": [{ type: "clear" }],
  "Invert mask": [{ type: "invert" }],
  "Fit and center": [{ type: "fit" }],
  "Actual size": [{ type: "actual" }],
  "Zoom in or out": [
    { type: "zoom", direction: 1 },
    { type: "zoom", direction: -1 },
  ],
  Maximize: [{ type: "maximize" }],
}

/** Entries the editor handles outside commandFor: held keys, the wheel, drags, the candidates. */
const OUTSIDE = (keys: string[], group: string) => group === "Candidates" || keys.some((key) => ["Shift", "Wheel", "Drag", "Right drag"].includes(key))

describe("the editor's shortcut list", () => {
  it("runs, for each key it names, the command it names", () => {
    for (const { keys, label, group } of shortcutList(false)) {
      if (OUTSIDE(keys, group)) continue
      const expected = COMMANDS[label]
      expect(expected, label).toBeDefined()
      const presses = keys[0] === "Ctrl" ? [press(keys[1]!, true)] : keys.map((key) => press(key))
      expect(presses.map((event) => commandFor(event)), `${label}: ${keys.join("+")}`).toEqual(expected)
    }
  })

  // Delete / Backspace cleared the layer and the staged candidates took ← → Enter Esc, unlisted
  it("lists Delete and the staged candidates' keys", () => {
    const list = shortcutList(false)
    expect(list).toContainEqual({ keys: ["Delete"], label: "Clear layer", group: "Edit" })
    expect(commandFor(press("Delete"))).toEqual({ type: "clear" })
    expect(list.filter((item) => item.group === "Candidates").map((item) => item.keys.join("+"))).toEqual(["←", "→", "Enter", "Esc"])
  })
})
