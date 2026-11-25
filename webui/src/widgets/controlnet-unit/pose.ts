/** OpenPose JSON text of the pose editor. */

/** The largest canvas side the server draws (`MAX_CANVAS` in lib_controlnet/openpose.py). */
export const MAX_CANVAS = 4096

/** Why editor text is not a pose: the server's reasons (`check_pose`), which the panel translates. */
export type PoseProblem = { reason: "json" } | { reason: "empty" } | { reason: "object" } | { reason: "size"; key: string } | { reason: "list"; key: string }

export type ParsedPose = { ok: true; pose: unknown } | { ok: false; problem: PoseProblem }

/** The pose as indented JSON ("" for none). */
export function poseText(pose: unknown): string {
  return pose === null || pose === undefined ? "" : JSON.stringify(pose, null, 2)
}

/** A pose from editor text: an OpenPose JSON object (or a list of them, the first is used by the server). */
export function parsePose(text: string): ParsedPose {
  let pose: unknown
  try {
    pose = JSON.parse(text)
  } catch {
    return { ok: false, problem: { reason: "json" } }
  }
  if (Array.isArray(pose) && pose.length === 0) return { ok: false, problem: { reason: "empty" } }
  const first = Array.isArray(pose) ? pose[0] : pose
  if (typeof first !== "object" || first === null || Array.isArray(first)) return { ok: false, problem: { reason: "object" } }
  const record = first as Record<string, unknown>
  for (const key of ["canvas_width", "canvas_height"]) {
    const size = record[key]
    if (typeof size !== "number" || !Number.isFinite(size) || size < 1 || size > MAX_CANVAS) return { ok: false, problem: { reason: "size", key } }
  }
  for (const key of ["people", "animals"]) {
    if (record[key] !== undefined && !Array.isArray(record[key])) return { ok: false, problem: { reason: "list", key } }
  }
  return { ok: true, pose }
}
