/**
 * The OpenPose editor: the preview's pose as JSON (edit, upload, download);
 * "Apply" draws it on the server and makes it the unit's input, as the old
 * editor's render did.
 */
import { Download, Play, Upload } from "@/components/icons"
import { useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { errorTextId } from "@/ui/a11y"
import { FieldError } from "@/ui/Field"
import { NS } from "./i18n"
import { MAX_CANVAS, parsePose, type PoseProblem, poseText } from "./pose"
import { panelClass, panelTitleClass } from "@/ui/layout/panel"
import { cn } from "@/lib/utils"

type PosePanelProps = {
  pose: unknown
  busy: boolean
  disabled?: boolean
  onApply: (pose: unknown) => void
}

export function PosePanel({ pose, busy, disabled, onApply }: PosePanelProps) {
  const { t: tw } = useTranslation(NS)
  const id = useId()
  const initial = poseText(pose)
  const [text, setText] = useState(initial)
  // the problem itself, said at render: a language picked after it says it again
  const [problem, setProblem] = useState<PoseProblem | null>(null)
  const file = useRef<HTMLInputElement>(null)

  const problemText = (problem: PoseProblem): string => {
    switch (problem.reason) {
      case "json":
        return tw("The pose is not valid JSON")
      case "empty":
        return tw("The pose list is empty")
      case "object":
        return tw("A pose is a JSON object")
      case "size":
        return tw("The pose's {{key}} must be a size between 1 and {{max}}", { replace: { key: problem.key, max: MAX_CANVAS } })
      case "list":
        return tw("The pose's {{key}} must be a list", { replace: { key: problem.key } })
    }
  }

  const error = problem ? problemText(problem) : null

  const apply = (source: string) => {
    const parsed = parsePose(source)
    if (!parsed.ok) {
      setProblem(parsed.problem)
      return
    }
    setProblem(null)
    onApply(parsed.pose)
  }

  const upload = async (files: FileList | null) => {
    const picked = files?.[0]
    if (file.current) file.current.value = ""
    if (!picked) return
    const source = await picked.text()
    setText(source)
    apply(source)
  }

  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }))
    const link = document.createElement("a")
    link.href = url
    link.download = "pose.json"
    document.body.append(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div role="group" aria-labelledby={`${id}-title`} className={cn("flex flex-col gap-2", panelClass(1))}>
      <div className="flex flex-wrap items-center gap-1.5">
        <span id={`${id}-title`} className={cn("mr-auto", panelTitleClass(1))}>
          {tw("Pose (OpenPose JSON)")}
        </span>
        <Button size="xs" variant="ghost" onClick={() => file.current?.click()} disabled={disabled || busy}>
          <Upload aria-hidden />
          {tw("Upload JSON")}
        </Button>
        <Button size="xs" variant="ghost" onClick={download} disabled={!text.trim()}>
          <Download aria-hidden />
          {tw("Download JSON")}
        </Button>
        <Button size="xs" variant="secondary" onClick={() => apply(text)} disabled={disabled || busy || !text.trim()}>
          <Play aria-hidden />
          {tw("Apply pose")}
        </Button>
        <input ref={file} type="file" accept=".json,application/json" hidden onChange={(event) => void upload(event.target.files)} />
      </div>
      <Textarea
        id={id}
        aria-label={tw("Pose (OpenPose JSON)")}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? errorTextId(id) : undefined}
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder={tw("Run an openpose preprocessor, or upload a pose JSON")}
        spellCheck={false}
        disabled={disabled}
        className="max-h-56 min-h-24 font-mono text-xs"
      />
      {error ? <FieldError id={id} message={error} /> : null}
    </div>
  )
}
