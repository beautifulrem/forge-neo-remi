"""Real-model checks of the depth-expanded Anima variants (2.9B, 3.8B).

    python tools/smoke_anima_variants.py                 # every variant installed in models/
    python tools/smoke_anima_variants.py --only 2.9b     # one of them
    python tools/smoke_anima_variants.py --only 2b --checks pose   # some of the checks

Starts the server like ``tools/smoke_real.py`` and, for each variant whose checkpoint is in
``models/Stable-diffusion`` (3.8B also needs ``qwen35_4b.safetensors`` in ``models/text_encoder``):

1. a txt2img with the variant's recommended sampler gives a real picture (not flat, not noise);
2. a LoRA trained on 2B (28 blocks) is remapped to the variant's depth: the infotext's
   ``Lora remap`` names ``28→40`` / ``28→52`` and the image changes; once with inserted blocks
   ``copy`` and once ``skip`` (both images are saved for a side-by-side look);
3. ``<lora:name:1:src=28>`` gives the same picture as the detected depth;
4. a ControlNet-LLLite lineart model trained on 2B follows a line drawing on the variant;
5. ``/controlnet/model_list`` only lists Anima control models;
6. the Anima-Control-Pose control-LoRA (``anima_pose_preview2``) changes the picture at weight 1
   against weight 0 (the same LoRA, no control); both images and the skeleton are saved, since
   whether the figure takes the pose is for the eye;
7. (3.8B) the semantic connector is live: running it for the negative prompt too changes the
   picture, and the model without its Qwen3.5 encoder fails with an error that names the file.

The 2B variants (the turbo checkpoint ``tools/smoke_real.py`` uses, and the base one) have no
depth to re-map: they are there for the control-LoRA (``--only 2b-base --checks pose``).

Images and the server log go to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_real import CONTROLNET_MODULE, ROOT, Report, Server, b64png, control_drawing, decode, diff, line_correlation, other_launchers, parse_infotext, wait_job

VAE = "models/VAE/qwen_image_vae.safetensors"
QWEN3 = "models/text_encoder/qwen_3_06b_base.safetensors"
QWEN35 = "models/text_encoder/qwen35_4b.safetensors"
PROMPT = "1girl, solo, red hair, long hair, white dress, standing in a flower garden, looking at viewer, smile, masterpiece, best quality"
NEGATIVE = "worst quality, low quality, blurry, jpeg artifacts"
TURBO_LORA = "anima-turbo-lora-v0.2"
STYLE_LORA = "anima-greg-rutkowski-style"
LLLITE_LINEART = "anima-lllite-lineart-1"
POSE_CONTROL_LORA = "anima_pose_preview2"
CHECKS = ("lora", "src", "controlnet", "pose", "semantic")


@dataclass(frozen=True)
class Variant:
    key: str
    checkpoint: str
    depth: int
    modules: tuple[str, ...]
    sampler: str
    scheduler: str
    steps: int
    cfg: float


VARIANTS = (
    Variant("2b", "anima-turbo-v1.1.safetensors", 28, (VAE, QWEN3), "Euler", "Simple", 8, 1.0),
    Variant("2b-base", "anima-base-v1.0.safetensors", 28, (VAE, QWEN3), "Euler", "Simple", 30, 4.0),
    Variant("2.9b", "Anima-2.9B-preview-v1.safetensors", 40, (VAE, QWEN3), "Euler", "SGM Uniform", 28, 4.0),
    Variant("3.8b", "Anima-3.8B-v1.1.safetensors", 52, (VAE, QWEN3, QWEN35), "Res Multistep", "Beta", 28, 5.0),
)


def picture_stats(image) -> tuple[float, float]:
    """(pixel standard deviation, mean absolute difference between neighbouring pixels): a flat
    image has a low first value, pure noise a second value close to the first."""

    import numpy as np

    array = np.asarray(image.convert("L"), dtype=np.float32)
    return float(array.std()), float(np.abs(np.diff(array, axis=1)).mean())


# LoRA -> (steps, CFG) it is meant for (the turbo LoRA: few steps without CFG)
LORA_SETTINGS = {TURBO_LORA: (8, 1.0)}


def txt2img(server: Server, variant: Variant, out: Path, name: str, *, prompt: str = PROMPT, scripts: dict | None = None, size: int = 832, seed: int = 20251119, steps: int | None = None, cfg: float | None = None):
    body = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        "seed": seed,
        "steps": steps or variant.steps,
        "cfg_scale": cfg if cfg is not None else variant.cfg,
        "sampler_name": variant.sampler,
        "scheduler": variant.scheduler,
        "width": size,
        "height": size,
    }
    if scripts:
        body["alwayson_scripts"] = scripts
    started = time.monotonic()
    answer = server.request("POST", "/sdapi/v1/txt2img", body, timeout=3600)
    image = decode(answer["images"][0]).convert("RGB")
    image.save(out / f"{variant.key}_{name}.png")
    info = json.loads(answer["info"])["infotexts"][0]
    return image, info, time.monotonic() - started


def set_options(server: Server, **values) -> None:
    server.request("PATCH", "/api/v2/options", values)


def pose_skeleton(size: tuple[int, int]):
    """A pose in the style the Anima-Control-Pose adapter was trained on (its node's "R0_thin":
    COCO 17 keypoints, thin coloured limbs, white joints, on black): both arms raised and the legs
    apart, far from the prompt's standing pose."""

    from PIL import Image, ImageDraw

    width, height = size
    # nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles (left = the figure's left)
    points = [(0.50, 0.17), (0.52, 0.15), (0.48, 0.15), (0.54, 0.16), (0.46, 0.16), (0.60, 0.27), (0.40, 0.27), (0.68, 0.17), (0.32, 0.17), (0.73, 0.06), (0.27, 0.06)]
    points += [(0.56, 0.52), (0.44, 0.52), (0.67, 0.70), (0.33, 0.70), (0.76, 0.90), (0.24, 0.90)]
    limbs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6)]
    palette = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0), (0, 255, 128), (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128), (180, 180, 180), (220, 220, 220)]
    image = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(image)
    scale = max(width, height) / 768  # the node draws 2 px limbs and 3 px joints at 768
    xy = [(x * width, y * height) for x, y in points]
    for (a, b), colour in zip(limbs, palette, strict=True):
        draw.line((*xy[a], *xy[b]), fill=colour, width=max(2, round(2 * scale)))
    radius = max(3, round(3 * scale))
    for x, y in xy:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")
    return image


def run_pose(server: Server, variant: Variant, out: Path, report: Report, base_size: tuple[int, int]) -> None:
    label = variant.key
    models = server.request("GET", "/controlnet/model_list")["model_list"]
    model = next((name for name in models if POSE_CONTROL_LORA in name), None)
    if model is None:
        report.check(f"{label} control-LoRA (needs models/ControlNet/{POSE_CONTROL_LORA}.safetensors)", False, "not listed")
        return
    skeleton = pose_skeleton(base_size)
    skeleton.save(out / f"{label}_pose_skeleton.png")
    images = {}
    for weight in (1.0, 0.0):
        unit = {"enabled": True, "module": "None", "model": model, "image": b64png(skeleton), "weight": weight, "pixel_perfect": True}
        image, info, seconds = txt2img(server, variant, out, f"pose_w{weight:g}", prompt=PROMPT.replace("standing", "posing"), scripts={"ControlNet": {"args": [unit]}})
        images[weight] = image
        unit_info = parse_infotext(info).get("ControlNet 0", "").strip('"')
        report.check(f"{label} control-LoRA weight {weight:g} infotext", f"Model: {model}" in unit_info, f"{unit_info[:160]}, {seconds:.0f} s")
    effect, _ = diff(images[1.0], images[0.0])
    report.check(f"{label} control-LoRA changes the picture", effect > 2.0, f"mean diff weight 1 vs 0: {effect:.2f} (see {label}_pose_w1.png, _w0.png, _pose_skeleton.png)")


def run_semantic(server: Server, variant: Variant, out: Path, report: Report, base) -> None:
    label = variant.key
    set_options(server, anima_semantic_negative=True)
    try:
        image, _, seconds = txt2img(server, variant, out, "semantic_negative")
    finally:
        set_options(server, anima_semantic_negative=False)
    effect, _ = diff(image, base)
    report.check(f"{label} connector on the negative prompt changes the image", effect > 0.5, f"mean diff {effect:.2f}, {seconds:.0f} s")

    set_options(server, forge_additional_modules=[str(ROOT / path) for path in variant.modules if path != QWEN35])
    try:
        # through the SPA's job API: /sdapi answers an internal error with only its id
        body = {"tab": "txt2img", "core": {"prompt": PROMPT, "negative_prompt": NEGATIVE, "width": 512, "height": 512, "cfg_scale": variant.cfg, "batch_size": 1, "batch_count": 1}, "scripts": {"sampler": {"sampler_name": variant.sampler, "scheduler": variant.scheduler, "steps": 2}}}
        job = wait_job(server, server.request("POST", "/api/v2/jobs", body)["job_id"])
        error = job.get("error") or ""
        report.check(f"{label} without Qwen3.5 fails, naming the file", job["status"] != "completed" and "qwen35_4b" in error, f"{job['status']}: {error[:300]}")
    finally:
        set_options(server, forge_additional_modules=[str(ROOT / path) for path in variant.modules])


def run_variant(server: Server, variant: Variant, out: Path, report: Report, *, checks: tuple[str, ...] = CHECKS) -> None:
    label = variant.key
    print(f"{label}: selecting {variant.checkpoint}", flush=True)
    set_options(server, sd_model_checkpoint=variant.checkpoint, forge_additional_modules=[str(ROOT / path) for path in variant.modules], anima_lora_inserted_blocks="copy")

    base, info, seconds = txt2img(server, variant, out, "base")
    spread, detail = picture_stats(base)
    report.check(f"{label} txt2img gives a picture", spread > 25 and detail < spread * 0.5, f"std {spread:.1f}, neighbour diff {detail:.1f}, {seconds:.0f} s")
    report.check(f"{label} infotext model", variant.checkpoint.rsplit(".", 1)[0] in info, info.splitlines()[-1][:300])

    if "pose" in checks:
        run_pose(server, variant, out, report, base.size)
    if "semantic" in checks and QWEN35 in variant.modules:
        run_semantic(server, variant, out, report, base)
    if variant.depth == 28:
        return  # nothing to re-map

    loras = {path.stem for path in (ROOT / "models" / "Lora").glob("*.safetensors")}
    for lora in (STYLE_LORA, TURBO_LORA) if "lora" in checks else ():
        if lora not in loras:
            report.check(f"{label} LoRA {lora} (needs models/Lora/{lora}.safetensors)", False, "not installed")
            continue
        images = {}
        steps, cfg = LORA_SETTINGS.get(lora, (None, None))
        for policy in ("copy", "skip"):
            set_options(server, anima_lora_inserted_blocks=policy)
            image, info, seconds = txt2img(server, variant, out, f"{lora}_{policy}", prompt=f"{PROMPT} <lora:{lora}:1>", steps=steps, cfg=cfg)
            images[policy] = image
            remap = parse_infotext(info).get("Lora remap", "").strip('"')
            report.check(f"{label} {lora} ({policy}) remapped 28→{variant.depth}", f"28→{variant.depth}" in remap, remap or info.splitlines()[-1][:300])
            if steps is None:
                effect, _ = diff(image, base)
                report.check(f"{label} {lora} ({policy}) changes the image", effect > 2.0, f"mean diff to without LoRA {effect:.2f}, {seconds:.0f} s")
        between, _ = diff(images["copy"], images["skip"])
        report.check(f"{label} {lora}: copy and skip differ", between > 0.5, f"mean diff {between:.2f} (see {label}_{lora}_copy.png / _skip.png)")
    set_options(server, anima_lora_inserted_blocks="copy")

    if "src" in checks and STYLE_LORA in loras:
        image, info, _ = txt2img(server, variant, out, f"{STYLE_LORA}_src28", prompt=f"{PROMPT} <lora:{STYLE_LORA}:1:src=28>")
        same, peak = diff(image, decode_saved(out / f"{label}_{STYLE_LORA}_copy.png"))
        report.check(f"{label} src=28 matches the detected depth", same == 0 and peak == 0, f"mean {same:.4f}, max {peak}")

    if "controlnet" not in checks:
        return
    models = server.request("GET", "/controlnet/model_list")["model_list"]
    real = [name for name in models if name and name != "None"]
    report.check(f"{label} ControlNet list has only Anima models", all("anima" in name.lower() for name in real), ", ".join(real)[:300])
    lineart = next((name for name in real if LLLITE_LINEART in name), None)
    if lineart is None:
        report.check(f"{label} LLLite lineart (needs models/ControlNet/{LLLITE_LINEART}.safetensors)", False, "not listed")
        return
    drawing = control_drawing(base.size)
    unit = {"enabled": True, "module": CONTROLNET_MODULE, "model": lineart, "image": b64png(drawing), "weight": 1.0, "pixel_perfect": True}
    image, info, seconds = txt2img(server, variant, out, "lllite_lineart", scripts={"ControlNet": {"args": [unit]}})
    unit_info = parse_infotext(info).get("ControlNet 0", "").strip('"')
    report.check(f"{label} LLLite infotext", f"Model: {lineart}" in unit_info, unit_info[:300] or info.splitlines()[-1][:300])
    with_control, without = line_correlation(image, drawing), line_correlation(base, drawing)
    report.check(f"{label} LLLite follows the drawing", with_control > without + 0.05, f"edge correlation {with_control:.3f} (without {without:.3f}), {seconds:.0f} s")


def decode_saved(path: Path):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.load()
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=7881)
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "remi-smoke-variants")
    parser.add_argument("--only", choices=[variant.key for variant in VARIANTS], action="append")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--checks", default=",".join(CHECKS), help=f"comma-separated, of {', '.join(CHECKS)} (the base picture is always made)")
    options = parser.parse_args()
    checks = tuple(check for check in options.checks.split(",") if check)
    if unknown := set(checks) - set(CHECKS):
        parser.error(f"unknown checks: {', '.join(sorted(unknown))}")

    variants = [variant for variant in VARIANTS if not options.only or variant.key in options.only]
    missing = {variant.key: [path for path in (f"models/Stable-diffusion/{variant.checkpoint}", *variant.modules) if not (ROOT / path).is_file()] for variant in variants}
    for key, paths in missing.items():
        if paths:
            print(f"{key}: skipped, missing {', '.join(paths)}", flush=True)
    variants = [variant for variant in variants if not missing[variant.key]]
    if not variants:
        print("no variant installed", file=sys.stderr)
        return 2
    if busy := other_launchers():
        print("another launch.py is running:\n  " + "\n  ".join(busy), file=sys.stderr)
        return 2

    options.out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(tempfile.mkdtemp(prefix="remi-smoke-variants-"))
    if (ROOT / "config.json").is_file():
        shutil.copy2(ROOT / "config.json", data_dir / "config.json")
    report = Report()
    server = Server(options.python, options.port, data_dir, options.out / "server.log")
    try:
        if report.check("server starts", server.wait_ready(options.startup_timeout)):
            for variant in variants:
                try:
                    run_variant(server, variant, options.out, report, checks=checks)
                except Exception as error:  # one variant failing still runs the next
                    report.check(f"{variant.key} run", False, f"{type(error).__name__}: {error}"[:2000])
    finally:
        server.stop()
        shutil.rmtree(data_dir, ignore_errors=True)

    print(f"\n{'check':<56} result  detail")
    for name, ok, detail in report.rows:
        print(f"{name:<56} {'pass' if ok else 'FAIL':<7} {detail}")
    print(f"\nImages: {options.out}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
