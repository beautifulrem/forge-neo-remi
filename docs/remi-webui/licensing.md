# Licensing

Remi WebUI is licensed under the [GNU Affero General Public License v3.0](../../LICENSE),
like the Forge Neo, Forge and AUTOMATIC1111 WebUIs it derives from. Code
from other projects keeps its own license and notices: Settings → Licenses
lists the projects the shared runtime takes code from (`html/licenses.html`,
upstream's list), the web interface build's `THIRD-PARTY-NOTICES.txt` lists
every library it ships with its license text (`webui/tools/notices.mjs`), and
third-party code inside a built-in extension (DAVE, FreeFuse, JoyCaption's
prompts, Agent Scheduler, Spectrum, …) keeps its notice in that extension's
folder (`NOTICE`, `LICENSE-*`), listed under
[Built-in extensions](#built-in-extensions) below; the Licenses page does not
list those.

## Source code for users of the server (AGPL-3.0 section 13)

Section 13 asks a modified program that users interact with over a network
to offer those users its source. The web interface links the source of the
server it talks to in three places:

- **the sign-in screen** (under the form, also in the form shown over the
  app when a session ended): with `--auth`, people who cannot sign in reach
  nothing of the program but this form, and they are offered the source there;
- **Settings → Licenses**, under the WebUI's own license;
- **Settings → Server and models**, next to the version.

The link comes from the server: `source_url` of `GET /api/v2/auth/status`
(a public path) and of the bootstrap's `auth` (`remi/server/source.py`).

| case | link |
|---|---|
| default, the running commit is published in the repository (a remote-tracking ref of a remote pointing at `github.com/beautifulrem/forge-neo-remi` contains it) | `https://github.com/beautifulrem/forge-neo-remi/tree/<commit>` |
| default, the commit is unknown (no git checkout, such as a Docker image) or not published there (a local commit) | `https://github.com/beautifulrem/forge-neo-remi/tree/remi` |
| `--source-url URL` (or `REMI_SOURCE_URL` when the flag is absent) | `URL`, as given; an address that is not `http(s)`, or that carries a user name or password (the link is public), is ignored with a warning in the console |
| a client that has not signed in (with `--auth`) | the default's branch link, or `URL` as given: Remi never adds the commit it found, which would tell anyone who reaches the server which fixes it lacks (a `URL` naming a commit tells it as well) |

**Running a modified copy:** publish the source of what you run (your
changes included) and start the server with `--source-url` pointing at it,
for example `--source-url https://git.example.com/me/remi/src/commit/<commit>`.
Remi cannot tell where you publish which version, so it links that URL as
it is, on the sign-in screen too: with `--auth`, a branch URL keeps the
running commit from people who cannot sign in.

## Built-in extensions

Most directories under `extensions-builtin/` carry the license of the
project they come from (`LICENSE`, e.g. ADetailer, ControlNet, Tag
Autocomplete). The ones below have no license file of the WebUI's kind, or
only one for part of their code; each was traced from the repository's
history (the commit that added it, whether Forge Neo's `neo` branch has it),
its README, NOTICE and file headers, and the license of every project they
name. `tests/test_remi_builtin_licenses.py` fails when a directory has
neither a license file nor a row here.

Origins: **Forge Neo** — the directory is in upstream Forge Neo (`neo`
branch) and comes with it under the repository's AGPL-3.0, as it does
upstream; **Remi** — written for this repository (its NOTICE says so, it
was split from such a directory here, or it came with Remi's own Anima
changes to the upstream built-ins); **bundled** — added in one commit by
the Remi maintainer as a standalone extension (an install-into-`extensions/`
README, `<your-fork>` placeholders), with no public repository of its own
found: who wrote it before it was bundled, and under which license, is not
on record. Nothing in them names another author: the two that link to each
other do so at the same `<your-fork>` placeholder, one README was written for
this repository ("In Forge Neo Remi this extension is preinstalled"), and
all were committed in the same series as Remi's own Anima work (2026-09-25,
17:30:25 to 17:30:28). It is distributed here under the repository's
AGPL-3.0 by the maintainer who bundled it.

| directory | origin | license | third-party code inside | evidence |
|---|---|---|---|---|
| `agent-scheduler` | bundled third-party, then replaced: the Agent Scheduler of ArtVentureX ("Agent Scheduler Neo" fork); now only `preload.py` (its flags) and a Remi README | AGPL-3.0 (repository); Apache-2.0 for what is left of the original: `LICENSE-APACHE-2.0`, `NOTICE` | `preload.py`'s `--agent-scheduler-sqlite-file` flag | added 1fd1f69a4 (2026-09-25) with the original's CHANGELOG linking github.com/ArtVentureX/sd-webui-agent-scheduler, code removed in 4ae0d3732 / cd199f101; that repository's README states Apache License 2.0, it has no license file and names no copyright holder |
| `extra-options-section` | Forge Neo (from AUTOMATIC1111) | AGPL-3.0 (repository) | none named | added df02498d0 (AUTOMATIC, 2023-05-31); on `neo` and `classic` |
| `forge_preprocessor_inpaint` | Forge Neo (Forge, lllyasviel) | AGPL-3.0 (repository) | `lama/`: LaMa's network modules (the layout of advimman/lama `saicinpainting/training/modules`, Apache-2.0), `lama/modules/ffc.py` from pkumivision/FFC (Apache-2.0), `lama/modules/pix2pixhd.py` from NVIDIA/pix2pixHD (BSD 2-clause, Copyright (C) 2019 NVIDIA Corporation; the file keeps only its URL, so the copyright and license notice are restored in `lama/modules/NOTICE`) | added 7b0a202f8 (lllyasviel, 2024-01-30); on `neo` and `classic`; file headers name FFC and pix2pixHD; licenses read from the GitHub repositories |
| `forge_preprocessor_reference` | Forge Neo (Forge, lllyasviel) | AGPL-3.0 (repository) | none named | added 3aff8a5c6 (lllyasviel, 2024-01-30); on `neo` and `classic` |
| `forge_preprocessor_tile` | Forge Neo (Forge, lllyasviel) | AGPL-3.0 (repository) | none named | added 184bb9a04 (lllyasviel, 2024-01-30); on `neo` and `classic` |
| `sd-forge-anima-prompt-rescale` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established | none named | added 44af449db (2026-09-25); not on `neo`; README written for Forge Classic (neo) users, no author or license line |
| `sd-forge-anima-resolutions` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established | none named | added 44af449db (2026-09-25); not on `neo`; also described on a public Rentry page (rentry.co/mpihcm3y) that names no author, repository or license |
| `sd-forge-dynamic-shift` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established | none named | added 44af449db (2026-09-25); not on `neo`; README written for Forge Neo users, no author or license line |
| `sd-forge-prompt-blacklist` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established beyond its README, which says "MIT — do whatever you want with it." without a copyright line or license text | none named | added 44af449db (2026-09-25); not on `neo`; no repository of that name found |
| `sd-forge-prompt-workshop` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established | none named (booru tag data is fetched at run time) | added 44af449db (2026-09-25); not on `neo`; also described on rentry.co/mpihcm3y, which names no author, repository or license; its README: "In Forge Neo Remi this extension is preinstalled under `extensions-builtin/`" |
| `sd-lock-img2img-steps` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established beyond its README, which says "Do whatever you want with this." without a copyright line or license text | none named | added 44af449db (2026-09-25); not on `neo`; no repository of that name found; the README's License section, dropped when it was rewritten in 4ae0d3732, is restored |
| `sd-postprocess-suite` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established | none named | added 44af449db (2026-09-25); not on `neo`; README written for Forge Neo users, no author or license line |
| `sd-webui-joycaption-modified` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established; Apache-2.0 for JoyCaption's prompts: `LICENSE-APACHE-2.0`, `NOTICE` | `joycaption_lib/prompts.py`: JoyCaption's caption types and extra options (fpgaminer/joycaption, Apache-2.0, Copyright 2024 fpgaminer@bitcoin-mining.com) | added 1fd1f69a4 (2026-09-25); not on `neo`; README: "lifted verbatim from the official fancyfeast/joy-caption-beta-one HF Space" (no license there; the same prompts are in fpgaminer/joycaption `gradio-app/app.py`); links to `sd-webui-prompt-anchor` at `github.com/<your-fork>/sd-webui-prompt-anchor`, the placeholder of its own install line |
| `sd-webui-prompt-anchor` | bundled | AGPL-3.0 as distributed here (repository); the author's own license: not established beyond its README, which says "AGPL-3.0, matching the host webui" | none named | added 44af449db (2026-09-25); not on `neo`; no repository of that name found; `sd-webui-joycaption-modified`'s README links to it at the same `<your-fork>` placeholder |
| `sd_forge_anima_dave` | Remi | AGPL-3.0 (repository); MIT for the adapted DAVE code: `NOTICE` (license text and copyright) | DAVE block band, defaults and attenuation math from sorryhyun/ComfyUI-Anima-DAVE (MIT, Copyright (c) 2026 Seunghyun Ji) | added 87a6efd68 (2026-09-26, "ported from sorryhyun/ComfyUI-Anima-DAVE"); NOTICE: "Adapted for Forge Neo by the forge-neo-remi contributors"; the license checked on GitHub |
| `sd_forge_anima_freefuse` | Remi | AGPL-3.0 (repository); Apache-2.0 for the ported mask code: `LICENSE-APACHE-2.0`, `NOTICE` | `lib_anima_freefuse/masks.py` ports yaoliliu/FreeFuse `mask_utils.py` (Apache-2.0, no copyright holder named) | added eb99c7de6 (2026-09-25); NOTICE: "The Forge Neo/Anima runtime integration is an independent implementation"; the license checked on GitHub |
| `sd_forge_anima_guidance` | Remi | AGPL-3.0 (repository); Apache-2.0 and MIT parts: `LICENSE-APACHE-2.0`, `NOTICE` (with the MIT text) | Skimmed CFG predicates (Extraltodeus/Skimmed_CFG, Apache-2.0), CNS / DCW / SMC-CFG numerics (sorryhyun/ComfyUI-Spectrum-KSampler, MIT), APG reimplemented after diffusers (Apache-2.0), modulation guidance formula (quickjkee/modulation-guidance, MIT) | added 2e2e0cfbf (2026-09-25); NOTICE: "Modified for Forge Neo by the forge-neo-remi contributors" |
| `sd_forge_anima_hires_guard` | Remi | AGPL-3.0 (repository) | none named | added 6dfc5f7ff (2026-09-25) with Remi's Anima performance work; not on `neo`; README: "Forge Neo built-in, Anima-only" |
| `sd_forge_anima_presets` | Remi | AGPL-3.0 (repository) | none named | added 1ae2da334 (2026-09-26): the presets panel moved out of `sd_forge_anima_guidance` |
| `sd_forge_anima_regional` | Remi | AGPL-3.0 (repository); MIT for the adapted design: `NOTICE` (license text and copyright) | regional design from Sen-sou/Comfyui-Anima-Regional-Conditioning (MIT, Copyright (c) 2026 Vikas Vishwakarma) | added eb99c7de6 (2026-09-25); NOTICE: the rest "maintained in this repository"; the license checked on GitHub |
| `sd_forge_anima_xattn_boost` | Remi | AGPL-3.0 (repository); MIT for the method and defaults: `NOTICE` (license text and copyright) | re-implemented after sorryhyun/anima_lora (MIT, Copyright (c) 2026 Seunghyun Ji; "no files copied") | added f05e386a1 (2026-09-26); the license checked on GitHub |
| `sd_forge_compile` | Forge Neo (Haoming02) | AGPL-3.0 (repository) | `scripts/compile.py` names ComfyUI `comfy_extras/nodes_torch_compile.py` (GPL-3.0, compatible) | added d3707324e (Haoming, 2026-02-16); on `neo`; Remi forked `scripts/compile.py` (upstream-absorption.md §1.1) |
| `sd_forge_controlllite` | Forge Neo (Forge, lllyasviel) | AGPL-3.0 (repository) | `lib_controllllite.py` from kohya-ss/ControlNet-LLLite-ComfyUI (GPL-3.0, compatible); `lib_controllllite_anima.py` from kohya-ss/ComfyUI-Anima-LLLite (Apache-2.0) | added 8ee804a15 (lllyasviel, 2024-01-29); on `neo` and `classic`; file headers; licenses read from GitHub |
| `sd_forge_image_stitch` | Forge Neo (Haoming02) | AGPL-3.0 (repository) | none named | added 1518c7727 (Haoming, 2025-10-08); on `neo` |
| `sd_forge_json_builder` | Forge Neo (Haoming02) | AGPL-3.0 (repository) | none named | upstream be6fa5d43 (Haoming, 2026-10-21); ported here in baf97d131 (2026-10-27) |
| `sd_forge_lora` | Forge Neo (Forge, from AUTOMATIC1111's built-in Lora) | AGPL-3.0 (repository) | none named | added d1b8a2676 (layerdiffusion, 2024-08-02); on `neo` |
| `sd_forge_lora_schedule` | Forge Neo (Haoming02) | AGPL-3.0 (repository) | none named | added c2ae52e5f (Haoming, 2026-09-04); on `neo` |
| `sd_forge_multidiffusion` | Forge Neo (Forge, lllyasviel) | AGPL-3.0 (repository) as upstream ships it, **but see the third-party code**: | `lib_multidiffusion/tiled_diffusion.py`: "1st Edit by" shiimizu/ComfyUI-TiledDiffusion (no license file) and "Based on" pkuliyi2015/multidiffusion-upscaler-for-automatic1111, whose license is **CC BY-NC-SA 4.0: not compatible with AGPL-3.0** (non-commercial); Mixture of Diffusers after albarji/mixture-of-diffusers (MIT) | added dca0a1d5d (lllyasviel, 2024-02-09); on `neo` and `classic`; file header lines 1–4; licenses read from GitHub. Upstream's file: not changed here |
| `sd_forge_neveroom` | Forge Neo (Forge, lllyasviel) | AGPL-3.0 (repository) | none named | added 437c34892 (lllyasviel, 2024-02-24); on `neo` and `classic` |
| `sd_forge_pid` | Forge Neo (Haoming02) | AGPL-3.0 (repository) | none named (the PiD weights it downloads are for non-commercial research, as its description says: a model license, not code here) | added 10285899e (Haoming, 2026-06-26); on `neo` |
| `sd_forge_radial` | Forge Neo (Haoming02) | AGPL-3.0 (repository) | `scripts/forge_radial.py`, `lib_radial/attn_mask.py` from woct0rdho/ComfyUI-RadialAttn (Apache-2.0), after mit-han-lab/radial-attention (Apache-2.0) | added cd0c67cc5 (Haoming, 2026-02-21); on `neo`; file headers; licenses read from GitHub |
| `sd_forge_sparse` | Forge Neo (Haoming02) | AGPL-3.0 (repository) | `scripts/sparse.py` names ComfyUI `comfy_extras/nodes_sparse_attention.py` (GPL-3.0, compatible) | added bd69f1736 (Haoming, 2026-09-10); on `neo` |
| `sd_forge_spectrum` | Forge Neo (Haoming02); its processing forked by Remi | AGPL-3.0 (repository); MIT parts: `NOTICE` (license texts and copyrights) | Spectrum (hanjq17/Spectrum), ComfyUI-Spectrum-KSampler (sorryhyun) and comfyui-spectrum-sdxl (ruwwww), all MIT | added ae30d4eff (Haoming, 2026-03-09); on `neo`; Remi's NOTICE |
| `soft-inpainting` | Forge Neo (from AUTOMATIC1111) | AGPL-3.0 (repository) | none named | added 93eae6989 (AUTOMATIC1111, 2023-12-16); on `neo` and `classic` |
