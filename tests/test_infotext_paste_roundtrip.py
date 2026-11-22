"""Infotext paste round-trips (``POST /api/v2/infotext/parse``) for keys the
generic infotext mapping cannot restore on its own: the refiner checkpoint and
the img2img "Inpaint area"."""

from types import SimpleNamespace

import pytest

from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def _infotext(prompt, params):
    from modules.infotext_utils import quote

    return f"{prompt}\nNegative prompt: \n" + ", ".join(f"{key}: {quote(value)}" for key, value in params.items())


# region Refiner


@pytest.fixture
def checkpoints(webui, monkeypatch):
    """Two registered checkpoints, as sd_models.list_models() registers them."""
    from modules import sd_models

    monkeypatch.setattr(sd_models, "checkpoints_list", {})
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {})
    infos = []
    for name, shorthash in (("anima-copy.safetensors", "fba1195327"), ("sub/anima-turbo-v1.1.safetensors", None)):
        stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        info = SimpleNamespace(
            name=name,
            name_for_extra=stem,
            model_name=name.rsplit(".", 1)[0].replace("/", "_"),
            title=name if shorthash is None else f"{name} [{shorthash}]",
            short_title=stem if shorthash is None else f"{stem} [{shorthash}]",
            filename=f"/models/{name}",
        )
        info.ids = [info.title, info.name, info.name_for_extra, info.model_name] + ([shorthash, info.short_title] if shorthash else [])
        sd_models.checkpoints_list[info.title] = info
        for alias in info.ids:
            sd_models.checkpoint_aliases[alias] = info
        infos.append(info)
    return infos


@pytest.mark.parametrize("use_short", [False, True])
def test_refiner_checkpoint_pastes_as_one_of_the_choices(webui, checkpoints, use_short):
    from modules.processing_scripts.refiner import checkpoint_choices
    from remi.core.infotext import parse_infotext

    webui.opts.data["sd_checkpoint_dropdown_use_short"] = use_short
    try:
        choices = checkpoint_choices()
        for info in checkpoints:
            # sd_samplers_common.apply_refiner writes the short title
            text = _infotext("cat", {"Steps": 20, "Sampler": "Euler", "Refiner": info.short_title, "Refiner switch at": 0.5, "Refiner CFG scale": 4.5})
            result = parse_infotext(text, "txt2img")
            assert result["problems"] == []
            refiner = result["scripts"]["refiner"]
            assert refiner["enable"] is True and refiner["switch_at"] == 0.5 and refiner["cfg"] == 4.5
            assert refiner["checkpoint"] in choices
            assert refiner["checkpoint"] == (info.short_title if use_short else info.name)
            # ...and the pasted values validate for a new job
            from remi.core.tabs import processing_args

            processing_args("txt2img", dict(result["core"]), result["scripts"])
    finally:
        webui.opts.data.pop("sd_checkpoint_dropdown_use_short", None)


@pytest.mark.parametrize("use_short", [False, True])
def test_hires_checkpoint_pastes_as_one_of_the_choices(webui, checkpoints, use_short):
    """processing.py writes ``Hires checkpoint: <short title>``; the choices are
    the file names unless short names are on."""
    from remi.core.infotext import parse_infotext

    webui.opts.data["sd_checkpoint_dropdown_use_short"] = use_short
    try:
        for info in checkpoints:
            text = _infotext("cat", {"Steps": 20, "Denoising strength": 0.5, "Hires upscale": 2, "Hires upscaler": "Latent", "Hires checkpoint": info.short_title})
            result = parse_infotext(text, "txt2img")
            assert result["problems"] == []
            assert result["core"]["hr_checkpoint_name"] == (info.short_title if use_short else info.name)
        same = parse_infotext(_infotext("cat", {"Steps": 20, "Denoising strength": 0.5, "Hires upscale": 2, "Hires checkpoint": "Use same checkpoint"}), "txt2img")
        assert same["core"]["hr_checkpoint_name"] == "Use same checkpoint" and same["problems"] == []
    finally:
        webui.opts.data.pop("sd_checkpoint_dropdown_use_short", None)


def test_unknown_refiner_checkpoint_is_left_alone(webui, checkpoints):
    from remi.core.infotext import parse_infotext

    result = parse_infotext(_infotext("cat", {"Steps": 20, "Refiner": "missing-model [0123456789]", "Refiner switch at": 0.5}), "txt2img")
    assert "checkpoint" not in result["scripts"]["refiner"] and result["problems"] == []


def test_refiner_infotext_has_the_hash_even_before_the_checkpoint_was_hashed(webui, monkeypatch):
    """A refiner checkpoint not hashed yet wrote "Refiner: anima-copy", the same run
    pasted back "Refiner: anima-copy [fba1195327]" (hashed by then): the infotext
    did not round-trip. apply_refiner hashes it before writing the key."""
    from modules import sd_samplers_common, shared

    info = SimpleNamespace(name_for_extra="anima-copy", short_title="anima-copy", shorthash=None, filename="/models/anima-copy.safetensors")

    def calculate_shorthash():
        info.shorthash, info.short_title = "fba1195327", "anima-copy [fba1195327]"
        return info.shorthash

    info.calculate_shorthash = calculate_shorthash
    p = SimpleNamespace(refiner_switch_at=0.5, refiner_checkpoint_info=info, enable_hr=False, extra_generation_params={})
    denoiser = SimpleNamespace(p=p, step=8, total_steps=8)
    monkeypatch.setattr(shared, "sd_model", SimpleNamespace(sd_checkpoint_info=object(), forge_objects=SimpleNamespace(unet=None)), raising=False)
    # the module's own opts (other suites may swap shared.opts)
    monkeypatch.setattr(sd_samplers_common.opts, "refiner_use_steps", True, raising=False)
    monkeypatch.setattr(sd_samplers_common.opts, "refiner_fast_sd", False, raising=False)
    monkeypatch.setattr(sd_samplers_common, "ORIGINAL_CHECKPOINT", None)
    monkeypatch.setattr(sd_samplers_common, "sampling_cleanup", lambda unet: None)
    # stop before any model is loaded
    monkeypatch.setattr(sd_samplers_common.main_entry, "checkpoint_change", lambda *args, **kwargs: False)

    assert sd_samplers_common.apply_refiner(denoiser, None, 0.0) is False  # past the switch either way
    assert p.extra_generation_params["Refiner"] == "anima-copy [fba1195327]"


# endregion

# region Inpaint area


@pytest.mark.parametrize(
    ("params", "area"),
    [
        # written by processing since the round-trip fix
        ({"Mask blur": 4, "Inpaint area": "Whole picture"}, "Whole picture"),
        ({"Mask blur": 4, "Inpaint area": "Only masked", "Masked area padding": 32}, "Only masked"),
        # inpaint infotexts written before: no key means the whole picture
        ({"Mask blur": 4}, "Whole picture"),
        ({"Masked content": "latent noise"}, "Whole picture"),
        ({"Mask mode": "Inpaint not masked"}, "Whole picture"),
    ],
)
def test_inpaint_area_round_trips(webui, params, area):
    from remi.core.infotext import parse_infotext

    result = parse_infotext(_infotext("dog", {"Steps": 20, "Sampler": "Euler", "Denoising strength": 0.75, **params}), "img2img")
    assert result["problems"] == [] and result["core"]["inpaint_full_res"] == area


def test_plain_img2img_infotext_leaves_inpaint_area_alone(webui):
    from remi.core.infotext import parse_infotext

    result = parse_infotext(_infotext("dog", {"Steps": 20, "Sampler": "Euler", "Denoising strength": 0.75}), "img2img")
    assert "inpaint_full_res" not in result["core"]


def test_whole_picture_inpaint_writes_inpaint_area():
    """processing writes the key for "Whole picture" (the form's default is "Only masked")."""
    import inspect

    from modules.processing import StableDiffusionProcessingImg2Img

    src = inspect.getsource(StableDiffusionProcessingImg2Img.init)
    assert 'self.extra_generation_params["Inpaint area"] = "Whole picture"' in src
    assert 'self.extra_generation_params["Inpaint area"] = "Only masked"' in src


# endregion


# region Hires modules


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"Hires Module 1": "Built-in"}, []),
        ({"Hires Module 1": "Use same choices"}, ["Use same choices"]),
        ({"Hires Module 1": "ae", "Hires Module 2": "qwen"}, ["ae.safetensors", "qwen.safetensors"]),
        ({"Hires Module 1": "missing_vae"}, ["Use same choices"]),  # not installed here: not Built-in
        ({}, ["Use same choices"]),
    ],
)
def test_hires_modules_paste_as_written(webui, monkeypatch, params, expected):
    from modules.infotext_utils import parse_generation_parameters
    from modules_forge import main_entry

    monkeypatch.setattr(main_entry, "module_list", {"ae.safetensors": "/m/ae.safetensors", "qwen.safetensors": "/m/qwen.safetensors"})
    res = parse_generation_parameters(_infotext("dog", {"Steps": 20, "Hires upscale": 2, **params}))
    assert res["Hires VAE/TE"] == expected
    assert not any(key.startswith("Hires Module") for key in res)


def test_hires_modules_paste_into_the_form(webui, monkeypatch):
    from modules_forge import main_entry
    from remi.core.infotext import parse_infotext

    monkeypatch.setattr(main_entry, "module_list", {"ae.safetensors": "/m/ae.safetensors"})
    params = {"Steps": 20, "Sampler": "Euler", "Size": "512x512", "Denoising strength": 0.5, "Hires upscale": 2}
    for recorded, expected in (("Built-in", []), ("ae", ["ae.safetensors"]), ("Use same choices", ["Use same choices"])):
        result = parse_infotext(_infotext("dog", {**params, "Hires Module 1": recorded}), "txt2img")
        assert result["problems"] == [] and result["core"]["hr_additional_modules"] == expected


# endregion


# region Inpaint defaults


def test_inpaint_settings_left_out_at_their_defaults_paste_as_the_defaults(webui):
    # Processing writes Mask blur only above 0, Mask mode only inverted and
    # Masked content only other than original
    from remi.core.infotext import parse_infotext

    base = {"Steps": 20, "Sampler": "Euler", "Denoising strength": 0.75}
    result = parse_infotext(_infotext("dog", {**base, "Inpaint area": "Whole picture"}), "img2img")
    assert result["problems"] == []
    assert (result["core"]["mask_blur"], result["core"]["inpainting_mask_invert"], result["core"]["inpainting_fill"]) == (0, "Inpaint masked", "original")
    written = parse_infotext(_infotext("dog", {**base, "Inpaint area": "Whole picture", "Mask blur": 8, "Mask mode": "Inpaint not masked", "Masked content": "fill"}), "img2img")
    assert (written["core"]["mask_blur"], written["core"]["inpainting_mask_invert"], written["core"]["inpainting_fill"]) == (8, "Inpaint not masked", "fill")
    plain = parse_infotext(_infotext("dog", base), "img2img")["core"]
    assert not {"mask_blur", "inpainting_mask_invert", "inpainting_fill"} & set(plain)


# endregion


# region Hires fix toggle


@pytest.mark.parametrize(
    ("params", "enabled"),
    [
        ({"Denoising strength": 0.75}, False),  # an img2img infotext: no hires pass
        ({"Denoising strength": 0.5, "Hires upscale": 2, "Hires upscaler": "Latent"}, True),
        ({"Denoising strength": 0.5, "Hires resize": "1024x1536", "Hires upscaler": "Latent"}, True),
        ({"Denoising strength": 0.5, "Hires resize": "1024x1536"}, True),
        ({}, False),
    ],
)
def test_hires_fix_pastes_on_only_for_a_hires_infotext(webui, params, enabled):
    from remi.core.infotext import parse_infotext

    result = parse_infotext(_infotext("dog", {"Steps": 20, "Sampler": "Euler", "Size": "512x512", **params}), "txt2img")
    assert result["core"]["enable_hr"] is enabled


# endregion
