import numpy as np
import torch
from PIL import Image

from backend.args import dynamic_args
from modules import images, scripts, sd_models
from modules.api import api
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, logger
from modules.sd_samplers_common import images_tensor_to_samples
from modules.shared import device, opts
from remi import ui

# upstream 04772d93: one "Edit Models" entry (a hover list there; Markdown has no hover)
_EDIT_MODELS = "Flux.1-Kontext, Flux.2-Klein, Qwen-Image-Edit, Qwen-Image-2.1, Anima Edit, Krea2 Edit, Moss Image Edit"

T2I_INFO = f"""For **Edit Models** ({_EDIT_MODELS}): Use in **txt2img** to achieve the effect of empty latent with custom resolution\\
For **Wan 2.2 I2V**: Use in **txt2img** to set as the Last Frame to achieve LastFrameToVideo\\
**Note:** This doesn't actually stitch the images; **Tip:** paste images straight into the list"""

I2I_INFO = f"""For **Edit Models** ({_EDIT_MODELS}): Use in **img2img** to achieve the effect of multiple input images\\
For **Wan 2.2 I2V**: Use in **img2img** to set as the Last Frame to achieve FirstLastFrameToVideo\\
**Note:** This doesn't actually stitch the images; **Tip:** paste images straight into the list"""


class ImageStitch(scripts.Script):
    sorting_priority = 529
    cached_parameters: tuple | None = None

    def title(self):
        return "ImageStitch Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        return ui.Section(
            self.title(),
            enable=ui.Toggle("enable", False, label=self.title()),
            children=[
                ui.Markdown(I2I_INFO if is_img2img else T2I_INFO),
                # Uploads (never restored from drafts) edited by the image-list widget: select, replace, remove, reorder
                ui.ImageList("references", label="Reference Image(s)", widget="image-list", props={"columns": 3}),
                # upstream caaa2cca: the side limit is the setting "Maximum Side Length for Reference
                # Images" (reference_max_side), which also covers img2img's reference image
            ],
        )

    @staticmethod
    def reset_references(p: StableDiffusionProcessing):
        # re-encode conditioning
        p.clear_prompt_cache()
        p.sd_model.clear_references()

    def process(self, p: StableDiffusionProcessing, enable: bool, references: list[str | Image.Image | tuple[Image.Image, str]], *_legacy):
        # ``_legacy``: the former "Maximum Side Length" argument, still sent by older API clients
        reference_images: list[Image.Image] | None = None
        if enable and references and dynamic_args.can_reference():
            reference_images = self.extract_images(references)

        is_WanT2I: bool = dynamic_args.wan and isinstance(p, StableDiffusionProcessingTxt2Img)

        # the references are encoded again when the inputs, the model or a setting the encoding reads changes
        cache: tuple | None = None
        if reference_images:
            cache = (opts.reference_max_side, opts.klein_do_reference, opts.qwen21_do_reference, opts.anima_do_reference, opts.krea2_do_reference, opts.moss_do_reference, str(sd_models.model_data.forge_loading_parameters), opts.img2img_background_color, tuple(self.hash_image(ref) for ref in reference_images), (p.width, p.height, p.batch_size) if is_WanT2I else None)

        if ImageStitch.cached_parameters == cache:
            return

        self.reset_references(p)
        ImageStitch.cached_parameters = cache

        if not reference_images:
            return

        if dynamic_args.wan:
            if is_WanT2I and p.batch_size == 1:
                logger.error("Wan 2.2 I2V requires more than one frame...")
                return
            if len(reference_images) > 1:
                logger.warning("Wan 2.2 I2V only uses the first reference image...")
                reference_images = [reference_images[0]]

        dynamic_args.is_referencing = True

        try:
            for reference in reference_images:
                if is_WanT2I:
                    reference = images.resize_image(1, reference, p.width, p.height)
                image = images.flatten(reference, opts.img2img_background_color)
                image = np.array(image, dtype=np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                image = torch.from_numpy(image).unsqueeze(0)
                if dynamic_args.wan:
                    p.sd_model.set_end_image(image, p.batch_size if is_WanT2I else 1)
                else:
                    images_tensor_to_samples(image.to(device=device), 0, p.sd_model)
        finally:
            dynamic_args.is_referencing = False

    @staticmethod
    def extract_images(gallery: list[str | Image.Image | tuple[Image.Image, str]]) -> list[Image.Image]:
        """PIL images from the form (resolved uploads), base64 strings (API
        clients) or ``(image, caption)`` gallery pairs."""
        images = []
        for item in gallery:
            if isinstance(item, str):
                item = api.decode_base64_to_image(item)
            elif isinstance(item, (tuple, list)):
                item = item[0]
            images.append(item)
        return images

    @staticmethod
    def hash_image(img: Image.Image) -> int:
        return hash((img.mode, img.size, img.convert("L").tobytes()))
