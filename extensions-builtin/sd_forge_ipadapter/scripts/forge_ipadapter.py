from pathlib import Path

from lib_ipadapter.IPAdapterPlus import InsightFaceLoader, IPAdapterApply

from modules_forge.shared import add_supported_control_model, add_supported_preprocessor
from modules_forge.supported_controlnet import ControlModelPatcher
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorClipVision, PreprocessorParameter
from modules_forge.utils import numpy_to_pytorch


class PreprocessorClipVisionForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.tags = ["IP-Adapter"]
        self.model_filename_filters = ["IP-Adapter", "IP_Adapter"]
        self.sorting_priority = 20

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            unfold_batch=False,
        )
        return cond


class PreprocessorClipVisionWithInsightFaceForIPAdapter(PreprocessorClipVisionForIPAdapter):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.cached_insightface = None

    def load_insightface(self):
        if self.cached_insightface is None:
            self.cached_insightface = InsightFaceLoader.load_insight_face()
        return self.cached_insightface

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            insightface=self.load_insightface(),
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            unfold_batch=False,
        )
        return cond


class PreprocessorInsightFaceForInstantID(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ["Instant-ID"]
        self.model_filename_filters = ["Instant-ID", "Instant_ID"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.show_control_mode = False
        self.sorting_priority = 10
        self.cached_insightface = None

    def load_insightface(self):
        if self.cached_insightface is None:
            self.cached_insightface = InsightFaceLoader.load_insight_face(name="antelopev2")
        return self.cached_insightface

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(clip_vision=None, insightface=self.load_insightface(), image=numpy_to_pytorch(input_image), weight_type="original", noise=0.0, embeds=None, unfold_batch=False, instant_id=True)
        return cond


add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(name="CLIP-ViT-H (IPAdapter)", url="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", filename="CLIP-ViT-H-14.safetensors"))

add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(name="CLIP-ViT-bigG (IPAdapter)", url="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors", filename="CLIP-ViT-bigG.safetensors"))

add_supported_preprocessor(PreprocessorClipVisionWithInsightFaceForIPAdapter(name="InsightFace+CLIP-H (IPAdapter)", url="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", filename="CLIP-ViT-H-14.safetensors"))

add_supported_preprocessor(PreprocessorInsightFaceForInstantID(name="InsightFace (InstantID)"))


class IPAdapterPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not isinstance(state_dict, dict):
            return None
        model = state_dict
        ckpt_lower = str(ckpt_path).lower()

        has_flat_keys = "ip_adapter" not in model and any(isinstance(k, str) and k.startswith("ip_adapter.") for k in model)
        if ckpt_lower.endswith((".safetensors", ".sft")) or has_flat_keys:
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key, value in model.items():
                if not isinstance(key, str):
                    continue
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.removeprefix("image_proj.")] = value
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.removeprefix("ip_adapter.")] = value
            model = st_model

        ip_adapter_sd = model.get("ip_adapter")
        image_proj_sd = model.get("image_proj")
        if not isinstance(ip_adapter_sd, dict) or len(ip_adapter_sd) == 0 or not isinstance(image_proj_sd, dict):
            return None

        o = IPAdapterPatcher(model)

        model_filename = Path(ckpt_path).name.lower()
        if "v2" in model_filename:
            o.faceid_v2 = True
            o.weight_v2 = True

        return o

    def __init__(self, state_dict):
        super().__init__()
        self.ip_adapter = state_dict
        self.faceid_v2 = False
        self.weight_v2 = False
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        if not isinstance(cond, dict):
            raise ValueError("IP-Adapter requires an IP-Adapter preprocessor (such as CLIP-ViT-H or InsightFace)")

        unet = process.sd_model.forge_objects.unet
        attn_mask = (mask[:, 0] if getattr(mask, "ndim", 0) == 4 and mask.shape[1] == 1 else mask) if mask is not None else None

        unet = IPAdapterApply.apply_ipadapter(
            ipadapter=self.ip_adapter,
            model=unet,
            weight=self.strength,
            start_at=self.start_percent,
            end_at=self.end_percent,
            faceid_v2=self.faceid_v2,
            weight_v2=self.weight_v2,
            attn_mask=attn_mask,
            **cond,
        )

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(IPAdapterPatcher)
