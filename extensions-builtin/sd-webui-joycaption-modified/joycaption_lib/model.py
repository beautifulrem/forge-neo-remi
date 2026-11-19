"""
JoyCaption model loading + inference.

Loads the JoyCaption Beta One Llava model lazily on first use and caches
it in memory. Supports bf16 (~17 GB VRAM), 8-bit (~10 GB), and 4-bit
(~6 GB) loading via bitsandbytes. Auto-downloads weights to
<webui>/models/joycaption/ on first run.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Optional

from PIL import Image

# Defer torch / transformers imports so the extension can still register
# its UI even if those are missing (the UI will show an error then).
_torch = None
_transformers = None


def _lazy_imports():
    global _torch, _transformers
    if _torch is None:
        import torch as _t
        _torch = _t
    if _transformers is None:
        import transformers as _tr
        _transformers = _tr
    return _torch, _transformers


MODEL_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def _models_dir() -> Path:
    """Folder where the JoyCaption weights live: <webui>/models/joycaption/."""
    try:
        from modules import paths_internal
        base = Path(paths_internal.models_path)
    except Exception:
        # Fallback for running outside webui:
        # <webui>/extensions/<this-extension>/joycaption_lib/model.py
        # parents[3] -> <webui>
        base = Path(__file__).resolve().parents[3] / "models"
    d = base / "joycaption"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_local_path() -> Path:
    """Where the snapshot will be saved on disk."""
    return _models_dir() / MODEL_REPO.split("/")[-1]


def is_model_downloaded() -> bool:
    p = model_local_path()
    # A minimal check: the config and at least one safetensors shard.
    if not (p / "config.json").is_file():
        return False
    return any(f.suffix == ".safetensors" for f in p.glob("*.safetensors"))


ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.model",
    "tokenizer*",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template*",
    "generation_config.json",
]
"""The files of the repository the engine needs (optional / unused ones are left out to save disk)."""

_DOWNLOAD_CODE = (
    "import json, sys\n"
    "from huggingface_hub import snapshot_download\n"
    "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], allow_patterns=json.loads(sys.argv[3]))\n"
)


def ensure_model_downloaded(progress_print=print, check=None) -> Path:
    """Download the JoyCaption weights into <webui>/models/joycaption/ if not already there.

    ``check``: called while downloading, raises to stop it (a page task's
    ``check``, which raises once the user cancels). The download then runs in
    a child process that is stopped on the spot (huggingface_hub resumes its
    partial files next time): ``snapshot_download`` itself cannot be
    interrupted, and ~17 GB can take hours."""
    target = model_local_path()
    if is_model_downloaded():
        return target

    progress_print(f"[JoyCaption] Downloading {MODEL_REPO} to {target} (~17 GB, one-time)…")

    if check is None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_REPO, local_dir=str(target), allow_patterns=ALLOW_PATTERNS)
    else:
        _download_in_child(target, check, progress_print)
    progress_print("[JoyCaption] Download complete.")
    return target


def _downloaded_bytes(folder: Path) -> int:
    total = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:  # renamed while counting
            pass
    return total


def _download_in_child(target: Path, check, progress_print) -> None:
    import json
    import subprocess
    import sys
    import tempfile
    import time

    target.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-c", _DOWNLOAD_CODE, MODEL_REPO, str(target), json.dumps(ALLOW_PATTERNS)]
    with tempfile.TemporaryFile() as errors:
        child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
        reported = time.monotonic()
        try:
            while True:
                try:
                    child.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                check()
                if time.monotonic() - reported >= 5:
                    reported = time.monotonic()
                    progress_print(f"[JoyCaption] Downloading… {_downloaded_bytes(target) / 1e9:.1f} GB so far")
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if child.returncode != 0:
            errors.seek(0)
            tail = errors.read()[-2000:].decode("utf-8", "replace").strip()
            raise RuntimeError(f"JoyCaption download failed: {tail.splitlines()[-1] if tail else f'exit code {child.returncode}'}")


class JoyCaptionEngine:
    """Holds the loaded processor + model. Thread-safe (single GPU)."""

    _instance: Optional["JoyCaptionEngine"] = None
    # Re-entrant: load() unloads first, and generate() must not race either.
    _lock = threading.RLock()

    def __init__(self):
        self.processor = None
        self.model = None
        self.loaded_quant: Optional[str] = None  # "bf16" | "int8" | "nf4"
        self.device = "cuda" if self._has_cuda() else "cpu"

    @staticmethod
    def _has_cuda() -> bool:
        torch, _ = _lazy_imports()
        return torch.cuda.is_available()

    @classmethod
    def get(cls) -> "JoyCaptionEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, quantization: str = "bf16", progress_print=print):
        """quantization: 'bf16' | 'int8' | 'nf4'"""
        with self._lock:
            self._load(quantization, progress_print)

    def _load(self, quantization: str, progress_print):
        torch, transformers = _lazy_imports()

        if self.is_loaded() and self.loaded_quant == quantization:
            return  # already loaded as requested

        if self.is_loaded():
            self.unload()

        model_path = str(ensure_model_downloaded(progress_print))

        progress_print(f"[JoyCaption] Loading processor from {model_path}…")
        self.processor = transformers.AutoProcessor.from_pretrained(model_path)

        kwargs = {}
        if quantization == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = 0 if self.device == "cuda" else "cpu"
        elif quantization == "int8":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        elif quantization == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            raise ValueError(f"Unknown quantization: {quantization}")

        progress_print(f"[JoyCaption] Loading model ({quantization})…")
        self.model = transformers.LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.loaded_quant = quantization
        progress_print(f"[JoyCaption] Model loaded ({quantization}).")

    def unload(self):
        with self._lock:
            self._unload()

    def _unload(self):
        torch, _ = _lazy_imports()
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.loaded_quant = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ------------------------------------------------------------------

    def generate(  # noqa: PLR0913 - inherently many knobs
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str = "You are a helpful image captioner.",
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> str:
        torch, _ = _lazy_imports()

        with self._lock:
            if not self.is_loaded():
                raise RuntimeError("JoyCaption model is not loaded. Call load() first.")
            if image.mode != "RGB":
                image = image.convert("RGB")

            convo = []
            if system_prompt:
                convo.append({"role": "system", "content": system_prompt})
            # Chat template needs SOME user content — substitute a space when empty
            convo.append({"role": "user", "content": prompt if prompt else " "})
            convo_string = self.processor.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True,
            )
            assert isinstance(convo_string, str)

            inputs = self.processor(
                text=[convo_string],
                images=[image],
                return_tensors="pt",
            ).to(getattr(self.model, "device", self.device))

            # bf16 pixel values
            if "pixel_values" in inputs:
                if self.loaded_quant == "bf16":
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                else:
                    # For 4/8-bit, the vision tower stays in compute dtype
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                suppress_tokens=None,
                use_cache=True,
                temperature=float(temperature) if temperature > 0 else 1.0,
                top_p=float(top_p),
                top_k=int(top_k) if top_k and top_k > 0 else None,
            )

            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, **gen_kwargs)[0]

            # Trim off the prompt tokens
            generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
            caption = self.processor.tokenizer.decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            return caption


# Helper: a sentinel the UI can call without importing torch up front
def get_engine() -> JoyCaptionEngine:
    return JoyCaptionEngine.get()
