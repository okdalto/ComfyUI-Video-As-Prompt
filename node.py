"""
ComfyUI custom nodes for WAN Video-as-Prompt (VAP).

Provides two nodes:
- `WANVAP_Load`: loads/cache the WAN pipeline for a given `model_id`.
- `WANVAP_Sample`: runs sampling on a target image and reference video frames,
  producing output frames.

This file is intentionally self-contained and re-implements the minimal
helpers used by the original `infer/wan_vap.py` script.
"""
from __future__ import annotations
import os
import random
import shutil
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

import sys

_repo_dir = os.path.dirname(__file__)
if _repo_dir and _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from diffusers_vap import (
    AutoencoderKLWan,
    WanImageToVideoMOTPipeline,
    WanTransformer3DMOTModel,
)

from transformers import CLIPVisionModel

import folder_paths
from comfy.utils import ProgressBar
from huggingface_hub import snapshot_download

# Global cache for loaded pipelines
_GLOBAL = {"pipelines": {}}

def _normalize_device(device_hint: Optional[str] = None) -> str:
    if device_hint is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device_hint).strip()
    if not device or device.lower() in ("auto", "none"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device.lower()


def _resolve_model_source(model_id: str) -> str:
    if not model_id:
        raise ValueError("model_id must be provided.")

    candidates: List[str] = []
    candidates.append(model_id)
    candidates.append(os.path.join(_repo_dir, model_id))
    candidates.append(os.path.join(_repo_dir, "ckpts", model_id))

    base_models_dir = getattr(folder_paths, "models_dir", None)
    if base_models_dir:
        for subdir in ("", "video_as_prompt", "video-as-prompt", "Video-As-Prompt"):
            base_dir = os.path.join(base_models_dir, subdir) if subdir else base_models_dir
            candidates.append(os.path.join(base_dir, model_id))
        candidates.append(os.path.join(base_models_dir, "ckpts", model_id))

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        abs_path = os.path.abspath(os.path.expanduser(candidate))
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if os.path.isdir(abs_path):
            pw = os.path.join(abs_path, "pretrained_weights")
            if os.path.isdir(pw):
                return pw
            return abs_path

    return model_id


def _maybe_download_model(model_id: str, base_models_dir: Optional[str]) -> str:
    if not base_models_dir:
        return model_id
    try:
        repo_name = model_id.replace("/", "_")
        target_dir = os.path.join(base_models_dir, repo_name)
        if not os.path.exists(target_dir):
            hf_path = snapshot_download(repo_id=model_id)
            try:
                shutil.copytree(hf_path, target_dir)
            except Exception:
                target_dir = hf_path

        pw = os.path.join(target_dir, "pretrained_weights")
        if os.path.exists(pw):
            return pw
        return target_dir
    except Exception:
        return model_id

def _tensor_to_pil_list(frames: torch.Tensor) -> List[Image.Image]:
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    if frames.dim() != 4:
        raise ValueError("Expected a 3D or 4D tensor for image frames.")

    if frames.shape[-1] in (1, 3, 4):
        frames_bhwc = frames
    elif frames.shape[1] in (1, 3, 4):
        frames_bhwc = frames.permute(0, 2, 3, 1)
    else:
        raise ValueError("Expected BHWC or BCHW tensor with 1, 3, or 4 channels.")

    frames_bhwc = frames_bhwc.detach().cpu()
    if frames_bhwc.is_floating_point():
        frames_bhwc = (frames_bhwc.clamp(0, 1) * 255).round().to(torch.uint8)
    else:
        frames_bhwc = frames_bhwc.clamp(0, 255).to(torch.uint8)

    pil_frames = []
    for frame in frames_bhwc:
        arr = frame.numpy()
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        pil_frames.append(Image.fromarray(arr))
    return pil_frames


def _array_to_pil(frame: np.ndarray) -> Image.Image:
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.ndim != 3:
        raise ValueError("Expected an HWC array for image data.")
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.shape[2] > 3:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        if frame.dtype.kind == "f":
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0).round().astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def _coerce_pil_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if torch.is_tensor(image):
        frames = _tensor_to_pil_list(image)
        return frames[0] if frames else None
    if isinstance(image, np.ndarray):
        return _array_to_pil(image)
    if isinstance(image, list) and image:
        return _coerce_pil_image(image[0])
    return None


def _coerce_video_frames(ref_video) -> List[Image.Image]:
    if ref_video is None:
        return []
    if torch.is_tensor(ref_video):
        return _tensor_to_pil_list(ref_video)
    if isinstance(ref_video, list):
        frames: List[Image.Image] = []
        for item in ref_video:
            if torch.is_tensor(item):
                frames.extend(_tensor_to_pil_list(item))
                continue
            if isinstance(item, Image.Image):
                frames.append(item)
                continue
            if isinstance(item, np.ndarray):
                frames.append(_array_to_pil(item))
                continue
            coerced = _coerce_pil_image(item)
            if coerced is not None:
                frames.append(coerced)
        return frames
    if isinstance(ref_video, Image.Image):
        return [ref_video]
    if isinstance(ref_video, np.ndarray):
        return [_array_to_pil(ref_video)]
    coerced = _coerce_pil_image(ref_video)
    return [coerced] if coerced is not None else []


def _pil_list_to_tensor(frames: List[Image.Image]) -> torch.Tensor:
    if not frames:
        raise ValueError("No frames provided for conversion.")
    tensors = []
    for frame in frames:
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        arr = np.asarray(frame, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors, dim=0)


def select_frames(video_frames: List[Image.Image], num: int, mode: str) -> List[Image.Image]:
    if len(video_frames) == 0:
        return []
    if mode == "first":
        return video_frames[:num]
    if mode == "evenly":
        import torch as _torch

        idx = _torch.linspace(0, len(video_frames) - 1, num).long().tolist()
        return [video_frames[i] for i in idx]
    if mode == "random":
        if len(video_frames) <= num:
            return video_frames
        import random as _random

        start = _random.randint(0, len(video_frames) - num)
        return video_frames[start : start + num]
    return video_frames


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_and_dtype_for(device_hint: Optional[str] = None):
    device = _normalize_device(device_hint)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return device, dtype


def get_folder_list():
    base_dir = folder_paths.models_dir
    if not os.path.exists(base_dir):
        return ["video_as_prompt"]

    candidates = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            if os.path.exists(os.path.join(full_path, "pretrained_weights")):
                candidates.append(name)

    if "video_as_prompt" not in candidates:
        candidates.append("video_as_prompt")

    return sorted(candidates)


def load_model(model_id: str = "ByteDance/Video-As-Prompt-Wan2.1-14B", device: Optional[str] = None):
    model_source = _resolve_model_source(model_id)
    base_models_dir = getattr(folder_paths, "models_dir", None) or os.path.join(os.getcwd(), "models")

    if not os.path.exists(model_source):
        model_source = _maybe_download_model(model_source, base_models_dir)

    device, dtype = _device_and_dtype_for(device)
    cache_key = (model_source, device)
    cached = _GLOBAL["pipelines"].get(cache_key)
    if cached is not None:
        return cached

    # load components similarly to the original script using resolved model_source
    image_encoder = CLIPVisionModel.from_pretrained(model_source, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_source, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DMOTModel.from_pretrained(model_source, subfolder="transformer", torch_dtype=dtype)

    pipe = WanImageToVideoMOTPipeline.from_pretrained(
        model_source,
        vae=vae,
        image_encoder=image_encoder,
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    # try to enable sequential CPU offload when running on CUDA to reduce memory pressure
    if device.startswith("cuda"):
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

    _GLOBAL["pipelines"][cache_key] = pipe
    return pipe


def sample_from_pipe(
    pipe,
    image: Image.Image,
    ref_video,
    prompt: str,
    prompt_mot_ref: Optional[List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    frames_selection: str = "evenly",
    guidance_scale: float = 5.0,
    seed: int = 42,
):
    set_global_seed(int(seed))

    if pipe is None:
        raise ValueError("pipeline is required. Use WANVAP_Load to create it.")

    image = _coerce_pil_image(image) or image
    if image is None:
        raise ValueError("image must be a valid ComfyUI IMAGE or PIL image.")

    ref_frames = _coerce_video_frames(ref_video)
    if not ref_frames:
        raise ValueError("ref_video must be provided as ComfyUI IMAGE frames.")

    if isinstance(prompt_mot_ref, str):
        prompt_mot_ref = [prompt_mot_ref]
    if not prompt_mot_ref:
        raise ValueError("prompt_mot_ref is required. Provide one or more prompts separated by '|'.")

    ref_frames = select_frames(ref_frames, num_frames, frames_selection)

    num_inference_steps = max(int(num_inference_steps), 1)
    progress = ProgressBar(num_inference_steps)

    def _on_step_end(_pipe, _step, _timestep, callback_kwargs):
        progress.update(1)
        return callback_kwargs

    out_frames = pipe(
        image=image,
        ref_videos=[ref_frames],
        prompt=prompt,
        prompt_mot_ref=prompt_mot_ref,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        frames_selection=frames_selection,
        guidance_scale=guidance_scale,
        callback_on_step_end=_on_step_end,
        callback_on_step_end_tensor_inputs=[],
        output_type="pil",
    ).frames[0]

    frames_tensor = _pil_list_to_tensor(out_frames)
    return frames_tensor


# ------------------ ComfyUI node classes ------------------
class WANVAP_Load:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model_id": ("STRING", {"default": "ByteDance/Video-As-Prompt-Wan2.1-14B"})},
            "optional": {"device": (["auto", "cuda", "cpu"],)},
        }

    RETURN_TYPES = ("WANVAP_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "WAN/VAP"

    def load(self, model_id: str = "ByteDance/Video-As-Prompt-Wan2.1-14B", device: Optional[str] = None):
        """Load and cache a WAN VAP pipeline for `model_id`."""
        pipe = load_model(model_id, device=device)
        return (pipe,)


class WANVAP_Sample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("WANVAP_PIPE",),
                "image": ("IMAGE",),
                "ref_video": ("IMAGE",),
                "prompt": ("STRING",),
                "prompt_mot_ref": ("STRING",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 49}),
                "height": ("INT", {"default": 480}),
                "width": ("INT", {"default": 832}),
                "num_inference_steps": ("INT", {"default": 50}),
                "frames_selection": (["evenly", "first", "random"],),
                "guidance_scale": ("FLOAT", {"default": 5.0}),
                "seed": ("INT", {"default": 42}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "sample"
    CATEGORY = "WAN/VAP"

    def sample(
        self,
        pipe,
        image,
        ref_video,
        prompt: str,
        prompt_mot_ref: str,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        height: int = 480,
        width: int = 832,
        frames_selection: str = "evenly",
        guidance_scale: float = 5.0,
        seed: int = 42,
    ):
        """Run VAP sampling. `prompt_mot_ref` is required and may be a single string or
        multiple prompts separated by `|`, which will be converted to a list.
        """
        prompt_mot_ref_list = [s.strip() for s in prompt_mot_ref.split("|")] if prompt_mot_ref else []

        frames = sample_from_pipe(
            pipe=pipe,
            image=image,
            ref_video=ref_video,
            prompt=prompt,
            prompt_mot_ref=prompt_mot_ref_list,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            frames_selection=frames_selection,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return (frames,)


# Expose node classes in the module-level variable expected by many ComfyUI loaders
NODE_CLASS_MAPPINGS = {
    "WANVAP_Load": WANVAP_Load,
    "WANVAP_Sample": WANVAP_Sample,
}
