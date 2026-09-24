# SPDX-License-Identifier: GPL-3.0-or-later
"""AI model wrappers with a small in-memory cache, so repeated generations
in one Blender session do not reload gigabytes of weights."""

import gc
import os
import threading

from . import downloads

_cache = {}
_lock = threading.Lock()


def resolve_device(pref):
    import torch
    if pref == "CUDA" or (pref == "AUTO" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible: instala PyTorch con CUDA o usa CPU")
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if pref == "MPS" or (pref == "AUTO" and mps is not None and mps.is_available()):
        if mps is None or not mps.is_available():
            raise RuntimeError("Apple Metal (MPS) no está disponible")
        return "mps"
    return "cpu"


def release(keep=()):
    """Free cached models except those in `keep` (and GPU memory)."""
    with _lock:
        for key in [k for k in _cache if k[0] not in keep]:
            del _cache[key]
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def triposr(models_dir, device, job=None):
    key = ("triposr", device)
    with _lock:
        if key in _cache:
            return _cache[key]
    path = downloads.ensure_model(models_dir, "triposr", job)
    if job is not None:
        job.set(message="Cargando TripoSR")
    from ..third_party.triposr import TSR
    model = TSR.from_checkpoint(path).eval().to(device)
    with _lock:
        _cache[key] = model
    return model


def background_remover(models_dir, job=None):
    key = ("isnet", "cpu")
    with _lock:
        if key in _cache:
            return _cache[key]
    path = downloads.ensure_model(models_dir, "isnet", job)
    from .image_prep import BackgroundRemover
    remover = BackgroundRemover(path)
    with _lock:
        _cache[key] = remover
    return remover


def text_to_image(model_key, device, cache_dir, token=None, job=None):
    key = ("t2i_" + model_key, device)
    with _lock:
        if key in _cache:
            return _cache[key]
    import torch
    info = downloads.T2I_MODELS[model_key]
    if job is not None:
        job.set(message="Cargando %s (la primera vez descarga ~%s)" % (info["source"], info["size"]))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype, "cache_dir": cache_dir, "use_safetensors": True}
    if token:
        kwargs["token"] = token
    if model_key == "FLUX_SCHNELL":
        from diffusers import FluxPipeline
        if device == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16
        pipe = FluxPipeline.from_pretrained(info["repo"], **kwargs)
    else:
        from diffusers import StableDiffusionXLPipeline
        if device == "cuda":
            kwargs["variant"] = "fp16"
        pipe = StableDiffusionXLPipeline.from_pretrained(info["repo"], **kwargs)
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # fits 8-12 GB cards; moves modules on demand
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    with _lock:
        _cache[key] = pipe
    return pipe
