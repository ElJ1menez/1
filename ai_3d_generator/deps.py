# SPDX-License-Identifier: GPL-3.0-or-later
"""Python dependencies installed with pip into a private folder of the add-on.

Every package listed here is open source and allows commercial use
(see LICENSES.md).
The add-on itself only needs Blender; each group is installed on demand.
"""

import importlib
import importlib.util
import os
import subprocess
import sys

import bpy

ADDON_ID = __package__

GROUPS = {
    "torch": {
        "label": "PyTorch",
        "modules": ("torch",),
        "packages": ("torch",),
        "license": "BSD-3-Clause",
    },
    "gen3d": {
        "label": "Imagen → 3D (TripoSR, IS-Net)",
        "modules": ("einops", "skimage", "onnxruntime", "PIL"),
        "packages": ("einops", "scikit-image", "onnxruntime", "Pillow"),
        "license": "MIT / BSD-3 / HPND",
    },
    "text2image": {
        "label": "Texto → imagen (diffusers)",
        "modules": ("diffusers", "transformers", "accelerate", "sentencepiece", "google.protobuf"),
        "packages": ("diffusers", "transformers", "accelerate", "safetensors",
                     "sentencepiece", "protobuf"),
        "license": "Apache-2.0 / BSD-3",
    },
}

TORCH_INDEX = {
    "DEFAULT": None,
    "CPU": "https://download.pytorch.org/whl/cpu",
    "CU126": "https://download.pytorch.org/whl/cu126",
    "CU128": "https://download.pytorch.org/whl/cu128",
}

# Filled by init_paths() on the main thread; worker threads only read them.
paths = {"base": "", "site": "", "models": "", "hf": "", "outputs": ""}


def init_paths():
    try:
        base = bpy.utils.extension_path_user(ADDON_ID, create=True)
    except (ValueError, AttributeError):  # legacy add-on or Blender < 4.2
        base = bpy.utils.user_resource("DATAFILES", path="ai_3d_generator", create=True)
    tag = "py%d%d" % sys.version_info[:2]
    paths["base"] = base
    paths["site"] = os.path.join(base, "site-packages-" + tag)
    paths["models"] = os.path.join(base, "models")
    paths["hf"] = os.path.join(base, "huggingface")
    paths["outputs"] = os.path.join(base, "outputs")
    for key in ("site", "models", "hf", "outputs"):
        os.makedirs(paths[key], exist_ok=True)
    # Appended, not prepended: Blender's own numpy must keep priority.
    if paths["site"] not in sys.path:
        sys.path.append(paths["site"])
    os.environ.setdefault("TORCH_HOME", os.path.join(base, "torch"))


def available(group):
    importlib.invalidate_caches()
    return all(importlib.util.find_spec(m) is not None for m in GROUPS[group]["modules"])


def missing_for(*groups):
    return [GROUPS[g]["label"] for g in groups if not available(g)]


def install_worker(job, p):
    """Runs on a worker thread. p: group, variant, site, python."""
    py = p["python"]
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    job.set(message="Preparando pip")
    subprocess.run([py, "-m", "ensurepip", "--upgrade"], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Pin numpy to Blender's version so compiled wheels match the numpy
    # Blender actually loads.
    constraints = os.path.join(os.path.dirname(p["site"]), "constraints.txt")
    with open(constraints, "w") as f:
        f.write("numpy==%s\n" % p["numpy"])

    cmd = [py, "-m", "pip", "install", "--upgrade", "--target", p["site"],
           "--no-warn-script-location", "--disable-pip-version-check",
           "-c", constraints]
    cmd += list(GROUPS[p["group"]]["packages"])
    index = TORCH_INDEX.get(p["variant"]) if p["group"] == "torch" else None
    if index:
        cmd += ["--index-url", index, "--extra-index-url", "https://pypi.org/simple"]
    job.log(" ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    for line in proc.stdout:
        job.log(line.rstrip())
        if line.strip():
            job.set(message=line.strip()[:110])
        if job.cancelled:
            proc.terminate()
            job.check()
    if proc.wait() != 0:
        raise RuntimeError("pip falló (código %d). Revisa el log en Preferencias." % proc.returncode)
    importlib.invalidate_caches()
    return {"group": p["group"]}
