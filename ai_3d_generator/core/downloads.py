# SPDX-License-Identifier: GPL-3.0-or-later
"""Model registry and downloads. Every model here allows commercial use of
its outputs; the license travels with each generated asset (provenance.py).
Keep LICENSES.md in sync when adding models."""

import os
import ssl
import urllib.request

_HF = "https://huggingface.co"

# name -> url, license, source, approximate size
MODELS = {
    "triposr": {
        "url": _HF + "/stabilityai/TripoSR/resolve/main/model.ckpt",
        "license": "MIT",
        "source": "TripoSR (Tripo AI & Stability AI)",
        "size": "1.7 GB",
    },
    "isnet": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        "license": "Apache-2.0",
        "source": "IS-Net / DIS (Xuebin Qin), vía rembg",
        "size": "170 MB",
    },
}

# Text-to-image models, loaded with diffusers (Hugging Face cache).
T2I_MODELS = {
    "SDXL": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "license": "CreativeML Open RAIL++-M",
        "source": "Stable Diffusion XL 1.0 (Stability AI)",
        "size": "7 GB",
        "steps": 30, "guidance": 6.5, "size_px": 1024,
    },
    "FLUX_SCHNELL": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "license": "Apache-2.0",
        "source": "FLUX.1 [schnell] (Black Forest Labs)",
        "size": "34 GB",
        "steps": 4, "guidance": 0.0, "size_px": 1024,
    },
}


def _ssl_context():
    # Respect SSL_CERT_FILE (corporate proxies); otherwise prefer certifi,
    # which Blender bundles, because some platforms' Python has no CA store.
    if not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return ssl.create_default_context()


def model_path(models_dir, name):
    return os.path.join(models_dir, name + "_" + os.path.basename(MODELS[name]["url"]))


def ensure_model(models_dir, name, job=None):
    """Download model `name` into models_dir if missing; return its path."""
    dest = model_path(models_dir, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(models_dir, exist_ok=True)
    url = MODELS[name]["url"]
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "ai-3d-generator-blender"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                if job is not None:
                    job.check()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if job is not None:
                    pct = " %d%%" % (100 * done // total) if total else ""
                    job.set(message="Descargando %s (%s)%s" % (name, MODELS[name]["size"], pct))
    if total and done != total:
        raise RuntimeError("Descarga incompleta de %s (%d de %d bytes)" % (name, done, total))
    os.replace(tmp, dest)
    return dest
