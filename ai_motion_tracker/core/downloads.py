# SPDX-License-Identifier: GPL-3.0-or-later
"""Model registry and downloads. Every model here allows commercial use."""

import os
import ssl
import urllib.request

_MP = "https://storage.googleapis.com/mediapipe-models"

# name -> (url, license, source). Keep LICENSES.md in sync when adding models.
MODELS = {
    "bootstapir": (
        "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",
        "Apache-2.0",
        "Google DeepMind TAPNet (BootsTAPIR)",
    ),
    "pose_lite": (
        _MP + "/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "Apache-2.0",
        "Google MediaPipe Pose Landmarker",
    ),
    "pose_full": (
        _MP + "/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "Apache-2.0",
        "Google MediaPipe Pose Landmarker",
    ),
    "pose_heavy": (
        _MP + "/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "Apache-2.0",
        "Google MediaPipe Pose Landmarker",
    ),
    "face": (
        _MP + "/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "Apache-2.0",
        "Google MediaPipe Face Landmarker",
    ),
    "segmenter": (
        _MP + "/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite",
        "Apache-2.0",
        "Google MediaPipe Image Segmenter (DeepLab v3)",
    ),
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
    url = MODELS[name][0]
    return os.path.join(models_dir, name + "_" + os.path.basename(url))


def ensure_model(models_dir, name, job=None):
    """Download model `name` into models_dir if missing; return its path."""
    dest = model_path(models_dir, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(models_dir, exist_ok=True)
    url = MODELS[name][0]
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as r:
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
                    job.set(message="Descargando modelo %s%s" % (name, pct))
    os.replace(tmp, dest)
    return dest
