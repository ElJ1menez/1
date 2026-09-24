# SPDX-License-Identifier: GPL-3.0-or-later
"""Image preparation for single-image 3D reconstruction: background removal
(IS-Net, ONNX), cropping, centring and gray background, as TripoSR expects."""

import numpy as np


def load_rgba(path):
    """Load an image file as float32 RGBA in [0, 1]."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    return np.asarray(im, dtype=np.float32) / 255.0


def save_rgb(path, rgb):
    from PIL import Image
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def has_transparency(rgba):
    return rgba.shape[-1] == 4 and float(rgba[..., 3].min()) < 0.99


def _resize(arr, size, resample="LANCZOS"):
    from PIL import Image
    mode = "F" if arr.ndim == 2 else None
    if mode:
        im = Image.fromarray(arr.astype(np.float32), mode="F")
    else:
        im = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))
    im = im.resize(size, getattr(Image.Resampling, resample))
    out = np.asarray(im, dtype=np.float32)
    return out if mode else out / 255.0


class BackgroundRemover:
    """IS-Net general-use segmentation (same pre/post-processing as rembg)."""

    SIZE = 1024
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)

    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=providers or ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name

    def mask(self, rgb):
        """rgb: (H, W, 3) float in [0, 1]. Returns alpha (H, W) in [0, 1]."""
        h, w = rgb.shape[:2]
        x = _resize(rgb, (self.SIZE, self.SIZE))
        x = x / max(float(x.max()), 1e-6)
        x = (x - self.MEAN).transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.session.run(None, {self.input_name: x})[0][0, 0]
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / max(hi - lo, 1e-6)
        return np.clip(_resize(pred, (w, h), "BILINEAR"), 0.0, 1.0)


def clean_alpha(alpha, low=0.1, high=0.9):
    """Remap soft mask values so faint halos disappear and the core is solid."""
    return np.clip((alpha - low) / (high - low), 0.0, 1.0)


def foreground_bbox(alpha, threshold=0.5):
    ys, xs = np.nonzero(alpha > threshold)
    if len(ys) == 0:
        raise RuntimeError("No se encontró ningún objeto en la imagen (máscara vacía)")
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def center_on_gray(rgba, ratio=0.85, out_size=512, gray=0.5):
    """Crop to the foreground, pad to a square where the object fills `ratio`
    of the side, composite on gray and resize. Returns (out_size, out_size, 3)."""
    y0, y1, x0, x1 = foreground_bbox(rgba[..., 3])
    fg = rgba[y0:y1, x0:x1]
    side = max(fg.shape[0], fg.shape[1])
    canvas = int(np.ceil(side / float(ratio)))
    out = np.zeros((canvas, canvas, 4), np.float32)
    oy = (canvas - fg.shape[0]) // 2
    ox = (canvas - fg.shape[1]) // 2
    out[oy:oy + fg.shape[0], ox:ox + fg.shape[1]] = fg
    a = out[..., 3:4]
    rgb = out[..., :3] * a + gray * (1.0 - a)
    return _resize(rgb, (out_size, out_size))
