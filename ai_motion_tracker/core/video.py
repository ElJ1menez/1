# SPDX-License-Identifier: GPL-3.0-or-later
"""Frame access for movie files and image sequences. Needs OpenCV, no bpy.

`info` is a plain dict captured on Blender's main thread:
    path, source ('MOVIE' | 'SEQUENCE'), frame_offset
Clip frames are 1-based like Blender's movie clip frames.
"""

import os
import re

import numpy as np


def _cv2():
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    return cv2


def working_size(width, height, max_side):
    """Scale (w, h) so the long side is <= max_side, rounded to multiples of 8."""
    scale = min(1.0, float(max_side) / max(width, height))
    w = max(8, int(round(width * scale / 8.0)) * 8)
    h = max(8, int(round(height * scale / 8.0)) * 8)
    return w, h


def _to_rgb8(img, cv2):
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        # Scene-linear EXR/HDR -> display-ish sRGB for the networks.
        img = np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)
        img = (img * 255.0 + 0.5).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def sequence_map(path):
    """{frame_number: filepath} for the image sequence containing `path`."""
    folder, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    m = re.search(r"(\d+)(?!.*\d)", stem)
    if not m:
        return {0: path}, 0
    prefix, suffix = stem[: m.start()], stem[m.end():]
    pat = re.compile(
        re.escape(prefix) + r"(\d+)" + re.escape(suffix) + re.escape(ext) + r"$",
        re.IGNORECASE,
    )
    files = {}
    for f in os.listdir(folder or "."):
        mm = pat.match(f)
        if mm:
            files[int(mm.group(1))] = os.path.join(folder, f)
    return files, int(m.group(1))


def iter_frames(info, first, last, size=None):
    """Yield (clip_frame, rgb_uint8) for clip frames first..last (inclusive).

    size: optional (w, h) to resize to. Stops early at end of footage.
    """
    cv2 = _cv2()
    offset = int(info.get("frame_offset", 0))

    def fit(img):
        if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
            img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
        return img

    if info["source"] == "SEQUENCE":
        files, first_number = sequence_map(info["path"])
        prev = None
        for c in range(first, last + 1):
            p = files.get(first_number + c - 1 + offset)
            img = _to_rgb8(cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2) if p else None
            if img is None:
                if prev is None:
                    continue
                img = prev  # hole in the sequence: hold the previous frame
            else:
                img = fit(img)
            prev = img
            yield c, img
        return

    cap = cv2.VideoCapture(info["path"])
    if not cap.isOpened():
        raise RuntimeError("OpenCV no pudo abrir el video: %s" % info["path"])
    try:
        # Sequential grab instead of seeking: seeking is frame-inaccurate on
        # many long-GOP codecs, which would silently misalign every track.
        for _ in range(max(0, first - 1 + offset)):
            if not cap.grab():
                return
        for c in range(first, last + 1):
            ok, img = cap.read()
            if not ok:
                return
            yield c, fit(_to_rgb8(img, cv2))
    finally:
        cap.release()


def probe_size(info):
    """(width, height) of the footage's first frame."""
    for _, img in iter_frames(info, 1, 1):
        return img.shape[1], img.shape[0]
    raise RuntimeError("No se pudo leer ningún frame de %s" % info["path"])
