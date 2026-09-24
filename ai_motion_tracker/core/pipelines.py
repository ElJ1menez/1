# SPDX-License-Identifier: GPL-3.0-or-later
"""Background-thread workers. Input: plain dict `p` captured on the main
thread. Output: plain Python/numpy data applied to Blender on the main thread.
No bpy here: Blender's API is not thread-safe."""

import numpy as np

from . import downloads, tapir_engine, video
from . import tracks as tk


def _stage(job, a, b):
    def report(frac, msg=None):
        job.set(progress=a + (b - a) * min(1.0, max(0.0, frac)), message=msg)
    return report


def load_frames(p, job, progress, max_side):
    w, h = video.working_size(p["width"], p["height"], max_side)
    total = p["last"] - p["first"] + 1
    frames = []
    for c, img in video.iter_frames(p["info"], p["first"], p["last"], (w, h)):
        job.check()
        frames.append(img)
        progress(len(frames) / float(total), "Leyendo video: %d/%d" % (len(frames), total))
    if len(frames) < 2:
        raise RuntimeError("Se necesitan al menos 2 frames legibles del clip")
    return frames


def _seed_fn(frames, masks, count):
    import cv2
    height, width = frames[0].shape[:2]
    min_dist = max(4.0, 0.6 * np.sqrt(width * height / float(count)))

    def seed(i):
        valid = None
        cv_mask = None
        if masks is not None:
            valid = ~masks[i]
            cv_mask = cv2.resize(valid.astype(np.uint8) * 255, (width, height),
                                 interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=count, qualityLevel=0.005,
                                      minDistance=min_dist, mask=cv_mask, blockSize=7)
        # OpenCV returns pixel indices; +0.5 converts to raster (pixel centre).
        pts = np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2) + 0.5
        if len(pts) < count // 3:
            # Low-texture shot: top up with a regular grid so the AI still gets points.
            pts = np.concatenate([pts, tk.grid_points(width, height, count - len(pts), valid=valid)])
        return pts.astype(np.float32)
    return seed


def camera_track(job, p):
    frames = load_frames(p, job, _stage(job, 0.0, 0.1), p["max_side"])
    height, width = frames[0].shape[:2]
    num = len(frames)

    masks = None
    notes = []
    if p["dynamic_mask"]:
        try:
            seg_path = downloads.ensure_model(p["models_dir"], "segmenter", job)
            from . import landmarks
            masks = landmarks.dynamic_masks(frames, seg_path, job, _stage(job, 0.1, 0.2))
        except (ImportError, OSError) as exc:
            notes.append("Máscara de objetos móviles omitida (MediaPipe no disponible: %s)" % exc)

    ckpt = downloads.ensure_model(p["models_dir"], "bootstapir", job)
    device = tapir_engine.pick_device(p["device"])
    job.set(message="Cargando BootsTAPIR en %s" % device)
    model = tapir_engine.load_model(ckpt, device)
    raw = tapir_engine.track_segments(
        model, frames, _seed_fn(frames, masks, p["points_per_seed"]),
        p["window"], p["interval"], device, job, _stage(job, 0.2, 0.9), int(p["model_res"]))

    job.set(progress=0.9, message="Filtrando tracks")
    kept = raw
    if masks is not None:
        kept = tk.apply_dynamic_masks(kept, masks, width, height)
    kept = tk.filter_min_length(kept, p["min_length"])
    if p["geometric_filter"] and kept:
        ratio = tk.geometric_inlier_ratio(kept, num, threshold=p["geo_threshold"])
        for t, r in zip(kept, ratio):
            t.score = t.length * float(r) ** 2
        kept = [t for t, r in zip(kept, ratio) if r >= p["min_inlier_ratio"]]
    else:
        for t in kept:
            t.score = float(t.length)
    selected, coverage = tk.select_by_coverage(kept, num, p["per_frame"], p["max_tracks"])
    weak = int((coverage < 8).sum())
    if weak:
        notes.append("%d frames con menos de 8 tracks (solve inestable ahí)" % weak)
    return {
        "tracks": selected, "width": width, "height": height,
        "first": p["first"], "raw": len(raw), "notes": notes,
        "min_coverage": int(coverage.min()) if num else 0,
    }


def track_selected(job, p):
    frames = load_frames(p, job, _stage(job, 0.0, 0.15), p["max_side"])
    height, width = frames[0].shape[:2]
    t0 = p["query_frame"] - p["first"]
    if not 0 <= t0 < len(frames):
        raise RuntimeError("El frame actual está fuera del rango a trackear")
    pts = tk.from_blender(np.array(p["query_co"], dtype=np.float64), width, height).astype(np.float32)
    ckpt = downloads.ensure_model(p["models_dir"], "bootstapir", job)
    device = tapir_engine.pick_device(p["device"])
    job.set(message="Cargando BootsTAPIR en %s" % device)
    model = tapir_engine.load_model(ckpt, device)
    xy, vis = tapir_engine.track_points(
        model, frames, t0, pts, p["window"], device, job, _stage(job, 0.15, 1.0), int(p["model_res"]))
    return {"names": p["names"], "xy": xy, "vis": vis, "width": width,
            "height": height, "first": p["first"]}


def _frame_iter(p, job, max_side):
    size = video.working_size(p["width"], p["height"], max_side)
    return video.iter_frames(p["info"], p["first"], p["last"], size)


def pose(job, p):
    from . import landmarks
    path = downloads.ensure_model(p["models_dir"], "pose_" + p["pose_model"].lower(), job)
    total = p["last"] - p["first"] + 1
    return landmarks.run_pose(_frame_iter(p, job, 1280), total, p["fps"], path,
                              p["min_confidence"], job, _stage(job, 0.0, 1.0))


def face(job, p):
    from . import landmarks
    path = downloads.ensure_model(p["models_dir"], "face", job)
    total = p["last"] - p["first"] + 1
    return landmarks.run_face(_frame_iter(p, job, 1280), total, p["fps"], path,
                              p["min_confidence"], job, _stage(job, 0.0, 1.0))
