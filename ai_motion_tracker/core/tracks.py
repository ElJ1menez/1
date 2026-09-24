# SPDX-License-Identifier: GPL-3.0-or-later
"""Track containers and filters. Pure numpy (+ OpenCV where noted), no bpy.

Coordinates are raster pixels of the working-resolution frames:
(0, 0) is the top-left corner of the top-left pixel, x grows right, y grows down.
"""

import numpy as np


class Track:
    """A 2D point track that lives on frames [start, start + len(vis))."""

    __slots__ = ("start", "xy", "vis", "score")

    def __init__(self, start, xy, vis):
        self.start = int(start)
        self.xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        self.vis = np.asarray(vis, dtype=bool).reshape(-1)
        self.score = 1.0

    @property
    def end(self):
        return self.start + len(self.vis)

    @property
    def length(self):
        return int(self.vis.sum())

    def visible_frames(self):
        return self.start + np.flatnonzero(self.vis)

    def trimmed(self):
        """Copy without leading/trailing invisible frames, or None if empty."""
        idx = np.flatnonzero(self.vis)
        if idx.size == 0:
            return None
        a, b = idx[0], idx[-1] + 1
        t = Track(self.start + a, self.xy[a:b].copy(), self.vis[a:b].copy())
        t.score = self.score
        return t


def visible_runs(vis):
    """[(a, b), ...] half-open index ranges where vis is True."""
    vis = np.asarray(vis, dtype=bool)
    if vis.size == 0:
        return []
    padded = np.concatenate(([False], vis, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def tracks_from_arrays(xy, vis, offset, width, height):
    """Build Tracks from model output.

    xy:  (N, T, 2) raster coordinates.  vis: (N, T) bool.
    Points outside the image are marked invisible.
    """
    xy = np.asarray(xy, dtype=np.float32)
    vis = np.asarray(vis, dtype=bool).copy()
    inside = (
        (xy[..., 0] >= 0) & (xy[..., 0] <= width)
        & (xy[..., 1] >= 0) & (xy[..., 1] <= height)
    )
    vis &= inside & np.isfinite(xy).all(axis=-1)
    out = []
    for n in range(xy.shape[0]):
        t = Track(offset, xy[n], vis[n]).trimmed()
        if t is not None:
            out.append(t)
    return out


def dense(tracks, num_frames):
    """(T, N, 2) array with NaN where a track is absent or occluded."""
    arr = np.full((num_frames, len(tracks), 2), np.nan, dtype=np.float32)
    for i, t in enumerate(tracks):
        frames = t.visible_frames()
        keep = (frames >= 0) & (frames < num_frames)
        arr[frames[keep], i] = t.xy[np.flatnonzero(t.vis)[keep]]
    return arr


def filter_min_length(tracks, min_length):
    return [t for t in tracks if t.length >= min_length]


def apply_dynamic_masks(tracks, masks, width, height, max_fraction=0.3):
    """Hide track samples that land on dynamic objects (people, cars...).

    masks: (T, h, w) bool, True = dynamic. Tracks spending more than
    `max_fraction` of their visible frames on dynamic pixels are dropped.
    """
    num_frames, mh, mw = masks.shape
    sx, sy = mw / float(width), mh / float(height)
    out = []
    for t in tracks:
        idx = np.flatnonzero(t.vis)
        frames = t.start + idx
        ok = (frames >= 0) & (frames < num_frames)
        idx, frames = idx[ok], frames[ok]
        if idx.size == 0:
            continue
        px = np.clip((t.xy[idx, 0] * sx).astype(int), 0, mw - 1)
        py = np.clip((t.xy[idx, 1] * sy).astype(int), 0, mh - 1)
        dyn = masks[frames, py, px]
        if dyn.mean() > max_fraction:
            continue
        vis = t.vis.copy()
        vis[idx[dyn]] = False
        nt = Track(t.start, t.xy, vis).trimmed()
        if nt is not None:
            out.append(nt)
    return out


def geometric_inlier_ratio(tracks, num_frames, frame_step=4, threshold=1.5,
                           min_points=16, min_motion=0.5):
    """Epipolar consistency of each track against the dominant (camera) motion.

    For frame pairs (f, f + frame_step) a fundamental matrix is fitted with
    RANSAC; a track's ratio is how often it was an inlier when tested. Points
    on independently moving objects or bad tracks get low ratios.
    Tracks never tested get ratio 1. Requires OpenCV.
    """
    import cv2

    n = len(tracks)
    tested = np.zeros(n, dtype=np.int32)
    inliers = np.zeros(n, dtype=np.int32)
    if n < min_points or num_frames <= frame_step:
        return np.ones(n, dtype=np.float32)
    pos = dense(tracks, num_frames)
    for f in range(0, num_frames - frame_step, max(1, frame_step // 2)):
        g = f + frame_step
        ok = np.isfinite(pos[f, :, 0]) & np.isfinite(pos[g, :, 0])
        if ok.sum() < min_points:
            continue
        p1 = pos[f, ok].astype(np.float64)
        p2 = pos[g, ok].astype(np.float64)
        if np.median(np.linalg.norm(p2 - p1, axis=1)) < min_motion:
            continue  # no parallax information between these frames
        _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, threshold, 0.999)
        if mask is None:
            continue
        ids = np.flatnonzero(ok)
        tested[ids] += 1
        inliers[ids] += mask.ravel().astype(bool)
    ratio = np.ones(n, dtype=np.float32)
    has = tested > 0
    ratio[has] = inliers[has] / tested[has]
    return ratio


def select_by_coverage(tracks, num_frames, per_frame, max_total):
    """Greedy pick of the best-scored tracks until each frame has `per_frame`.

    Returns (selected_tracks, per_frame_counts).
    """
    order = sorted(range(len(tracks)), key=lambda i: -tracks[i].score)
    count = np.zeros(num_frames, dtype=np.int32)
    chosen = []
    for i in order:
        frames = tracks[i].visible_frames()
        frames = frames[(frames >= 0) & (frames < num_frames)]
        if frames.size and np.any(count[frames] < per_frame):
            chosen.append(tracks[i])
            count[frames] += 1
            if len(chosen) >= max_total:
                break
    return chosen, count


def grid_points(width, height, count, margin=0.04, valid=None):
    """~`count` points on a regular grid (x, y), skipping invalid mask cells."""
    aspect = width / float(height)
    ny = max(2, int(round(np.sqrt(count / aspect))))
    nx = max(2, int(round(count / ny)))
    xs = np.linspace(margin * width, (1 - margin) * width, nx)
    ys = np.linspace(margin * height, (1 - margin) * height, ny)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)
    if valid is not None:
        vh, vw = valid.shape
        px = np.clip((pts[:, 0] * vw / width).astype(int), 0, vw - 1)
        py = np.clip((pts[:, 1] * vh / height).astype(int), 0, vh - 1)
        pts = pts[valid[py, px]]
    return pts


def to_blender(xy, width, height):
    """Raster pixels -> Blender marker space (0..1, origin bottom-left)."""
    xy = np.asarray(xy, dtype=np.float64)
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] / width
    out[..., 1] = 1.0 - xy[..., 1] / height
    return out


def from_blender(co, width, height):
    """Blender marker space -> raster pixels."""
    co = np.asarray(co, dtype=np.float64)
    out = np.empty_like(co)
    out[..., 0] = co[..., 0] * width
    out[..., 1] = (1.0 - co[..., 1]) * height
    return out
