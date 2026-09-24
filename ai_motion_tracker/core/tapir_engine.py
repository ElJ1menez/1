# SPDX-License-Identifier: GPL-3.0-or-later
"""Point tracking with BootsTAPIR (Google DeepMind, Apache-2.0 code + weights)."""

import numpy as np

from . import tracks as tk

_cache = {}


def pick_device(pref="AUTO"):
    import torch
    if pref == "CUDA" or (pref == "AUTO" and torch.cuda.is_available()):
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if pref == "MPS" or (pref == "AUTO" and mps is not None and mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint, device):
    key = (checkpoint, str(device))
    if key not in _cache:
        import torch
        from ..third_party.tapir import tapir_model
        _cache.clear()  # keep at most one copy in (V)RAM
        model = tapir_model.TAPIR(pyramid_level=1)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        _cache[key] = model.to(device).eval()
    return _cache[key]


def infer(model, frames, queries_tyx, device, size=512, query_chunk_size=64):
    """frames: (T, H, W, 3) uint8.  queries_tyx: (N, 3) as (frame, y, x).

    TAPIR is only accurate on square inputs of the size it was trained on
    (256 or 512): on e.g. 640x360 the error grows from ~0.2 px to >20 px.
    Frames are therefore squashed to size x size and coordinates mapped back.

    Returns xy (N, T, 2) raster coords of the input frames and vis (N, T) bool.
    """
    import cv2
    import torch
    height, width = frames[0].shape[:2]
    sx, sy = size / float(width), size / float(height)
    square = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
    q = np.array(queries_tyx, dtype=np.float32).reshape(-1, 3)
    q[:, 1] *= sy
    q[:, 2] *= sx
    with torch.no_grad():
        video = torch.from_numpy(square).to(device)
        video = (video.float() / 255.0 * 2.0 - 1.0)[None]
        out = model(video, torch.from_numpy(q).to(device)[None], query_chunk_size=query_chunk_size)
        occ = torch.sigmoid(out["occlusion"][0])
        unc = torch.sigmoid(out["expected_dist"][0])
        vis = ((1.0 - occ) * (1.0 - unc) > 0.5).cpu().numpy()
        xy = out["tracks"][0].float().cpu().numpy()
    xy[..., 0] /= sx
    xy[..., 1] /= sy
    return xy, vis


def segment_starts(num_frames, window, interval):
    """Seed frames and their [a, b) windows covering the whole clip."""
    seeds = list(range(0, num_frames, max(1, interval)))
    if num_frames - 1 - seeds[-1] >= max(1, interval // 2):
        seeds.append(num_frames - 1)
    half = window // 2
    out = []
    for s in seeds:
        a = max(0, s - half)
        b = min(num_frames, a + window)
        a = max(0, b - window)
        out.append((s, a, b))
    return out


def track_segments(model, frames, seed_fn, window, interval, device, job, progress, size=512):
    """Camera-tracking mode: re-seed points every `interval` frames.

    Each seed frame gets fresh points (from seed_fn(frame_index) -> (N, 2) xy)
    tracked forward and backward inside its own window, so memory stays
    bounded and newly revealed areas always receive tracks.
    """
    height, width = frames[0].shape[:2]
    segments = segment_starts(len(frames), window, interval)
    result = []
    for k, (s, a, b) in enumerate(segments):
        job.check()
        progress(k / len(segments), "IA siguiendo puntos: segmento %d/%d" % (k + 1, len(segments)))
        pts = seed_fn(s)
        if len(pts) == 0 or b - a < 2:
            continue
        q = np.zeros((len(pts), 3), dtype=np.float32)
        q[:, 0] = s - a
        q[:, 1] = pts[:, 1]
        q[:, 2] = pts[:, 0]
        xy, vis = infer(model, frames[a:b], q, device, size)
        result.extend(tk.tracks_from_arrays(xy, vis, a, width, height))
    progress(1.0, "Seguimiento terminado")
    return result


def _track_forward(model, frames, t0, pts, window, device, job, progress, size):
    """Track pts (N, 2) from frame t0 to the end, chaining windows.

    Consecutive windows overlap by a quarter; each point is re-queried at its
    last visible frame inside the overlap, which keeps identity across windows
    without holding the whole clip on the GPU.
    """
    num = len(frames)
    n = len(pts)
    xy = np.zeros((n, num, 2), dtype=np.float32)
    vis = np.zeros((n, num), dtype=bool)
    xy[:, t0] = pts
    vis[:, t0] = True
    qf = np.full(n, t0, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    overlap = max(1, window // 4)
    a = t0
    while a < num - 1:
        job.check()
        b = min(num, a + window)
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        progress((a - t0) / max(1, num - t0), None)
        q = np.stack([qf[idx] - a, xy[idx, qf[idx], 1], xy[idx, qf[idx], 0]], axis=1)
        rxy, rvis = infer(model, frames[a:b], q, device, size)
        for j, i in enumerate(idx):
            s = qf[i] + 1
            xy[i, s:b] = rxy[j, s - a:]
            vis[i, s:b] = rvis[j, s - a:]
        if b >= num:
            break
        nxt = b - overlap
        for i in idx:
            seen = np.flatnonzero(vis[i, nxt:b])
            if seen.size:
                qf[i] = nxt + seen[-1]
            else:
                active[i] = False  # lost for longer than the overlap
        a = nxt
    return xy, vis


def track_points(model, frames, t0, pts, window, device, job, progress, size=512):
    """Track pts (N, 2) given on frame t0 both forward and backward in time."""
    num = len(frames)
    fxy, fvis = _track_forward(
        model, frames, t0, pts, window, device, job,
        lambda p, m: progress(0.5 * p, m), size)
    rev = frames[::-1]
    bxy, bvis = _track_forward(
        model, rev, num - 1 - t0, pts, window, device, job,
        lambda p, m: progress(0.5 + 0.5 * p, m), size)
    bxy, bvis = bxy[:, ::-1], bvis[:, ::-1]
    xy = np.where((np.arange(num) < t0)[None, :, None], bxy, fxy)
    vis = np.where((np.arange(num) < t0)[None, :], bvis, fvis)
    return xy, vis
