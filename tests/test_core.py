# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the Blender-independent core. Run: python -m pytest tests"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_motion_tracker.core import smoothing, tapir_engine, video  # noqa: E402
from ai_motion_tracker.core import tracks as tk  # noqa: E402


def test_visible_runs():
    assert tk.visible_runs([]) == []
    assert tk.visible_runs([1, 1, 0, 0, 1, 0, 1, 1]) == [(0, 2), (4, 5), (6, 8)]
    assert tk.visible_runs([0, 0]) == []


def test_tracks_from_arrays_trims_and_hides_outside():
    xy = np.zeros((2, 5, 2), np.float32) + 10
    xy[0, 4] = (-5, 10)  # leaves the image on the last frame
    vis = np.array([[0, 1, 1, 1, 1], [0, 0, 0, 0, 0]], bool)
    out = tk.tracks_from_arrays(xy, vis, offset=7, width=64, height=64)
    assert len(out) == 1
    assert out[0].start == 8 and out[0].end == 11 and out[0].length == 3


def test_blender_coordinate_round_trip():
    pts = np.array([[0.0, 0.0], [640.0, 360.0], [320.0, 90.0]])
    co = tk.to_blender(pts, 640, 360)
    assert np.allclose(co[0], (0.0, 1.0))  # top-left raster -> Blender (0, 1)
    assert np.allclose(co[1], (1.0, 0.0))
    assert np.allclose(tk.from_blender(co, 640, 360), pts)


def test_segments_cover_every_frame_with_full_windows():
    for num, window, interval in [(100, 48, 16), (30, 48, 16), (7, 4, 3), (1000, 64, 20)]:
        segs = tapir_engine.segment_starts(num, window, interval)
        covered = np.zeros(num, bool)
        for s, a, b in segs:
            assert a <= s < b and 0 <= a and b <= num
            assert b - a == min(window, num)
            covered[a:b] = True
        assert covered.all()


def _homography(t):
    return np.array([[1.0, 0.002 * t, 3.0 * t], [0.0, 1.0, 1.5 * t], [0.0, 1e-5 * t, 1.0]])


def test_geometric_filter_rejects_independent_motion():
    rng = np.random.default_rng(0)
    num, width, height = 30, 640, 360
    tracks = []
    for _ in range(80):  # static scene points: consistent with the camera
        p = np.append(rng.uniform([50, 50], [590, 310]), 1.0)
        xy = []
        for t in range(num):
            q = _homography(t) @ p
            xy.append(q[:2] / q[2])
        tracks.append(tk.Track(0, np.array(xy), np.ones(num, bool)))
    movers = []
    for _ in range(10):  # independently moving object points
        xy = rng.uniform([50, 50], [590, 310]) + np.cumsum(rng.normal(0, 6, (num, 2)), axis=0)
        movers.append(tk.Track(0, xy, np.ones(num, bool)))
    ratio = tk.geometric_inlier_ratio(tracks + movers, num, threshold=1.0)
    assert ratio[:80].mean() > 0.95
    assert ratio[80:].mean() < 0.5


def test_dynamic_mask_hides_and_drops():
    masks = np.zeros((4, 10, 10), bool)
    masks[:, :, 5:] = True  # right half is a "person"
    left = tk.Track(0, [[10, 50]] * 4, [1, 1, 1, 1])
    right = tk.Track(0, [[90, 50]] * 4, [1, 1, 1, 1])
    crossing = tk.Track(0, [[10, 50], [20, 50], [30, 50], [90, 50]], [1, 1, 1, 1])
    out = tk.apply_dynamic_masks([left, right, crossing], masks, 100, 100)
    assert len(out) == 2
    assert out[1].length == 3 and out[1].end == 3


def test_coverage_selection_prefers_score_and_fills_gaps():
    long = tk.Track(0, np.zeros((10, 2)), np.ones(10, bool))
    long.score = 10
    late = tk.Track(8, np.zeros((5, 2)), np.ones(5, bool))
    late.score = 1
    dup = tk.Track(0, np.zeros((10, 2)), np.ones(10, bool))
    dup.score = 5
    chosen, count = tk.select_by_coverage([late, dup, long], 13, per_frame=1, max_total=10)
    assert chosen[0] is long and late in chosen and dup not in chosen
    assert (count >= 1).all()


def test_one_euro_smooths_and_respects_gaps():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 200)[:, None] + rng.normal(0, 0.02, (200, 1))
    x[100] = np.nan
    y = smoothing.one_euro(x, fps=30, min_cutoff=1.0, beta=0.0)
    assert np.isnan(y[100]).all()
    assert np.nanstd(np.diff(y[:100, 0])) < np.nanstd(np.diff(x[:100, 0]))


def test_working_size_multiple_of_8():
    assert video.working_size(1920, 1080, 640) == (640, 360)
    w, h = video.working_size(1000, 563, 512)
    assert w % 8 == 0 and h % 8 == 0 and w <= 512


def test_sequence_map(tmp_path):
    for n in (1, 2, 3, 10):
        (tmp_path / ("shot_%04d.png" % n)).write_bytes(b"")
    (tmp_path / "other_0001.png").write_bytes(b"")
    files, first = video.sequence_map(str(tmp_path / "shot_0002.png"))
    assert first == 2 and sorted(files) == [1, 2, 3, 10]


def test_grid_points_respects_mask():
    valid = np.ones((9, 16), bool)
    valid[:, 8:] = False
    pts = tk.grid_points(1600, 900, 100, valid=valid)
    assert len(pts) > 20 and (pts[:, 0] < 800).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
