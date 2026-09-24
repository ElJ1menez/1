# SPDX-License-Identifier: GPL-3.0-or-later
"""Ground-truth accuracy of the AI tracker on a synthetic clip with known motion.

Needs torch, opencv, dm-tree, einshape and the BootsTAPIR checkpoint:
    AIMT_BOOTSTAPIR=/path/bootstapir_checkpoint_v2.pt python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CKPT = os.environ.get("AIMT_BOOTSTAPIR", "")
pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(not os.path.isfile(CKPT), reason="set AIMT_BOOTSTAPIR")

from ai_motion_tracker.core import tapir_engine as te  # noqa: E402


def _clip(num=16, width=640, height=360):
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur((rng.random((900, 1400, 3)) * 255).astype(np.uint8), (0, 0), 3)
    hs = []
    frames = []
    for t in range(num):
        s = 0.6 + 0.004 * t
        h = np.array([[s, 0.0008 * t, -200 - 6.0 * t], [0, s, -150 - 2.5 * t], [0, 2e-6 * t, 1.0]])
        hs.append(h)
        frames.append(cv2.warpPerspective(tex, h, (width, height)))
    return frames, hs


def _gt(hs, p, s, t):
    q = hs[t] @ np.linalg.inv(hs[s]) @ np.array([p[0] - 0.5, p[1] - 0.5, 1.0])
    return q[:2] / q[2] + 0.5


class _Job:
    cancelled = False

    def check(self):
        pass


@pytest.mark.parametrize("size,limit", [(256, 1.0), (512, 0.8)])
def test_bidirectional_tracking_is_subpixel(size, limit):
    frames, hs = _clip()
    device = te.pick_device("CPU")
    model = te.load_model(CKPT, device)
    pts = np.array([[100.5, 80.5], [320.5, 180.5], [500.5, 300.5]], np.float32)
    t0 = 8
    xy, vis = te.track_points(model, frames, t0, pts, 8, device, _Job(), lambda p, m: None, size)
    err = [np.linalg.norm(xy[n, t] - _gt(hs, pts[n], t0, t))
           for n in range(3) for t in range(len(frames)) if vis[n, t]]
    assert vis.mean() > 0.9
    assert np.median(err) < limit
