# SPDX-License-Identifier: GPL-3.0-or-later
"""Temporal smoothing for landmark streams. Pure numpy, no bpy."""

import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(values, fps, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
    """One Euro filter (Casiez et al. 2012) along axis 0.

    values: (T, ...) float array. NaN samples (missed detections) are kept as
    NaN and restart the filter, so gaps are never bridged with stale data.
    """
    x = np.asarray(values, dtype=np.float64)
    out = np.full_like(x, np.nan)
    dt = 1.0 / float(fps)
    a_d = _alpha(d_cutoff, dt)
    prev = None
    dprev = None
    for i in range(x.shape[0]):
        cur = x[i]
        if not np.all(np.isfinite(cur)):
            prev = dprev = None
            continue
        if prev is None:
            prev, dprev = cur, np.zeros_like(cur)
            out[i] = cur
            continue
        dx = (cur - prev) / dt
        dhat = a_d * dx + (1.0 - a_d) * dprev
        cutoff = min_cutoff + beta * np.abs(dhat)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        filt = a * cur + (1.0 - a) * prev
        out[i] = filt
        prev, dprev = filt, dhat
    return out
