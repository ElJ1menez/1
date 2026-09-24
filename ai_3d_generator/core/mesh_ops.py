# SPDX-License-Identifier: GPL-3.0-or-later
"""Numpy mesh clean-up applied to raw AI meshes before they reach Blender:
floating fragments, marching-cubes staircase, orientation and axes."""

import numpy as np


def edges_of(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def connected_components(n_verts, faces):
    """Label of each vertex's connected component (scipy if available)."""
    edges = edges_of(faces)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components as cc
        graph = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])),
                           shape=(n_verts, n_verts))
        return cc(graph, directed=False)[1]
    except ImportError:
        labels = np.arange(n_verts)
        while True:  # min-label propagation, slower but dependency-free
            m = np.minimum(labels[edges[:, 0]], labels[edges[:, 1]])
            new = labels.copy()
            np.minimum.at(new, edges[:, 0], m)
            np.minimum.at(new, edges[:, 1], m)
            new = new[new]
            if np.array_equal(new, labels):
                return np.unique(labels, return_inverse=True)[1]
            labels = new


def remove_small_parts(verts, faces, colors=None, min_ratio=0.02):
    """Drop connected parts with fewer than min_ratio of the largest part's faces."""
    if len(faces) == 0:
        return verts, faces, colors, 0
    labels = connected_components(len(verts), faces)
    face_label = labels[faces[:, 0]]
    counts = np.bincount(face_label)
    keep_label = counts >= max(1, min_ratio * counts.max())
    keep_face = keep_label[face_label]
    removed = int(np.count_nonzero(counts[~keep_label] > 0))
    verts, faces, colors = compact(verts, faces[keep_face], colors)
    return verts, faces, colors, removed


def compact(verts, faces, colors=None):
    used = np.zeros(len(verts), bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    colors = colors[used] if colors is not None else None
    return verts[used], remap[faces].astype(np.int32), colors


def _neighbour_mean(verts, edges, degree):
    acc = np.zeros_like(verts)
    np.add.at(acc, edges[:, 0], verts[edges[:, 1]])
    np.add.at(acc, edges[:, 1], verts[edges[:, 0]])
    return acc / degree[:, None]


def taubin_smooth(verts, faces, iterations=10, lamb=0.5, mu=-0.53):
    """Volume-preserving smoothing that removes marching-cubes terracing."""
    if iterations <= 0 or len(faces) == 0:
        return verts
    edges = edges_of(faces)
    degree = np.bincount(edges.ravel(), minlength=len(verts)).astype(np.float64)
    degree[degree == 0] = 1.0
    v = verts.astype(np.float64)
    for _ in range(iterations):
        v = v + lamb * (_neighbour_mean(v, edges, degree) - v)
        v = v + mu * (_neighbour_mean(v, edges, degree) - v)
    return v.astype(np.float32)


def signed_volume(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def orient_outwards(verts, faces):
    """Flip winding if the (closed) mesh has negative signed volume."""
    if signed_volume(verts, faces) < 0:
        return faces[:, ::-1].copy(), True
    return faces, False


def triposr_to_blender(verts):
    """TripoSR space (z up, input camera on +x, image right = +y) to Blender
    (Z up, object front facing -Y as seen from the Front view)."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    return np.stack([y, -x, z], axis=1).astype(np.float32)


def normalize_to_ground(verts, size=1.0, axis="HEIGHT"):
    """Centre in X/Y, rest on Z=0 and scale so height (or longest side) = size."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    extent = hi - lo
    ref = extent[2] if axis == "HEIGHT" else extent.max()
    scale = size / ref if ref > 0 else 1.0
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], np.float32)
    return ((verts - center) * scale).astype(np.float32)


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def clean(verts, faces, colors, min_part_ratio=0.02, smooth_iterations=10):
    """Full numpy clean-up. Returns (verts, faces, colors, notes)."""
    notes = []
    verts, faces, colors, removed = remove_small_parts(verts, faces, colors, min_part_ratio)
    if removed:
        notes.append("%d fragmentos sueltos eliminados" % removed)
    verts = taubin_smooth(verts, faces, smooth_iterations)
    faces, flipped = orient_outwards(verts, faces)
    if flipped:
        notes.append("normales invertidas corregidas")
    return verts, faces, colors, notes
