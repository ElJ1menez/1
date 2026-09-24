# SPDX-License-Identifier: GPL-3.0-or-later
"""AI 3D Generator: tests for the Blender-independent core.
Run: python -m pytest tests/test_ai3d_core.py"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_3d_generator.core import image_prep, mesh_ops  # noqa: E402


def grid_sphere(res=24, radius=0.5, offset=(0, 0, 0)):
    """Closed triangulated sphere (UV sphere with poles)."""
    verts = [(0, 0, radius)]
    for i in range(1, res):
        th = np.pi * i / res
        for j in range(2 * res):
            ph = 2 * np.pi * j / (2 * res)
            verts.append((radius * np.sin(th) * np.cos(ph), radius * np.sin(th) * np.sin(ph),
                          radius * np.cos(th)))
    verts.append((0, 0, -radius))
    faces = []
    ring = 2 * res
    for j in range(ring):
        faces.append((0, 1 + j, 1 + (j + 1) % ring))
    for i in range(res - 2):
        a0 = 1 + i * ring
        b0 = a0 + ring
        for j in range(ring):
            a, a1 = a0 + j, a0 + (j + 1) % ring
            b, b1 = b0 + j, b0 + (j + 1) % ring
            faces += [(a, b, b1), (a, b1, a1)]
    last = len(verts) - 1
    base = 1 + (res - 2) * ring
    for j in range(ring):
        faces.append((base + j, last, base + (j + 1) % ring))
    return np.array(verts, np.float32) + np.array(offset, np.float32), np.array(faces, np.int32)


def test_sphere_is_outward_and_closed():
    v, f = grid_sphere()
    assert mesh_ops.signed_volume(v, f) > 0.45  # ≈ 4/3·π·0.125 = 0.52
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_remove_small_parts_keeps_main_body():
    v1, f1 = grid_sphere(24)
    v2, f2 = grid_sphere(4, 0.05, offset=(2, 0, 0))  # small floater
    v = np.concatenate([v1, v2])
    f = np.concatenate([f1, f2 + len(v1)])
    c = np.random.rand(len(v), 3).astype(np.float32)
    v_out, f_out, c_out, removed = mesh_ops.remove_small_parts(v, f, c, 0.05)
    assert removed == 1
    assert len(v_out) == len(v1) and len(f_out) == len(f1) and len(c_out) == len(v1)
    np.testing.assert_allclose(v_out, v1)
    np.testing.assert_allclose(c_out, c[:len(v1)])


def test_connected_components_fallback_matches_scipy(monkeypatch):
    v1, f1 = grid_sphere(6)
    v2, f2 = grid_sphere(4, offset=(3, 0, 0))
    f = np.concatenate([f1, f2 + len(v1)])
    n = len(v1) + len(v2)
    ref = mesh_ops.connected_components(n, f)
    import builtins
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_scipy)
    fb = mesh_ops.connected_components(n, f)
    assert len(np.unique(fb)) == len(np.unique(ref)) == 2
    assert np.all(fb[:len(v1)] == fb[0]) and np.all(fb[len(v1):] != fb[0])


def test_taubin_reduces_noise_without_shrinking():
    v, f = grid_sphere(24)
    rng = np.random.default_rng(0)
    noisy = v + rng.normal(0, 0.01, v.shape).astype(np.float32)
    smooth = mesh_ops.taubin_smooth(noisy, f, 10)
    err = lambda x: np.abs(np.linalg.norm(x, axis=1) - 0.5).mean()  # noqa: E731
    assert err(smooth) < err(noisy) * 0.6
    assert abs(np.linalg.norm(smooth, axis=1).mean() - 0.5) < 0.01  # no shrinkage


def test_orient_outwards_flips_inverted_mesh():
    v, f = grid_sphere(8)
    f2, flipped = mesh_ops.orient_outwards(v, f[:, ::-1])
    assert flipped and mesh_ops.signed_volume(v, f2) > 0


def test_axes_and_ground():
    # TripoSR: +x towards the input camera, +y image right, +z up.
    v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    b = mesh_ops.triposr_to_blender(v)
    np.testing.assert_allclose(b, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # front faces -Y
    v, _ = grid_sphere(8)
    g = mesh_ops.normalize_to_ground(v * [1, 1, 2], size=1.8)
    assert abs(g[:, 2].min()) < 1e-6 and abs(g[:, 2].max() - 1.8) < 1e-5
    assert abs(g[:, 0].min() + g[:, 0].max()) < 1e-5
    g = mesh_ops.normalize_to_ground(v * [3, 1, 1], size=2.0, axis="LONGEST")
    assert abs(np.ptp(g[:, 0]) - 2.0) < 1e-5


def test_srgb_to_linear():
    np.testing.assert_allclose(mesh_ops.srgb_to_linear(np.array([0.0, 0.5, 1.0])),
                               [0.0, 0.214, 1.0], atol=1e-3)


def test_center_on_gray_ratio_and_background():
    rgba = np.zeros((200, 300, 4), np.float32)
    rgba[50:150, 100:140] = (1, 0, 0, 1)  # 100 tall, 40 wide
    out = image_prep.center_on_gray(rgba, ratio=0.8, out_size=100)
    assert out.shape == (100, 100, 3)
    np.testing.assert_allclose(out[2, 2], 0.5, atol=0.01)
    red = (out[..., 0] > 0.9) & (out[..., 1] < 0.1)
    ys, xs = np.nonzero(red)
    assert 76 <= ys.max() - ys.min() + 1 <= 81  # object fills ~80% of the side
    assert abs((ys.min() + ys.max()) / 2 - 49.5) <= 1.5
    assert abs((xs.min() + xs.max()) / 2 - 49.5) <= 1.5


def test_empty_mask_raises():
    with pytest.raises(RuntimeError):
        image_prep.center_on_gray(np.zeros((10, 10, 4), np.float32))


def test_clean_alpha():
    np.testing.assert_allclose(image_prep.clean_alpha(np.array([0.05, 0.5, 0.95])), [0, 0.5, 1])


# --- vendored TripoSR ---------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("einops")


def test_vit_matches_transformers():
    transformers = pytest.importorskip("transformers")
    from ai_3d_generator.third_party.triposr.vit import ViTModel
    torch.manual_seed(0)
    ref = transformers.ViTModel(transformers.ViTConfig(qkv_bias=True)).eval()
    mine = ViTModel().eval()
    mine.load_state_dict(ref.state_dict(), strict=True)  # same parameter names
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        a = ref(x).last_hidden_state
        b, _ = mine(x)
    assert torch.allclose(a, b, atol=1e-5)
    with torch.no_grad():  # 512 px input uses interpolated position embeddings
        assert mine(torch.rand(1, 3, 512, 512))[0].shape == (1, 1 + 32 * 32, 768)


def test_triposr_architecture_and_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("skimage")
    from ai_3d_generator.third_party.triposr import TSR, TRIPOSR_CONFIG
    model = TSR(TRIPOSR_CONFIG).eval()
    n = sum(p.numel() for p in model.state_dict().values())
    assert abs(n - 419.28e6) < 0.1e6  # matches the 1.68 GB official checkpoint
    # Checkpoint saved without the unused ViT pooler still loads.
    sd = {k: v for k, v in model.state_dict().items() if ".pooler." not in k}
    path = tmp_path / "model.ckpt"
    torch.save(sd, path)
    loaded = TSR.from_checkpoint(str(path))
    torch.save({**sd, "extra.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError):
        TSR.from_checkpoint(str(path))
    with torch.no_grad():
        codes = loaded(np.random.rand(64, 48, 3).astype(np.float32), "cpu")
    assert codes.shape == (1, 3, 40, 64, 64)


def test_extract_mesh_sphere_outward():
    pytest.importorskip("skimage")
    from ai_3d_generator.third_party.triposr import TSR, TRIPOSR_CONFIG
    model = TSR(TRIPOSR_CONFIG)

    def fake_query(decoder, pos, code):
        r = pos.norm(dim=-1, keepdim=True)
        return {"density_act": torch.where(r < 0.5, 100.0, 0.0) * torch.ones_like(r),
                "color": (pos + 1) / 2}
    model.renderer.query_triplane = fake_query
    v, f, c = model.extract_mesh(None, resolution=48, threshold=25.0)
    assert v.shape[1] == 3 and f.shape[1] == 3 and c.shape == v.shape
    assert np.abs(np.linalg.norm(v, axis=1) - 0.5).max() < 0.05
    assert mesh_ops.signed_volume(v, f) > 0  # normals point outwards
