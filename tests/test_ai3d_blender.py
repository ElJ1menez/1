# SPDX-License-Identifier: GPL-3.0-or-later
"""AI 3D Generator inside Blender (headless), using the `bpy` module from PyPI
(pip install bpy==4.2.0 on Python 3.11). The generation pipeline runs for
real except TripoSR's weights: the model is built with the real architecture
but its density field is replaced by a known shape, since the checkpoint is a
1.7 GB download. Set AI3D_ISNET=/path/isnet-general-use.onnx to also run the
real background remover.

Run: python -m pytest tests/test_ai3d_blender.py
"""

import json
import os
import sys
import time

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
torch = pytest.importorskip("torch")
pytest.importorskip("skimage")
pytest.importorskip("PIL")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def addon():
    import addon_utils
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mod = addon_utils.enable("ai_3d_generator", default_set=True)
    assert mod is not None
    from ai_3d_generator import api, deps
    yield api, deps
    addon_utils.disable("ai_3d_generator")


def _fake_triposr(models_dir, device, job=None):
    from ai_3d_generator.third_party.triposr import TSR, TRIPOSR_CONFIG
    model = TSR(TRIPOSR_CONFIG).eval()

    def query(decoder, pos, code):
        # Ellipsoid, taller than wide, plus a small floating fragment that
        # the clean-up must remove. Colour: red top, blue bottom.
        q = pos / torch.tensor([0.3, 0.25, 0.45])
        inside = (q.norm(dim=-1, keepdim=True) < 1.0) | ((pos - torch.tensor([0.0, 0.6, 0.0])).norm(dim=-1, keepdim=True) < 0.035)
        density = torch.where(inside, 100.0, 0.0) * torch.ones_like(pos[..., :1])
        top = (pos[..., 2:3] > 0).float()
        color = torch.cat([top * 0.9 + 0.05, torch.full_like(top, 0.2), (1 - top) * 0.9 + 0.05], -1)
        return {"density_act": density, "color": color}
    model.renderer.query_triplane = query
    return model


def _drive(context, key, obj=None, timeout=300):
    from ai_3d_generator import jobs, tools
    label, worker, params, finish = tools.prepare(context, key, obj)
    job = jobs.start(label, worker, params)
    driver = jobs.Driver(job, lambda c, r: finish(c, r, job))
    t0 = time.time()
    while True:
        done = driver.tick(context)
        if done is not None:
            jobs.complete(job, *done)
            if done[0] == "ERROR":
                print("\n".join(job.lines[-30:]))
            return done
        assert time.time() - t0 < timeout
        time.sleep(0.05)


def test_generate_optimize_export(addon, tmp_path, monkeypatch):
    api, deps = addon
    from ai_3d_generator.core import engines
    monkeypatch.setattr(engines, "triposr", _fake_triposr)
    context = bpy.context

    # Input image: an object on white, RGBA with transparency unless the real
    # IS-Net model is available.
    from PIL import Image
    img = np.full((256, 256, 4), 255, np.uint8)
    yy, xx = np.mgrid[:256, :256]
    blob = ((yy - 128) / 90.0) ** 2 + ((xx - 128) / 60.0) ** 2 < 1
    img[blob, :3] = (200, 60, 30)
    isnet = os.environ.get("AI3D_ISNET")
    if isnet:
        monkeypatch.setitem(deps.paths, "models", str(tmp_path))
        os.symlink(isnet, tmp_path / "isnet_isnet-general-use.onnx")
    else:
        img[~blob, 3] = 0
    path = tmp_path / "input.png"
    Image.fromarray(img).save(path)
    monkeypatch.setitem(deps.paths, "outputs", str(tmp_path / "outputs"))

    api._apply_settings(dict(source="IMAGE", image_path=str(path), name="Jarron",
                             mc_resolution=96, real_size=0.4, target_faces=2000,
                             texture_size="512", lods=1, keep_source=True,
                             export_dir=str(tmp_path / "export")))
    level, text = _drive(context, "generate")
    assert level == "INFO", text
    assert "fragmentos sueltos eliminados" in text

    obj = bpy.data.objects["Jarron"]
    r = api.inspect("Jarron")
    assert r["watertight"] and r["uv_maps"] and len(r["textures"]) == 2
    assert 1000 <= r["triangles"] <= 2000
    assert abs(r["dimensions_m"][2] - 0.4) < 0.01
    # TripoSR depth axis (x, semi-axis 0.3) becomes Blender Y, image width (y, 0.25) becomes X.
    assert r["dimensions_m"][2] > r["dimensions_m"][1] > r["dimensions_m"][0]
    assert abs(r["dimensions_m"][0] - 0.4 * 0.25 / 0.45) < 0.01  # fragment gone
    lo = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    assert abs(lo - obj.location.z) < 1e-4  # origin on the ground
    assert bpy.data.objects.get("Jarron_LOD1") is not None
    assert bpy.data.objects.get("Jarron_original") is not None  # keep_source

    # The baked base colour reproduces the AI vertex colours (red top, blue bottom).
    base = bpy.data.images["Jarron_BaseColor"]
    px = np.array(base.pixels[:]).reshape(base.size[1], base.size[0], 4)
    uv = np.array([d.uv[:] for d in obj.data.uv_layers.active.data])
    loops_z = np.array([obj.data.vertices[lp.vertex_index].co.z for lp in obj.data.loops])
    ij = np.clip((uv * base.size[0]).astype(int), 0, base.size[0] - 1)
    sampled = px[ij[:, 1], ij[:, 0], :3]
    top, bottom = sampled[loops_z > 0.25].mean(0), sampled[loops_z < 0.15].mean(0)
    assert top[0] > top[2] + 0.3 and bottom[2] > bottom[0] + 0.3, (top, bottom)

    out = api.export("Jarron", str(tmp_path / "export"), formats=("glb", "fbx", "obj", "stl"))
    names = sorted(os.path.basename(f) for f in out["files"])
    assert "Jarron.glb" in names and "Jarron_LOD1.glb" in names and "Jarron_licencias.json" in names
    assert os.path.getsize(tmp_path / "export" / "Jarron.glb") > 50000  # textures embedded
    sheet = json.load(open(tmp_path / "export" / "Jarron_licencias.json", encoding="utf-8"))
    assert "MIT" in sheet["licenses"]
    assert sheet["report"]["watertight"] is True
    stages = [s["stage"] for s in sheet["generation"]["steps"]]
    assert stages[0] == "input_image" and "image_to_3d" in stages
    assert ("background_removal" in stages) == bool(isnet)


def test_optimize_any_mesh_quads_without_bake(addon):
    api, _deps = addon
    context = bpy.context
    bpy.ops.mesh.primitive_monkey_add()
    monkey = context.object
    monkey.modifiers.new("s", "SUBSURF").levels = 2
    monkey.name = "Mono"
    api._apply_settings(dict(remesh_mode="QUADS", target_faces=3000, bake_textures=False,
                             lods=0, keep_source=False, apply_size=True, real_size=0.3))
    level, text = _drive(context, "optimize", monkey)
    assert level == "INFO", text
    r = api.inspect("Mono")
    assert r["triangles"] < 5000 and abs(r["dimensions_m"][2] - 0.3) < 0.01
    assert bpy.data.objects.get("Mono_original") is None
    assert r["watertight"]  # voxel remesh closed the eyes/ears gaps


def test_optimize_failure_leaves_original(addon, monkeypatch):
    api, _deps = addon
    from ai_3d_generator import optimize
    context = bpy.context
    bpy.ops.mesh.primitive_uv_sphere_add()
    context.object.name = "Bola"
    names = {o.name for o in bpy.data.objects}

    def broken(*args, **kwargs):
        raise RuntimeError("UV roto")
    monkeypatch.setattr(optimize, "smart_uv", broken)
    api._apply_settings(dict(remesh_mode="DECIMATE", bake_textures=True))
    level, text = _drive(context, "optimize", context.object)
    assert level == "ERROR" and "UV roto" in text
    assert {o.name for o in bpy.data.objects} == names


def test_api_surface(addon):
    api, _deps = addon
    s = api.settings()
    assert s["remesh_mode"]["options"] == ["DECIMATE", "QUADS", "NONE"]
    with pytest.raises(ValueError):
        api._apply_settings({"no_existe": 1})
    st = api.status()
    json.dumps(st)
    assert set(st["dependencies"]) == {"torch", "gen3d", "text2image"}
    lic = api.licenses()
    assert lic["triposr"]["license"] == "MIT" and lic["t2i_FLUX_SCHNELL"]["license"] == "Apache-2.0"
    assert "ai3d.generate" in api.help() or "generate(" in api.help()
