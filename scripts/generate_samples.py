# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate a batch of test models with the AI 3D Generator add-on.

Needs Blender with the add-on installed and its dependencies installed
(Preferences > Add-ons > AI 3D Generator). Run from a terminal:

    blender -b --python scripts/generate_samples.py -- --out C:/ai3d_pruebas
    blender -b --python scripts/generate_samples.py -- --out ~/ai3d --images ~/fotos
    blender -b --python scripts/generate_samples.py -- --out ~/ai3d --only taza,silla

Without --images it generates the built-in text prompts (needs the
"Texto → imagen" dependencies; SDXL downloads ~7 GB the first time). With
--images it reconstructs every .png/.jpg/.webp in that folder instead.

Writes per model: <name>.glb + <name>_licencias.json, the AI input image,
and for the batch: muestras.png (a render of all models side by side),
muestras.blend and resumen.json.
"""

import argparse
import json
import math
import os
import sys
import time

import bpy

# name, English prompt (SDXL/FLUX understand English best), real size in m, measured axis
SAMPLES = [
    ("taza", "a glossy blue ceramic coffee mug with a handle", 0.10, "HEIGHT"),
    ("silla", "a carved wooden nordic style chair", 0.90, "HEIGHT"),
    ("lampara", "an art deco brass table lamp with a white fabric shade", 0.55, "HEIGHT"),
    ("zapatilla", "a red and white running sneaker", 0.30, "LONGEST"),
    ("cofre", "a fantasy wooden treasure chest with iron bands and gold coins", 0.60, "LONGEST"),
    ("maceta", "a terracotta flower pot with a small green succulent plant", 0.25, "HEIGHT"),
]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def addon_modules():
    api = sys.modules.get("ai_3d_generator_api")
    if api is None:
        raise SystemExit("AI 3D Generator no está activado en este Blender "
                         "(Preferences > Add-ons). Si usas --factory-startup, quítalo.")
    pkg = api.__name__.rsplit(".", 1)[0]
    return api, sys.modules[pkg + ".tools"], sys.modules[pkg + ".jobs"]


def run_tool(key, obj=None, timeout=3600):
    """Run a tool synchronously (in `blender -b` timers do not fire while the
    script runs, so the job is driven here instead of by bpy.app.timers)."""
    _api, tools, jobs = addon_modules()
    context = bpy.context
    label, worker, params, finish = tools.prepare(context, key, obj)
    job = jobs.start(label, worker, params)
    driver = jobs.Driver(job, lambda c, r: finish(c, r, job))
    t0, last = time.time(), ""
    while True:
        done = driver.tick(context)
        if done is not None:
            jobs.complete(job, *done)
            return done[0], done[1], job.lines
        if job.message != last:
            last = job.message
            print("   %3d%%  %s" % (job.progress * 100, last), flush=True)
        if time.time() - t0 > timeout:
            job.cancelled = True
        time.sleep(0.1)


def contact_sheet(objects, path):
    """Render every model in a row, scaled to the same height, on neutral grey."""
    scene = bpy.context.scene
    hidden = {o: o.hide_render for o in scene.objects}
    for o in hidden:  # only the laid-out copies appear in the render
        o.hide_render = True
    tmp = []
    x = 0.0
    for obj in objects:
        dup = obj.copy()
        dup.hide_render = False
        scene.collection.objects.link(dup)
        dims = obj.dimensions
        s = 1.0 / max(dims.z, 1e-6)
        dup.scale = (s, s, s)
        width = max(dims.x, dims.y) * s
        dup.location = (x + width / 2, 0, 0)
        dup.rotation_euler = (0, 0, math.radians(-25))
        x += width + 0.3
        tmp.append(dup)
    cam_data = bpy.data.cameras.new("AI3D_SheetCam")
    cam_data.type = "ORTHO"
    width = max(x - 0.3, 1.0) + 0.4  # models are 1 unit tall, 0.3 apart
    height = 1.3
    cam_data.ortho_scale = max(width, height)
    cam = bpy.data.objects.new("AI3D_SheetCam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = ((x - 0.3) / 2, -10, 0.5)
    cam.rotation_euler = (math.radians(90), 0, 0)
    sun = bpy.data.objects.new("AI3D_SheetSun", bpy.data.lights.new("AI3D_SheetSun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(35))
    scene.collection.objects.link(sun)
    world = scene.world or bpy.data.worlds.new("AI3D_World")
    scene.world = world
    if bpy.app.version < (5, 0, 0):
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.35, 0.35, 0.37, 1)
        bg.inputs[1].default_value = 0.8
    scene.camera = cam
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x = max(640, int(420 * width))
    scene.render.resolution_y = int(scene.render.resolution_x * height / width)
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    for o in tmp + [cam, sun]:
        bpy.data.objects.remove(o)
    for o, h in hidden.items():
        o.hide_render = h
    return path


def main(argv):
    p = argparse.ArgumentParser(prog="generate_samples.py")
    p.add_argument("--out", required=True, help="carpeta de salida")
    p.add_argument("--images", help="carpeta con imágenes (si no, usa los prompts de ejemplo)")
    p.add_argument("--only", help="nombres separados por comas, p. ej. taza,silla")
    p.add_argument("--faces", type=int, default=20000, help="triángulos objetivo")
    p.add_argument("--texture", default="2048", choices=["512", "1024", "2048", "4096"])
    p.add_argument("--model", default="SDXL", choices=["SDXL", "FLUX_SCHNELL"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", default="glb", help="p. ej. glb,fbx,usdz")
    p.add_argument("--no-render", action="store_true", help="no renderizar muestras.png")
    args = p.parse_args(argv)

    api, _tools, _jobs = addon_modules()
    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.read_homefile(use_empty=True)

    if args.images:
        folder = os.path.abspath(os.path.expanduser(args.images))
        items = [(os.path.splitext(f)[0], os.path.join(folder, f), 1.0, "HEIGHT")
                 for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXT)]
    else:
        items = list(SAMPLES)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        items = [it for it in items if it[0] in wanted]
    if not items:
        raise SystemExit("No hay nada que generar")

    common = dict(auto_optimize=True, remesh_mode="DECIMATE", target_faces=args.faces,
                  texture_size=args.texture, bake_textures=True, bake_normal=True,
                  apply_size=True, keep_source=False, lods=0, export_dir=out,
                  t2i_model=args.model, write_provenance=True)
    summary = []
    made = []
    for n, (name, what, size, axis) in enumerate(items, 1):
        print("\n[%d/%d] %s" % (n, len(items), name), flush=True)
        settings = dict(common, name=name, real_size=size, size_axis=axis, seed=args.seed + n)
        if args.images:
            settings.update(source="IMAGE", image_path=what)
        else:
            settings.update(source="TEXT", prompt=what)
        api._apply_settings(settings)
        t0 = time.time()
        level, text, log = run_tool("generate")
        entry = {"name": name, "input": what, "status": level, "result": text,
                 "seconds": round(time.time() - t0, 1)}
        obj = bpy.data.objects.get(name)
        if level == "ERROR" or obj is None:
            entry["log_tail"] = log[-15:]
            print("   ERROR: %s" % text, flush=True)
        else:
            files = api.export(name, out, formats=args.formats.split(","))["files"]
            entry["files"] = [os.path.basename(f) for f in files]
            entry["report"] = api.inspect(name)
            made.append(obj)
            print("   OK: %s" % text, flush=True)
        summary.append(entry)

    if made and not args.no_render:
        print("\nRenderizando muestras.png", flush=True)
        contact_sheet(made, os.path.join(out, "muestras.png"))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "muestras.blend"))
    with open(os.path.join(out, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    ok = sum(e["status"] != "ERROR" for e in summary)
    print("\n%d/%d modelos generados en %s" % (ok, len(summary), out), flush=True)
    return summary


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
