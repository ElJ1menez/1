# SPDX-License-Identifier: GPL-3.0-or-later
"""Generation pipeline run on the worker thread (no bpy):

prompt ─(text-to-image)─► image ─(IS-Net)─► object on gray ─(TripoSR)─►
triplanes ─(marching cubes)─► mesh with vertex colours ─(numpy clean-up)─►
Blender (builders.py / optimize.py).
"""

import os
import random
import time

import numpy as np

from . import downloads, engines, image_prep, mesh_ops

# Appended to the user's prompt so the image works for single-view 3D:
# one whole object, centred, neutral light, no ground or clutter.
ASSET_STYLE = ("single object, full object in frame, centered, three-quarter front view, "
               "plain white background, soft even studio lighting, no shadows, "
               "product render, highly detailed")
ASSET_NEGATIVE = ("multiple objects, cropped, cut off, text, watermark, logo, frame, "
                  "busy background, scenery, floor, harsh shadows, blurry")


def generate_image(job, p, device):
    info = downloads.T2I_MODELS[p["t2i_model"]]
    pipe = engines.text_to_image(p["t2i_model"], device, p["hf_cache"], p.get("hf_token"), job)
    import torch
    steps = p["t2i_steps"] or info["steps"]
    prompt = p["prompt"].strip()
    if p["asset_style"]:
        prompt = "%s, %s" % (prompt, ASSET_STYLE)
    generator = torch.Generator("cpu").manual_seed(p["seed"])

    def on_step(pipe_, step, timestep, kwargs):
        job.check()
        job.set(progress=0.05 + 0.35 * (step + 1) / steps,
                message="Generando imagen (%d/%d)" % (step + 1, steps))
        return kwargs

    kwargs = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=info["guidance"],
                  height=info["size_px"], width=info["size_px"], generator=generator,
                  callback_on_step_end=on_step)
    if p["t2i_model"] == "FLUX_SCHNELL":
        kwargs["max_sequence_length"] = 256
    else:
        neg = p["negative_prompt"].strip()
        if p["asset_style"]:
            neg = ", ".join(x for x in (neg, ASSET_NEGATIVE) if x)
        kwargs["negative_prompt"] = neg or None
    image = pipe(**kwargs).images[0]
    if not p["keep_models"]:
        engines.release(keep=("triposr", "isnet"))
    rgba = np.asarray(image.convert("RGBA"), np.float32) / 255.0
    return rgba, {"model": info["source"], "license": info["license"], "repo": info["repo"],
                  "prompt": prompt, "steps": steps}


def generate(job, p):
    """Worker entry point. p: see tools._generate_params."""
    t0 = time.time()
    device = engines.resolve_device(p["device"])
    if p["seed"] < 0:
        p["seed"] = random.randint(0, 2**31 - 1)
    os.makedirs(p["out_dir"], exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    provenance = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device,
                  "seed": p["seed"], "steps": []}

    job.set(0.02, "Preparando imagen")
    if p["source"] == "TEXT":
        if not p["prompt"].strip():
            raise RuntimeError("Escribe una descripción del objeto")
        rgba, meta = generate_image(job, p, device)
        provenance["steps"].append(dict(stage="text_to_image", **meta))
        image_prep.save_rgb(os.path.join(p["out_dir"], stamp + "_imagen.png"), rgba[..., :3])
    else:
        if not os.path.isfile(p["image_path"]):
            raise RuntimeError("No existe la imagen: %s" % p["image_path"])
        rgba = image_prep.load_rgba(p["image_path"])
        provenance["steps"].append({"stage": "input_image", "path": p["image_path"],
                                    "note": "Imagen aportada por el usuario: asegúrate de tener "
                                            "derechos comerciales sobre ella"})
    job.check()

    if p["remove_background"] and not image_prep.has_transparency(rgba):
        job.set(0.42, "Quitando el fondo (IS-Net)")
        remover = engines.background_remover(p["models_dir"], job)
        rgba = rgba.copy()
        rgba[..., 3] = image_prep.clean_alpha(remover.mask(rgba[..., :3]))
        info = downloads.MODELS["isnet"]
        provenance["steps"].append({"stage": "background_removal", "model": info["source"],
                                    "license": info["license"]})
    elif not image_prep.has_transparency(rgba):
        rgba = rgba.copy()
        rgba[..., 3] = 1.0
    cond = image_prep.center_on_gray(rgba, p["foreground_ratio"])
    cond_path = os.path.join(p["out_dir"], stamp + "_entrada_ia.png")
    image_prep.save_rgb(cond_path, cond)
    job.check()

    job.set(0.5, "Reconstruyendo en 3D (TripoSR)")
    model = engines.triposr(p["models_dir"], device, job)
    import torch
    model.renderer.set_chunk_size(p["chunk_size"])
    model.renderer.check = job.check
    with torch.no_grad():
        scene_codes = model(cond, device)
    job.check()
    job.set(0.65, "Extrayendo superficie (%d³)" % p["mc_resolution"])
    verts, faces, colors = model.extract_mesh(scene_codes[0], p["mc_resolution"], p["threshold"])
    model.renderer.check = None
    if not p["keep_models"]:
        engines.release()
    info = downloads.MODELS["triposr"]
    provenance["steps"].append({"stage": "image_to_3d", "model": info["source"],
                                "license": info["license"], "resolution": p["mc_resolution"],
                                "threshold": p["threshold"]})

    job.set(0.85, "Limpiando malla")
    verts = mesh_ops.triposr_to_blender(verts)
    verts, faces, colors, notes = mesh_ops.clean(verts, faces, colors, p["min_part_ratio"],
                                                 p["smooth_iterations"])
    verts = mesh_ops.normalize_to_ground(verts, p["real_size"], p["size_axis"])
    provenance["seconds"] = round(time.time() - t0, 1)
    return {
        "verts": verts, "faces": faces, "colors": mesh_ops.srgb_to_linear(colors),
        "name": p["name"], "image": cond_path, "notes": notes, "provenance": provenance,
    }
