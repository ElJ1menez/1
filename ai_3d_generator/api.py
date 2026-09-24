# SPDX-License-Identifier: GPL-3.0-or-later
"""Scripting API, designed to be driven through an MCP server for Blender
(blender-mcp or similar) via its "execute code" tool, or from any script:

    import ai_3d_generator_api as ai3d
    print(ai3d.help())

Every function returns plain JSON-serialisable data. AI jobs run in the
background (bpy.app.timers), so calls return immediately: poll status().
"""

import os

import bpy

from . import deps, jobs, tools
from . import export as export_mod
from . import optimize as optimize_mod
from .core import downloads

ALIAS = "ai_3d_generator_api"

HELP = """AI 3D Generator — API (import ai_3d_generator_api as ai3d)

ai3d.status()                        dependencias, tarea en curso, último resultado, objeto activo
ai3d.settings()                      ajustes actuales {nombre: valor} con descripción y opciones
ai3d.generate(prompt=None, image=None, **ajustes)
    prompt: texto → imagen (SDXL/FLUX) → 3D      image: ruta de una foto/render → 3D
    ajustes: cualquier nombre de settings(), p. ej. real_size=0.45, target_faces=15000,
             remesh_mode="QUADS", texture_size="2048", lods=2, seed=123
    devuelve al instante; la IA trabaja en segundo plano
ai3d.optimize(object=None, **ajustes)   limpia, retopologiza, UVs, texturas PBR, LODs
ai3d.export(object=None, directory=None, formats=("glb",), provenance=True)
ai3d.inspect(object=None)            control de calidad: triángulos, medidas, estanqueidad, UVs, texturas
ai3d.licenses()                      modelos de IA usados y sus licencias
ai3d.wait_hint()                     cuánto esperar antes de volver a llamar a status()
ai3d.cancel()                        cancela la tarea en curso

Flujo típico:
    ai3d.generate(prompt="taza de cerámica esmaltada azul", real_size=0.1, name="Taza")
    ai3d.status()      # repetir hasta que "running" sea False; leer "last_result"
    ai3d.inspect()     # watertight True, triángulos ≈ target_faces, texturas presentes
    ai3d.export(directory=r"C:/assets", formats=("glb", "fbx"))
Consejos: una imagen de un único objeto completo, centrado y con fondo liso da el mejor
resultado. Si sale hueco o roto: threshold más bajo (15–20); si sale hinchado: más alto (30–40).
"""


def help():  # noqa: A001 - intentionally mirrors the builtin for discoverability
    return HELP


def _context():
    return bpy.context


def _value(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    try:
        return list(v)
    except TypeError:
        return str(v)


def settings():
    s = _context().scene.ai3d
    out = {}
    for prop in s.bl_rna.properties:
        if prop.identifier in {"rna_type"}:
            continue
        entry = {"value": _value(getattr(s, prop.identifier)), "description": prop.description}
        if prop.type == "ENUM":
            entry["options"] = [i.identifier for i in prop.enum_items]
        elif prop.type in {"INT", "FLOAT"}:
            entry["min"], entry["max"] = prop.hard_min, prop.hard_max
        out[prop.identifier] = entry
    return out


def _apply_settings(values):
    s = _context().scene.ai3d
    valid = set(s.bl_rna.properties.keys()) - {"rna_type"}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError("Ajustes desconocidos: %s. Usa settings() para ver los válidos." % unknown)
    for key, value in values.items():
        setattr(s, key, value)


def _start(tool, obj=None):
    ctx = _context()
    label, worker, params, finish = tools.prepare(ctx, tool, obj)
    job = jobs.run_in_background(label, worker, params,
                                 lambda c, result: finish(c, result, job))
    return job


def generate(prompt=None, image=None, **overrides):
    if prompt and image:
        raise ValueError("Pasa prompt o image, no ambos")
    if prompt:
        overrides.update(source="TEXT", prompt=prompt)
    elif image:
        path = os.path.abspath(bpy.path.abspath(image))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        overrides.update(source="IMAGE", image_path=path)
    _apply_settings(overrides)
    _start("generate")
    s = _context().scene.ai3d
    return {"started": "generate", "source": s.source, "name": s.name,
            "auto_optimize": s.auto_optimize, "hint": "Llama a status() para ver el progreso"}


def optimize(object=None, **overrides):  # noqa: A002 - matches the Blender term
    _apply_settings(overrides)
    obj = tools.target_object(_context(), object)
    _start("optimize", obj)
    return {"started": "optimize", "object": obj.name if obj else None}


def export(object=None, directory=None, formats=("glb",), provenance=True):  # noqa: A002
    ctx = _context()
    obj = tools.target_object(ctx, object)
    if obj is None:
        raise ValueError("No hay objeto de malla activo; pasa object='Nombre'")
    folder = export_mod.export_dir(directory or ctx.scene.ai3d.export_dir)
    files = export_mod.export_asset(ctx, obj, folder, [f.lower().lstrip(".") for f in formats],
                                    provenance)
    return {"object": obj.name, "directory": folder, "files": files}



def inspect(object=None):  # noqa: A002
    obj = tools.target_object(_context(), object)
    if obj is None:
        return {"error": "no hay objeto de malla activo"}
    return optimize_mod.mesh_report(obj)



def licenses():
    out = {name: {k: m[k] for k in ("source", "license", "size")}
           for name, m in downloads.MODELS.items()}
    out.update({"t2i_" + name: {k: m[k] for k in ("source", "license", "size", "repo")}
                for name, m in downloads.T2I_MODELS.items()})
    return out


def cancel():
    job = jobs.current()
    if job is None or job.finished:
        return {"cancelled": False, "reason": "no hay tarea en curso"}
    job.cancelled = True
    return {"cancelled": True, "job": job.name}


def status():
    ctx = _context()
    job = jobs.current()
    obj = tools.target_object(ctx)
    out = {
        "addon": ADDON_VERSION,
        "blender": bpy.app.version_string,
        "dependencies": {k: deps.available(k) for k in deps.GROUPS},
        "device": tools.get_prefs(ctx).device,
        "active_object": obj.name if obj else None,
        "running": jobs.busy(),
    }
    if job is not None:
        out["job"] = {"name": job.name, "progress": round(job.progress, 3), "message": job.message,
                      "finished": job.finished}
    if jobs.last_report["text"]:
        out["last_result"] = {"level": jobs.last_report["level"], "text": jobs.last_report["text"]}
        if jobs.last_report["level"] == "ERROR" and jobs.last_log:
            out["last_result"]["log_tail"] = jobs.last_log[-15:]
    return out


def wait_hint():
    job = jobs.current()
    if not jobs.busy():
        return {"seconds": 0, "reason": "nada en marcha"}
    if "Descargando" in job.message:
        return {"seconds": 30, "reason": job.message}
    return {"seconds": 10, "reason": job.message}


ADDON_VERSION = export_mod.ADDON_VERSION
