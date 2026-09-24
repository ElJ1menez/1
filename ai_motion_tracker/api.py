# SPDX-License-Identifier: GPL-3.0-or-later
"""Scripting API, designed to be driven through an MCP server for Blender
(blender-mcp or similar) via its "execute code" tool, or from any script:

    import ai_motion_tracker_api as aimt
    print(aimt.help())

Every function returns plain JSON-serialisable data. AI jobs run in the
background (bpy.app.timers), so calls return immediately: poll status().
"""

import os

import bpy

from . import deps, jobs, tools

ALIAS = "ai_motion_tracker_api"

HELP = """AI Motion Tracker — API (import ai_motion_tracker_api as aimt)

aimt.status()                      estado: dependencias, clip activo, tarea en curso, último resultado
aimt.load_clip(path)               carga un video o secuencia y lo pone como clip activo
aimt.settings()                    ajustes actuales {nombre: valor} y opciones válidas
aimt.run(tool, clip=None, **ajustes)
    tool: "camera_track" | "track_selected" | "solve" | "pose" | "face"
    ajustes: cualquier nombre de settings(), p. ej. per_frame=100, auto_focal=True
    devuelve al instante; la IA trabaja en segundo plano
aimt.wait_hint()                   cuánto esperar antes de volver a llamar a status()
aimt.cancel()                      cancela la tarea en curso
aimt.solve_info(clip=None)         error del solve, focal, frames resueltos, nº de tracks

Flujo típico de tracking de cámara:
    aimt.load_clip(r"C:/ruta/plano.mp4")
    aimt.run("camera_track", auto_focal=True, setup_scene=True)
    aimt.status()   # repetir hasta que "running" sea False; leer "last_result"
    aimt.solve_info()
Un solve con error < 0.5 px es bueno. Si falla: run("camera_track", per_frame=120)
o run("solve", auto_focal=True, focal_min=10, focal_max=150).
"""


def help():  # noqa: A001 - intentionally mirrors the builtin for discoverability
    return HELP


def _context():
    return bpy.context


def _clip(name=None):
    ctx = _context()
    if name:
        clip = bpy.data.movieclips.get(name)
        if clip is None:
            raise ValueError("No existe el clip %r. Clips: %s" % (name, [c.name for c in bpy.data.movieclips]))
        return clip
    return tools.active_clip(ctx)


def _value(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    try:
        return list(v)
    except TypeError:
        return str(v)


def settings():
    s = _context().scene.aimt
    out = {}
    for prop in s.bl_rna.properties:
        if prop.identifier in {"rna_type", "name"}:
            continue
        entry = {"value": _value(getattr(s, prop.identifier)), "description": prop.description}
        if prop.type == "ENUM":
            entry["options"] = [i.identifier for i in prop.enum_items]
        elif prop.type in {"INT", "FLOAT"}:
            entry["min"], entry["max"] = prop.hard_min, prop.hard_max
        out[prop.identifier] = entry
    return out


def _apply_settings(values):
    s = _context().scene.aimt
    valid = set(s.bl_rna.properties.keys())
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError("Ajustes desconocidos: %s. Usa settings() para ver los válidos." % unknown)
    for key, value in values.items():
        setattr(s, key, value)


def load_clip(path):
    path = os.path.abspath(bpy.path.abspath(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    clip = bpy.data.movieclips.load(path, check_existing=True)
    ctx = _context()
    ctx.scene.active_clip = clip
    for window in ctx.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "CLIP_EDITOR":
                area.spaces.active.clip = clip
    return {"clip": clip.name, "frames": clip.frame_duration, "size": list(clip.size),
            "fps": tools.clip_fps(ctx, clip)}


def run(tool, clip=None, **overrides):
    if tool not in tools.TOOLS:
        raise ValueError("Herramienta desconocida %r. Opciones: %s" % (tool, list(tools.TOOLS)))
    _apply_settings(overrides)
    ctx = _context()
    target = _clip(clip)
    label, worker, params, finish = tools.prepare(ctx, tool, target)
    job = jobs.run_in_background(label, worker, params,
                                 lambda c, result: finish(c, result, job))
    return {"started": tool, "clip": target.name, "frames": params["last"] - params["first"] + 1,
            "hint": "Llama a status() para ver el progreso"}


def cancel():
    job = jobs.current()
    if job is None or job.finished:
        return {"cancelled": False, "reason": "no hay tarea en curso"}
    job.cancelled = True
    return {"cancelled": True, "job": job.name}


def status():
    ctx = _context()
    job = jobs.current()
    clip = tools.active_clip(ctx)
    out = {
        "addon": ADDON_VERSION,
        "blender": bpy.app.version_string,
        "dependencies": {k: deps.available(k) for k in deps.GROUPS},
        "device": tools.get_prefs(ctx).device,
        "active_clip": clip.name if clip else None,
        "clips": [c.name for c in bpy.data.movieclips],
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
    if job.progress < 0.2:
        return {"seconds": 15, "reason": "cargando video/modelos"}
    return {"seconds": 10, "reason": job.message}


def solve_info(clip=None):
    target = _clip(clip)
    if target is None:
        return {"error": "no hay clip activo"}
    obj = target.tracking.objects.active
    rec = obj.reconstruction
    tracks = list(obj.tracks)
    return {
        "clip": target.name,
        "valid": rec.is_valid,
        "average_error_px": round(rec.average_error, 4) if rec.is_valid else None,
        "solved_frames": len(rec.cameras),
        "clip_frames": target.frame_duration,
        "focal_mm": round(target.tracking.camera.focal_length, 3),
        "sensor_mm": target.tracking.camera.sensor_width,
        "tracks": len(tracks),
        "ai_tracks": sum(t.name.startswith("AI_") for t in tracks),
        "bundles": sum(t.has_bundle for t in tracks),
    }


ADDON_VERSION = "1.1.0"
