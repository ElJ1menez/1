# SPDX-License-Identifier: GPL-3.0-or-later
"""The add-on's tools, defined once and shared by the UI operators and the
scripting/MCP API: dependencies, parameters to capture, worker, and how the
result is applied to Blender."""

import os

import bpy
import numpy as np

from . import builders, deps, solve
from .core import pipelines
from .core import tracks as tk
from .prefs import get_prefs


def active_clip(context):
    space = getattr(context, "space_data", None)
    clip = getattr(space, "clip", None) if space and space.type == "CLIP_EDITOR" else None
    return clip or context.scene.active_clip


def clip_params(context, clip):
    """Everything the worker thread needs, captured on the main thread."""
    scene = context.scene
    s = scene.aimt
    width, height = clip.size
    if width == 0 or height == 0:
        raise RuntimeError("El clip no se pudo cargar (¿ruta o códec?)")
    first, last = 1, clip.frame_duration
    if s.use_scene_range:
        first = max(first, scene.frame_start - clip.frame_start + 1)
        last = min(last, scene.frame_end - clip.frame_start + 1)
    if last - first < 1:
        raise RuntimeError("El rango de frames a procesar está vacío")
    path = os.path.normpath(bpy.path.abspath(clip.filepath, library=clip.library))
    return {
        "info": {"path": path, "source": clip.source, "frame_offset": clip.frame_offset},
        "first": first, "last": last, "width": width, "height": height, "fps": clip_fps(context, clip),
        "device": get_prefs(context).device, "models_dir": deps.paths["models"],
    }


def clip_fps(context, clip):
    if getattr(clip, "fps", 0) > 0:
        return clip.fps
    return context.scene.render.fps / context.scene.render.fps_base


# ---------------------------------------------------------------------------
# Per-tool parameter capture (main thread) and result application
# ---------------------------------------------------------------------------

_CAMERA_KEYS = ("model_res", "max_side", "points_per_seed", "window", "interval", "min_length",
                "per_frame", "max_tracks", "dynamic_mask", "geometric_filter", "geo_threshold",
                "min_inlier_ratio")


def _camera_params(context, clip):
    s = context.scene.aimt
    return {k: getattr(s, k) for k in _CAMERA_KEYS}


def _camera_finish(context, clip, r, job):
    s = context.scene.aimt
    if s.replace_tracks:
        builders.delete_tracks(context, clip, lambda t: t.name.startswith("AI_"))
    n = builders.write_ai_tracks(clip, r["tracks"], r["width"], r["height"], r["first"])
    msg = "%d tracks IA (de %d candidatos, mín. %d/frame)" % (n, r["raw"], r["min_coverage"])
    if r["notes"]:
        msg += " · " + "; ".join(r["notes"])
    if not s.solve or n == 0:
        return msg

    def then_solve():
        summary = yield from solve.solve_steps(context, clip, s, job)
        return msg + " · " + summary
    return then_solve()


def _selected_params(context, clip):
    frame = context.scene.frame_current - clip.frame_start + 1
    names, co = [], []
    for t in clip.tracking.objects.active.tracks:
        if not t.select or t.hide:
            continue
        m = t.markers.find_frame(frame, exact=True)
        if m is None or m.mute:
            continue
        names.append(t.name)
        co.append(tuple(m.co))
    if not names:
        raise RuntimeError("Selecciona marcadores que tengan posición en el frame actual")
    s = context.scene.aimt
    return {"names": names, "query_co": co, "query_frame": frame,
            "max_side": s.max_side, "window": s.window, "model_res": s.model_res}


def _selected_finish(context, clip, r, job):
    tracks = clip.tracking.objects.active.tracks
    done = 0
    for n, name in enumerate(r["names"]):
        track = tracks.get(name)
        if track is None:
            continue
        co = tk.to_blender(r["xy"][n], r["width"], r["height"])
        builders.write_markers(track, r["first"], co, r["vis"][n])
        done += 1
    return "%d marcadores seguidos (%d posiciones)" % (done, int(np.sum(r["vis"])))


def _pose_params(context, clip):
    s = context.scene.aimt
    return {"pose_model": s.pose_model, "min_confidence": s.min_confidence}


def _pose_finish(context, clip, r, job):
    return builders.build_pose(context, clip, r, context.scene.aimt, clip_fps(context, clip))


def _face_params(context, clip):
    return {"min_confidence": context.scene.aimt.min_confidence}


def _face_finish(context, clip, r, job):
    return builders.build_face(context, clip, r, context.scene.aimt, clip_fps(context, clip))


def _solve_finish(context, clip, r, job):
    return solve.solve_steps(context, clip, context.scene.aimt, job)


class Tool:
    def __init__(self, label, needs, worker, params, finish, description):
        self.label = label
        self.needs = needs
        self.worker = worker
        self.params = params
        self.finish = finish
        self.description = description


TOOLS = {
    "camera_track": Tool(
        "Tracking de cámara IA", ("vision", "torch"), pipelines.camera_track,
        _camera_params, _camera_finish,
        "Siembra y sigue puntos con BootsTAPIR, filtra y (si solve=True) resuelve la cámara"),
    "track_selected": Tool(
        "Siguiendo marcadores", ("vision", "torch"), pipelines.track_selected,
        _selected_params, _selected_finish,
        "Sigue los marcadores seleccionados desde el frame actual, hacia delante y atrás"),
    "solve": Tool(
        "Resolviendo cámara", (), lambda job, p: None,
        lambda context, clip: {}, _solve_finish,
        "Resuelve la cámara con los tracks existentes (auto_focal, clean_error, setup_scene)"),
    "pose": Tool(
        "Captura de cuerpo IA", ("vision", "mediapipe"), pipelines.pose,
        _pose_params, _pose_finish,
        "33 puntos del cuerpo: tracks 2D, empties 3D y esqueleto"),
    "face": Tool(
        "Captura facial IA", ("vision", "mediapipe"), pipelines.face,
        _face_params, _face_finish,
        "Movimiento de cabeza, 52 blendshapes y tracks faciales 2D"),
}


def prepare(context, key, clip):
    """Validate and capture everything a tool needs. Raises RuntimeError.

    Returns (label, worker, params, finish(context, result, job)).
    """
    tool = TOOLS[key]
    missing = deps.missing_for(*tool.needs)
    if missing:
        raise RuntimeError("Faltan dependencias: %s (Preferencias > Add-ons > AI Motion Tracker)"
                           % ", ".join(missing))
    if clip is None:
        raise RuntimeError("No hay clip: abre uno en el Clip Editor o usa load_clip()")
    params = clip_params(context, clip)
    params.update(tool.params(context, clip))
    clip_name = clip.name

    def finish(ctx, result, job):
        live = bpy.data.movieclips.get(clip_name)
        if live is None:
            raise RuntimeError("El clip ya no existe")
        return tool.finish(ctx, live, result, job)
    return tool.label, tool.worker, params, finish
