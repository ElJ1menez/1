# SPDX-License-Identifier: GPL-3.0-or-later
import os

import bpy
import numpy as np

from . import builders, deps, jobs, solve
from .core import pipelines
from .core import tracks as tk
from .prefs import get_prefs


def active_clip(context):
    space = context.space_data
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
    fps = clip.fps if getattr(clip, "fps", 0) > 0 else scene.render.fps / scene.render.fps_base
    path = os.path.normpath(bpy.path.abspath(clip.filepath, library=clip.library))
    return {
        "info": {"path": path, "source": clip.source, "frame_offset": clip.frame_offset},
        "first": first, "last": last, "width": width, "height": height, "fps": fps,
        "device": get_prefs(context).device, "models_dir": deps.paths["models"],
    }


class _ClipJob(jobs.JobOperator):
    needs = ("vision", "torch")

    @classmethod
    def poll(cls, context):
        return active_clip(context) is not None and not jobs.busy()

    def start(self, context, name, worker, extra=None):
        missing = deps.missing_for(*self.needs)
        if missing:
            self.report({"ERROR"}, "Faltan dependencias: %s (Preferencias > Add-ons > AI Motion Tracker)"
                        % ", ".join(missing))
            return {"CANCELLED"}
        clip = active_clip(context)
        try:
            params = clip_params(context, clip)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        params.update(extra or {})
        self._clip_name = clip.name
        return self.launch(context, name, worker, params)

    def clip(self):
        clip = bpy.data.movieclips.get(self._clip_name)
        if clip is None:
            raise RuntimeError("El clip ya no existe")
        return clip


class AIMT_OT_camera_track(_ClipJob, bpy.types.Operator):
    bl_idname = "aimt.camera_track"
    bl_label = "Tracking de cámara con IA"
    bl_description = ("Siembra puntos, los sigue con BootsTAPIR, descarta objetos móviles y "
                      "tracks inconsistentes y resuelve la cámara")

    def execute(self, context):
        s = context.scene.aimt
        extra = {k: getattr(s, k) for k in (
            "model_res", "max_side", "points_per_seed", "window", "interval", "min_length", "per_frame",
            "max_tracks", "dynamic_mask", "geometric_filter", "geo_threshold", "min_inlier_ratio")}
        return self.start(context, "Tracking de cámara IA", pipelines.camera_track, extra)

    def finish(self, context, r):
        clip = self.clip()
        s = context.scene.aimt
        if s.replace_tracks:
            builders.delete_tracks(context, clip, lambda t: t.name.startswith("AI_"))
        n = builders.write_ai_tracks(clip, r["tracks"], r["width"], r["height"], r["first"])
        msg = "%d tracks IA (de %d candidatos, mín. %d/frame)" % (n, r["raw"], r["min_coverage"])
        if r["notes"]:
            msg += " · " + "; ".join(r["notes"])
        if not s.solve or n == 0:
            return msg
        return self._then_solve(context, clip, s, msg)

    def _then_solve(self, context, clip, s, msg):
        summary = yield from solve.solve_steps(context, clip, s, self._job)
        return msg + " · " + summary


class AIMT_OT_solve(_ClipJob, bpy.types.Operator):
    bl_idname = "aimt.solve"
    bl_label = "Resolver cámara"
    bl_description = "Resuelve la cámara con los tracks actuales (con búsqueda de focal opcional)"
    needs = ()

    def execute(self, context):
        return self.start(context, "Resolviendo cámara", lambda job, p: None)

    def finish(self, context, _result):
        return solve.solve_steps(context, self.clip(), context.scene.aimt, self._job)


class AIMT_OT_track_selected(_ClipJob, bpy.types.Operator):
    bl_idname = "aimt.track_selected"
    bl_label = "Seguir marcadores seleccionados con IA"
    bl_description = ("Sigue los marcadores seleccionados hacia delante y hacia atrás desde el "
                      "frame actual con BootsTAPIR (robusto a oclusiones y desenfoque)")

    def execute(self, context):
        clip = active_clip(context)
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
            self.report({"ERROR"}, "Selecciona marcadores que tengan posición en el frame actual")
            return {"CANCELLED"}
        s = context.scene.aimt
        extra = {"names": names, "query_co": co, "query_frame": frame,
                 "max_side": s.max_side, "window": s.window, "model_res": s.model_res}
        return self.start(context, "Siguiendo %d marcadores" % len(names), pipelines.track_selected, extra)

    def finish(self, context, r):
        clip = self.clip()
        tracks = clip.tracking.objects.active.tracks
        done = 0
        for n, name in enumerate(r["names"]):
            track = tracks.get(name)
            if track is None:
                continue
            co = tk.to_blender(r["xy"][n], r["width"], r["height"])
            builders.write_markers(track, r["first"], co, r["vis"][n])
            done += 1
        frames = int(np.sum(r["vis"]))
        return "%d marcadores seguidos (%d posiciones)" % (done, frames)


class AIMT_OT_pose(_ClipJob, bpy.types.Operator):
    bl_idname = "aimt.pose_capture"
    bl_label = "Captura de cuerpo con IA"
    bl_description = "Detecta 33 puntos del cuerpo por frame y crea tracks, empties y esqueleto"
    needs = ("vision", "mediapipe")

    def execute(self, context):
        s = context.scene.aimt
        extra = {"pose_model": s.pose_model, "min_confidence": s.min_confidence}
        return self.start(context, "Captura de cuerpo IA", pipelines.pose, extra)

    def finish(self, context, r):
        clip = self.clip()
        fps = clip.fps if getattr(clip, "fps", 0) > 0 else context.scene.render.fps
        return builders.build_pose(context, clip, r, context.scene.aimt, fps)


class AIMT_OT_face(_ClipJob, bpy.types.Operator):
    bl_idname = "aimt.face_capture"
    bl_label = "Captura facial con IA"
    bl_description = "Movimiento de cabeza, 52 expresiones (blendshapes) y tracks faciales 2D"
    needs = ("vision", "mediapipe")

    def execute(self, context):
        s = context.scene.aimt
        return self.start(context, "Captura facial IA", pipelines.face,
                          {"min_confidence": s.min_confidence})

    def finish(self, context, r):
        clip = self.clip()
        fps = clip.fps if getattr(clip, "fps", 0) > 0 else context.scene.render.fps
        return builders.build_face(context, clip, r, context.scene.aimt, fps)


class AIMT_OT_open_prefs(bpy.types.Operator):
    bl_idname = "aimt.open_prefs"
    bl_label = "Instalar dependencias"
    bl_description = "Abre las preferencias del add-on para instalar las librerías de IA"

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = "ADDONS"
        bpy.ops.preferences.addon_show(module=__package__)
        return {"FINISHED"}


classes = (
    AIMT_OT_camera_track, AIMT_OT_solve, AIMT_OT_track_selected,
    AIMT_OT_pose, AIMT_OT_face, AIMT_OT_open_prefs, jobs.AIMT_OT_cancel,
)
