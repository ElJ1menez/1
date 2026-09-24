# SPDX-License-Identifier: GPL-3.0-or-later
import bpy

from . import deps, jobs
from .prefs import draw_progress


class _Panel:
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "IA Tracking"

    @classmethod
    def poll(cls, context):
        return context.space_data.clip is not None


def _missing(layout, *groups):
    missing = deps.missing_for(*groups)
    if missing:
        box = layout.box()
        box.label(text="Falta: " + ", ".join(missing), icon="ERROR")
        box.operator("aimt.open_prefs", icon="PREFERENCES")
    return bool(missing)


class AIMT_PT_main(_Panel, bpy.types.Panel):
    bl_label = "AI Motion Tracker"

    def draw(self, context):
        layout = self.layout
        s = context.scene.aimt
        job = jobs.current()
        if job is not None and not job.finished:
            layout.label(text=job.name, icon="SORTTIME")
            draw_progress(layout, job)
            layout.label(text="ESC o ✕ para cancelar")
        elif jobs.last_report["text"]:
            icon = {"ERROR": "ERROR", "WARNING": "INFO"}.get(jobs.last_report["level"], "CHECKMARK")
            col = layout.column(align=True)
            col.scale_y = 0.8
            for i, part in enumerate(jobs.last_report["text"].split(" · ")):
                col.label(text=part, icon=icon if i == 0 else "BLANK1")
        layout.prop(s, "use_scene_range")


class AIMT_PT_camera(_Panel, bpy.types.Panel):
    bl_label = "Cámara"
    bl_parent_id = "AIMT_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.aimt
        if _missing(layout, "vision", "torch"):
            return
        layout.prop(s, "model_res")
        col = layout.column(align=True)
        col.prop(s, "max_side")
        col.prop(s, "points_per_seed")
        col.prop(s, "window")
        col.prop(s, "interval")
        col = layout.column(align=True)
        col.prop(s, "per_frame")
        col.prop(s, "min_length")
        col.prop(s, "max_tracks")
        col = layout.column(heading="Filtros")
        row = col.row()
        row.enabled = deps.available("mediapipe")
        row.prop(s, "dynamic_mask")
        col.prop(s, "geometric_filter")
        if s.geometric_filter:
            sub = col.column(align=True)
            sub.prop(s, "geo_threshold")
            sub.prop(s, "min_inlier_ratio")
        layout.prop(s, "replace_tracks")
        layout.prop(s, "solve")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("aimt.camera_track", icon="CAMERA_DATA")
        layout.operator("aimt.track_selected", icon="TRACKER")


class AIMT_PT_solve(_Panel, bpy.types.Panel):
    bl_label = "Solve"
    bl_parent_id = "AIMT_PT_camera"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.aimt
        cam = context.space_data.clip.tracking.camera
        col = layout.column(align=True)
        col.prop(cam, "sensor_width")
        row = col.row()
        row.enabled = not s.auto_focal
        row.prop(cam, "focal_length")
        layout.prop(s, "auto_focal")
        if s.auto_focal:
            row = layout.row(align=True)
            row.prop(s, "focal_min", text="Mín")
            row.prop(s, "focal_max", text="Máx")
        col = layout.column(heading="Refinar")
        col.prop(s, "refine_focal")
        col.prop(s, "refine_distortion")
        layout.prop(s, "clean_error")
        layout.prop(s, "setup_scene")
        layout.operator("aimt.solve", icon="CON_CAMERASOLVER")


class AIMT_PT_body(_Panel, bpy.types.Panel):
    bl_label = "Cuerpo y cara"
    bl_parent_id = "AIMT_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.aimt
        if _missing(layout, "vision", "mediapipe"):
            return
        layout.prop(s, "min_confidence")
        layout.prop(s, "make_tracks_2d")
        layout.prop(s, "smoothing")
        if s.smoothing:
            row = layout.row(align=True)
            row.prop(s, "smooth_cutoff")
            row.prop(s, "smooth_beta")

        box = layout.box()
        box.label(text="Cuerpo", icon="ARMATURE_DATA")
        box.prop(s, "pose_model")
        box.prop(s, "make_empties")
        box.prop(s, "make_armature")
        box.operator("aimt.pose_capture", icon="POSE_HLT")

        box = layout.box()
        box.label(text="Cara", icon="USER")
        box.prop(s, "face_head")
        box.prop(s, "face_blendshapes")
        box.operator("aimt.face_capture", icon="SHAPEKEY_DATA")


classes = (AIMT_PT_main, AIMT_PT_camera, AIMT_PT_solve, AIMT_PT_body)
