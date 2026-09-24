# SPDX-License-Identifier: GPL-3.0-or-later
"""UI operators: thin wrappers around tools.TOOLS run as modal jobs."""

import bpy

from . import jobs, tools


class _ToolOperator(jobs.JobOperator):
    tool = ""

    @classmethod
    def poll(cls, context):
        return tools.active_clip(context) is not None and not jobs.busy()

    def execute(self, context):
        try:
            label, worker, params, finish = tools.prepare(context, self.tool, tools.active_clip(context))
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return self.launch(context, label, worker, params,
                           lambda ctx, result: finish(ctx, result, self._job))


class AIMT_OT_camera_track(_ToolOperator, bpy.types.Operator):
    bl_idname = "aimt.camera_track"
    bl_label = "Tracking de cámara con IA"
    bl_description = ("Siembra puntos, los sigue con BootsTAPIR, descarta objetos móviles y "
                      "tracks inconsistentes y resuelve la cámara")
    tool = "camera_track"


class AIMT_OT_solve(_ToolOperator, bpy.types.Operator):
    bl_idname = "aimt.solve"
    bl_label = "Resolver cámara"
    bl_description = "Resuelve la cámara con los tracks actuales (con búsqueda de focal opcional)"
    tool = "solve"


class AIMT_OT_track_selected(_ToolOperator, bpy.types.Operator):
    bl_idname = "aimt.track_selected"
    bl_label = "Seguir marcadores seleccionados con IA"
    bl_description = ("Sigue los marcadores seleccionados hacia delante y hacia atrás desde el "
                      "frame actual con BootsTAPIR (robusto a oclusiones y desenfoque)")
    tool = "track_selected"


class AIMT_OT_pose(_ToolOperator, bpy.types.Operator):
    bl_idname = "aimt.pose_capture"
    bl_label = "Captura de cuerpo con IA"
    bl_description = "Detecta 33 puntos del cuerpo por frame y crea tracks, empties y esqueleto"
    tool = "pose"


class AIMT_OT_face(_ToolOperator, bpy.types.Operator):
    bl_idname = "aimt.face_capture"
    bl_label = "Captura facial con IA"
    bl_description = "Movimiento de cabeza, 52 expresiones (blendshapes) y tracks faciales 2D"
    tool = "face"


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
