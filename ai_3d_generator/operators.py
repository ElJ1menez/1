# SPDX-License-Identifier: GPL-3.0-or-later
"""UI operators: thin wrappers around tools.TOOLS run as modal jobs."""

import os

import bpy

from . import deps, jobs, tools


class _ToolOperator(jobs.JobOperator):
    tool = ""

    @classmethod
    def poll(cls, context):
        if jobs.busy():
            return False
        if tools.TOOLS[cls.tool].needs_object:
            return tools.target_object(context) is not None
        return True

    def execute(self, context):
        try:
            label, worker, params, finish = tools.prepare(context, self.tool,
                                                          tools.target_object(context))
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return self.launch(context, label, worker, params,
                           lambda ctx, result: finish(ctx, result, self._job))


class AI3D_OT_generate(_ToolOperator, bpy.types.Operator):
    bl_idname = "ai3d.generate"
    bl_label = "Generar modelo 3D"
    bl_description = ("Crea un modelo 3D desde la imagen o la descripción con IA open source "
                      "(TripoSR) y, si está activado, lo optimiza para producción")
    tool = "generate"


class AI3D_OT_optimize(_ToolOperator, bpy.types.Operator):
    bl_idname = "ai3d.optimize"
    bl_label = "Optimizar para producción"
    bl_description = ("Limpia la malla, ajusta tamaño y origen, reduce polígonos o hace quads, "
                      "despliega UVs, hornea texturas PBR y crea LODs")
    bl_options = {"UNDO"}
    tool = "optimize"


class AI3D_OT_export(_ToolOperator, bpy.types.Operator):
    bl_idname = "ai3d.export"
    bl_label = "Exportar"
    bl_description = "Exporta el objeto activo y sus LODs con una ficha de licencias"
    tool = "export"


class AI3D_OT_open_prefs(bpy.types.Operator):
    bl_idname = "ai3d.open_prefs"
    bl_label = "Instalar dependencias"
    bl_description = "Abre las preferencias del add-on para instalar las librerías de IA"

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = "ADDONS"
        bpy.ops.preferences.addon_show(module=__package__)
        return {"FINISHED"}


class AI3D_OT_open_folder(bpy.types.Operator):
    bl_idname = "ai3d.open_folder"
    bl_label = "Abrir carpeta"
    bl_description = "Abre la carpeta de exportación (o la de imágenes generadas)"

    outputs: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    def execute(self, context):
        folder = deps.paths["outputs"] if self.outputs else bpy.path.abspath(context.scene.ai3d.export_dir)
        if not os.path.isdir(folder):
            self.report({"WARNING"}, "La carpeta aún no existe: %s" % folder)
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=folder)
        return {"FINISHED"}


classes = (
    AI3D_OT_generate, AI3D_OT_optimize, AI3D_OT_export,
    AI3D_OT_open_prefs, AI3D_OT_open_folder, jobs.AI3D_OT_cancel,
)
