# SPDX-License-Identifier: GPL-3.0-or-later
import sys

import bpy
from bpy.props import EnumProperty

from . import deps, jobs


class AIMT_OT_install_deps(jobs.JobOperator, bpy.types.Operator):
    bl_idname = "aimt.install_deps"
    bl_label = "Instalar"
    bl_description = "Descarga e instala las librerías de IA con pip (requiere internet)"

    group: EnumProperty(items=[(k, v["label"], "") for k, v in deps.GROUPS.items()])

    def execute(self, context):
        import numpy
        prefs = get_prefs(context)
        params = {
            "group": self.group, "variant": prefs.torch_variant,
            "site": deps.paths["site"], "python": sys.executable,
            "numpy": numpy.__version__,
        }
        return self.launch(context, "Instalando " + deps.GROUPS[self.group]["label"],
                           deps.install_worker, params)

    def finish(self, context, result):
        if deps.available(result["group"]):
            return "Instalado. Si ya habías usado la IA en esta sesión, reinicia Blender."
        return "pip terminó, pero el módulo no se puede importar: reinicia Blender"


class AIMT_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    device: EnumProperty(
        name="Dispositivo",
        items=[("AUTO", "Automático", "GPU si está disponible"),
               ("CUDA", "NVIDIA CUDA", ""),
               ("MPS", "Apple Metal (MPS)", ""),
               ("CPU", "CPU", "Lento, pero funciona en cualquier equipo")],
        default="AUTO",
    )
    torch_variant: EnumProperty(
        name="Variante de PyTorch",
        items=[("DEFAULT", "Por defecto (PyPI)", "Linux: CUDA · macOS: Metal · Windows: CPU"),
               ("CU128", "CUDA 12.8", "NVIDIA RTX 20xx en adelante, incluidas RTX 50xx"),
               ("CU126", "CUDA 12.6", "NVIDIA con drivers algo más antiguos"),
               ("CPU", "Solo CPU", "Descarga más pequeña")],
        default="DEFAULT",
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "device")
        col.prop(self, "torch_variant")

        box = layout.box()
        box.label(text="Dependencias (todas open source con uso comercial permitido)")
        for key, g in deps.GROUPS.items():
            row = box.row()
            ok = deps.available(key)
            row.label(text="%s  [%s]" % (g["label"], g["license"]),
                      icon="CHECKMARK" if ok else "ERROR")
            sub = row.row()
            sub.enabled = not jobs.busy()
            op = sub.operator("aimt.install_deps", text="Reinstalar" if ok else "Instalar",
                              icon="IMPORT")
            op.group = key
        box.label(text="Carpeta: " + deps.paths["site"])

        job = jobs.current()
        if job is not None and not job.finished:
            draw_progress(layout, job)
        if jobs.last_log:
            logbox = layout.box()
            logbox.label(text="Último log:")
            for line in jobs.last_log[-8:]:
                logbox.label(text=line[:140])


def draw_progress(layout, job):
    row = layout.row(align=True)
    if hasattr(row, "progress"):  # Blender 4.0+
        row.progress(factor=job.progress, text=job.message)
    else:
        row.label(text="%d%% %s" % (job.progress * 100, job.message))
    row.operator("aimt.cancel", text="", icon="CANCEL")


def get_prefs(context):
    return context.preferences.addons[__package__].preferences


classes = (AIMT_OT_install_deps, AIMT_Preferences)
