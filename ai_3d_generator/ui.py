# SPDX-License-Identifier: GPL-3.0-or-later
import bpy

from . import deps, jobs, optimize, tools
from .prefs import draw_progress


class _Panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "IA 3D"


def _missing(layout, groups):
    missing = deps.missing_for(*groups)
    if missing:
        box = layout.box()
        box.label(text="Falta: " + ", ".join(missing), icon="ERROR")
        box.operator("ai3d.open_prefs", icon="PREFERENCES")
    return bool(missing)


class AI3D_PT_main(_Panel, bpy.types.Panel):
    bl_label = "AI 3D Generator"

    def draw(self, context):
        layout = self.layout
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


class AI3D_PT_generate(_Panel, bpy.types.Panel):
    bl_label = "1 · Generar"
    bl_parent_id = "AI3D_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai3d
        layout.row().prop(s, "source", expand=True)
        if _missing(layout, tools._generate_needs(context)):
            return
        if s.source == "IMAGE":
            layout.prop(s, "image_path", text="")
            layout.prop(s, "remove_background")
        else:
            layout.prop(s, "prompt", text="")
            layout.prop(s, "t2i_model", text="")
            layout.prop(s, "asset_style")
        layout.prop(s, "name")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("ai3d.generate", icon="SHADERFX")
        layout.prop(s, "auto_optimize")


class AI3D_PT_generate_advanced(_Panel, bpy.types.Panel):
    bl_label = "Avanzado"
    bl_parent_id = "AI3D_PT_generate"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        s = context.scene.ai3d
        if s.source == "TEXT":
            layout.prop(s, "negative_prompt")
            layout.prop(s, "t2i_steps")
        layout.prop(s, "seed")
        layout.prop(s, "foreground_ratio")
        layout.prop(s, "mc_resolution")
        layout.prop(s, "threshold")
        layout.prop(s, "min_part_ratio")
        layout.prop(s, "smooth_iterations")
        layout.operator("ai3d.open_folder", text="Ver imágenes generadas", icon="FILE_FOLDER").outputs = True


class AI3D_PT_optimize(_Panel, bpy.types.Panel):
    bl_label = "2 · Optimizar"
    bl_parent_id = "AI3D_PT_main"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        s = context.scene.ai3d
        col = layout.column(heading="Tamaño real")
        col.prop(s, "apply_size", text="")
        sub = col.column()
        sub.active = s.apply_size
        sub.prop(s, "real_size")
        sub.prop(s, "size_axis")
        layout.prop(s, "remesh_mode")
        if s.remesh_mode != "NONE":
            layout.prop(s, "target_faces")
        if s.remesh_mode == "QUADS":
            layout.prop(s, "voxel_detail")
        layout.prop(s, "fill_holes")
        layout.prop(s, "smooth_angle")
        col = layout.column(heading="Texturas")
        col.prop(s, "bake_textures", text="UVs + PBR")
        sub = col.column()
        sub.active = s.bake_textures
        sub.prop(s, "texture_size")
        sub.prop(s, "bake_normal")
        sub.prop(s, "roughness")
        sub.prop(s, "metallic")
        layout.prop(s, "lods")
        layout.prop(s, "keep_source")
        layout.use_property_split = False
        obj = tools.target_object(context)
        row = layout.row()
        row.scale_y = 1.3
        row.operator("ai3d.optimize", icon="MOD_REMESH",
                     text="Optimizar '%s'" % obj.name if obj else "Optimizar (selecciona una malla)")


class AI3D_PT_export(_Panel, bpy.types.Panel):
    bl_label = "3 · Exportar"
    bl_parent_id = "AI3D_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai3d
        layout.prop(s, "export_dir", text="")
        row = layout.row(align=True)
        for fmt in ("glb", "fbx", "obj", "usdz", "stl"):
            row.prop(s, "export_" + fmt, toggle=True)
        layout.prop(s, "write_provenance")
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("ai3d.export", icon="EXPORT")
        row.operator("ai3d.open_folder", text="", icon="FILE_FOLDER").outputs = False


class AI3D_PT_report(_Panel, bpy.types.Panel):
    bl_label = "Control de calidad"
    bl_parent_id = "AI3D_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return tools.target_object(context) is not None

    def draw(self, context):
        obj = tools.target_object(context)
        if obj.mode != "OBJECT":
            self.layout.label(text="Sal del modo edición para ver el informe")
            return
        r = optimize.mesh_report(obj)
        col = self.layout.column(align=True)

        def check(ok, text):
            col.label(text=text, icon="CHECKMARK" if ok else "ERROR")
        col.label(text="%s · %d triángulos" % (r["object"], r["triangles"]), icon="MESH_DATA")
        col.label(text="%.3f × %.3f × %.3f m" % tuple(r["dimensions_m"]), icon="EMPTY_ARROWS")
        if r["watertight"] is None:
            col.label(text="Malla muy densa: optimízala para comprobarla", icon="INFO")
        else:
            check(r["watertight"], "Malla cerrada" if r["watertight"] else
                  "%d aristas abiertas / no-manifold" % (r["open_edges"] + r["non_manifold_edges"]))
        check(bool(r["uv_maps"]), "UVs" if r["uv_maps"] else "Sin UVs")
        check(bool(r["textures"]), "%d texturas" % len(r["textures"]) if r["textures"] else "Sin texturas")
        check(r["provenance"], "Ficha de origen/licencias" if r["provenance"] else "Sin ficha de origen")


classes = (AI3D_PT_main, AI3D_PT_generate, AI3D_PT_generate_advanced, AI3D_PT_optimize,
           AI3D_PT_export, AI3D_PT_report)
