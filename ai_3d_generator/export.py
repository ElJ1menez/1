# SPDX-License-Identifier: GPL-3.0-or-later
"""Export an asset (and its LODs) to the formats stores and engines expect,
with a provenance/licence sheet next to the files."""

import json
import os
import time

import bpy
from mathutils import Matrix

from . import builders, optimize

FORMATS = ("glb", "fbx", "obj", "usdz", "stl")

LICENSE_NOTES = {
    "MIT": "Permite uso comercial. Conserva el aviso de copyright si redistribuyes el modelo de IA.",
    "Apache-2.0": "Permite uso comercial. Conserva el aviso de licencia si redistribuyes el modelo de IA.",
    "CreativeML Open RAIL++-M": (
        "Permite uso comercial de las imágenes generadas, con restricciones de uso "
        "(Anexo A: nada ilegal, dañino, desinformación, etc.). Esas restricciones se "
        "transmiten a quien redistribuya el modelo, no a tus assets."),
}


def _call(op, **kwargs):
    """Call an exporter with only the options this Blender version knows
    (exporter options get renamed between releases)."""
    known = set(op.get_rna_type().properties.keys())
    return op(**{k: v for k, v in kwargs.items() if k in known})


def _export_one(fmt, path, has_colors):
    if fmt == "glb":
        _call(bpy.ops.export_scene.gltf, filepath=path, export_format="GLB", use_selection=True,
              export_apply=True, export_yup=True)
    elif fmt == "fbx":
        _call(bpy.ops.export_scene.fbx, filepath=path, use_selection=True, path_mode="COPY",
              embed_textures=True, mesh_smooth_type="FACE", add_leaf_bones=False,
              use_mesh_modifiers=True)
    elif fmt == "obj":
        _call(bpy.ops.wm.obj_export, filepath=path, export_selected_objects=True,
              path_mode="COPY", export_materials=True, export_colors=has_colors,
              apply_modifiers=True)
    elif fmt == "usdz":
        _call(bpy.ops.wm.usd_export, filepath=path, selected_objects_only=True,
              export_materials=True, generate_preview_surface=True, export_textures=True)
    elif fmt == "stl":
        if hasattr(bpy.ops.wm, "stl_export"):
            _call(bpy.ops.wm.stl_export, filepath=path, export_selected_objects=True,
                  apply_modifiers=True)
        else:
            _call(bpy.ops.export_mesh.stl, filepath=path, use_selection=True)


def export_dir(path):
    folder = bpy.path.abspath(path or "//ai3d_export/")
    if folder.startswith("//") or not os.path.isabs(folder):
        raise RuntimeError("Guarda el .blend o elige una carpeta absoluta para exportar")
    os.makedirs(folder, exist_ok=True)
    return folder


def export_asset(context, obj, folder, formats, provenance=True):
    """Export obj (at the world origin) and its LODs. Returns written paths."""
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Selecciona el objeto de malla a exportar")
    formats = [f for f in formats if f in FORMATS]
    if not formats:
        raise RuntimeError("Elige al menos un formato de exportación")
    if context.object is not None and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    base = bpy.path.clean_name(obj.name)
    lods = sorted((c for c in obj.children if c.type == "MESH" and "_LOD" in c.name),
                  key=lambda c: c.name)
    has_colors = bool(obj.data.color_attributes)
    written = []
    saved = obj.matrix_world.copy()
    hidden = {o: o.hide_get() for o in [obj] + lods}
    try:
        obj.matrix_world = Matrix.Translation(-saved.to_translation()) @ saved
        for variant, name in [(obj, base)] + [(lod, bpy.path.clean_name(lod.name)) for lod in lods]:
            builders.select_only(context, [variant])
            for fmt in formats:
                path = os.path.join(folder, "%s.%s" % (name, fmt))
                _export_one(fmt, path, has_colors)
                written.append(path)
    finally:
        obj.matrix_world = saved
        for o, h in hidden.items():
            o.hide_set(h)
        builders.select_only(context, [obj])
    if provenance:
        written.append(write_provenance(obj, folder, base, written))
    return written


def provenance_sheet(obj, files=()):
    prov = builders.get_provenance(obj)
    licenses = sorted({s["license"] for s in prov.get("steps", []) if s.get("license")})
    return {
        "asset": obj.name,
        "files": [os.path.basename(f) for f in files],
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "AI 3D Generator %s para Blender %s" % (ADDON_VERSION, bpy.app.version_string),
        "report": optimize.mesh_report(obj),
        "generation": prov,
        "licenses": {name: LICENSE_NOTES.get(name, "") for name in licenses},
        "commercial_use": (
            "Todos los modelos de IA usados permiten uso comercial de sus resultados. "
            "Si partiste de una imagen propia, necesitas derechos sobre esa imagen. "
            "Revisa que el objeto no reproduzca marcas, logotipos o diseños protegidos."),
    }


def write_provenance(obj, folder, base, files):
    path = os.path.join(folder, base + "_licencias.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(provenance_sheet(obj, files), f, ensure_ascii=False, indent=2)
    return path


ADDON_VERSION = "1.0.0"
