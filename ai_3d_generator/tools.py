# SPDX-License-Identifier: GPL-3.0-or-later
"""The add-on's tools, defined once and shared by the UI operators and the
scripting/MCP API: dependencies, parameters to capture, worker, and how the
result is applied to Blender."""

import os

import bpy

from . import builders, deps, export, optimize
from .core import pipeline
from .prefs import get_prefs

_GEN_KEYS = ("source", "prompt", "negative_prompt", "t2i_model", "t2i_steps", "asset_style",
             "seed", "remove_background", "foreground_ratio", "mc_resolution", "threshold",
             "min_part_ratio", "smooth_iterations", "size_axis", "name")


def target_object(context, name=None):
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError("No existe el objeto %r" % name)
        return obj
    obj = context.view_layer.objects.active
    return obj if obj is not None and obj.type == "MESH" else None


def _generate_needs(context):
    needs = ["torch", "gen3d"]
    if context.scene.ai3d.source == "TEXT":
        needs.append("text2image")
    return needs


def _generate_params(context, _obj):
    s = context.scene.ai3d
    prefs = get_prefs(context)
    p = {k: getattr(s, k) for k in _GEN_KEYS}
    p["image_path"] = os.path.normpath(bpy.path.abspath(s.image_path)) if s.image_path else ""
    if s.source == "IMAGE" and not os.path.isfile(p["image_path"]):
        raise RuntimeError("Elige una imagen existente (o cambia el origen a Texto)")
    if s.source == "TEXT" and not s.prompt.strip():
        raise RuntimeError("Escribe una descripción del objeto")
    p["real_size"] = s.real_size if s.apply_size else 1.0
    p.update(device=prefs.device, keep_models=prefs.keep_models, hf_token=prefs.hf_token.strip(),
             models_dir=deps.paths["models"], hf_cache=deps.paths["hf"],
             out_dir=deps.paths["outputs"], chunk_size=8192)
    return p


def _optimize(context, obj, job):
    device = get_prefs(context).bake_device
    return optimize.optimize_steps(context, obj, context.scene.ai3d, device, job)


def _generate_finish(context, _obj, r, job):
    obj = builders.build_generated(context, r)
    msg = "%s creado (%d triángulos, semilla %d)" % (
        obj.name, len(r["faces"]), r["provenance"]["seed"])
    if r["notes"]:
        msg += " · " + "; ".join(r["notes"])
    if not context.scene.ai3d.auto_optimize:
        return msg

    def then_optimize():
        summary = yield from _optimize(context, obj, job)
        return msg + " · " + summary
    return then_optimize()


def _optimize_finish(context, obj, _r, job):
    return _optimize(context, obj, job)


def _export_finish(context, obj, _r, _job):
    s = context.scene.ai3d
    folder = export.export_dir(s.export_dir)
    formats = [f for f in export.FORMATS if getattr(s, "export_" + f)]
    files = export.export_asset(context, obj, folder, formats, s.write_provenance)
    return "%d archivos exportados en %s" % (len(files), folder)


class Tool:
    def __init__(self, label, needs, worker, params, finish, description, needs_object):
        self.label = label
        self.needs = needs
        self.worker = worker
        self.params = params
        self.finish = finish
        self.description = description
        self.needs_object = needs_object


TOOLS = {
    "generate": Tool(
        "Generando modelo 3D", _generate_needs, pipeline.generate,
        _generate_params, _generate_finish,
        "Imagen o texto → malla 3D con TripoSR; si auto_optimize, la deja lista para producción",
        False),
    "optimize": Tool(
        "Optimizando modelo", lambda c: [], lambda job, p: None,
        lambda c, o: {}, _optimize_finish,
        "Limpia, retopologiza, despliega UVs, hornea texturas PBR y crea LODs del objeto activo",
        True),
    "export": Tool(
        "Exportando", lambda c: [], lambda job, p: None,
        lambda c, o: {}, _export_finish,
        "Exporta el objeto activo y sus LODs (GLB/FBX/OBJ/USDZ/STL) con ficha de licencias",
        True),
}


def prepare(context, key, obj=None):
    """Validate and capture everything a tool needs. Raises RuntimeError.

    Returns (label, worker, params, finish(context, result, job)).
    """
    tool = TOOLS[key]
    missing = deps.missing_for(*tool.needs(context))
    if missing:
        raise RuntimeError("Faltan dependencias: %s (Preferencias > Add-ons > AI 3D Generator)"
                           % ", ".join(missing))
    if tool.needs_object and (obj is None or obj.type != "MESH"):
        raise RuntimeError("Selecciona un objeto de malla")
    params = tool.params(context, obj)
    obj_name = obj.name if obj is not None else None

    def finish(ctx, result, job):
        live = None
        if obj_name is not None:
            live = bpy.data.objects.get(obj_name)
            if live is None:
                raise RuntimeError("El objeto ya no existe")
        return tool.finish(ctx, live, result, job)
    return tool.label, tool.worker, params, finish
