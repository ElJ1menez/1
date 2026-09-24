# SPDX-License-Identifier: GPL-3.0-or-later
"""AI 3D Generator: text or image to production-ready 3D models, using AI
models that are open source and licensed for commercial use, with Blender
doing the clean-up, retopology, UVs, PBR baking and export."""

bl_info = {
    "name": "AI 3D Generator",
    "author": "ElJ1menez",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > IA 3D",
    "description": "Genera modelos 3D con IA open source (TripoSR, SDXL/FLUX) y los deja listos para uso comercial",
    "category": "3D View",
}

# Submodules are imported lazily so core/ stays importable without Blender.
_modules = None


def _load():
    from . import api, deps, jobs, operators, prefs, props, ui
    return api, deps, jobs, operators, prefs, props, ui


def register():
    import sys
    import bpy
    global _modules
    _modules = _load()
    api, deps, _jobs, operators, prefs, props, ui = _modules
    deps.init_paths()
    for cls in props.classes + prefs.classes + operators.classes + ui.classes:
        bpy.utils.register_class(cls)
    props.register()
    # Stable import name for scripts and MCP servers, whatever the extension
    # repository ("bl_ext.<repo>.ai_3d_generator") the add-on lives in.
    sys.modules[api.ALIAS] = api


def unregister():
    import sys
    import bpy
    api, _deps, jobs, operators, prefs, props, ui = _modules
    sys.modules.pop(api.ALIAS, None)
    job = jobs.current()
    if job is not None:
        job.cancelled = True
    props.unregister()
    for cls in reversed(props.classes + prefs.classes + operators.classes + ui.classes):
        bpy.utils.unregister_class(cls)
