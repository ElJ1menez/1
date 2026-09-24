# SPDX-License-Identifier: GPL-3.0-or-later
"""AI Motion Tracker: camera, point, body and face tracking driven by AI
models that are open source and licensed for commercial use."""

bl_info = {
    "name": "AI Motion Tracker",
    "author": "ElJ1menez",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "Clip Editor > Sidebar (N) > IA Tracking",
    "description": "Motion tracking con IA: cámara (BootsTAPIR), cuerpo y cara (MediaPipe)",
    "category": "Video Tools",
}

# Submodules are imported lazily so core/ stays importable without Blender.
_modules = None


def _load():
    from . import deps, jobs, operators, prefs, props, ui
    return deps, jobs, operators, prefs, props, ui


def register():
    import bpy
    global _modules
    _modules = _load()
    deps, _jobs, operators, prefs, props, ui = _modules
    deps.init_paths()
    for cls in props.classes + prefs.classes + operators.classes + ui.classes:
        bpy.utils.register_class(cls)
    props.register()


def unregister():
    import bpy
    _deps, jobs, operators, prefs, props, ui = _modules
    job = jobs.current()
    if job is not None:
        job.cancelled = True
    props.unregister()
    for cls in reversed(props.classes + prefs.classes + operators.classes + ui.classes):
        bpy.utils.unregister_class(cls)
