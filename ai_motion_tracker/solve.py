# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera solve using Blender's own solver (libmv) on the AI tracks."""

import math

import bpy

from . import builders


def _set(obj, attr, value):
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def _configure(tracking, refine_focal, refine_distortion):
    settings = tracking.settings
    _set(settings, "use_keyframe_selection", True)
    _set(tracking.objects.active, "use_keyframe_selection", True)
    _set(settings, "refine_intrinsics_focal_length", refine_focal)
    _set(settings, "refine_intrinsics_principal_point", False)
    _set(settings, "refine_intrinsics_radial_distortion", refine_distortion)
    _set(settings, "refine_intrinsics_tangential_distortion", False)


def solve_steps(context, clip, s, job):
    """Generator: yields between solves; returns a summary string."""
    override = builders.clip_editor_override(context, clip)
    if override is None:
        raise RuntimeError("Abre un Clip Editor para resolver la cámara")
    context.scene.active_clip = clip
    tracking = clip.tracking
    camera = tracking.camera
    camera.units = "MILLIMETERS"

    def run():
        with context.temp_override(**override):
            bpy.ops.clip.solve_camera()
        rec = tracking.objects.active.reconstruction
        return rec.average_error if rec.is_valid else math.inf

    notes = []
    if s.auto_focal:
        _configure(tracking, False, False)
        lo, hi = sorted((s.focal_min, s.focal_max))
        n = 7
        cands = [lo * (hi / lo) ** (i / (n - 1.0)) for i in range(n)]
        errors = []
        for i, f in enumerate(cands):
            job.set(progress=i / (n + 6.0), message="Buscando focal: %.1f mm" % f)
            yield
            camera.focal_length = f
            errors.append(run())
        best = min(range(n), key=errors.__getitem__)
        if math.isinf(errors[best]):
            raise RuntimeError("Ninguna focal produjo un solve válido: revisa los tracks")
        # Golden-section refinement between the neighbours of the best sample.
        a = cands[max(0, best - 1)]
        b = cands[min(n - 1, best + 1)]
        g = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - g * (b - a), a + g * (b - a)
        cache = {}

        def err(f):
            if f not in cache:
                camera.focal_length = f
                cache[f] = run()
            return cache[f]

        for i in range(5):
            job.set(progress=(n + i) / (n + 6.0), message="Refinando focal: %.1f–%.1f mm" % (a, b))
            yield
            if err(c) < err(d):
                b, d = d, c
                c = b - g * (b - a)
            else:
                a, c = c, d
                d = a + g * (b - a)
        focal = min(list(cache.items()) + [(cands[best], errors[best])], key=lambda kv: kv[1])[0]
        camera.focal_length = focal
        notes.append("focal %.1f mm" % focal)

    _configure(tracking, s.refine_focal, s.refine_distortion)
    job.set(message="Resolviendo cámara")
    yield
    error = run()
    if math.isinf(error):
        raise RuntimeError(
            "El solve falló. Prueba 'Buscar focal automáticamente', más tracks por frame "
            "o introduce la focal/sensor reales en Track > Camera")

    if s.clean_error > 0.0:
        removed = builders.delete_tracks(
            context, clip,
            lambda t: t.name.startswith("AI_") and t.has_bundle and t.average_error > s.clean_error)
        if removed:
            notes.append("%d tracks limpiados" % removed)
            job.set(message="Re-resolviendo sin los %d peores tracks" % removed)
            yield
            error = run()

    if s.setup_scene:
        with context.temp_override(**override):
            bpy.ops.clip.setup_tracking_scene()
        notes.append("escena configurada")

    extra = (" · " + ", ".join(notes)) if notes else ""
    return "Solve: error %.3f px%s" % (error, extra)
