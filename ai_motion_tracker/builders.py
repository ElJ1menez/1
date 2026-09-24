# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply AI results to Blender data (main thread only)."""

import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .core import tracks as tk
from .core.landmarks import FACE_TRACK_POINTS, POSE_NAMES
from .core.smoothing import one_euro

AI_COLOR = (0.1, 0.85, 0.55)

# ---------------------------------------------------------------------------
# Clip editor context
# ---------------------------------------------------------------------------


def clip_editor_override(context, clip):
    """temp_override kwargs for a Clip Editor showing `clip` (assigns it if needed)."""
    candidates = []
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "CLIP_EDITOR":
                candidates.append((area.spaces.active.clip == clip, window, area))
    if not candidates:
        return None
    candidates.sort(key=lambda c: not c[0])
    _, window, area = candidates[0]
    space = area.spaces.active
    if space.clip != clip:
        space.clip = clip
    if space.mode != "TRACKING":
        space.mode = "TRACKING"
    region = next(r for r in area.regions if r.type == "WINDOW")
    return {"window": window, "area": area, "region": region}


def delete_tracks(context, clip, predicate):
    """Delete tracks of the active tracking object for which predicate(track)."""
    tracks = clip.tracking.objects.active.tracks
    doomed = [t for t in tracks if predicate(t)]
    if not doomed:
        return 0
    override = clip_editor_override(context, clip)
    if override is None:
        raise RuntimeError("Abre un Clip Editor para poder borrar tracks")
    doomed_names = {t.name for t in doomed}
    for t in tracks:
        t.select = t.name in doomed_names
    with context.temp_override(**override):
        bpy.ops.clip.delete_track()
    return len(doomed)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def write_markers(track, first_frame, co, vis):
    """Write a visibility-aware marker stream starting at clip frame first_frame.

    co: (T, 2) Blender-normalised positions, vis: (T,) bool. Blender holds a
    marker until the next one, so every occlusion starts with a disabled
    marker; otherwise the solver would see a frozen, wrong point.
    """
    n = len(vis)
    for f in range(first_frame, first_frame + n):
        track.markers.delete_frame(f)
    runs = tk.visible_runs(vis)
    if not runs:
        return
    if runs[0][0] > 0:
        m = track.markers.insert_frame(first_frame, co=tuple(co[runs[0][0]]))
        m.mute = True
    for a, b in runs:
        for i in range(a, b):
            track.markers.insert_frame(first_frame + i, co=(float(co[i, 0]), float(co[i, 1])))
        if b < n:
            m = track.markers.insert_frame(first_frame + b, co=(float(co[b - 1, 0]), float(co[b - 1, 1])))
            m.mute = True


def new_track(clip, name, frame):
    track = clip.tracking.objects.active.tracks.new(name=name, frame=frame)
    track.use_custom_color = True
    track.color = AI_COLOR
    track.select = False
    return track


def write_ai_tracks(clip, tracks, width, height, first, prefix="AI_"):
    """Create one Blender track per tk.Track. Returns number written."""
    for i, t in enumerate(tracks):
        frame = first + t.start
        track = new_track(clip, "%s%04d" % (prefix, i + 1), frame)
        write_markers(track, frame, tk.to_blender(t.xy, width, height), t.vis)
    return len(tracks)


def write_named_tracks(context, clip, prefix, names, first, co, vis):
    """co: (N, T, 2) Blender coords, vis: (N, T). Replaces same-named tracks."""
    delete_tracks(context, clip, lambda t: t.name.startswith(prefix))
    for n, name in enumerate(names):
        idx = np.flatnonzero(vis[n])
        if idx.size == 0:
            continue
        track = new_track(clip, prefix + name, first + int(idx[0]))
        write_markers(track, first, co[n], vis[n])


# ---------------------------------------------------------------------------
# Fast keyframing (handles Blender 4.4+ slotted actions and older versions)
# ---------------------------------------------------------------------------


def _fcurves(idblock):
    ad = idblock.animation_data or idblock.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(idblock.name + "Action")
    action = ad.action
    if hasattr(ad, "action_slot") and hasattr(action, "slots"):
        try:
            from bpy_extras import anim_utils
            if ad.action_slot is None:
                ad.action_slot = action.slots.new(id_type=idblock.id_type, name=idblock.name)
            return anim_utils.action_ensure_channelbag_for_slot(action, ad.action_slot).fcurves
        except (ImportError, AttributeError):
            pass
    return action.fcurves


def set_keys(idblock, data_path, index, frames, values):
    """Key a property from arrays in one go. NaN values are skipped."""
    frames = np.asarray(frames, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    if not ok.any():
        return
    fcurves = _fcurves(idblock)
    old = fcurves.find(data_path, index=index)
    if old is not None:
        fcurves.remove(old)
    fc = fcurves.new(data_path, index=index)
    co = np.stack([frames[ok], values[ok]], axis=1).ravel()
    fc.keyframe_points.add(int(ok.sum()))
    fc.keyframe_points.foreach_set("co", co)
    linear = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value
    fc.keyframe_points.foreach_set("interpolation", [linear] * int(ok.sum()))
    fc.update()


def key_vectors(obj, data_path, frames, values):
    for axis in range(values.shape[1]):
        set_keys(obj, data_path, axis, frames, values[:, axis])


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

_VIRTUAL = {"hips_center": (23, 24), "shoulders_center": (11, 12), "ears_center": (7, 8)}

BONES = (
    ("spine", "hips_center", "shoulders_center"),
    ("neck", "shoulders_center", "ears_center"),
    ("head", "ears_center", "nose"),
    ("clavicle.L", "shoulders_center", "left_shoulder"),
    ("upper_arm.L", "left_shoulder", "left_elbow"),
    ("forearm.L", "left_elbow", "left_wrist"),
    ("hand.L", "left_wrist", "left_index"),
    ("clavicle.R", "shoulders_center", "right_shoulder"),
    ("upper_arm.R", "right_shoulder", "right_elbow"),
    ("forearm.R", "right_elbow", "right_wrist"),
    ("hand.R", "right_wrist", "right_index"),
    ("pelvis.L", "hips_center", "left_hip"),
    ("thigh.L", "left_hip", "left_knee"),
    ("shin.L", "left_knee", "left_ankle"),
    ("foot.L", "left_ankle", "left_foot_index"),
    ("pelvis.R", "hips_center", "right_hip"),
    ("thigh.R", "right_hip", "right_knee"),
    ("shin.R", "right_knee", "right_ankle"),
    ("foot.R", "right_ankle", "right_foot_index"),
)


def mediapipe_to_blender(p):
    """MediaPipe world (x right, y down, z away from camera) -> Blender Z-up."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _new_collection(context, name):
    coll = bpy.data.collections.new(name)
    context.scene.collection.children.link(coll)
    return coll


def _empty(coll, name, parent=None, size=0.03, shape="SPHERE"):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = shape
    obj.empty_display_size = size
    obj.parent = parent
    coll.objects.link(obj)
    return obj


def build_pose(context, clip, res, s, fps):
    frames = np.asarray(res["frames"])
    image, world = res["image"], res["world"]
    detected = int(np.isfinite(world[:, 0, 0]).sum())
    if detected == 0:
        raise RuntimeError("La IA no detectó ninguna persona en el clip")
    first = int(frames[0])

    if s.make_tracks_2d:
        co = np.stack([image[..., 0], 1.0 - image[..., 1]], axis=-1).transpose(1, 0, 2)
        vis = (np.nan_to_num(image[..., 3]) >= s.min_confidence).T & np.isfinite(co[..., 0])
        write_named_tracks(context, clip, "AIPose_", POSE_NAMES, first, co, vis)

    if not (s.make_empties or s.make_armature):
        return "Cuerpo: %d/%d frames detectados" % (detected, len(frames))

    names = list(POSE_NAMES) + list(_VIRTUAL)
    extra = [(world[:, a] + world[:, b]) * 0.5 for a, b in _VIRTUAL.values()]
    pts = mediapipe_to_blender(np.concatenate([world, np.stack(extra, axis=1)], axis=1))
    if s.smoothing:
        flat = one_euro(pts.reshape(len(pts), -1), fps, s.smooth_cutoff, s.smooth_beta)
        pts = flat.reshape(pts.shape)

    scene_frames = clip.frame_start + frames - 1
    ref = int(np.flatnonzero(np.isfinite(pts[:, 0, 0]))[0])
    feet = [POSE_NAMES.index(n) for n in ("left_heel", "right_heel", "left_foot_index", "right_foot_index")]
    floor = -float(np.nanmin(pts[ref, feet, 2]))  # put the feet on Z = 0

    coll = _new_collection(context, "AI Pose")
    root = _empty(coll, "AI_Pose_Root", size=0.3, shape="PLAIN_AXES")
    root.location.z = floor
    empties = {}
    for j, name in enumerate(names):
        e = _empty(coll, "AIP_" + name, root, size=0.025 if name in POSE_NAMES else 0.04)
        e.location = pts[ref, j]
        key_vectors(e, "location", scene_frames, pts[:, j])
        e.hide_set(not s.make_empties)
        empties[name] = e

    if s.make_armature:
        _build_armature(context, coll, root, empties, {n: pts[ref, j] for j, n in enumerate(names)})
    return "Cuerpo: %d/%d frames detectados" % (detected, len(frames))


def _build_armature(context, coll, root, empties, rest):
    arm = bpy.data.armatures.new("AI_Pose_Rig")
    obj = bpy.data.objects.new("AI_Pose_Rig", arm)
    obj.parent = root
    obj.show_in_front = True
    coll.objects.link(obj)
    view_layer = context.view_layer
    active = view_layer.objects.active
    if active is not None and active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for name, head, tail in BONES:
        eb = arm.edit_bones.new(name)
        eb.head = Vector(rest[head].tolist())
        eb.tail = Vector(rest[tail].tolist())
        if (eb.tail - eb.head).length < 1e-4:
            eb.tail = eb.head + Vector((0.0, 0.0, 0.05))
    bpy.ops.object.mode_set(mode="OBJECT")
    for name, head, tail in BONES:
        pb = obj.pose.bones[name]
        c = pb.constraints.new("COPY_LOCATION")
        c.target = empties[head]
        c = pb.constraints.new("STRETCH_TO")
        c.target = empties[tail]
        c.volume = "NO_VOLUME"
    return obj


# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------

MEDIAPIPE_FACE_VFOV = 63.0  # perspective used by MediaPipe's face geometry


def build_face(context, clip, res, s, fps):
    frames = np.asarray(res["frames"])
    image, mats = res["image"], res["matrix"]
    detected = int(np.isfinite(mats[:, 0, 0]).sum())
    if detected == 0:
        raise RuntimeError("La IA no detectó ninguna cara en el clip")
    first = int(frames[0])
    scene_frames = clip.frame_start + frames - 1

    if s.make_tracks_2d:
        names = list(FACE_TRACK_POINTS)
        idx = [FACE_TRACK_POINTS[n] for n in names]
        sub = image[:, idx]
        co = np.stack([sub[..., 0], 1.0 - sub[..., 1]], axis=-1).transpose(1, 0, 2)
        vis = np.isfinite(co[..., 0])
        write_named_tracks(context, clip, "AIFace_", names, first, co, vis)

    if not (s.face_head or s.face_blendshapes):
        return "Cara: %d/%d frames detectados" % (detected, len(frames))

    coll = _new_collection(context, "AI Face")
    cam_data = bpy.data.cameras.new("AI_Face_Cam")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(MEDIAPIPE_FACE_VFOV)
    cam_data.show_background_images = True
    bg = cam_data.background_images.new()
    bg.source = "MOVIE_CLIP"
    bg.clip = clip
    cam = bpy.data.objects.new("AI_Face_Cam", cam_data)
    cam.rotation_euler = (math.pi / 2.0, 0.0, 0.0)  # look down +Y
    coll.objects.link(cam)

    # MediaPipe's face matrix maps the canonical face (cm) into an OpenGL-style
    # camera frame, which is exactly a Blender camera's local frame.
    head = _empty(coll, "AI_Head", cam, size=0.1, shape="ARROWS")
    if s.face_head:
        loc = mats[:, :3, 3] * 0.01
        quats = np.full((len(mats), 4), np.nan)
        prev = None
        for i, m in enumerate(mats):
            if not np.isfinite(m).all():
                continue
            q = Matrix(m[:3, :3].tolist()).to_quaternion().normalized()
            if prev is not None and q.dot(prev) < 0.0:
                q.negate()  # stay on one hemisphere so smoothing is valid
            quats[i] = q
            prev = q
        if s.smoothing:
            loc = one_euro(loc, fps, s.smooth_cutoff, s.smooth_beta)
            quats = one_euro(quats, fps, s.smooth_cutoff, s.smooth_beta)
        eulers = np.full((len(mats), 3), np.nan)
        prev_e = None
        for i, q in enumerate(quats):
            if not np.isfinite(q).all():
                continue
            qq = Quaternion(q.tolist()).normalized()
            e = qq.to_euler("XYZ", prev_e) if prev_e is not None else qq.to_euler("XYZ")
            eulers[i] = e
            prev_e = e
        key_vectors(head, "location", scene_frames, loc)
        key_vectors(head, "rotation_euler", scene_frames, eulers)

    if s.face_blendshapes and res["names"]:
        blend = res["blend"]
        if s.smoothing:
            blend = one_euro(blend, fps, s.smooth_cutoff * 2.0, s.smooth_beta)
        for k, name in enumerate(res["names"]):
            head[name] = 0.0
            try:
                head.id_properties_ui(name).update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)
            except AttributeError:
                pass
            set_keys(head, '["%s"]' % name, 0, scene_frames, np.clip(blend[:, k], 0.0, 1.0))
    return "Cara: %d/%d frames detectados" % (detected, len(frames))
