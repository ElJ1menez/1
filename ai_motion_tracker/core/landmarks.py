# SPDX-License-Identifier: GPL-3.0-or-later
"""Body, face and semantic segmentation with Google MediaPipe (Apache-2.0)."""

import numpy as np

POSE_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)

# Face-mesh indices worth exporting as 2D tracks (corners, lips, brows, jaw).
FACE_TRACK_POINTS = {
    "nose_tip": 1, "forehead": 10, "chin": 152,
    "eye_R_outer": 33, "eye_R_inner": 133, "eye_L_inner": 362, "eye_L_outer": 263,
    "mouth_R": 61, "mouth_L": 291, "lip_upper": 13, "lip_lower": 14,
    "brow_R_outer": 70, "brow_R_mid": 105, "brow_R_inner": 107,
    "brow_L_inner": 336, "brow_L_mid": 334, "brow_L_outer": 300,
    "cheek_R": 234, "cheek_L": 454, "nose_R": 129, "nose_L": 358,
}

# PASCAL VOC classes of MediaPipe's DeepLab v3 that usually move on their own.
DYNAMIC_VOC_CLASSES = (1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 17, 19)


def _vision():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    return mp, mp_tasks.BaseOptions, vision


def _timestamps(n, fps):
    ts, last = [], -1
    for i in range(n):
        t = max(last + 1, int(round(i * 1000.0 / fps)))
        ts.append(t)
        last = t
    return ts


def _opt(v, default=1.0):
    return default if v is None else float(v)


def run_pose(frame_iter, total, fps, model_path, min_conf, job, progress):
    """Returns dict(frames, image (T,33,4)=[x,y,z,visibility], world (T,33,3))."""
    mp, BaseOptions, vision = _vision()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    frames, image, world = [], [], []
    ts = _timestamps(total, fps)
    with vision.PoseLandmarker.create_from_options(options) as lm:
        for i, (c, rgb) in enumerate(frame_iter):
            job.check()
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts[i])
            img = np.full((33, 4), np.nan)
            wld = np.full((33, 3), np.nan)
            if res.pose_landmarks:
                img[:] = [(p.x, p.y, p.z, _opt(p.visibility)) for p in res.pose_landmarks[0]]
            if res.pose_world_landmarks:
                wld[:] = [(p.x, p.y, p.z) for p in res.pose_world_landmarks[0]]
            frames.append(c)
            image.append(img)
            world.append(wld)
            progress((i + 1) / float(total), "IA analizando cuerpo: frame %d/%d" % (i + 1, total))
    return {"frames": frames, "image": np.array(image), "world": np.array(world)}


def run_face(frame_iter, total, fps, model_path, min_conf, job, progress):
    """Returns dict(frames, image (T,478,3), blend (T,K), names, matrix (T,4,4))."""
    mp, BaseOptions, vision = _vision()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    frames, image, blend, mats = [], [], [], []
    names = None
    ts = _timestamps(total, fps)
    with vision.FaceLandmarker.create_from_options(options) as lm:
        for i, (c, rgb) in enumerate(frame_iter):
            job.check()
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts[i])
            img = np.full((478, 3), np.nan)
            bl = None
            mat = np.full((4, 4), np.nan)
            if res.face_landmarks:
                pts = res.face_landmarks[0]
                img[: len(pts)] = [(p.x, p.y, p.z) for p in pts]
            if res.face_blendshapes:
                cats = res.face_blendshapes[0]
                if names is None:
                    names = [cat.category_name for cat in cats]
                bl = [cat.score for cat in cats]
            if res.facial_transformation_matrixes:
                mat = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float64)
            frames.append(c)
            image.append(img)
            blend.append(bl)
            mats.append(mat)
            progress((i + 1) / float(total), "IA analizando cara: frame %d/%d" % (i + 1, total))
    k = len(names or [])
    blend = np.array([b if b is not None else [np.nan] * k for b in blend]).reshape(len(frames), k)
    return {"frames": frames, "image": np.array(image), "blend": blend,
            "names": names or [], "matrix": np.array(mats)}


def dynamic_masks(frames, model_path, job, progress, downscale=4, dilate=2):
    """(T, h, w) bool masks of people/vehicles/animals for each frame."""
    import cv2
    mp, BaseOptions, vision = _vision()
    options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    height, width = frames[0].shape[:2]
    size = (max(8, width // downscale), max(8, height // downscale))
    kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    dyn = np.zeros(256, dtype=bool)
    dyn[list(DYNAMIC_VOC_CLASSES)] = True
    out = np.zeros((len(frames), size[1], size[0]), dtype=bool)
    with vision.ImageSegmenter.create_from_options(options) as seg:
        for i, rgb in enumerate(frames):
            job.check()
            res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            cat = res.category_mask.numpy_view()
            cat = cv2.resize(cat, size, interpolation=cv2.INTER_NEAREST)
            m = dyn[cat].astype(np.uint8)
            out[i] = cv2.dilate(m, kernel) > 0
            progress((i + 1) / float(len(frames)), "IA segmentando objetos móviles: %d/%d" % (i + 1, len(frames)))
    return out
