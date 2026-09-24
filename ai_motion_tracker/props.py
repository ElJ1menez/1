# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty


class AIMT_Settings(bpy.types.PropertyGroup):
    # --- common -----------------------------------------------------------
    use_scene_range: BoolProperty(
        name="Solo rango de la escena",
        description="Procesa solo los frames entre inicio y fin de la escena en vez de todo el clip",
        default=False)

    # --- camera tracking ----------------------------------------------------
    model_res: EnumProperty(
        name="Calidad IA",
        items=[("512", "Precisa (512)", "Error sub-píxel. Recomendado con GPU"),
               ("256", "Rápida (256)", "~4x más rápida, algo menos precisa. Útil en CPU")],
        default="512")
    max_side: IntProperty(
        name="Resolución IA", min=256, max=1536, default=640,
        description="Lado largo de los frames que ve la IA. Más = más preciso y más lento")
    points_per_seed: IntProperty(
        name="Puntos por semilla", min=16, max=2000, default=250,
        description="Puntos nuevos que la IA siembra en cada frame semilla")
    window: IntProperty(
        name="Ventana", min=8, max=256, default=48,
        description="Frames que la IA procesa de una vez. Más = tracks más largos y más memoria")
    interval: IntProperty(
        name="Resembrar cada", min=2, max=256, default=16,
        description="Cada cuántos frames se siembran puntos nuevos")
    min_length: IntProperty(
        name="Longitud mínima", min=3, max=500, default=10,
        description="Descarta tracks visibles en menos frames que esto")
    per_frame: IntProperty(
        name="Tracks por frame", min=8, max=1000, default=60,
        description="Objetivo de tracks activos en cada frame tras la selección")
    max_tracks: IntProperty(
        name="Máximo de tracks", min=16, max=20000, default=1500)
    dynamic_mask: BoolProperty(
        name="Ignorar objetos móviles (IA)", default=True,
        description="Segmenta personas, vehículos y animales y no trackea sobre ellos")
    geometric_filter: BoolProperty(
        name="Filtro geométrico", default=True,
        description="Elimina tracks que no cumplen la geometría epipolar de la cámara (RANSAC)")
    geo_threshold: FloatProperty(
        name="Umbral (px)", min=0.2, max=10.0, default=1.5)
    min_inlier_ratio: FloatProperty(
        name="Consistencia mínima", min=0.0, max=1.0, default=0.7, subtype="FACTOR")
    replace_tracks: BoolProperty(
        name="Reemplazar tracks IA anteriores", default=True)

    # --- solve --------------------------------------------------------------
    solve: BoolProperty(name="Resolver cámara al terminar", default=True)
    auto_focal: BoolProperty(
        name="Buscar focal automáticamente", default=False,
        description="Prueba varias focales y se queda con la de menor error de solve")
    focal_min: FloatProperty(name="Focal mín (mm)", min=4.0, max=1000.0, default=14.0)
    focal_max: FloatProperty(name="Focal máx (mm)", min=4.0, max=1000.0, default=100.0)
    refine_focal: BoolProperty(name="Refinar focal", default=False)
    refine_distortion: BoolProperty(name="Refinar distorsión", default=False)
    clean_error: FloatProperty(
        name="Limpiar error >", min=0.0, max=20.0, default=3.0,
        description="Borra tracks con error de reproyección mayor (px) y vuelve a resolver. 0 = no")
    setup_scene: BoolProperty(
        name="Configurar escena de tracking", default=False,
        description="Crea cámara con Camera Solver, fondo y suelo (Setup Tracking Scene)")

    # --- body / face ------------------------------------------------------------
    pose_model: EnumProperty(
        name="Modelo",
        items=[("LITE", "Rápido", ""), ("FULL", "Equilibrado", ""), ("HEAVY", "Preciso", "")],
        default="HEAVY")
    min_confidence: FloatProperty(
        name="Confianza mínima", min=0.05, max=0.95, default=0.5, subtype="FACTOR")
    make_tracks_2d: BoolProperty(
        name="Tracks 2D en el clip", default=True,
        description="Crea marcadores en el Clip Editor (útiles para 2D, estabilizar, compositing)")
    make_empties: BoolProperty(name="Empties 3D animados", default=True)
    make_armature: BoolProperty(
        name="Esqueleto (armature)", default=True,
        description="Armature cuyos huesos siguen a los empties (bakeable y retargeteable)")
    smoothing: BoolProperty(name="Suavizado (One Euro)", default=True)
    smooth_cutoff: FloatProperty(
        name="Suavidad", min=0.05, max=10.0, default=1.0,
        description="Frecuencia de corte mínima (Hz). Menos = más suave, más lag")
    smooth_beta: FloatProperty(
        name="Reactividad", min=0.0, max=10.0, default=0.5,
        description="Cuánto se reduce el suavizado en movimientos rápidos")
    face_head: BoolProperty(name="Movimiento de cabeza", default=True)
    face_blendshapes: BoolProperty(
        name="Expresiones (52 blendshapes ARKit)", default=True,
        description="Propiedades animadas 0..1 listas para drivers de shape keys")


classes = (AIMT_Settings,)


def register():
    bpy.types.Scene.aimt = bpy.props.PointerProperty(type=AIMT_Settings)


def unregister():
    del bpy.types.Scene.aimt
