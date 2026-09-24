# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty


class AI3D_Settings(bpy.types.PropertyGroup):
    # --- generation --------------------------------------------------------
    source: EnumProperty(
        name="Origen",
        items=[("IMAGE", "Imagen", "Reconstruye el objeto de una foto o ilustración"),
               ("TEXT", "Texto", "Genera primero una imagen desde tu descripción y luego el 3D")],
        default="IMAGE")
    image_path: StringProperty(
        name="Imagen", subtype="FILE_PATH",
        description="Foto o render de UN objeto completo, bien iluminado, sin recortar")
    prompt: StringProperty(
        name="Descripción",
        description="Qué objeto quieres, p. ej. 'silla de madera tallada, estilo nórdico'")
    negative_prompt: StringProperty(
        name="Evitar", description="Lo que no quieres en la imagen (solo SDXL)")
    t2i_model: EnumProperty(
        name="Modelo de imagen",
        items=[("SDXL", "SDXL 1.0 (8 GB VRAM)",
                "Stable Diffusion XL · CreativeML Open RAIL++-M (uso comercial permitido)"),
               ("FLUX_SCHNELL", "FLUX.1 schnell (16 GB+ VRAM)",
                "Black Forest Labs · Apache-2.0. Mejor calidad, descarga de 34 GB")],
        default="SDXL")
    t2i_steps: IntProperty(
        name="Pasos", min=0, max=100, default=0,
        description="Pasos de difusión (0 = recomendado para el modelo)")
    asset_style: BoolProperty(
        name="Estilo de asset 3D", default=True,
        description="Añade al prompt 'objeto único, centrado, fondo blanco, luz de estudio', "
                    "que es lo que mejor reconstruye la IA 3D")
    seed: IntProperty(
        name="Semilla", min=-1, default=-1,
        description="-1 = aleatoria. Repite una semilla para reproducir un resultado")
    remove_background: BoolProperty(
        name="Quitar fondo (IA)", default=True,
        description="Recorta el objeto con IS-Net. Se omite si la imagen ya tiene transparencia")
    foreground_ratio: FloatProperty(
        name="Tamaño en el encuadre", min=0.5, max=1.0, default=0.85,
        description="Fracción de la imagen que ocupa el objeto antes de reconstruir")
    mc_resolution: IntProperty(
        name="Resolución de malla", min=64, max=512, default=256, step=32,
        description="Rejilla de marching cubes. 256 = buen detalle; 384+ necesita mucha memoria")
    threshold: FloatProperty(
        name="Umbral de densidad", min=1.0, max=100.0, default=25.0,
        description="Más bajo = objeto más grueso y relleno; más alto = más fino")
    min_part_ratio: FloatProperty(
        name="Quitar fragmentos <", min=0.0, max=0.5, default=0.02, subtype="FACTOR",
        description="Borra piezas flotantes más pequeñas que esta fracción de la pieza principal")
    smooth_iterations: IntProperty(
        name="Suavizado", min=0, max=50, default=10,
        description="Iteraciones de suavizado Taubin (quita el escalonado sin encoger)")
    auto_optimize: BoolProperty(
        name="Optimizar al terminar", default=True,
        description="Aplica automáticamente la optimización de abajo al modelo generado")
    name: StringProperty(name="Nombre", default="AI_Modelo")

    # --- optimisation --------------------------------------------------------
    apply_size: BoolProperty(
        name="Ajustar tamaño real", default=True,
        description="Escala el modelo a su tamaño real en metros")
    real_size: FloatProperty(
        name="Tamaño", min=0.001, max=1000.0, default=1.0, unit="LENGTH",
        description="Medida real del objeto (altura o lado mayor)")
    size_axis: EnumProperty(
        name="Medida",
        items=[("HEIGHT", "Altura", "El tamaño es la altura (Z)"),
               ("LONGEST", "Lado mayor", "El tamaño es la dimensión más grande")],
        default="HEIGHT")
    fill_holes: BoolProperty(name="Rellenar agujeros", default=True)
    remesh_mode: EnumProperty(
        name="Topología",
        items=[("DECIMATE", "Triángulos optimizados",
                "Decimado que conserva la forma. Ideal para juegos, web, AR y tiendas de assets"),
               ("QUADS", "Quads (QuadriFlow)",
                "Retopología en quads limpia para editar, esculpir o animar"),
               ("NONE", "Sin cambios", "Mantiene la malla densa original")],
        default="DECIMATE")
    target_faces: IntProperty(
        name="Triángulos objetivo", min=100, max=2000000, default=20000,
        description="Presupuesto de polígonos. Móvil/AR: 5–20k · PC/consola: 20–100k")
    voxel_detail: IntProperty(
        name="Detalle voxel", min=16, max=1000, default=200,
        description="Solo quads: vóxeles a lo largo de la diagonal antes de QuadriFlow")
    smooth_angle: FloatProperty(
        name="Ángulo de suavizado", min=0.0, max=180.0, default=40.0,
        description="Aristas más agudas que esto se muestran duras")
    bake_textures: BoolProperty(
        name="UVs + texturas PBR", default=True,
        description="Despliega UVs y hornea color y normales del modelo denso al optimizado")
    texture_size: EnumProperty(
        name="Resolución",
        items=[("512", "512", ""), ("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", "")],
        default="2048")
    bake_normal: BoolProperty(
        name="Mapa de normales", default=True,
        description="Conserva el detalle fino del modelo denso en el optimizado")
    roughness: FloatProperty(name="Rugosidad", min=0.0, max=1.0, default=0.6, subtype="FACTOR")
    metallic: FloatProperty(name="Metálico", min=0.0, max=1.0, default=0.0, subtype="FACTOR")
    lods: IntProperty(
        name="LODs", min=0, max=4, default=0,
        description="Versiones con 1/2, 1/4… de polígonos para juegos")
    keep_source: BoolProperty(
        name="Conservar malla original", default=False,
        description="Guarda la malla densa en la colección oculta 'AI 3D Originales'")

    # --- export ----------------------------------------------------------------
    export_dir: StringProperty(
        name="Carpeta", subtype="DIR_PATH", default="//ai3d_export/",
        description="Dónde guardar modelos y texturas ('//' = junto al .blend)")
    export_glb: BoolProperty(name="GLB", default=True, description="glTF binario: web, AR, Unity, Godot, tiendas")
    export_fbx: BoolProperty(name="FBX", default=False, description="Unreal, Unity, Maya, 3ds Max")
    export_obj: BoolProperty(name="OBJ", default=False, description="Universal, impresión 3D")
    export_usdz: BoolProperty(name="USDZ", default=False, description="AR en iPhone/iPad (Quick Look)")
    export_stl: BoolProperty(name="STL", default=False, description="Impresión 3D (sin color)")
    write_provenance: BoolProperty(
        name="Ficha de licencias", default=True,
        description="Guarda un .json con los modelos de IA usados, sus licencias, prompt y semilla")


classes = (AI3D_Settings,)


def register():
    bpy.types.Scene.ai3d = bpy.props.PointerProperty(type=AI3D_Settings)


def unregister():
    del bpy.types.Scene.ai3d
