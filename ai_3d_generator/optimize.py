# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender side of "make it look right": turns a raw AI mesh (dense,
triangle soup, vertex colours) into a production asset: clean manifold
geometry at a controlled polygon budget, UVs, baked PBR textures (base colour
and normal map from the dense original), LODs, real-world scale and the
origin on the ground.

Works on any mesh object, not only generated ones. optimize_steps() is a
generator so the UI stays responsive between steps (see jobs.Driver).
"""

import math
import os
import tempfile

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import builders

SOURCE_COLLECTION = "AI 3D Originales"


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def world_bbox(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector(min(p[i] for p in pts) for i in range(3))
    hi = Vector(max(p[i] for p in pts) for i in range(3))
    return lo, hi


def _override(context, obj, selected=None):
    selected = selected or [obj]
    return context.temp_override(object=obj, active_object=obj, edit_object=obj,
                                 selected_objects=selected, selected_editable_objects=selected)


def _apply_modifier(context, obj, mod):
    """Apply one modifier without operators (works with no window)."""
    depsgraph = context.evaluated_depsgraph_get()
    depsgraph.update()
    others = [m for m in obj.modifiers if m != mod]
    states = [(m, m.show_viewport) for m in others]
    for m in others:
        m.show_viewport = False
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    new_mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                               depsgraph=depsgraph)
    for m, shown in states:
        m.show_viewport = shown
    obj.modifiers.remove(mod)
    old = obj.data
    new_mesh.materials.clear()
    for mat in old.materials:
        new_mesh.materials.append(mat)
    obj.data = new_mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)


def duplicate(context, obj, name, coll=None):
    new = obj.copy()
    new.data = obj.data.copy()
    new.name = name
    new.data.name = name
    for c in (coll,) if coll else obj.users_collection:
        c.objects.link(new)
    return new


def bake_transform(obj):
    """Apply rotation and scale to the mesh data (keeps location)."""
    loc = obj.matrix_world.to_translation()
    obj.data.transform(Matrix.Translation(-loc) @ obj.matrix_world)
    obj.matrix_world = Matrix.Translation(loc)
    obj.data.update()


def set_size_and_ground(obj, size, axis="HEIGHT", resize=True):
    """Scale to real-world size and put the origin at the bottom centre."""
    bake_transform(obj)
    co = [v.co for v in obj.data.vertices]
    lo = Vector(min(c[i] for c in co) for i in range(3))
    hi = Vector(max(c[i] for c in co) for i in range(3))
    extent = hi - lo
    ref = extent.z if axis == "HEIGHT" else max(extent)
    scale = size / ref if (resize and ref > 0) else 1.0
    pivot = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    obj.data.transform(Matrix.Scale(scale, 4) @ Matrix.Translation(-pivot))
    obj.data.update()
    return extent * scale


def cleanup(obj, merge_dist, fill_holes=True, max_hole_sides=64):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.dissolve_degenerate(bm, dist=merge_dist * 0.1, edges=bm.edges)
    filled = 0
    if fill_holes:
        boundary = [e for e in bm.edges if e.is_boundary]
        if boundary:
            filled = len(bmesh.ops.holes_fill(bm, edges=boundary, sides=max_hole_sides)["faces"])
            bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    merged = before - len(bm.verts)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return merged, filled


def mesh_report(obj, max_topology_check=400000):
    non_manifold = boundary = None
    if len(obj.data.polygons) <= max_topology_check:  # keep UI redraws fast
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        non_manifold = sum(not e.is_manifold for e in bm.edges)
        boundary = sum(e.is_boundary for e in bm.edges)
        bm.free()
    lo, hi = world_bbox(obj)
    mats = [m for m in obj.data.materials if m]
    images = sorted({n.image.name for m in mats if m.node_tree for n in m.node_tree.nodes
                     if n.type == "TEX_IMAGE" and n.image})
    return {
        "object": obj.name,
        "triangles": tri_count(obj),
        "vertices": len(obj.data.vertices),
        "dimensions_m": [round(v, 4) for v in (hi - lo)],
        "non_manifold_edges": non_manifold,
        "open_edges": boundary,
        "watertight": None if boundary is None else (non_manifold == 0 and boundary == 0),
        "uv_maps": [uv.name for uv in obj.data.uv_layers],
        "color_attributes": [a.name for a in obj.data.color_attributes],
        "materials": [m.name for m in mats],
        "textures": images,
        "provenance": bool(obj.get(builders.PROVENANCE_KEY)),
    }


def decimate(context, obj, target_tris):
    tris = tri_count(obj)
    if tris <= target_tris:
        return
    mod = obj.modifiers.new("AI3D_Decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = max(target_tris / float(tris), 0.0005)
    mod.use_collapse_triangulate = True
    _apply_modifier(context, obj, mod)


def voxel_remesh(context, obj, voxel_size):
    mod = obj.modifiers.new("AI3D_Remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel_size
    mod.adaptivity = 0.0
    mod.use_smooth_shade = True
    _apply_modifier(context, obj, mod)


def quad_remesh(context, obj, target_quads, voxel_size):
    """Voxel remesh (makes it manifold) then QuadriFlow. Returns False if
    QuadriFlow failed, leaving the voxel mesh in place."""
    voxel_remesh(context, obj, voxel_size)
    decimate(context, obj, max(target_quads * 8, 50000))  # QuadriFlow is slow on huge input
    before = len(obj.data.polygons)
    try:
        with _override(context, obj):
            res = bpy.ops.object.quadriflow_remesh(
                target_faces=int(target_quads), use_mesh_symmetry=False, use_preserve_sharp=False,
                use_preserve_boundary=False, smooth_normals=False, mode="FACES", seed=0)
    except RuntimeError:
        return False
    return "FINISHED" in res and len(obj.data.polygons) != before


def shade_smooth(obj, angle_deg):
    mesh = obj.data
    if hasattr(mesh, "shade_smooth"):
        mesh.shade_smooth()
    else:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if hasattr(mesh, "set_sharp_from_angle"):  # Blender 4.1+
        mesh.set_sharp_from_angle(angle=math.radians(angle_deg))
    elif hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.radians(angle_deg)


def smart_uv(context, obj, margin=0.004):
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    obj.data.uv_layers.new(name="UVMap")
    with _override(context, obj):
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bpy.ops.mesh.reveal()
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=margin,
                                     area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
            try:
                bpy.ops.uv.pack_islands(rotate=True, margin=margin)
            except (RuntimeError, TypeError):
                pass
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")


def transfer_colors(context, src, dst):
    """Copy the AI vertex colours from the dense mesh to the optimized one."""
    if not src.data.color_attributes:
        return False
    mod = dst.modifiers.new("AI3D_ColorTransfer", "DATA_TRANSFER")
    mod.object = src
    mod.use_object_transform = True
    mod.use_vert_data = True
    mod.data_types_verts = {"COLOR_VERTEX"}
    mod.vert_mapping = "POLYINTERP_NEAREST"
    mod.layers_vcol_vert_select_src = "ALL"
    mod.layers_vcol_vert_select_dst = "NAME"
    _apply_modifier(context, dst, mod)
    attrs = dst.data.color_attributes
    if attrs:
        attrs.active_color = attrs[0]
        attrs.render_color_index = attrs.active_color_index
    return bool(attrs)


def texture_dir(export_dir):
    path = bpy.path.abspath(export_dir) if export_dir else ""
    if not path or (export_dir.startswith("//") and not bpy.data.filepath):
        path = os.path.join(tempfile.gettempdir(), "ai3d_export")
    path = os.path.join(path, "textures")
    os.makedirs(path, exist_ok=True)
    return path


def _new_image(name, size, non_color=False, fill=(0.5, 0.5, 0.5, 1.0)):
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    img.generated_color = fill
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def _save_image(img, folder):
    path = os.path.join(folder, bpy.path.clean_name(img.name) + ".png")
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    img.filepath = path
    return path


def pbr_material(name, base_img=None, normal_img=None, roughness=0.6, metallic=0.0):
    mat = bpy.data.materials.get(name) or builders.new_node_material(name)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if base_img is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = base_img
        tex.location = (-400, 250)
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if normal_img is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = normal_img
        tex.location = (-600, -250)
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (-250, -250)
        links.new(tex.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


class _BakeSettings:
    """Switch the scene to a fast Cycles bake and restore it afterwards."""

    def __init__(self, scene, device):
        self.scene = scene
        self.device = device

    def __enter__(self):
        s = self.scene
        self.saved = (s.render.engine, s.cycles.samples, s.cycles.device,
                      s.render.bake.margin, s.cycles.use_denoising)
        s.render.engine = "CYCLES"
        s.cycles.samples = 1
        s.cycles.use_denoising = False
        s.cycles.device = self.device
        return s

    def __exit__(self, *exc):
        s = self.scene
        (s.render.engine, s.cycles.samples, s.cycles.device,
         s.render.bake.margin, s.cycles.use_denoising) = self.saved


def _bake(context, high, low, bake_type, image, extrusion, max_ray):
    mat = low.data.materials[0]
    nodes = mat.node_tree.nodes
    node = nodes.new("ShaderNodeTexImage")
    node.image = image
    for n in nodes:
        n.select = False
    node.select = True
    nodes.active = node
    bake = context.scene.render.bake
    bake.use_selected_to_active = True
    bake.cage_extrusion = extrusion
    bake.max_ray_distance = max_ray
    bake.margin = 16
    bake.margin_type = "EXTEND"
    bake.target = "IMAGE_TEXTURES"
    kwargs = dict(type=bake_type, use_selected_to_active=True, cage_extrusion=extrusion,
                  max_ray_distance=max_ray, margin=16, use_clear=True)
    if bake_type == "DIFFUSE":
        kwargs["pass_filter"] = {"COLOR"}
    else:
        kwargs["normal_space"] = "TANGENT"
    try:
        with _override(context, low, [high, low]):
            bpy.ops.object.bake(**kwargs)
    finally:
        nodes.remove(node)


def bake_textures(context, high, low, size, name, folder, normal=True, device="CPU"):
    """Bake base colour (and normal map) from the dense mesh onto low's UVs."""
    lo, hi = world_bbox(high)
    diag = (hi - lo).length
    extrusion, max_ray = 0.02 * diag, 0.08 * diag
    base = _new_image(name + "_BaseColor", size)
    nrm = _new_image(name + "_Normal", size, non_color=True, fill=(0.5, 0.5, 1.0, 1.0)) if normal else None
    low.data.materials.clear()
    low.data.materials.append(pbr_material(name + "_Bake"))
    for o in (high, low):
        o.hide_set(False)
        o.hide_render = False
    builders.select_only(context, [high, low], active=low)
    with _BakeSettings(context.scene, device):
        _bake(context, high, low, "DIFFUSE", base, extrusion, max_ray)
        yield "Textura de color horneada"
        if nrm is not None:
            _bake(context, high, low, "NORMAL", nrm, extrusion, max_ray)
            yield "Mapa de normales horneado"
    paths = [_save_image(base, folder)]
    if nrm is not None:
        paths.append(_save_image(nrm, folder))
    bpy.data.materials.remove(low.data.materials[0])
    low.data.materials.clear()
    return base, nrm, paths


def make_lods(context, obj, count):
    lods = []
    for i in range(1, count + 1):
        lod = duplicate(context, obj, "%s_LOD%d" % (obj.name, i))
        lod.data.materials.clear()
        for mat in obj.data.materials:
            lod.data.materials.append(mat)
        lod.parent = obj
        lod.matrix_parent_inverse = obj.matrix_world.inverted()
        decimate(context, lod, max(64, int(tri_count(obj) * 0.5 ** i)))
        lod.hide_set(True)
        lod.hide_render = True
        lods.append(lod)
    return lods


def _stash_source(context, obj):
    coll = builders.collection(context, SOURCE_COLLECTION)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    obj.hide_render = True
    # Collection-level hiding: excluding it from the view layer instead can
    # crash Blender 4.2 when the object list is read right afterwards.
    coll.hide_viewport = True
    coll.hide_render = True


def optimize_steps(context, obj, s, device="CPU", job=None):
    """Generator: optimise `obj` in place of the original. Returns a summary.
    On error the partial copies are removed and the original keeps its name."""
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Selecciona un objeto de malla")
    name = obj.name
    before = set(bpy.data.objects)
    try:
        return (yield from _optimize(context, obj, s, device, job))
    except Exception:
        for extra in set(bpy.data.objects) - before:
            bpy.data.objects.remove(extra)
        if obj.name != name:
            obj.name = name
        raise


def _optimize(context, obj, s, device, job):
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Selecciona un objeto de malla")
    if context.object is not None and context.object.mode != "OBJECT":
        with _override(context, context.object):
            bpy.ops.object.mode_set(mode="OBJECT")

    def step(progress, text):
        if job is not None:
            job.check()
            job.set(progress=progress, message=text)

    name = obj.name
    high = obj
    high.name = name + "_original"
    notes = []

    step(0.05, "Escala real y origen en el suelo")
    dims = set_size_and_ground(high, s.real_size, s.size_axis, s.apply_size)
    low = duplicate(context, high, name)
    yield

    step(0.12, "Limpiando geometría")
    diag = dims.length
    merged, filled = cleanup(low, merge_dist=diag * 1e-5, fill_holes=s.fill_holes)
    if filled:
        notes.append("%d agujeros rellenados" % filled)
    yield

    raw_tris = tri_count(low)
    if s.remesh_mode == "QUADS":
        step(0.2, "Retopología en quads (QuadriFlow)")
        voxel = max(diag / max(s.voxel_detail, 16), 1e-5)
        if not quad_remesh(context, low, max(s.target_faces // 2, 100), voxel):
            notes.append("QuadriFlow falló: se usó decimado")
            decimate(context, low, s.target_faces)
    elif s.remesh_mode == "DECIMATE":
        step(0.2, "Reduciendo polígonos")
        decimate(context, low, s.target_faces)
    yield

    step(0.35, "Suavizado")
    shade_smooth(low, s.smooth_angle)
    yield

    textures = []
    if s.bake_textures:
        step(0.45, "Desplegando UVs")
        smart_uv(context, low)
        yield
        step(0.55, "Horneando texturas (Cycles)")
        folder = texture_dir(s.export_dir)
        gen = bake_textures(context, high, low, int(s.texture_size), name, folder,
                            s.bake_normal, device)
        while True:
            try:
                msg = next(gen)
            except StopIteration as stop:
                base, nrm, textures = stop.value
                break
            step(0.75, msg)
            yield
        low.data.materials.append(pbr_material(name + "_MAT", base, nrm, s.roughness, s.metallic))
        if low.data.color_attributes:
            for attr in list(low.data.color_attributes):
                low.data.color_attributes.remove(attr)
    else:
        if s.remesh_mode == "QUADS":
            transfer_colors(context, high, low)
        if low.data.color_attributes:  # AI mesh: show its colours; else keep the materials
            low.data.materials.clear()
            low.data.materials.append(builders.vertex_color_material())
    yield

    # Remeshing moves the lowest point slightly: rest exactly on the ground.
    set_size_and_ground(low, 1.0, resize=False)

    lods = []
    if s.lods:
        step(0.85, "Creando LODs")
        lods = make_lods(context, low, s.lods)

    low["ai3d_optimized"] = True
    if builders.PROVENANCE_KEY not in low:
        builders.set_provenance(low, {"steps": [{"stage": "user_mesh", "object": name}]})
    if s.keep_source:
        _stash_source(context, high)
    else:
        mesh = high.data
        bpy.data.objects.remove(high)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    builders.select_only(context, [low])

    msg = "%s: %d → %d triángulos · %.2f × %.2f × %.2f m" % (
        name, raw_tris, tri_count(low), dims.x, dims.y, dims.z)
    if textures:
        msg += " · texturas %spx" % s.texture_size
    if lods:
        msg += " · %d LODs" % len(lods)
    if notes:
        msg += " · " + "; ".join(notes)
    return msg
