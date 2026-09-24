# SPDX-License-Identifier: GPL-3.0-or-later
"""Create Blender data from the AI results."""

import json

import bpy
import numpy as np

COLLECTION = "AI 3D"
COLOR_ATTR = "AI_Color"
PROVENANCE_KEY = "ai3d_provenance"


def collection(context, name=COLLECTION):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)
    return coll


def mesh_from_arrays(name, verts, faces, colors=None):
    verts = np.ascontiguousarray(verts, np.float32)
    faces = np.ascontiguousarray(faces, np.int32)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3, dtype=np.int32))
    if colors is not None:
        attr = mesh.color_attributes.new(COLOR_ATTR, "FLOAT_COLOR", "POINT")
        rgba = np.ones((len(verts), 4), np.float32)
        rgba[:, :3] = colors
        attr.data.foreach_set("color", rgba.ravel())
        mesh.color_attributes.active_color = attr
        mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index
    mesh.update()
    mesh.validate(clean_customdata=False)
    return mesh


def vertex_color_material(name="AI3D_VertexColor"):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    attr = nodes.new("ShaderNodeVertexColor")
    attr.layer_name = COLOR_ATTR
    attr.location = (-350, 250)
    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def set_provenance(obj, data):
    obj[PROVENANCE_KEY] = json.dumps(data, ensure_ascii=False)


def get_provenance(obj):
    try:
        return json.loads(obj.get(PROVENANCE_KEY, "{}"))
    except (TypeError, ValueError):
        return {}


def build_generated(context, r):
    """Create the object for a pipeline.generate() result."""
    name = r["name"] or "AI_Model"
    mesh = mesh_from_arrays(name, r["verts"], r["faces"], r["colors"])
    mesh.materials.append(vertex_color_material())
    obj = bpy.data.objects.new(name, mesh)
    collection(context).objects.link(obj)
    for poly_obj in context.view_layer.objects:
        poly_obj.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    obj.location = context.scene.cursor.location
    set_provenance(obj, r["provenance"])
    obj["ai3d_input_image"] = r["image"]
    return obj


def select_only(context, objs, active=None):
    for o in context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    context.view_layer.objects.active = active or (objs[0] if objs else None)
