import bpy
from bpy.props import FloatVectorProperty, BoolVectorProperty, BoolProperty, StringProperty, FloatProperty
from mathutils import Matrix
import numpy as np

class ExportParameters(bpy.types.PropertyGroup):
    flip_loc: BoolVectorProperty(
        name="Location Flips",
        size=3,
        default=[False, True, False]
    )
    flip_rot: BoolVectorProperty(
        name="Rotation Flips",
        size=3,
        default=[True, True, True]
    )
    adjust_rot: BoolProperty(
        name="Adjust rot to -pi + rot",
        default=False
    )

    flip_scale: BoolVectorProperty(
        name="Scale Flips",
        size=3,
        default=[False, False, False]
    )
    abs_scale: BoolVectorProperty(
        name="Scale Abs",
        size=3,
        default=[False, False, False]
    )
    swizzle_neg_scale: BoolProperty(
        name="Swizzle on -scale",
        default=False
    )
    rotation_offset: FloatVectorProperty(
        name="Offset Rotation (deg)",
        size=3,
        default=[0.0, 0.0, 0.0]
    )

    json_path: StringProperty(
        name="JSON File",
        subtype="FILE_PATH",
        description="JSON file to export the scene data into")

    export_path: StringProperty(
        name="Export Dir",
        subtype="DIR_PATH",
        description="Directory to export the scene fbx into")

    target_density: FloatProperty(
        name="Target Density",
        description="In p/m on 4k texture",
        default=256.0
    )

    def draw(self, layout):
        for prop in list(self.__annotations__):
            row = layout.row()
            row.prop(self, prop)


def get_polygon_data(data) -> np.ndarray:
    """Returns flat list of all face positions extended with face areas"""
    centers = np.zeros(len(data.polygons) * 3, dtype=float)
    areas = np.zeros(len(data.polygons), dtype=float)
    data.polygons.foreach_get("center", centers)
    data.polygons.foreach_get("area", areas)
    return np.concatenate((centers, areas))


def hash_uv_maps(data):
    """Hash UV coordinates of all UV maps and return list of hashes"""
    if len(data.uv_layers) == 0:
        return 0

    hashes = []
    container = np.zeros(len(data.uv_layers[0].uv.values()) * 2, dtype=float)
    for uvmap in data.uv_layers:
        uvmap.uv.foreach_get("vector", container)
        hashes.append(hash(container.tobytes()))

    return hashes


def hash_geometry(obj) -> int:
    face_hash = get_polygon_data(obj.data).tobytes()
    uv_hash = str(hash_uv_maps(obj.data))
    return hash(tuple([face_hash, uv_hash]))


def get_world_transform(obj):
    world_matrix = obj.matrix_world

    loc = world_matrix.to_translation()
    rot = world_matrix.to_euler()
    scale = world_matrix.to_scale()

    scale_matrix = Matrix.Diagonal(scale)

    for s in scale:
        if s > 0:
            continue
    else:
        rot = (rot.to_matrix() @ scale_matrix).to_euler()

    return loc, rot, scale


def copy_transform(source, target) -> None:
    s_loc, s_rot, s_scale = get_world_transform(source)
    target.location = s_loc
    target.scale = s_scale
    target.rotation_euler = s_rot


def vector_angle(v1, v2, normalized):
    if v1 == v2:
        return 0

    if not normalized:
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def select_smooth_faces(obj, slope_epsilon=1):
    mesh = obj.data

    if mesh.normals_domain == 'FACE':        # face normal domain means mesh is flat shaded
        return
    elif mesh.normals_domain == 'POINT':     # point  normal domain means mesh is entirely smooth shaded
        mesh.polygons.foreach_set('select',
                                  np.ones(len(mesh.polygons), dtype=bool))
        return
    else:
        # Since normals are stored in face-corner domain in every polygon loop,
        # check every loop of a polygon against the first loop's normal,
        # and if the angle between them is >0deg - set-select them to utilize
        # "Select Similar -> delimit UV" operator to mark the entire island without using bmesh

        # reset selection
        mesh.polygons.foreach_set('select',
                                  np.zeros(len(mesh.polygons), dtype=bool))
        mesh.edges.foreach_set('select',
                               np.zeros(len(mesh.edges), dtype=bool))
        mesh.vertices.foreach_set('select',
                                  np.zeros(len(mesh.vertices), dtype=bool))

        # select a
        for poly in mesh.polygons:
            ref_normal = mesh.loops[poly.loop_indices[0]].normal

            for index in poly.loop_indices:
                if vector_angle(mesh.loops[index].normal, ref_normal, True) > (0 + slope_epsilon):
                    mesh.polygons[poly.index].select = True
                    break
    return
