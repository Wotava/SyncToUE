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
