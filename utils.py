import bpy
from bpy.props import *
from mathutils import Matrix
import numpy as np


class AtlasMatSlot(bpy.types.PropertyGroup):
    material: PointerProperty(
        type=bpy.types.Material,
        name="Material"
    )


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
        name="ObjJSON",
        subtype="FILE_PATH",
        description="JSON file to export the scene data into")

    json_mat_path: StringProperty(
        name="MatJSON",
        subtype="FILE_PATH",
        description="JSON file to export materials data into")

    export_path: StringProperty(
        name="Export Dir",
        subtype="DIR_PATH",
        description="Directory to export the scene fbx into")

    # Asset export
    asset_export_path: StringProperty(
        name="Asset Export Dir",
        subtype="DIR_PATH",
        description="Directory to export assets into")

    copy_textures: EnumProperty(
        name="Copy Textures to Asset Folder",
        description="Copy ALL textures to destination folder",
        items=[
            ('NONE', "Don't Copy", ""),
            ('COPY', "Full Copy", ""),
            ('LINK', "Symlink", ""),
        ],
        default='COPY'
    )

    target_density: FloatProperty(
        name="Target Density",
        description="In p/m on 4k texture",
        default=16
    )
    target_resolution: FloatProperty(
        name="Target Resolution",
        description="Target image texture resolution",
        default=4096
    )
    bake_ue: BoolProperty(
        name="Output UE fbx",
        default=True
    )
    bake_houdini: BoolProperty(
        name="Output Houdini fbx",
        default=True
    )
    internal_padding: FloatProperty(
        name="Extra Padding",
        description="Extra padding between instances",
        soft_min=0,
        soft_max=1,
        default=0.001
    )
    accepted_td_loss: FloatProperty(
        name="TD Loss MAX",
        description="Maximum loss of target TD when squeezing udims",
        min=0,
        max=1,
        default=0
    )

    active_material_index: IntProperty(
        name="Active Mat Slot",
        default=-1
    )
    materials: CollectionProperty(
        type=AtlasMatSlot,
        name="Atlas-baked Materials"
    )
    baked_material_suffixes: StringProperty(
        name="Bake Tags",
        description="Split by coma",
        default='_ARCH_, _PAN_'
    )

    def add_material_slot(self):
        self.materials.add()
        self.active_material_index = len(self.materials) - 1

    def remove_material_slot(self):
        self.materials.remove(self.active_material_index)
        self.active_material_index -= 1

    def __contains__(self, item):
        suffixes = self.baked_material_suffixes.replace(' ', '').split(',')
        for suffix in suffixes:
            if suffix in item:
                return True
        return False

    def check_obj(self, obj) -> bool:
        materials = [slot.material.name for slot in obj.material_slots if slot.material is not None]
        for material in materials:
            if material not in self:
                return False
        return True

    def draw(self, layout):
        box = layout.box()
        box.label(text='JSON Transforms', icon='CON_TRANSFORM')
        row = box.row()
        col = row.column()
        col.label(text='')
        col = row.column()
        col.label(text='X')
        col = row.column()
        col.label(text='Y')
        col = row.column()
        col.label(text='Z')

        for prop in list(self.__annotations__):

            if prop == 'json_path':
                box = layout.box()
                box.label(text='Filepaths', icon='FILE_FOLDER')
            if prop == 'target_density':
                box = layout.box()
                box.label(text='Scene Export', icon='SCENE')

            row = box.row()
            if prop not in ['materials', 'active_material_index']:
                row.prop(self, prop)

        box = layout.box()
        box.label(text='Atlas Materials', icon='MATERIAL')
        row = box.row()
        row.template_list("DATA_UL_AtlasMaterials", "", self, "materials", self, "active_material_index", rows=3)

        row = box.row()
        row.operator("scene.add_atlas_material_slot", text="+")
        row.operator("scene.remove_atlas_material_slot", text="-")

        if len(self.materials) > 0:
            row = box.row()
            row.template_ID(self.materials[self.active_material_index], "material", new="material.new")


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


def hash_geometry(data) -> int:
    face_hash = get_polygon_data(data).tobytes()
    uv_hash = str(hash_uv_maps(data))
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
