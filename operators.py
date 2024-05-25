import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty

import json
import os
from mathutils import Vector, Euler, Matrix
from math import pi

import numpy as np

unit_multiplier = 100.0
light_multiplier = 1.0

table_justify = 40

instancing_nodes = ["MG_Scatter", "MG_Array"]
morph_nodes = ["MG_StairGenerator"]

last_mode = 'OBJECT'


def snake_case(string: str) -> str:
    return string.lower().replace(" ", "_")


def get_node_group_input(modifier, input_display_name):
    """Get a value by input display name"""
    node_group = modifier.node_group
    for field in node_group.inputs:
        if field.name == input_display_name:
            return modifier[field.identifier]
    return None


def get_modifier_pseudohash(mod) -> str:
    values = ""
    if mod.type == 'NODES':
        for prop in mod.keys():
            if prop.find('attribute') == -1:
                values += str(mod.get(prop))
    else:
        properties = [p.identifier for p in list(mod.bl_rna.properties) if not p.is_readonly]
        for prop in properties:
            values += str(getattr(mod, prop))
    return values


def hash_modifier_stack(obj) -> int:
    stack = ""
    for mod in obj.modifiers:
        stack += get_modifier_pseudohash(mod)
    return hash(stack)


class Report:
    __text = None
    __specialities = dict()

    def init(self):
        self.__text = bpy.data.texts.get('SyncReport')
        if not self.__text:
            self.__text = bpy.data.texts.new('SyncReport')
        else:
            self.__text.clear()
        self.__text.write(">BEGIN REPORT")
        self.nl(2)

    def add_category(self, category: str):
        if not self.__specialities.get(category):
            self.__specialities.update({category: []})

    def message(self, text: str, nl_count=1, category='Log') -> None:
        if category == 'Log':
            self.__text.write(text)
            self.nl(nl_count)
        else:
            self.add_category(category)
            self.__specialities.get(category).append(text + "\n")

    def verdict(self):
        for key, value in zip(self.__specialities.keys(), self.__specialities.values()):
            self.__text.write(">" + key.upper() + '\n')
            for val in value:
                self.__text.write(val)
            self.nl(2)
        self.__specialities.clear()

    def nl(self, count=1):
        for _ in range(count):
            self.__text.write("\n")


class SCENE_OP_DumpToJSON(bpy.types.Operator):
    """Dump scene to JSON"""
    bl_label = "Dump Scene to JSON"
    bl_idname = "scene.json_dump"
    bl_options = {'REGISTER', 'UNDO'}

    exported_variants = []
    data_variants = {}

    hash_data = {}
    hash_count = {}

    export_scene = None

    write_meshes: BoolProperty(
        name="Write Meshes",
        default=True
    )

    @classmethod
    def poll(cls, context):
        return True

    # data_pack: [data.name, geo_hash]
    # this data allows unique identification of instances
    def get_variation_index(self, data_pack) -> int:
        variants = self.data_variants.get(data_pack[0])
        if not variants:
            self.data_variants.update({data_pack[0]: [data_pack[1]]})
            return 0
        else:
            if data_pack[1] not in variants:
                variants.append(data_pack[1])
                return len(variants) - 1
            else:
                return variants.index(data_pack[1])

    def get_polygon_data(self, data) -> np.ndarray:
        """Returns flat list of all face positions extended with face areas"""
        centers = np.zeros(len(data.polygons) * 3, dtype=float)
        areas = np.zeros(len(data.polygons), dtype=float)
        data.polygons.foreach_get("center", centers)
        data.polygons.foreach_get("area", areas)
        return np.concatenate((centers, areas))

    def hash_uv_maps(self, data):
        """Hash UV coordinates of all UV maps and return list of hashes"""
        if len(data.uv_layers) == 0:
            return 0

        hashes = []
        container = np.zeros(len(data.uv_layers[0].uv.values()) * 2, dtype=float)
        for uvmap in data.uv_layers:
            uvmap.uv.foreach_get("vector", container)
            hashes.append(hash(container.tobytes()))

        return hashes

    def hash_geometry(self, obj) -> int:
        face_hash = self.get_polygon_data(obj.data).tobytes()
        uv_hash = str(self.hash_uv_maps(obj.data))
        return hash(tuple([face_hash, uv_hash]))

    # expects already "prepared" object on input
    def export_fbx(self, obj, name=None):
        if not obj:
            raise Exception
        bpy.ops.object.select_all(action='DESELECT')

        export_path = bpy.context.scene.stu_parameters.export_path
        if len(export_path) == 0:
            filepath = bpy.data.filepath
            export_path = os.path.dirname(filepath)
        else:
            export_path = bpy.path.abspath(export_path)

        if obj.is_evaluated:
            data = bpy.data.meshes.new_from_object(obj)
            dummy = bpy.data.objects.new(data.name, data)
        else:
            data = obj.data
            dummy = obj.copy()
            dummy.data = obj.data.copy()

        bpy.context.scene.collection.objects.link(dummy)
        dummy.select_set(True)

        bpy.context.view_layer.objects.active = dummy

        if not obj.is_evaluated:
            bpy.ops.object.convert(target='MESH')

        dummy.parent = None
        dummy.location = Vector((0.0, 0.0, 0.0))
        dummy.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
        dummy.scale = Vector((1.0, 1.0, 1.0))

        if not name:
            name = data.name
        bpy.ops.export_scene.fbx(check_existing=False,
                                 filepath=export_path + "/" + name + ".fbx",
                                 filter_glob="*.fbx",
                                 use_selection=True,
                                 object_types={'MESH'},
                                 bake_space_transform=True,
                                 mesh_smooth_type='OFF',
                                 add_leaf_bones=False,
                                 path_mode='ABSOLUTE',
                                 axis_forward='X',
                                 axis_up='Z',
                                 apply_unit_scale=True,
                                 apply_scale_options='FBX_SCALE_NONE',
                                 global_scale=1.0,
                                 use_triangles=False
                                 )
        bpy.data.objects.remove(dummy)

    def try_export(self, obj, geo_hash, name) -> bool:
        if obj.type != 'MESH':
            return False

        # evaluated mesh data loses is_library_indirect flag, so we need to get the "real" mesh
        if obj.is_evaluated:
            if hasattr(obj, 'original'):
                data = bpy.data.meshes.get(obj.original.data.name)
            else:
                data = bpy.data.meshes.get(obj.data.name)

            if not data:
                return False
        else:
            data = obj.data

        # only export unique mesh-data that is not linked from another blend (asset)
        # TODO calc delta between original "linked" data and inpit to catch unique variations of assets
        if not data.is_library_indirect and [data, geo_hash] not in self.exported_variants:

            if self.write_meshes:
                self.export_fbx(obj, name)
                self.exported_variants.append([data, geo_hash])

            return True
        else:
            return False

    def make_obj_name(self, mesh_data, count) -> str:
        base_name = mesh_data.name
        if base_name[:3] != 'SM_':
            base_name = 'SM_' + base_name
        base_name = base_name.replace('.', '_')
        if count > 0:
            base_name += f"_{count}"
        return base_name

    def make_dict(self, obj, name=None) -> dict:
        container = dict()

        params = bpy.context.scene.stu_parameters

        # general object parameters
        if not name:
            name = obj.name

        container["OBJ_NAME"] = name
        container["OBJ_TYPE"] = obj.type

        loc, rot, scale = self.get_world_transform(obj)
        rot = list(rot[0:3])

        swizzle_rot = False
        for i in range(3):
            loc[i] = loc[i] * (1 - 2 * params.flip_loc[i])
            rot[i] = rot[i] + params.rotation_offset[i]
            rot[i] = rot[i] * (1 - 2 * params.flip_rot[i])
            scale[i] = scale[i] * (1 - 2 * params.flip_scale[i])

            if scale[i] < 0:
                swizzle_rot = True

            if params.abs_scale[i]:
                scale[i] = abs(scale[i])

            if params.adjust_rot:
                rot[i] = -1 * pi + abs(rot[i])

        if swizzle_rot and params.swizzle_neg_scale:
            rot.reverse()

        container["LOCATION"] = (loc * unit_multiplier).to_tuple()
        container["ROTATION"] = rot[0:3]
        container["SCALE"] = scale.to_tuple()

        # light shenanigans
        if obj.type == 'LIGHT':
            container["LIGHT_COLOR"] = obj.data.color[0:3]

            if obj.data.type == 'POINT':
                container["LIGHT_TYPE"] = 0
            elif obj.data.type == 'SUN':
                container["LIGHT_TYPE"] = 1
            elif obj.data.type == 'SPOT':
                container["LIGHT_TYPE"] = 2
            else:
                container["LIGHT_TYPE"] = 3

            container["LIGHT_RADIUS"] = obj.data.shadow_soft_size

            if obj.data.type == 'AREA':
                container["LIGHT_SIZE"] = obj.data.size
            elif obj.data.type == 'SPOT':
                container["LIGHT_SIZE"] = obj.data.spot_size

            container["LIGHT_INTENSITY"] = obj.data.energy / (4 * pi)

        return container

    def get_world_transform(self, obj):
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

    def copy_transform(self, source, target) -> None:
        s_loc, s_rot, s_scale = self.get_world_transform(source)
        target.location = s_loc
        target.scale = s_scale
        target.rotation_euler = s_rot

    def duplicate_to_export(self, obj) -> None:
        geo_hash = self.hash_geometry(obj)
        if geo_hash not in self.hash_data:
            # make evaluated copy mesh data
            new_mesh = bpy.data.meshes.new_from_object(obj)
            self.hash_data.update({geo_hash: new_mesh})
            self.hash_count.update({geo_hash: 0})
        else:
            # make linked copy of already evaluated and realised mesh data
            # and increment counter
            new_mesh = self.hash_data.get(geo_hash)
            self.hash_count[geo_hash] += 1

        # link to export scene, generate name accordingly
        # write object custom property with count for later use with UV offset
        count = self.hash_count.get(geo_hash)
        name = self.make_obj_name(new_mesh, count)
        new_obj = bpy.data.objects.new(name, new_mesh)
        new_obj['instance_count'] = count
        self.copy_transform(obj, new_obj)

        self.export_scene.collection.objects.link(new_obj)
        return

    def execute(self, context):
        self.exported_variants.clear()

        json_target = bpy.data.texts.get("JSON_base")
        if not json_target:
            json_target = bpy.data.texts.new("JSON_base")
        else:
            json_target.clear()

        object_array = []

        depsgraph = bpy.context.evaluated_depsgraph_get()

        if bpy.data.scenes.get('ExportScene'):
            self.export_scene = bpy.data.scenes.get('ExportScene')
        else:
            self.export_scene = bpy.data.scenes.new('ExportScene')

        # PASS 1: make copy of current scene with all objects without modifiers
        for obj in context.visible_objects:
            if obj.type not in ['MESH', 'LIGHT']:
                continue

            if obj.type == 'LIGHT':
                object_array.append(self.make_dict(obj))
                continue

            evaluated_obj = obj.evaluated_get(depsgraph)

            for inst in depsgraph.object_instances:
                if inst.is_instance and inst.parent == evaluated_obj:
                    if inst.object.type == 'MESH' and inst.object.data and len(inst.object.data.polygons) > 1:
                        self.duplicate_to_export(inst.object)

                    elif inst.object.type == 'LIGHT':
                        # we don't export lights, just write it straight to json and forget
                        object_array.append(self.make_dict(inst.object, inst.object.name))

            if evaluated_obj.data and len(evaluated_obj.data.polygons) > 0:
                self.duplicate_to_export(evaluated_obj)

        return {'FINISHED'}

        json_target.write("{\"array\":")
        json_target.write(json.dumps(object_array, sort_keys=True, indent=4))
        json_target.write("}")

        json_path = context.scene.stu_parameters.json_path
        if len(json_path) == 0:
            filepath = bpy.data.filepath
            directory = os.path.dirname(filepath)
            json_disk = open(directory + "\\level_data.json", "a")
            self.report({'INFO'}, "JSON was exported in .blend directory")
        else:
            json_disk = open(bpy.path.abspath(json_path), "a")

        json_disk.truncate(0)
        json_disk.write(json_target.as_string())
        json_disk.close()

        return {'FINISHED'}
