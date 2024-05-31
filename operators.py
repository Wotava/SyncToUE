import random

import bpy
import mathutils
from bpy.props import BoolProperty, IntProperty

import json
import os
from mathutils import Vector, Euler, Matrix
from math import pi, sqrt

import numpy as np

unit_multiplier = 100.0
light_multiplier = 1.0

table_justify = 40

target_uv_names = ['UVMap', 'UVAtlas', 'UV_SlopePreset', 'UVInset']
bake_atlas_layer = target_uv_names[1]


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

    hash_data = {}
    hash_count = {}

    export_scene = None

    target_UV = target_uv_names[0]

    write_meshes: BoolProperty(
        name="Write Meshes",
        default=True
    )

    @classmethod
    def poll(cls, context):
        if context.workspace.name != 'UV Editing':
            cls.poll_message_set("Only works from default UV Editing workspace. Workspace should have exact name 'UV Editing' and contain UV area")
        return context.workspace.name == 'UV Editing'

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
    def export_fbx(self, obj, target='UE'):
        if not obj and target == 'UE':
            raise Exception

        if not self.write_meshes:
            return

        export_path = bpy.context.scene.stu_parameters.export_path
        if len(export_path) == 0:
            filepath = bpy.data.filepath
            export_path = os.path.dirname(filepath)
        else:
            export_path = bpy.path.abspath(export_path)

        if target == 'UE':
            bpy.ops.object.select_all(action='DESELECT')

            export_path += '\\UnrealEngine'
            name = obj.name
            dummy = obj.copy()
            bpy.context.scene.collection.objects.link(dummy)
            dummy.select_set(True)

            bpy.context.view_layer.objects.active = dummy

            dummy.parent = None
            dummy.location = Vector((0.0, 0.0, 0.0))
            dummy.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
            dummy.scale = Vector((1.0, 1.0, 1.0))

            export_scale = 'FBX_SCALE_NONE'
            export_global_scale = 1.0
        else:
            bpy.ops.object.select_all(action='SELECT')
            export_path += '\\Houdini'
            name = bpy.path.basename(bpy.context.blend_data.filepath)
            export_scale = 'FBX_SCALE_UNITS'
            export_global_scale = 0.01

        is_exist = os.path.exists(export_path)
        if not is_exist:
            os.makedirs(export_path)

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
                                 apply_scale_options=export_scale,
                                 global_scale=export_global_scale,
                                 use_triangles=False
                                 )

        if target == 'UE':
            bpy.data.objects.remove(dummy)

    def make_obj_name(self, mesh_data, count) -> str:
        base_name = mesh_data.name
        if base_name[:3] != 'SM_':
            base_name = 'SM_' + base_name
        base_name = base_name.replace('.', '_')
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
        self.hash_count.clear()
        self.hash_data.clear()
        object_array = []

        target_td = context.scene.stu_parameters.target_density

        json_target = bpy.data.texts.get("JSON_base")
        if not json_target:
            json_target = bpy.data.texts.new("JSON_base")
        else:
            json_target.clear()

        depsgraph = bpy.context.evaluated_depsgraph_get()

        if bpy.data.scenes.get('ExportScene'):
            self.export_scene = bpy.data.scenes.get('ExportScene')
        else:
            self.export_scene = bpy.data.scenes.new('ExportScene')

        # PASS 1: make copy of current scene with all objects without modifiers
        t_len = len(context.visible_objects)
        print(f"Pass 1 start")
        for i, obj in enumerate(context.visible_objects):
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
            print(f"Processed {(i+1)/t_len * 100:.2f}% of objects")

        # PASS 2: pack all geometry, set texel density
        # switch scene to export and context to uv-view/3d-view by the name 'UV Editing'
        print(f"Pass 2 start")
        context.window.scene = self.export_scene
        bpy.ops.object.select_all(action='SELECT')

        context.view_layer.objects.active = context.selected_objects[0]

        if context.workspace.name != 'UV Editing':
            self.report({'ERROR'}, "Please open default UV Editing workspace with exact name 'UV Editing'")
            return {'CANCELLED'}

        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                uv_area = area
                break
        else:
            self.report({'ERROR'}, "Failed to find UV area, reset workspace")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        # Check if UVAtlas is available and ensure all UVs are in place
        print("Ensure UV slots")
        for obj in context.visible_objects:
            if len(obj.data.uv_layers) > 0:
                obj.data.uv_layers.active_index = 0

            if len(obj.data.uv_layers) < 4:
                for layer in target_uv_names[len(obj.data.uv_layers):]:
                    obj.data.uv_layers.new().name = layer
            else:
                for i, name in enumerate(target_uv_names):
                    obj.data.uv_layers[i].name = name

            uv_temp = np.zeros(len(obj.data.uv_layers[0].uv.data.uv), dtype=float)

            # refresh UVAtlas content
            obj.data.uv_layers[0].uv.data.uv.foreach_get('uv', uv_temp)
            obj.data.uv_layers[1].uv.data.uv.foreach_set('uv', uv_temp)
            obj.data.uv_layers.active_index = 1
            del uv_temp

        # override context to UV editor, try unwrap
        print(f"UV packing began")
        with bpy.context.temp_override(area=uv_area):
            bpy.ops.uv.select_all(action='SELECT')

            self.export_scene.uvpm3_props.normalize_scale = True
            bpy.ops.uvpackmaster3.pack(mode_id="pack.single_tile", pack_to_others=False)
        print(f"UV packing done")
        bpy.ops.object.mode_set(mode='OBJECT')

        # calculate approx. texel density from a random object
        td_target = context.visible_objects[random.randint(0, len(context.visible_objects))]
        uv_area = np.zeros(len(td_target.data.polygons), dtype=float)
        for polygon in td_target.data.polygons:
            loops = polygon.loop_indices
            if len(loops) > 4:
                continue
            else:
                uv_verts = [td_target.data.uv_layers.active.uv.data.uv[x].vector for x in loops]
            face_uv_area = 0
            face_uv_area += mathutils.geometry.area_tri(uv_verts[0], uv_verts[1], uv_verts[2])
            if len(uv_verts) == 4:
                face_uv_area += mathutils.geometry.area_tri(uv_verts[0], uv_verts[2], uv_verts[3])
            uv_area[polygon.index] = sqrt(face_uv_area) / (sqrt(polygon.area) * 100) * 100
        texel_density_approx = np.average(uv_area) * 4096

        uv_multiplier = target_td / texel_density_approx

        # scale UVs and find V max to determine base offset for instances
        print("Start UV scaling")
        max_v = uv_multiplier
        center = (0.0, 0.0)
        S = Matrix.Diagonal((uv_multiplier, uv_multiplier))
        for obj in context.visible_objects:
            if obj.name[-2:] == '_0':
                uv = obj.data.uv_layers[self.target_UV]
                uv_temp = np.zeros(len(uv.data) * 2, dtype=float)
                uv.data.foreach_get('uv', uv_temp)
                scaled_uv = np.dot(uv_temp.reshape((-1, 2)) - center, S) + center
                uv.data.foreach_set('uv', scaled_uv.ravel())

        # PASS 3: export objects to UE/Houdini folders and write JSON

        # I can't get "fresh" mesh data blocks from here for some reason if I don't switch back and forth
        # between edit and object modes after making all objects single-user
        # So first I write JSON and make mesh data single user, then switch context mode and then offset UVs
        print(f"Pass 3 start")
        for obj in bpy.context.visible_objects:
            object_array.append(self.make_dict(obj, obj.data.name))

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True, material=False,
                                        animation=False, obdata_animation=True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()

        print(f"Begin export")
        t_len = len(self.hash_count)
        i = 0
        for obj in bpy.context.visible_objects:
            if obj.name[-2:] == '_0':
                # export as unique
                self.export_fbx(obj, target='UE')
                print(f"Export progress: {(i + 1) / t_len * 100:.1f}% of unique objects exported")
                i += 1
            else:
                target_uv = obj.data.uv_layers.get(self.target_UV)
                if not target_uv:
                    self.report({'ERROR'}, "UVAtlas not found, this should NOT happen ever")
                    return {'CANCELLED'}

                offset = max_v * (int(obj.name.split('_')[-1]))
                uv_temp = np.zeros(len(target_uv.data) * 2, dtype=float)
                target_uv.data.foreach_get('uv', uv_temp)
                uv_temp[1::2] += offset
                target_uv.data.foreach_set('uv', uv_temp)

        print("Large scene export")
        self.export_fbx(None, target='Houdini')
        print("Finish export")

        # PASS 4: fill and save internal and on-disk JSON files
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
