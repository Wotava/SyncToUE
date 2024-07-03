import random

import bpy
import mathutils
from bpy.props import BoolProperty, FloatProperty

import json
import os
from mathutils import Vector, Euler, Matrix
from math import pi, sqrt, ceil

from .utils import hash_geometry, get_world_transform, copy_transform, select_smooth_faces

import numpy as np
import bmesh

unit_multiplier = 100.0
light_multiplier = 1.0

table_justify = 40

target_uv_names = ['UVMap', 'UVAtlas', 'UV_SlopePreset', 'UVInset']
panel_preset_attribute = 'panel_preset_index'
inset_uv_layer = target_uv_names[3]
slope_uv_layer = target_uv_names[2]
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

    write_meshes: BoolProperty(
        name="Write Meshes",
        default=True
    )

    @classmethod
    def poll(cls, context):
        if context.workspace.name != 'UV Editing':
            cls.poll_message_set(
                "Only works from default UV Editing workspace. Workspace should have exact name 'UV Editing' and contain UV area")
        return context.workspace.name == 'UV Editing'

    # expects already "prepared" object on input
    def export_fbx(self, obj, target='UE'):
        if target == 'UE' and not self.stu_params.bake_ue:
            return
        elif target != 'UE' and not self.stu_params.bake_houdini:
            return

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
            export_global_scale = 1

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

        loc, rot, scale = get_world_transform(obj)
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

    def duplicate_to_export(self, obj) -> None:
        geo_hash = hash_geometry(obj)
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
        copy_transform(obj, new_obj)

        self.export_scene.collection.objects.link(new_obj)
        return

    def execute(self, context):
        self.exported_variants.clear()
        self.hash_count.clear()
        self.hash_data.clear()
        object_array = []

        self.stu_params = context.scene.stu_parameters
        target_td = self.stu_params.target_density
        target_resolution = self.stu_params.target_resolution

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
            print(f"Processed {(i + 1) / t_len * 100:.2f}% of objects")

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

        # Write all important data to UV channels
        # Start with selecting smooth faces
        for obj in context.visible_objects:
            select_smooth_faces(obj, 1)

        # TODO switch to bmesh here
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_linked(delimit={'UV'})
        bpy.ops.object.mode_set(mode='OBJECT')

        print("Convert data to UV slots")
        for obj in context.visible_objects:
            uv_temp = np.zeros(len(obj.data.uv_layers[0].uv) * 2, dtype=float)

            if len(obj.data.uv_layers) > 0:
                obj.data.uv_layers.active_index = 0

            has_inset = obj.data.uv_layers.get(inset_uv_layer) is not None
            if has_inset:
                obj.data.uv_layers.get(inset_uv_layer).data.foreach_get('uv', uv_temp)

            if len(obj.data.uv_layers) < 4:
                for layer in target_uv_names[len(obj.data.uv_layers):]:
                    obj.data.uv_layers.new().name = layer

            for i, name in enumerate(target_uv_names):
                obj.data.uv_layers[i].name = name

            # restore inset if it was present
            if has_inset:
                obj.data.uv_layers.get(inset_uv_layer).data.foreach_set('uv', uv_temp)

            # refresh UVAtlas content
            obj.data.uv_layers[0].data.foreach_get('uv', uv_temp)
            obj.data.uv_layers[1].data.foreach_set('uv', uv_temp)
            obj.data.uv_layers.active_index = 1

            # write slope from selection
            preset_temp = np.zeros(len(obj.data.polygons), dtype=float)
            slope_uv = obj.data.uv_layers[slope_uv_layer]

            # get a list of selected
            slope_loops = [[*x.loop_indices] for x in obj.data.polygons if x.select]
            slope_loops = [y for x in slope_loops for y in x]  # extend a list of lists with nested comprehension

            clean_loops = [[*x.loop_indices] for x in obj.data.polygons if not x.select]
            clean_loops = [y for x in clean_loops for y in x]  # extend a list of lists with nested comprehension

            for i in slope_loops:
                uv_temp[i * 2] = -1

            for i in clean_loops:
                uv_temp[i * 2] = +1

            if obj.data.attributes.get(panel_preset_attribute):
                obj.data.attributes['panel_preset_index'].data.foreach_get('value', preset_temp)
                for polygon in obj.data.polygons:
                    preset_index = preset_temp[polygon.index]
                    loops = polygon.loop_indices
                    for loop in loops:
                        uv_temp[(loop * 2) + 1] = preset_index

            slope_uv.data.foreach_set("uv", uv_temp)
            del uv_temp

        # Deselect all un-important UVs by material (only building materials matter)
        # and calculate TD approx in the meantime
        print("Deselecting non-Atlas UV islands")
        for obj in context.visible_objects:
            me = obj.data
            bm = bmesh.new()
            bm.from_mesh(me)

            uv_layer = bm.loops.layers.uv.verify()

            for face in bm.faces:
                is_relevant = obj.material_slots[face.material_index].name in self.stu_params
                for loop in face.loops:
                    loop_uv = loop[uv_layer]
                    if is_relevant:
                        loop_uv.select = True
                        loop.vert.select_set(True)
                    else:
                        loop_uv.select = False
                        loop.vert.select_set(False)
                        loop_uv.uv = Vector((-1, -1))
            bm.to_mesh(me)
            bm.free()

        # override context to UV editor, try unwrap
        print(f"UV packing began")
        bpy.ops.object.mode_set(mode='EDIT')
        with bpy.context.temp_override(area=uv_area):
            self.export_scene.uvpm3_props.normalize_scale = True
            self.export_scene.uvpm3_props.margin = 0.006

            bpy.ops.uvpackmaster3.pack(mode_id="pack.single_tile", pack_to_others=False)
        print(f"UV packing done")
        bpy.ops.object.mode_set(mode='OBJECT')

        # calculate approx. texel density from a number of random objects
        print("Start TD calculation")
        i = 0
        max_samples = 10
        td_averages = np.zeros(max_samples, dtype=float)

        for obj in context.visible_objects:
            me = obj.data
            bm = bmesh.new()
            bm.from_mesh(me)

            uv_layer = bm.loops.layers.uv.verify()
            uv_areas = np.zeros(len(obj.data.polygons), dtype=float)

            for face in bm.faces:
                is_relevant = obj.material_slots[face.material_index].name in self.stu_params

                if is_relevant:
                    uv_verts = [loop[uv_layer].uv for loop in face.loops]
                    face_uv_area = mathutils.geometry.area_tri(uv_verts[0], uv_verts[1], uv_verts[2])
                    if len(uv_verts) == 4:
                        face_uv_area += mathutils.geometry.area_tri(uv_verts[0], uv_verts[2], uv_verts[3])
                    uv_areas[face.index] = sqrt(face_uv_area) / (sqrt(face.calc_area()) * 100) * 100

            if len(np.nonzero(uv_areas)[0]) > 0:
                td_averages[i] = np.average(uv_areas[np.nonzero(uv_areas)]) * target_resolution
                i += 1

            bm.to_mesh(me)
            bm.free()

            if i == max_samples:
                break
        if len(td_averages.nonzero()) != 0 and np.average(td_averages[td_averages.nonzero()]) > 0:
            texel_density_approx = np.average(td_averages[td_averages.nonzero()])
        else:
            self.report({'ERROR'}, "Failed to calculate texel density")
            return {'CANCELLED'}

        uv_multiplier = target_td / texel_density_approx

        # scale UVs and find V max to determine base offset for instances
        print("Calculate max UV values")
        max_v = 0
        max_u = 0
        for obj in context.visible_objects:
            uv_temp = np.zeros(len(obj.data.uv_layers[bake_atlas_layer].data) * 2, dtype=float)
            obj.data.uv_layers[bake_atlas_layer].data.foreach_get('uv', uv_temp)
            max_v = max(max_v, np.max(uv_temp[1::2]))
            max_u = max(max_u, np.max(uv_temp[::2]))

        max_v *= uv_multiplier
        max_u *= uv_multiplier

        max_u += self.stu_params.internal_padding
        max_v += self.stu_params.internal_padding

        # squeeze more instances in one UDIM
        squeezed_v_mult = 1 / ceil(1 / max_v) / max_v
        squeezed_u_mult = 1 / ceil(1 / max_u) / max_u

        global_multiplier = uv_multiplier
        global_multiplier *= min(squeezed_u_mult, squeezed_v_mult)

        max_u = max_u / uv_multiplier * global_multiplier
        max_v = max_v / uv_multiplier * global_multiplier

        print(f"Requested TD {target_td}, actual TD {target_td / uv_multiplier * global_multiplier}")
        # calc amount of horizontal udims
        horizontal_udims = int(1 / max_u)
        print(f"Horizontal UDIMs: {horizontal_udims}")

        print("Start UV scaling")
        center = (0.0, 0.0)
        S = Matrix.Diagonal((global_multiplier, global_multiplier))
        for obj in context.visible_objects:
            if obj.name[-2:] == '_0':
                uv = obj.data.uv_layers.get(bake_atlas_layer)
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
                v_offset = max_v * ((int(obj.name.split('_')[-1])) // horizontal_udims)
                u_offset = max_u * ((int(obj.name.split('_')[-1])) % horizontal_udims)
                offset = Vector((u_offset, v_offset))

                if self.stu_params.check_obj(obj):
                    target_uv = obj.data.uv_layers.get(bake_atlas_layer)
                    uv_temp = np.zeros(len(target_uv.data) * 2, dtype=float)
                    target_uv.data.foreach_get('uv', uv_temp)
                    uv_temp[1::2] += v_offset
                    uv_temp[::2] += u_offset
                    target_uv.data.foreach_set('uv', uv_temp)
                else:
                    me = obj.data
                    bm = bmesh.new()
                    bm.from_mesh(me)
                    uv_layer = bm.loops.layers.uv.verify()

                    for face in bm.faces:
                        is_relevant = obj.material_slots[face.material_index].name in self.stu_params
                        for loop in face.loops:
                            loop_uv = loop[uv_layer]
                            if is_relevant:
                                loop_uv.uv += offset

                    bm.to_mesh(me)
                    bm.free()

        print("Large scene export")
        for obj in context.visible_objects:
            if obj.scale.x < 0 or obj.scale.y < 0 or obj.scale.z < 0:
                # obj.data = obj.data.copy()
                mat = obj.matrix_local
                mat_scale = Matrix.LocRotScale(None, None, mat.decompose()[2])
                obj.data.transform(mat_scale)
                obj.scale = 1, 1, 1
                obj.data.flip_normals()
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
        else:
            json_disk = open(bpy.path.abspath(json_path), "a")

        json_disk.truncate(0)
        json_disk.write(json_target.as_string())
        json_disk.close()

        self.report({'INFO'}, "Finished as expected. UNDO now to return to previous state")
        return {'FINISHED'}


class SCENE_OP_ValidateUVs(bpy.types.Operator):
    """Check all UVs and print report"""
    bl_label = "Validate UVs"
    bl_idname = "scene.validate_uvs"
    bl_options = {'REGISTER', 'UNDO'}

    hashes = {}
    reported = []

    tolerance: FloatProperty(
        name="Zero-area tolerance",
        default=0.001
    )

    max_allowed_zeros: FloatProperty(
        name="Max allowed",
        description="Maximum percent of zero-area UV faces allowed",
        default=10.0
    )

    @classmethod
    def poll(cls, context):
        return True

    def check_uv(self, obj) -> int:
        if obj.type != 'MESH' or not obj.data:
            return -1

        # calc hash
        container = np.zeros(len(obj.data.uv_layers[0].uv.values()) * 2, dtype=float)
        obj.data.uv_layers[0].uv.foreach_get("vector", container)
        hashed_uv = hash(container.tobytes())
        if hashed_uv in self.hashes:
            return self.hashes.get(hashed_uv)

        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()
        uv_areas = np.zeros(len(obj.data.polygons), dtype=float)

        for face in bm.faces:
            uv_verts = [loop[uv_layer].uv for loop in face.loops]
            face_uv_area = mathutils.geometry.area_tri(uv_verts[0], uv_verts[1], uv_verts[2])
            if len(uv_verts) == 4:
                face_uv_area += mathutils.geometry.area_tri(uv_verts[0], uv_verts[2], uv_verts[3])
            uv_areas[face.index] = sqrt(face_uv_area) / (sqrt(face.calc_area()) * 100) * 100

        bm.to_mesh(me)
        bm.free()

        zero_faces = len(uv_areas) - len(np.where(uv_areas > 0 + self.tolerance)[0])

        self.hashes.update({hashed_uv: zero_faces})

        return zero_faces

    def execute(self, context):
        self.hashes.clear()

        if bpy.data.scenes.get('UV_Cleanup'):
            cleanup_scene = bpy.data.scenes.get('UV_Cleanup')
            unlink = list(cleanup_scene.collection.objects)
            for obj in unlink:
                cleanup_scene.collection.objects.unlink(obj)
        else:
            cleanup_scene = bpy.data.scenes.new('UV_Cleanup')

        depsgraph = context.evaluated_depsgraph_get()

        report_target = bpy.data.texts.get("UV_Report")
        if not report_target:
            report_target = bpy.data.texts.new("UV_Report")
        else:
            report_target.clear()

        for obj in context.visible_objects:
            has_header = False
            if obj.type not in ['MESH']:
                continue

            evaluated_obj = obj.evaluated_get(depsgraph)

            for inst in depsgraph.object_instances:
                if inst.is_instance and inst.parent == evaluated_obj:
                    if inst.object.type == 'MESH' and inst.object.data and len(inst.object.data.uv_layers):
                        zero_area_faces = self.check_uv(inst.object)
                        percentile = zero_area_faces / len(inst.object.data.polygons) / 0.01
                        if percentile > self.max_allowed_zeros and inst.object.data.name not in self.reported:
                            if not has_header:
                                report_target.write(f"\n{obj.name} instances: \n")
                                has_header = True

                            report_target.write(f">>{inst.object.name}({inst.object.data.name}) has "
                                                f"{percentile:.2f}% zero-area faces in UV \n")
                            for mod in inst.object.modifiers:
                                if mod.type == 'SOLIDIFY':
                                    report_target.write(f"  Possibly due to solidify modifier, can be ignored\n")
                                    break

                            cleanup_scene.collection.objects.link(bpy.data.objects[inst.object.name])
                            self.reported.append(inst.object.data.name)

            if evaluated_obj.data and len(evaluated_obj.data.polygons) > 0:
                zero_area_faces = self.check_uv(evaluated_obj)
                percentile = zero_area_faces / len(evaluated_obj.data.polygons) / 0.01
                if percentile > self.max_allowed_zeros and obj.data.name not in self.reported:
                    if not has_header:
                        report_target.write(f"\n{obj.name} instances: \n")

                    report_target.write(f">>{evaluated_obj.name}({evaluated_obj.data.name}) evaluated has "
                                        f"{percentile:.2f}% zero-faces in UV \n")
                    for mod in evaluated_obj.modifiers:
                        if mod.type == 'SOLIDIFY':
                            report_target.write(f"  Possibly due to solidify modifier, can be ignored\n")
                            break

                    cleanup_scene.collection.objects.link(bpy.data.objects[evaluated_obj.name])
                    self.reported.append(obj.data.name)

        context.window.scene = cleanup_scene
        return {'FINISHED'}


class SCENE_OP_AddAtlasMaterialSlot(bpy.types.Operator):
    """Add Material Slot to Atlas Materials List"""
    bl_label = "Add Slot to Atlas Material List"
    bl_idname = "scene.add_atlas_material_slot"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        context.scene.stu_parameters.add_material_slot()
        return {'FINISHED'}


class SCENE_OP_RemoveAtlasMaterialSlot(bpy.types.Operator):
    """Remove Material Slot to Atlas Materials List"""
    bl_label = "Remove Slot to Atlas Material List"
    bl_idname = "scene.remove_atlas_material_slot"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        slots = len(context.scene.stu_parameters.materials)
        return slots > 0 and (-1 < context.scene.stu_parameters.active_material_index < slots)

    def execute(self, context):
        context.scene.stu_parameters.remove_material_slot()
        return {'FINISHED'}


class OBJECT_OP_MatchDataNames(bpy.types.Operator):
    """Match Data-block name with Object name (both ways)"""
    bl_label = "Match Object/Data Names"
    bl_idname = "object.match_data_name"
    bl_options = {'REGISTER', 'UNDO'}

    data_to_object: BoolProperty(
        name="Copy from OBJ to DATA",
        default=False
    )
    only_meshes: BoolProperty(
        name="Only Meshes",
        default=False
    )

    @classmethod
    def poll(cls, context):
        return len(context.visible_objects)

    def execute(self, context):
        if len(context.selected_objects) > 0:
            targets = context.selected_objects
        else:
            targets = context.visible_objects

        for obj in targets:
            if self.only_meshes and obj.type != 'MESH':
                continue

            if obj.data:
                if self.data_to_object:
                    obj.name = obj.data.name
                else:
                    obj.data.name = obj.name
        return {'FINISHED'}


class OBJECT_OP_AddCollectionNamePrefix(bpy.types.Operator):
    """Add Collection name prefix to objects"""
    bl_label = "Add Collection Prefix to Name"
    bl_idname = "object.add_collection_name_prefix"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.visible_objects)

    def execute(self, context):
        if len(context.selected_objects) > 0:
            targets = context.selected_objects
        else:
            targets = context.visible_objects

        for obj in targets:
            obj.name = f"{obj.users_collection[0].name}_{obj.name}"

        return {'FINISHED'}


class OBJECT_OP_WrapInCollection(bpy.types.Operator):
    """Wrap this object (or objects) in collection with the name of Active Object"""
    bl_label = "Wrap in Collection with Same Name"
    bl_idname = "object.wrap_in_collection"
    bl_options = {'REGISTER', 'UNDO'}

    per_object: BoolProperty(
        name="Wrap Individually",
        default=False
    )

    @classmethod
    def poll(cls, context):
        return context.active_object

    def execute(self, context):
        if len(context.selected_objects) > 1:
            targets = context.selected_objects
        else:
            targets = [context.active_object]

        if self.per_object:
            for obj in targets:
                base_collection = obj.users_collection[0]
                base_collection.objects.unlink(obj)
                new_collection = bpy.data.collections.new(obj.name)
                base_collection.children.link(new_collection)
                new_collection.objects.link(obj)

        else:
            base_collection = context.active_object.users_collection[0]
            new_collection = bpy.data.collections.new(context.active_object.name)
            base_collection.children.link(new_collection)

            for obj in targets:
                base_collection.objects.unlink(obj)
                new_collection.objects.link(obj)

        return {'FINISHED'}