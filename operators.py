import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty

import json
import os
from mathutils import Vector, Euler, Matrix
from math import pi

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

    export_list = []
    reporter = Report()

    write_meshes: BoolProperty(
        name="Write Meshes",
        default=True
    )
    write_all: BoolProperty(
        name="Write All",
        default=False
    )

    @classmethod
    def poll(cls, context):
        return True

    def debug_spawn(self, matrix):
        obj = bpy.data.objects.new("Test", None)
        bpy.context.scene.collection.objects.link(obj)
        obj.matrix_world = matrix

    def get_real_name(self, obj) -> str:
        if obj.data:
            return obj.data.name
        else:
            return obj.name

    def export_fbx(self):
        if self.export_list == 0:
            return

        bpy.ops.object.select_all(action='DESELECT')

        export_path = bpy.context.scene.stu_parameters.export_path
        if len(export_path) == 0:
            filepath = bpy.data.filepath
            export_path = os.path.dirname(filepath)
        else:
            export_path = bpy.path.abspath(export_path)

        # build ref list
        export_ref = []
        for data in self.export_list:
            if data.get('clean') is True and not self.write_all:
                continue
            else:
                self.reporter.message(f"{data.name.ljust(table_justify)} was dirty",
                                      category="Export-Filter")
            for obj in bpy.data.objects:
                if obj.data is data:
                    export_ref.append(obj)
                    break
            else:
                self.reporter.message(f"{data.name.ljust(table_justify)} -> failed to find obj ref",
                                      category="Export-Errors")
                continue
            data['clean'] = True

        for obj in export_ref:
            data = obj.data

            dummy = obj.copy()
            dummy.data = obj.data.copy()

            bpy.context.scene.collection.objects.link(dummy)
            dummy.select_set(True)

            bpy.context.view_layer.objects.active = dummy

            bpy.ops.object.convert(target='MESH')
            dummy.parent = None

            dummy.location = Vector((0.0, 0.0, 0.0))
            dummy.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
            dummy.scale = Vector((1.0, 1.0, 1.0))

            bpy.ops.export_scene.fbx(check_existing=False,
                                     filepath=export_path + "/" + data.name + ".fbx",
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
        return

    def add_to_export(self, obj) -> bool:
        if obj.type != 'MESH':
            return False

        # evaluated mesh data loses is_library_indirect flag, so we need to get the "real" mesh
        if obj.is_evaluated:
            data = bpy.data.meshes.get(obj.data.name)
            if not data:
                return False
        else:
            data = obj.data

        # only export unique mesh-data that is not linked from another blend (asset)
        if not data.is_library_indirect and data not in self.export_list:
            self.export_list.append(data)
            self.reporter.message(f"{obj.name.ljust(table_justify)} -> {data.name}", category="Export-List")
            return True
        else:
            return False

    def make_dict(self, obj, world_matrix=None) -> dict:
        container = dict()

        params = bpy.context.scene.stu_parameters

        # general object parameters
        container["OBJ_NAME"] = self.get_real_name(obj).replace('.', '_')
        container["OBJ_TYPE"] = obj.type

        if world_matrix:
            container["ORIGIN"] = "Scatter"
        else:
            world_matrix = obj.matrix_world
            container["ORIGIN"] = "Placed"

        loc = world_matrix.to_translation()
        rot = world_matrix.to_euler()
        scale = world_matrix.to_scale()

        for s in scale:
            if s > 0:
                continue
        else:
            scale_matrix_x = Matrix.Scale(scale[0], 3, (1.0, 0.0, 0.0))
            scale_matrix_y = Matrix.Scale(scale[1], 3, (0.0, 1.0, 0.0))
            scale_matrix_z = Matrix.Scale(scale[2], 3, (0.0, 0.0, 1.0))
            scale_matrix_all = scale_matrix_x @ scale_matrix_y @ scale_matrix_z
            rot = (rot.to_matrix() @ scale_matrix_all).to_euler()

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

            container["LIGHT_INTENSITY"] = obj.data.energy * light_multiplier

        return container

    def execute(self, context):
        self.reporter.init()
        self.export_list.clear()

        json_target = bpy.data.texts.get("JSON_base")
        if not json_target:
            json_target = bpy.data.texts.new("JSON_base")
        else:
            json_target.clear()

        object_array = []

        depsgraph = bpy.context.evaluated_depsgraph_get()
        global_count = 0

        for obj in context.visible_objects:
            is_instanced = False

            self.reporter.message(f"{obj.name}: Begin write")

            if obj.type not in ['MESH', 'LIGHT']:
                self.reporter.message(f"{obj.name}: unsupported type {obj.type}", category="Skip")
                continue

            if obj.name[:-4] not in obj.data.name or obj.data.name[:-4] not in obj.name:
                self.reporter.message(f"{obj.name.ljust(table_justify)} -> {obj.data.name}",
                                      category="Naming mismatch")
            if len(obj.data.name) > 4 and obj.data.name[-4] == '.' and obj.data.name[-3:].isnumeric():
                self.reporter.message(f"{obj.name.ljust(table_justify)} -> {obj.data.name}",
                                      category="Possible name dupes")

            if len(obj.modifiers) > 0:
                for mod in obj.modifiers:
                    if mod.type == 'NODES' and mod.node_group.name in instancing_nodes:
                        if get_node_group_input(mod, "Decal Mode"):
                            self.reporter.message(f"{obj.name} should be baked: Decal Mode",
                                                  category="Export")
                        elif get_node_group_input(mod, "Realize Instances"):
                            self.reporter.message(f"{obj.name} possibly should be baked: Realize Instances",
                                                  category="Export")
                        else:
                            is_instanced = True
                            self.reporter.message(f"{obj.name}: found {mod.node_group.name}")
                    else:
                        self.reporter.message(f"{obj.name}: {mod.name} was ignored")

            if is_instanced:
                # unwrap geonodes stack if present
                evaluated_obj = obj.evaluated_get(depsgraph)
                i = 0
                for inst in depsgraph.object_instances:
                    if inst.is_instance and inst.parent == evaluated_obj:
                        object_array.append(self.make_dict(inst.object, inst.matrix_world))
                        self.add_to_export(inst.object)
                        i += 1
                self.reporter.message(f"{obj.name}: Wrote {i} spawned instances", 2)
                if i == 0:
                    self.reporter.message(f"{obj.name}: 0 instances were written, check stack", category='Error')
                global_count += i
            else:
                object_array.append(self.make_dict(obj))
                self.add_to_export(obj)
                self.reporter.message(f"{obj.name}: Written as individual object", 2)

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

        self.reporter.message(f"INSTANCES COUNT: {global_count}", category="Final")

        if self.write_meshes:
            self.export_fbx()

        self.reporter.verdict()
        return {'FINISHED'}


last_state = {'OBJECT'}


@persistent
def edit_mode_handler(scene):
    if last_state != {bpy.context.mode}:
        last_state.pop()
        last_state.add(bpy.context.mode)
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                obj.data['clean'] = False


class SCENE_OP_StartHandler(bpy.types.Operator):
    """Toggle Handler to catch edit mode changes"""
    bl_label = "Toggle StU handler"
    bl_idname = "scene.stu_toggle_handler"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        if context.scene.get('stu_handler_loaded') is True or context.scene.get('stu_handler_loaded') is None:
            for h in bpy.app.handlers.depsgraph_update_post:
                if h.__name__ == 'edit_mode_handler':
                    bpy.app.handlers.depsgraph_update_post.remove(h)
            context.scene['stu_handler_loaded'] = False
        else:
            bpy.app.handlers.depsgraph_update_post.append(edit_mode_handler)
            context.scene['stu_handler_loaded'] = True
        return {'FINISHED'}
