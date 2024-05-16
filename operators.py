import bpy
import json

unit_multiplier = 100.0
light_multiplier = 1.0

table_justify = 40

instancing_nodes = ["MG_Scatter", "MG_Array"]
morph_nodes = ["MG_StairGenerator"]


def snake_case(string: str) -> str:
    return string.lower().replace(" ", "_")


def get_node_group_input(modifier, input_display_name):
    """Get a value by input display name"""
    node_group = modifier.node_group
    for field in node_group.inputs:
        if field.name == input_display_name:
            return modifier[field.identifier]
    return None


class Report():
    text = None
    specialities = dict()

    def init(self):
        self.text = bpy.data.texts.get('SyncReport')
        if not self.text:
            self.text = bpy.data.texts.new('SyncReport')
        else:
            self.text.clear()
        self.text.write(">BEGIN REPORT")
        self.nl(2)

    def message(self, text: str, nl_count=1, category='Log') -> None:
        if category == 'Log':
            self.text.write(text)
            self.nl(nl_count)
        else:
            if self.specialities.get(category):
                self.specialities[category].append(text + "\n")
            else:
                self.specialities[category] = [text + "\n"]

    def verdict(self):
        for key, value in zip(self.specialities.keys(), self.specialities.values()):
            self.text.write(">" + key.upper() + '\n')
            for val in value:
                self.text.write(val)
            self.nl(2)

    def nl(self, count=1):
        for _ in range(count):
            self.text.write("\n")


class SCENE_OP_DumpToJSON(bpy.types.Operator):
    """Dump scene to JSON"""
    bl_label = "Dump Scene to JSON"
    bl_idname = "scene.json_dump"
    bl_options = {'REGISTER', 'UNDO'}

    export_list = []
    reporter = Report()

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

    def add_to_export(self, obj, instanced=False) -> bool:
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

        # general object parameters
        container["OBJ_NAME"] = self.get_real_name(obj)
        container["OBJ_TYPE"] = obj.type

        if world_matrix:
            loc, rot, scale = world_matrix.decompose()
            rot = rot.to_euler()
            container["ORIGIN"] = "Scatter"
        else:
            loc, rot, scale = obj.location, obj.rotation_euler, obj.scale
            container["ORIGIN"] = "Placed"

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

                    elif mod.type in ['NODES', 'ARRAY']:
                        needs_export = True
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

        json_target.write(json.dumps(object_array, sort_keys=True, indent=4))
        self.reporter.message(f"INSTANCES COUNT: {global_count}", category="Final")
        self.reporter.verdict()
        return {'FINISHED'}
