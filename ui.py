import bpy

class VIEW3D_PT_ModifierManager(bpy.types.Panel):
    bl_label = "Sync to UE"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'StU'

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        op = row.operator("scene.json_dump", text="Dump and Export")
        op.write_meshes = True

        row = layout.row()
        op = row.operator("scene.json_dump", text="Dump only")
        op.write_meshes = False

        row = layout.row()
        row.prop(context.scene.stu_parameters, "flip_loc")

        row = layout.row()
        row.prop(context.scene.stu_parameters, "flip_rot")

        row = layout.row()
        row.prop(context.scene.stu_parameters, "adjust_rot")

        row = layout.row()
        row.prop(context.scene.stu_parameters, "rotation_offset")

        row = layout.row()
        row.prop(context.scene.stu_parameters, "flip_scale")

        row = layout.row()
        row.prop(context.scene.stu_parameters, "abs_scale")

        row = layout.row()
        row.prop(context.scene.stu_parameters, "swizzle_neg_scale")