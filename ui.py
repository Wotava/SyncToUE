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

        context.scene.stu_parameters.draw(layout)
