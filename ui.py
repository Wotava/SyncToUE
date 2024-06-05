import bpy


class DATA_UL_AtlasMaterials(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        slot = item
        mat = slot.material
        row = layout.row()
        col = row.column(align=True)

        if mat is not None:
            col.prop(slot.material, "name", text="", emboss=False, icon='MATERIAL')
        else:
            col.label(text="EMPTY SLOT", icon='QUESTION')
        return



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
