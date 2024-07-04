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

        row = layout.row()
        op = row.operator("export_scene.assets", text="Export Assets", icon='ASSET_MANAGER')

        context.scene.stu_parameters.draw(layout)

        box = layout.box()
        row = box.row()
        row.label(text="Asset Quickies")
        row = box.row()
        op = row.operator("object.match_data_name", icon='OBJECT_DATA', text="Data=Obj")
        op.data_to_object = False
        op = row.operator("object.match_data_name", icon='MESH_DATA', text="Obj=Data")
        op.data_to_object = True

        row = box.row()
        op = row.operator("object.add_collection_name_prefix")

        row = box.row()
        op = row.operator("object.wrap_in_collection", text="Wrap in Collection")