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

        box = layout.box()
        box.label(text="JSON", icon='TEXT')
        row = box.row()
        op = row.operator("scene.json_dump", text="Scene")
        op.write_meshes = False
        op = row.operator("scene.json_dump", text="Scene+FBX")
        op.write_meshes = True
        row = box.row()
        op = row.operator("material.json_dump", text="Materials")

        box = layout.box()
        box.label(text="Asset Tools", icon='ASSET_MANAGER')
        row = box.row()
        op = row.operator("export_scene.assets", text="Export Assets", icon='EXPORT')

        row = box.row()
        op = row.operator("ed.fix_assets", text="Fix Previews")
        op = row.operator("ed.update_asset_previews", text="Update Previews")

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
        op = row.operator("object.add_collection_name_prefix", text="Col. Prefix")
        op = row.operator("object.add_collection_name_prefix", text="Col. Name")
        op.set_full_name = True

        row = box.row()
        op = row.operator("object.wrap_in_collection", text="Wrap in Collection")
        op = row.operator("object.make_base_collection", text="Make Base")

        # Morphs
        box = layout.box()
        box.operator_context = 'EXEC_DEFAULT'
        row = box.row()
        row.label(text="Morph Quickies")
        
        row = box.row()
        op = row.operator("transform.translate", text="X -0.5")
        op.value = (-0.5, 0, 0)
        op = row.operator("transform.translate", text="X 0.5")
        op.value = (0.5, 0, 0)

        row = box.row()
        op = row.operator("transform.translate", text="X -0.25")
        op.value = (-0.25, 0, 0)
        op = row.operator("transform.translate", text="X 0.25")
        op.value = (0.25, 0, 0)

        row = box.row()
        op = row.operator("transform.translate", text="X -0.125")
        op.value = (-0.125, 0, 0)
        op = row.operator("transform.translate", text="X 0.125")
        op.value = (0.125, 0, 0)

        row = box.row()
        op = row.operator("transform.translate", text="Y -0.5")
        op.value = (0, -0.5, 0)
        op = row.operator("transform.translate", text="Y 0.5")
        op.value = (0, 0.5, 0)

        row = box.row()
        op = row.operator("transform.translate", text="Y -0.25")
        op.value = (0, -0.25, 0)
        op = row.operator("transform.translate", text="Y 0.25")
        op.value = (0, 0.25, 0)

        row = box.row()
        op = row.operator("transform.translate", text="Y -0.125")
        op.value = (0, -0.125, 0)
        op = row.operator("transform.translate", text="Y 0.125")
        op.value = (0, 0.125, 0)

        row = box.row()
        op = row.operator("transform.translate", text="Z -0.5")
        op.value = (0, 0, -0.5)
        op = row.operator("transform.translate", text="Z 0.5")
        op.value = (0, 0, 0.5)

        row = box.row()
        op = row.operator("transform.translate", text="Z -0.25")
        op.value = (0, 0, -0.25)
        op = row.operator("transform.translate", text="Z 0.25")
        op.value = (0, 0, 0.25)

        row = box.row()
        op = row.operator("transform.translate", text="Z -0.125")
        op.value = (0, 0, -0.125)
        op = row.operator("transform.translate", text="Z 0.125")
        op.value = (0, 0, 0.125)
        