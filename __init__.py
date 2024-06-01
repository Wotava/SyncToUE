if "bpy" in locals():
    import importlib
    importlib.reload(utils)
    importlib.reload(operators)
    importlib.reload(ui)

else:
    import bpy
    from . import utils
    from . import operators
    from . import ui

bl_info = {
    'name': 'Sync to UE',
    'description': 'Simple helper addon to sync scene in UE with Blender',
    'location': '3D View -> Toolbox',
    'author': 'wotava',
    'version': (1, 1),
    'blender': (4, 1, 0),
    'category': 'Object'
}
classes = [
    operators.SCENE_OP_DumpToJSON,
    ui.VIEW3D_PT_ModifierManager,
    utils.ExportParameters,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.stu_parameters = bpy.props.PointerProperty(type=utils.ExportParameters)


def unregister():
    # Unregister this addon
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)