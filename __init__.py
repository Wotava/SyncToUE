if "bpy" in locals():
    import importlib
    importlib.reload(operators)
else:
    import bpy
    from . import operators

bl_info = {
    'name': 'Sync to UE',
    'description': 'Simple helper addon to sync scene in UE with Blender',
    'location': '3D View -> Toolbox',
    'author': 'wotava',
    'version': (1, 0),
    'blender': (3, 0, 0),
    'category': 'Object'
}
classes = [
    operators.SCENE_OP_DumpToJSON
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    # Unregister this addon
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)