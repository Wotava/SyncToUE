import bpy
from bpy.props import FloatVectorProperty, BoolVectorProperty, BoolProperty, StringProperty, FloatProperty


class ExportParameters(bpy.types.PropertyGroup):
    flip_loc: BoolVectorProperty(
        name="Location Flips",
        size=3,
        default=[False, True, False]
    )
    flip_rot: BoolVectorProperty(
        name="Rotation Flips",
        size=3,
        default=[True, True, True]
    )
    adjust_rot: BoolProperty(
        name="Adjust rot to -pi + rot",
        default=False
    )

    flip_scale: BoolVectorProperty(
        name="Scale Flips",
        size=3,
        default=[False, False, False]
    )
    abs_scale: BoolVectorProperty(
        name="Scale Abs",
        size=3,
        default=[False, False, False]
    )
    swizzle_neg_scale: BoolProperty(
        name="Swizzle on -scale",
        default=False
    )
    rotation_offset: FloatVectorProperty(
        name="Offset Rotation (deg)",
        size=3,
        default=[0.0, 0.0, 0.0]
    )

    json_path: StringProperty(
        name="JSON File",
        subtype="FILE_PATH",
        description="JSON file to export the scene data into")

    export_path: StringProperty(
        name="Export Dir",
        subtype="DIR_PATH",
        description="Directory to export the scene fbx into")

    target_density: FloatProperty(
        name="Target Density",
        description="In p/m on 4k texture",
        default=256.0
    )

    def draw(self, layout):
        for prop in list(self.__annotations__):
            if getattr(self, prop):
                row = layout.row()
                row.prop(self, prop)
