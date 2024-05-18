import bpy
from bpy.props import FloatVectorProperty, BoolVectorProperty, BoolProperty


class ExportParameters(bpy.types.PropertyGroup):
    flip_loc: BoolVectorProperty(
        name="Location Flips",
        size=3,
        default=[False, False, False]
    )
    flip_rot: BoolVectorProperty(
        name="Rotation Flips",
        size=3,
        default=[False, False, False]
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