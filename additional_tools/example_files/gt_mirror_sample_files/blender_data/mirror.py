import bpy

for obj in bpy.context.selected_objects:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type='MIRROR')
    bpy.context.object.modifiers["Mirror"].use_axis = [False, False, True]
    bpy.context.object.modifiers["Mirror"].mirror_object = bpy.context.scene.objects["Front wall"]