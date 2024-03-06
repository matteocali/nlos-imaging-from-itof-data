import bpy
import math
import itertools
from pathlib import Path
import pickle


obj_names = ["Cube", "Cone", "Cylinder", "Parallelepiped", "Sphere", "Concave plane", "Random"]
file = Path("PATH_TO_tr_rot_list")
with open(file, "rb") as fp:
        data = pickle.load(fp)

coll = bpy.context.scene.collection.children.get("NLOS Objects")

bpy.ops.object.select_all(action='DESELECT')
for b_index, batch in enumerate(data):
    batch_name = f"batch0{b_index + 1}"
    for o_index, object in enumerate(batch):
        obj_name = obj_names[o_index].lower()
        coll_target = bpy.context.scene.collection.children.get("All objects locations and rotations").children.get(obj_name)
        for elm in object:
            if obj_name != "random":
                tr = elm[1][0]
                rot = elm[1][1]
                name_str =  f"{obj_name}_{batch_name}_tr({str(tr[0])}_{str(tr[1])}_{str(tr[2])})_rot({str(rot[0])}_{str(rot[1])}_{str(rot[2])})"
                
                obj = bpy.context.scene.objects[obj_name.capitalize()].copy()
                obj.name = name_str
                coll_target.objects.link(obj)
                obj.rotation_euler.x = math.radians(rot[0])
                if obj_name != "concave plane":
                    obj.rotation_euler.y = math.radians(rot[1])
                else:
                    obj.rotation_euler.y += math.radians(rot[1])
                obj.rotation_euler.z = math.radians(rot[2])
                obj.location.x += tr[0]
                obj.location.y += tr[1]
                obj.location.z += tr[2]
            else:
                obj_name1 = elm[1][2][0].lower()
                obj_name2 = elm[1][2][1].lower()
                tr1 = elm[1][0][0]
                tr2 = elm[1][1][0]
                rot1 = elm[1][0][1]
                rot2 = elm[1][1][1]
                name_str1 = f"{obj_name1}_{batch_name}_tr({tr1[0]}_{tr1[1]}_{tr1[2]})_rot({rot1[0]}_{rot1[1]}_{rot1[2]})"
                name_str2 = f"{obj_name2}_{batch_name}_tr({tr2[0]}_{tr2[1]}_{tr2[2]})_rot({rot2[0]}_{rot2[1]}_{rot2[2]})"
            
                obj1 = bpy.context.scene.objects[obj_name1.capitalize()].copy()
                obj1.name = name_str1
                coll_target.objects.link(obj1)
                obj1.rotation_euler.x = math.radians(rot1[0])
                if obj_name1 != "concave plane":
                    obj1.rotation_euler.y = math.radians(rot1[1])
                else:
                    obj1.rotation_euler.y += math.radians(rot1[1])
                obj1.rotation_euler.z = math.radians(rot1[2])
                obj1.location.x += tr1[0]
                obj1.location.y += tr1[1]
                obj1.location.z += tr1[2]
                
                obj2 = bpy.context.scene.objects[obj_name2.capitalize()].copy()
                obj2.name = name_str2
                coll_target.objects.link(obj2)
                obj2.rotation_euler.x = math.radians(rot2[0])
                if obj_name1 != "concave plane":
                    obj2.rotation_euler.y = math.radians(rot2[1])
                else:
                    obj2.rotation_euler.y += math.radians(rot2[1])
                obj2.rotation_euler.z = math.radians(rot2[2])
                obj2.location.x += tr2[0]
                obj2.location.y += tr2[1]
                obj2.location.z += tr2[2]
                