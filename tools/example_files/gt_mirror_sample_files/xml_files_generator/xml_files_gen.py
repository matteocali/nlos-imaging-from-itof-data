from pathlib import Path
from os import path, mkdir
from tqdm import tqdm
import pickle

current_folder = Path(path.realpath(__file__)).parent.absolute()
template_single = Path(current_folder / "template_single_obj.xml")
template_multi = Path(current_folder / "template_multi_obj.xml")
out_folder = current_folder.parent.absolute() / "xml_files"
mesh_folder = current_folder.parent.absolute() / "meshes"
list_folder = Path(current_folder / "data_configuration" / "tr_rot_list")
obj_names = ["cube", "cone", "cylinder", "parallelepiped", "sphere", "concave plane"]

with open(list_folder, "rb") as fp:
        data = pickle.load(fp)
data = data

try:
    mkdir(out_folder)
except:
    pass

for b_index, batch in tqdm(enumerate(data), desc="batches", leave=True):
    batch_name = f"batch0{b_index + 1}"
    final_folder = out_folder / batch_name
    try:
        final_folder.mkdir(parents=True)
    except:
        pass

    if b_index == 2 or b_index == 3 or b_index == 6 or b_index	== 7:
        cam_var = True
    else:
        cam_var = False

    for obj_index, obj in tqdm(enumerate(batch), desc="objects", leave=False):
        if obj_index < (len(batch) - 1):
            obj_name = obj_names[obj_index]
            obj_name_no_space = obj_name.replace(" ", "_")
            multi = False
        else:
            multi = True
        for elm in tqdm(obj, desc="elements", leave=False):
            if cam_var:
                cam_pos = str(elm[0][0]).replace(" ", "_").replace(",", "")
                cam_rot = str(elm[0][1]).replace(" ", "_").replace(",", "")
            else:
                cam_pos = "(1.0_-1.0_1.65)"
                cam_rot = "(90_0_50)"

            if not multi:
                obj_tr = str(elm[1][0]).replace(" ", "_").replace(",", "")
                obj_rot = str(elm[1][1]).replace(" ", "_").replace(",", "")

                mesh_name = f"{obj_name}_{batch_name}_tr{obj_tr}_rot{obj_rot}"
                
                with open(str(template_single), 'r', newline="\n") as file:
                    filedata = file.read()

                filedata = filedata.replace('&', str(mesh_name))

                # Write the file out again
                with open(str(final_folder / f"transient_nlos_cam_pos{cam_pos}_cam_rot_{cam_rot}_{obj_name_no_space}_tr{obj_tr}_rot{obj_rot}.xml"), 'w', newline="\n") as file:
                    file.write(filedata)
                
            else:
                obj_tr_1 = str(elm[1][0][0]).replace(" ", "_").replace(",", "")
                obj_tr_2 = str(elm[1][1][0]).replace(" ", "_").replace(",", "")
                obj_rot_1 = str(elm[1][0][1]).replace(" ", "_").replace(",", "")
                obj_rot_2 = str(elm[1][1][1]).replace(" ", "_").replace(",", "")
                obj_name_1 = elm[1][2][0].lower()
                obj_name_2 = elm[1][2][1].lower()
                obj_name_no_space_1 = obj_name_1.replace(" ", "_")
                obj_name_no_space_2 = obj_name_2.replace(" ", "_")

                mesh_name_1 = f"{obj_name_1}_{batch_name}_tr{obj_tr_1}_rot{obj_rot_1}"
                mesh_name_2 = f"{obj_name_2}_{batch_name}_tr{obj_tr_2}_rot{obj_rot_2}"
                
                with open(str(template_multi), 'r', newline="\n") as file:
                    filedata = file.read()

                filedata = filedata.replace('&', str(mesh_name_1))
                filedata = filedata.replace('@', str(mesh_name_2))

                # Write the file out again
                with open(str(final_folder / f"transient_nlos_cam_pos{cam_pos}_cam_rot_{cam_rot}_{obj_name_no_space_1}_tr{obj_tr_1}_rot{obj_rot_1}_{obj_name_no_space_2}_tr{obj_tr_2}_rot{obj_rot_2}.xml"), 'w', newline="\n") as file:
                    file.write(filedata)
                