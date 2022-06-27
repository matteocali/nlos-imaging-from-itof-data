import itertools
import random
from pathlib import Path
import lxml.etree as et
from tqdm import tqdm, trange
from modules import utilities as ut


## CONSTANTS ##
SEED = 2019283
N_BATCH = 4
TEMPLATE_PATH = Path("Z:\\decaligm\\linux_server\\mitsuba_renders\\nlos_scenes\\dataset\\template\\template.xml")
BATCH_PATH = Path("Z:\\decaligm\\linux_server\\mitsuba_renders\\nlos_scenes\\dataset\\xml_files")
PERM_LIST_LOC = Path("Z:\\decaligm\\linux_server\\mitsuba_renders\\nlos_scenes\\dataset\\dataset_setup\\data\\lists")
DATASET_FILE = Path("Z:\\decaligm\\linux_server\\mitsuba_renders\\nlos_scenes\\dataset\\dataset_setup\\data")
DEF_CAM_POS = (1.5, -1, 1.65)
DEF_CAM_ROT = (90, 0, 50)
N_TR_OBJ = [18, 16, 16, 14]
N_TR_SPHERE = [32, 27, 27, 22]
N_ROT_OBJ = [20, 17, 17, 14]
N_TR_ROT_CAM = 45
OBJ_NAMES = ["Cube", "Cone", "Cylinder", "Parallelepiped", "Sphere", "Concave plane", "Cube + sphere", "Cylinder + cone", "Sphere + cone"]

## PARAMETERS LIST ##
obj_tr_list = [[i / 10 for i in range(0, 21, 5)], [i / 10 for i in range(-4, 13, 2)], [i / 10 for i in range(-10, 11, 5)]]
obj_full_rot_list = [[i for i in range(-90, 91, 45)], [i for i in range(-90, 91, 45)], [i for i in range(-90, 91, 45)]]
obj_partial_rot_list = [[i for i in range(-90, 91, 45)], [i for i in range(-90, 91, 45)], [0]]
cam_rot_list = [[i for i in range(90, 101, 5)], [i for i in range(-30, 31, 11)], [i for i in range(50, 101, 10)]]
cam_pos_list = [[i / 10 for i in range(10, 31, 5)], [i / 10 for i in range(-30, -11, 5)], [i / 10 for i in range(10, 31, 5)]]
'''
# Compute all the permutations of the parameter lists
obj_tr_list = ut.permute_list(obj_tr_list, SEED)
obj_full_rot_list = ut.permute_list(obj_full_rot_list, SEED)
obj_partial_rot_list = ut.permute_list(obj_partial_rot_list, SEED)
cam_rot_list = ut.permute_list(cam_rot_list, SEED)
cam_pos_list = ut.permute_list(cam_pos_list, SEED)

for b_index in trange(N_BATCH, desc="Batches"):
    for name in tqdm(OBJ_NAMES, desc="Objects"):
        # Set the cam translation and rotation parameters
        # Translation
        if (b_index + 1) == 1 or (b_index + 1) == 2:
            cam_tr = [(DEF_CAM_POS_X, DEF_CAM_POS_Y, DEF_CAM_POS_Z)]
        else:
            cam_tr = random.sample(cam_pos_list, N_TR_ROT_CAM)
        # Rotations
        if (b_index + 1) == 1 or (b_index + 1) == 3:
            cam_rot = [(DEF_CAM_ROT_X, DEF_CAM_ROT_Y, DEF_CAM_ROT_Z)]
        else:
            cam_rot = random.sample(cam_rot_list, N_TR_ROT_CAM)
        # Sample the correct number of rotation and translation couple (at random) of the camera
        if (b_index + 1) == 1:
            cam_tr_rot = [(cam_tr[0], cam_rot[0])]
        else:
            cam_tr_rot = random.sample(list(itertools.product(*[cam_tr, cam_rot])), len(cam_tr) + len(cam_rot))

        # Set the object translation and rotations parameter
        # Translations
        if name != "Sphere":
            obj_tr = random.sample(obj_tr_list, N_TR_OBJ[b_index])
        else:
            obj_tr = random.sample(obj_tr_list, N_TR_SPHERE[b_index])
        # Rotations
        if name == "Cube" or name == "Parallelepiped" or name == "Concave plane" or name == "Cube + sphere":
            obj_rot = random.sample(obj_full_rot_list, N_ROT_OBJ[b_index])
        elif name == "Cone" or name == "Cylinder" or name == "Cylinder + cone" or name == "Sphere + cone":
            obj_rot = random.sample(obj_partial_rot_list, N_ROT_OBJ[b_index])
        # Sample the correct number of rotation and translation couple (at random) of the object
        if name != "Sphere":
            obj_tr_rot = random.sample(list(itertools.product(*[obj_tr, obj_rot])), len(obj_tr) + len(obj_rot))
            ut.save_list(obj_tr_rot, PERM_LIST_LOC / f"obj_tr_rot_{name.lower()}")
        else:
            ut.save_list(obj_tr, PERM_LIST_LOC / f"obj_tr_rot_{name.lower()}")

        # Sample the correct number of rotation and translation couple (at random) for both the object and the camera
        if (b_index + 1) == 1:
            tot_tx_rot = random.sample(list(itertools.product(*[cam_tr_rot, obj_tr_rot])), len(obj_tr_rot))
        else:
            tot_tx_rot = random.sample(list(itertools.product(*[cam_tr_rot, obj_tr_rot])), len(cam_tr_rot) + len(obj_tr_rot))

        for data in tqdm(tot_tx_rot, desc="Data"):  ### FIX CAMERA ROTATION AND CREATE OUTPUT FILE TO STORE FINAL DATASET SETUP
            cam_t_x = str(data[0][0][0])
            cam_t_y = str(data[0][0][1])
            cam_t_z = str(data[0][0][2])
            cam_r_x = str(data[0][1][0])
            cam_r_y = str(data[0][1][1])
            cam_r_z = str(data[0][1][2])
            obj_t_x = str(data[1][0][0])
            obj_t_y = str(data[1][0][1])
            obj_t_z = str(data[1][0][2])

            pos, angle = ut.blender2mitsuba_coord_mapping(float(cam_r_x), float(cam_r_y), float(cam_r_z), float(cam_t_x), float(cam_t_x), float(cam_t_z))
            print(f"ax: {round(angle[0], 0)}, ay: {round(angle[1], 0)}, az: {round(angle[2], 0)}")
            print(f"px: {round(pos[0], 2)}, py: {round(pos[1], 2)}, pz: {round(pos[2], 2)}")

            if name != "Sphere":
                obj_r_x = str(data[1][1][0])
                obj_r_y = str(data[1][1][1])
                obj_r_z = str(data[1][1][2])

            if name != "Sphere":
                obj_file_name = "rotations/" + name + "(" + str(data[1][1][0]) + "_" + str(data[1][1][1]) + "_" + str(data[1][1][2]) + ")"
                file_name = "transient_nlos_" + name.lower() + "_[cam_pos_(" + cam_t_x + "_" + cam_t_y + "_" + cam_t_z + ")_cam_rot_(" + cam_r_x + "_" + cam_r_y + "_" + cam_r_z + ")_obj_tr_(" + obj_t_x + "_" + obj_t_y + "_" + obj_t_z + ")_obj_rot_(0_0_0)].xml"
            else:
                obj_file_name = name
                file_name = "transient_nlos_" + name.lower() + "_[cam_pos_(" + cam_t_x + "_" + cam_t_y + "_" + cam_t_z + ")_cam_rot_(" + cam_r_x + "_" + cam_r_y + "_" + cam_r_z + ")_obj_tr_(" + obj_t_x + "_" + obj_t_y + "_" + obj_t_z + ")_obj_rot_(" + obj_r_x + "_" + obj_r_y + "_" + obj_r_z + ")].xml"

            # Correct the value of the camera rotation in order to be coherent with the axis displacement of Mitsuba
            # (different from Blender and from the provided ones in the list, since these last one suppose that the rotation [0, 0, 0] correspond to the camera looking straight to the wall
            # and in Blender is [90, 0, 50], in mitsuba [180, -50, 180])
            if int(cam_r_x) < 0:
                m_cam_r_x = str(-(180 + int(cam_r_x)))
                m_cam_r_z = str(-int(cam_r_z))
            else:
                m_cam_r_x = str(180 - int(cam_r_x))
                m_cam_r_z = str(180 - int(cam_r_z))
            m_cam_r_y = str(-90 + int(cam_r_y))

            # Modify the template inserting the desired data
            # (code from: https://stackoverflow.com/questions/37868881/how-to-search-and-replace-text-in-an-xml-file-using-python)
            with open(str(TEMPLATE_PATH), encoding="utf8") as f:
                tree = et.parse(f)
                root = tree.getroot()

                for elem in root.getiterator():
                    try:
                        if elem.attrib["value"] == "obj_name":
                            elem.attrib["value"] = obj_file_name
                        elif elem.attrib["value"] == "t_obj_x":
                            elem.attrib["value"] = obj_t_x
                        elif elem.attrib["value"] == "t_obj_y":
                            elem.attrib["value"] = obj_t_x
                        elif elem.attrib["value"] == "t_obj_z":
                            elem.attrib["value"] = obj_t_x
                        elif elem.attrib["value"] == "t_cam_x":
                            elem.attrib["value"] = cam_t_x
                        elif elem.attrib["value"] == "t_cam_y":
                            elem.attrib["value"] = cam_t_y
                        elif elem.attrib["value"] == "t_cam_z":
                            elem.attrib["value"] = cam_t_z
                        elif elem.attrib["value"] == "r_cam_x":
                            elem.attrib["value"] = m_cam_r_x
                        elif elem.attrib["value"] == "r_cam_y":
                            elem.attrib["value"] = m_cam_r_y
                        elif elem.attrib["value"] == "r_cam_z":
                            elem.attrib["value"] = m_cam_r_z
                    except KeyError:
                        pass
            tree.write(str(BATCH_PATH / BATCH_NAMES[b_index] / file_name), xml_declaration=True, method="xml", encoding="utf8")
'''
print("Compute all the permutations of the camera and objects locations and rotations (batch by batch):")
tr_rot_list = ut.generate_dataset_list(obj_tr_list=obj_tr_list,
                                       obj_full_rot_list=obj_full_rot_list, obj_partial_rot_list=obj_partial_rot_list,
                                       cam_rot_list=cam_rot_list,
                                       cam_pos_list=cam_pos_list,
                                       def_cam_pos=DEF_CAM_POS, def_cam_rot=DEF_CAM_ROT,
                                       n_batches=N_BATCH, obj_names=OBJ_NAMES,
                                       n_tr_rot_cam=N_TR_ROT_CAM, n_tr_obj=N_TR_OBJ, n_rot_obj=N_ROT_OBJ, n_tr_sphere=N_TR_SPHERE,
                                       folder_path=PERM_LIST_LOC, seed=SEED)

ut.generate_dataset_file(tx_rt_list=tr_rot_list, folder_path=DATASET_FILE, obj_names=OBJ_NAMES)

print("\nGenerate all the .xml files:")
ut.generate_dataset_xml(tr_rot_list=tr_rot_list, template=TEMPLATE_PATH, folder_path=BATCH_PATH, obj_names=OBJ_NAMES)