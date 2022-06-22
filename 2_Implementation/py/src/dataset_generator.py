import itertools
import random
from pathlib import Path
import lxml.etree as et
from tqdm import tqdm, trange


random.seed(2019283)  # Define the random seed

## CONSTANTS ##
N_BATCH = 4
TEMPLATE_PATH = Path("Z:\\decaligm\\linux_server\\mitsuba_renders\\nlos_scenes\\dataset\\template\\template.xml")
BATCH_PATH = Path("Z:\\decaligm\\linux_server\\mitsuba_renders\\nlos_scenes\\dataset\\xml_files")
BATCH_NAMES = ["batch_01", "batch_02", "batch_03", "batch_04"]
DEF_CAM_TX_X = 1
DEF_CAM_TX_Y = 1.65
DEF_CAM_TX_Z = 1.5
DEF_CAM_ROT_X = 180
DEF_CAM_ROT_Y = -50
DEF_CAM_ROT_Z = 180
N_TR_OBJ = [18, 16, 16, 10]
N_TR_SPHERE = [32, 30, 30, 18]
N_ROT_OBJ = [20, 16, 16, 11]
N_TR_ROT_CAM = 50
OBJ_NAMES = ["Cube", "Cone", "Cylinder", "Parallelepiped", "Sphere", "Concave plane", "Cube + sphere", "Cylinder + cone", "Sphere + cone"]

## Parameter lists ##
single_obj_tr_list = [[i/10 for i in range(-6, 17, 6)], [i/10 for i in range(-16, 17, 6)], [i/10 for i in range(-2, 19, 6)]]
multi_cone_tr_list = [[i/100 for i in range(-45, 170, 40)], [i/10 for i in range(-16, 17, 6)], [i/10 for i in range(-2, 19, 6)]]
cube_sphere_tr_list = [[i/100 for i in range(-55, 170, 15)], [i/10 for i in range(-16, 17, 6)], [i/10 for i in range(-2, 19, 6)]]
full_rot_list = [[i for i in range(-45, 46, 45)], [i for i in range(-45, 46, 45)], [i for i in range(-45, 46, 45)]]
partial_rot_list = [[i for i in range(-60, 61, 30)], [0], [i for i in range(-60, 61, 30)]]
cam_rot_list = [[i for i in range(-20, 21, 10) if i != 0], [i for i in range(-10, 41, 10) if i != 0], [i for i in range(-20, 21, 10) if i != 0]]
cam_rot_list_w_zeros = [[i for i in range(-20, 21, 10)], [i for i in range(-10, 41, 10)], [i for i in range(-10, 11, 10)]]
cam_tr_list = [[i/10 for i in range(-20, 0, 5)], [i/10 for i in range(-10, 11, 5)], [i/10 for i in range(0, 21, 5)]]
# Compute all the permutations of the parameter lists
single_obj_tr_list = list(itertools.product(*single_obj_tr_list))
multi_cone_tr_list = list(itertools.product(*multi_cone_tr_list))
cube_sphere_tr_list = list(itertools.product(*cube_sphere_tr_list))
full_rot_list = list(itertools.product(*full_rot_list))
partial_rot_list = list(itertools.product(*partial_rot_list))
cam_rot_list = list(itertools.product(*cam_rot_list))
cam_rot_list_w_zeros = list(itertools.product(*cam_rot_list_w_zeros))
cam_tr_list = list(itertools.product(*cam_tr_list))

for b_index in trange(N_BATCH, desc="Batches"):
    for name in tqdm(OBJ_NAMES, desc="Objects"):
        # Set the cam translation and rotation parameters
        # Translation
        if (b_index + 1) == 1 or (b_index + 1) == 2:
            cam_tr = [(DEF_CAM_TX_X, DEF_CAM_TX_Y, DEF_CAM_TX_Z)]
        else:
            cam_tr = random.sample(cam_tr_list, N_TR_ROT_CAM)
        # Rotations
        if (b_index + 1) == 1 or (b_index + 1) == 3:
            cam_rot = [(DEF_CAM_ROT_X, DEF_CAM_ROT_Y, DEF_CAM_ROT_Z)]
        elif (b_index + 1) == 2:
            cam_rot = random.sample(cam_rot_list, N_TR_ROT_CAM)
        else:
            cam_rot = random.sample(cam_rot_list_w_zeros, N_TR_ROT_CAM)
        # Sample the correct number of rotation and translation couple (at random) of the camera
        if (b_index + 1) == 1:
            cam_tr_rot = [(cam_tr[0], cam_rot[0])]
        else:
            cam_tr_rot = random.sample(list(itertools.product(*[cam_tr, cam_rot])), len(cam_tr) + len(cam_rot))

        # Set the object translation and rotations parameter
        # Translations
        if name == "Cube" or name == "Cone" or name == "Cylinder" or name == "Parallelepiped" or name == "Concave plane":
            obj_tr = random.sample(single_obj_tr_list, N_TR_OBJ[b_index])
        elif name == "Sphere":
            obj_tr = random.sample(single_obj_tr_list, N_TR_SPHERE[b_index])
        elif name == "Cylinder + cone" or name == "Sphere + cone":
            obj_tr = random.sample(multi_cone_tr_list, N_TR_OBJ[b_index])
        else:
            obj_tr = random.sample(cube_sphere_tr_list, N_TR_OBJ[b_index])
        # Rotations
        if name == "Cube" or name == "Parallelepiped" or name == "Concave plane" or name == "Cube + sphere":
            obj_rot = random.sample(full_rot_list, N_ROT_OBJ[b_index])
        elif name == "Cone" or name == "Cylinder" or name == "Cylinder + cone" or name == "Sphere + cone":
            obj_rot = random.sample(partial_rot_list, N_ROT_OBJ[b_index])
        # Sample the correct number of rotation and translation couple (at random) of the object
        if name != "Sphere":
            obj_tr_rot = random.sample(list(itertools.product(*[obj_tr, obj_rot])), len(obj_tr) + len(obj_rot))

        # Sample the correct number of rotation and translation couple (at random) for both the object and the camera
        if (b_index + 1) == 1:
            tot_tx_rot = random.sample(list(itertools.product(*[cam_tr_rot, obj_tr_rot])), len(obj_tr_rot))
        else:
            tot_tx_rot = random.sample(list(itertools.product(*[cam_tr_rot, obj_tr_rot])), len(cam_tr_rot) + len(obj_tr_rot))

        for data in tqdm(tot_tx_rot, desc="Data"):
            cam_t_x = str(data[0][0][0])
            cam_t_y = str(data[0][0][1])
            cam_t_z = str(data[0][0][2])
            cam_r_x = str(data[0][1][0])
            cam_r_y = str(data[0][1][1])
            cam_r_z = str(data[0][1][2])
            obj_t_x = str(data[1][0][0])
            obj_t_y = str(data[1][0][1])
            obj_t_z = str(data[1][0][2])
            if name != "Sphere":
                obj_r_x = str(data[1][1][0])
                obj_r_y = str(data[1][1][1])
                obj_r_z = str(data[1][1][2])

            if name != "Sphere":
                obj_file_name = name + "(" + str(data[1][1][0]) + "_" + str(data[1][1][1]) + "_" + str(data[1][1][2]) + ")"
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
