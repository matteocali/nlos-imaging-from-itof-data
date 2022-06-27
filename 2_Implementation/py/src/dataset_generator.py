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