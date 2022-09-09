from pathlib import Path
from random import seed as rnd_seed, sample as rnd_sample, shuffle as rnd_shuffle

import matplotlib.pyplot as plt
from lxml import etree as et
from tqdm import trange, tqdm
from numpy import nonzero, unique, zeros as np_zeros, where, sum as np_sum, swapaxes, moveaxis, ndarray, \
    copy as np_copy, array, float32, zeros, count_nonzero, empty, matmul, nanmax as np_nanmax, Inf, full, convolve, \
    flip as np_flip, sort as np_sort, argsort as np_argsort, nan, size as np_size, square as np_square, exp as np_exp, \
    maximum, all as np_all, nanmin as np_nanmin
from scipy.signal import find_peaks
import open3d as o3d
from typing import Union
from modules.utilities import permute_list, load_list, save_list, blender2mitsuba_coord_mapping, spot_bitmap_gen, \
    read_folders, save_h5, load_h5, read_files, k_matrix_calculator
from modules.transient_handler import transient_loader, grid_transient_loader, compute_distance_map, phi, \
    amp_phi_compute
from modules.fermat_tools import undistort_depthmap, compute_bin_center, prepare_fermat_data


def generate_dataset_file(tx_rt_list: list, folder_path: Path, obj_names: tuple, obj_base_pos: tuple, wall_roughness: list) -> None:
    """
    Function that build a .txt file containing all the information about how the dataset has been created
    :param tx_rt_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param folder_path: path of the folder where to save the dataset file
    :param obj_names: tuple containing all the name of the used objects
    :param obj_base_pos: tuple containing the position of the objects
    :param wall_roughness: list containing the wall roughness for each object
    """

    if not folder_path.exists():  # Create the output folder if not already present
        folder_path.mkdir(parents=True)

    with open(str(folder_path / "dataset.txt"), "w") as f:  # Open the target .txt file (in ase create it)
        for b_index, batch in enumerate(tx_rt_list):  # Cycle through each batches
            f.write(f"BATCH 0{b_index + 1}:\n")  # Write which batch it is under consideration
            for obj_index, obj in enumerate(batch):  # Cycle through each object
                f.write(f"{obj_names[obj_index]}:\n")  # Write which object is under consideration
                for i, data in enumerate(obj):  # Cycle through each cam_pos/cam_rot/obj_tr_obj_rot combination for the given object
                    f.write(f"\t- Object {i + 1}:\n")
                    f.write(f"\t\t- camera position -> (x: {data[0][0][0]}, y: {data[0][0][1]}, z: {data[0][0][2]})\n")  # Write the position of the camera
                    f.write(f"\t\t- camera rotation -> (x: {data[0][1][0]}, y: {data[0][1][1]}, z: {data[0][1][2]})\n")  # Write the rotation of the camera
                    if obj_names[obj_index] != "Random":
                        f.write(f"\t\t- object position -> (x: {round(obj_base_pos[0] + data[1][0][0], 2)}, y: {round(obj_base_pos[1] + data[1][0][1], 2)}, z: {round(obj_base_pos[2] + data[1][0][2], 2)})\n")  # Write the position of the object summing its default position with the applied translation
                        f.write(f"\t\t- object rotation -> (x: {data[1][1][0]}, y: {data[1][1][1]}, z: {data[1][1][2]})\n")  # Write the rotation of the object
                        if len(wall_roughness[b_index][0]) == 0:
                            f.write(f"\t\t- wall roughness -> diffuse\n")  # Write the wall roughness of the object
                        else:  # If the current batch requires roughness
                            f.write(f"\t\t- wall roughness -> {wall_roughness[b_index][obj_index][i]}\n")  # Write the wall roughness of the object
                    else:
                        f.write(f"\t\t- considered objects -> {data[1][2][0]} + {data[1][2][0]}\n")
                        f.write(f"\t\t- {data[1][2][0]} position -> (x: {round(obj_base_pos[0] + data[1][0][0][0], 2)}, y: {round(obj_base_pos[1] + data[1][0][0][1], 2)}, z: {round(obj_base_pos[2] + data[1][0][0][2], 2)})\n")  # Write the position of the object summing its default position with the applied translation
                        f.write(f"\t\t- {data[1][2][1]} position -> (x: {round(obj_base_pos[0] + data[1][1][0][0], 2)}, y: {round(obj_base_pos[1] + data[1][1][0][1], 2)}, z: {round(obj_base_pos[2] + data[1][1][0][2], 2)})\n")  # Write the position of the object summing its default position with the applied translation
                        f.write(f"\t\t- {data[1][2][0]} rotation -> (x: {data[1][0][1][0]}, y: {data[1][0][1][1]}, z: {data[1][0][1][2]})\n")  # Write the rotation of the object
                        f.write(f"\t\t- {data[1][2][1]} rotation -> (x: {data[1][1][1][0]}, y: {data[1][1][1][1]}, z: {data[1][1][1][2]})\n")  # Write the rotation of the object
                        if len(wall_roughness[b_index][0]) == 0:  # If the current batch doesn't require roughness
                            f.write(f"\t\t- wall roughness -> diffuse\n")  # Write the wall roughness of the object
                        else:  # If the current batch requires roughness
                            f.write(f"\t\t- wall roughness -> {wall_roughness[b_index][obj_index][i]}\n")  # Write the wall roughness of the object
            f.write("\n-------------------------------------------------------------------\n\n")


def generate_dataset_list(obj_tr_list: list, obj_full_rot_list: list, obj_partial_rot_list: list, cam_rot_list: list,
                          cam_pos_list: list, n_classes: int, n_batches: int, n_elm_per_class: int, obj_names: tuple,
                          def_cam_pos: tuple, def_cam_rot: tuple, roughness_values: tuple, folder_path: Path = None,
                          seed: int = None) -> tuple[list[list[list]], list]:
    """
    Function that generate the list of list containing all the final combinations of camera and object location/translation/rotations
    :param obj_tr_list: List that contains all the possible translations that is granted to an object [[possible x translations], [possible y translations], [possible z translations]]
    :param obj_full_rot_list: List that contains all the possible rotations that is granted to an object [[possible x rotations], [possible y rotations], [possible z rotations]]
    :param obj_partial_rot_list: List that contains all the possible rotations that is granted to an object with no rotations over the z axis [[possible x rotations], [possible y rotations], 0]
    :param cam_rot_list: List that contains all the possible rotations that is granted to the camera [[possible x rotations], [possible y rotations], [possible z rotations]]
    :param cam_pos_list: List that contains all the possible positions that is granted to the camera [[possible x positions], [possible y positions], [possible z positions]]
    :param n_classes: Number of classes of objects to consider
    :param n_batches: Number of different batches for each class
    :param n_elm_per_class: Number of elements per class
    :param obj_names: List that contains the name of every object that will be considered
    :param def_cam_pos: Default camera position (x, y, z)
    :param def_cam_rot: Default camera orientation (x, y, z)
    :param roughness_values: Tuple that contains the possible roughness values for the material of the front wall
    :param folder_path: path of the folder where to store the object translation and rotations permutation needed by blender
    :param seed: random seed
    :return: [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    """

    if not folder_path.exists():  # Create the output folder if not already present
        folder_path.mkdir(parents=True)

    if seed is not None:
        rnd_seed(seed)  # Set the random seed

    if folder_path is not None and (folder_path / "tr_rot_list").is_file() and (folder_path / "roughness_list").is_file():  # If the file already exists, load it
        tr_rot_list = load_list(folder_path / "tr_rot_list")  # Load the list of all the possible translations and rotations
        roughness_list = load_list(folder_path / "roughness_list")  # Load the list of all the possible roughness values
        return tr_rot_list, roughness_list

        # Compute all the permutations of the parameter lists
    obj_tr_list = permute_list(obj_tr_list, seed)
    obj_full_rot_list = permute_list(obj_full_rot_list, seed)
    obj_partial_rot_list = permute_list(obj_partial_rot_list, seed)
    cam_rot_list = permute_list(cam_rot_list, seed)
    cam_pos_list = permute_list(cam_pos_list, seed)

    # Compute how many elements is required for each class
    n_objs_cl_1_3 = int(n_elm_per_class / len(obj_names))        # fixed camera location
    n_objs_cl_2_4 = int(n_elm_per_class / (len(obj_names) + 1))  # no fixed position

    tr_rot_list = []     # define the list that will contain all the data
    roughness_list = []  # define the list that will contain all the roughness data

    for c_index in trange(n_classes, desc="Generating all the camera/object location, class by class", leave=True):  # Cycle through each batch
        obj_data = []  # Define the list that will contain the data of the current class
        rough_data = []  # Define the list that will contain the roughness data of the current class

        # Avoid that two consecutive batches have the same data
        cam_rot_tmp = cam_rot_list.copy()
        cam_pos_tmp = cam_pos_list.copy()
        obj_tr_tmp = obj_tr_list.copy()
        obj_full_rot_tmp = obj_full_rot_list.copy()
        obj_partial_rot_tmp = obj_partial_rot_list.copy()


        # Set the cam translation and rotation parameters considering that in some batches the camera position and/or rotation is fixed
        if c_index == 0 or c_index == 2:
            cam_tr = [(def_cam_pos[0], def_cam_pos[1], def_cam_pos[2])]
            cam_rot = [(def_cam_rot[0], def_cam_rot[1], def_cam_rot[2])]
            cam_tr_rot = [(cam_tr[0], cam_rot[0])]
        else:
            # noinspection PyUnboundLocalVariable
            cam_tr = rnd_sample(cam_pos_tmp, int(n_objs_cl_2_4 / 2))
            cam_pos_tmp = [x for x in cam_pos_tmp if x not in cam_tr]  # Avoid that two consecutive batches have the same data
            # noinspection PyUnboundLocalVariable
            cam_rot = rnd_sample(cam_rot_tmp, int(n_objs_cl_2_4 / 2))
            cam_rot_tmp = [x for x in cam_rot_tmp if x not in cam_rot]  # Avoid that two consecutive batches have the same data
            cam_tr_rot = rnd_sample(permute_list([cam_tr, cam_rot]), n_objs_cl_2_4)  # Randomly select the camera translation and rotation (the correct number of combinations)
            rnd_shuffle(cam_tr_rot)  # Randomly shuffle the camera translation and rotation

        for name in tqdm(obj_names, desc="Defining all the objects locations", leave=False):  # Cycle through all the objects
            # Set the object translation and rotations parameter
            if c_index == 0 or c_index == 2:
                n_obj_to_sample = n_objs_cl_1_3
            else:
                n_obj_to_sample = n_objs_cl_2_4

            # Translations
            if name != "Sphere":
                obj_tr = rnd_sample(obj_tr_tmp, int(n_obj_to_sample / 2))
                if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                    obj_tr_tmp = [x for x in obj_tr_tmp if x not in obj_tr]
            else:
                obj_tr = rnd_sample(obj_tr_tmp, n_obj_to_sample)
                if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                    obj_tr_tmp = [x for x in obj_tr_tmp if x not in obj_tr]
            # Rotations
            if name == "Cube" or name == "Parallelepiped" or name == "Concave plane" or name == "Random":
                # noinspection PyUnboundLocalVariable
                obj_rot = rnd_sample(obj_full_rot_tmp, int(n_obj_to_sample / 2))
                if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                    obj_full_rot_tmp = [x for x in obj_full_rot_tmp if x not in obj_rot]
            elif name == "Cone" or name == "Cylinder":
                # noinspection PyUnboundLocalVariable
                obj_rot = rnd_sample(obj_partial_rot_tmp, int(n_obj_to_sample / 2))
                if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                    obj_partial_rot_tmp = [x for x in obj_partial_rot_tmp if x not in obj_rot]
            else:
                obj_rot = [(0, 0, 0)]

            # Sample the correct number of rotation and translation couple (at random) of the object
            obj_tr_rot = rnd_sample(permute_list([obj_tr, obj_rot]), n_obj_to_sample)
            rnd_shuffle(obj_tr_rot)  # Randomly shuffle the list of translation and rotation

            # Sample the correct number of rotation and translation couple (at random) for both the object and the camera
            obj_data.append(rnd_sample(permute_list([cam_tr_rot, obj_tr_rot]), n_objs_cl_1_3))

            # Sample the roughness of the wall
            if c_index > 1:
                elm_roughness = []
                for i in range(n_objs_cl_1_3):
                    elm_roughness.append(rnd_sample(roughness_values, 1)[0])
                rough_data.append(elm_roughness)

        for b_index in range(n_batches):
            batch_data = []  # Define the list that will contain the data of the current batch
            batch_roughness = []  # Define the list that will contain the roughness data of the current batch
            for obj_data_i in obj_data:
                batch_data.append(obj_data_i[(b_index * int(len(obj_data_i) / n_batches)):((b_index + 1) * int(len(obj_data_i) / n_batches))])
            if c_index <= 1:
                batch_roughness.append([])
            else:
                for rough_data_i in rough_data:
                    batch_roughness.append(rough_data_i[(b_index * int(len(rough_data_i) / n_batches)):((b_index + 1) * int(len(rough_data_i) / n_batches))])

            tr_rot_list.append(batch_data)  # Add the current batch to the list of all the batches
            roughness_list.append(batch_roughness)  # Add the current batch to the list of all the batches

    tr_rot_list = rnd_obj_data_generator(tr_rot_list, obj_names[:-1])  # Add the information relative to the random object

    if folder_path is not None:
        save_list(tr_rot_list, folder_path / "tr_rot_list")  # Save the list of the object translation and rotations
        save_list(roughness_list, folder_path / "roughness_list")  # Save the list of the wall roughness

    return tr_rot_list, roughness_list


def rnd_obj_data_generator(tr_rot_list: list[list[list]], obj_names: list) -> list[list[list]]:
    """
    Function that define the rotation, translation and parent objects of each randomly generated one
    :param tr_rot_list: list of all the rotations and translation
    :param obj_names: list of the names of the possible object to combine
    :return: Updated input list [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], ..., obj_rand[(camera_pos, camera_rot), ((obj1_tr, obj1_rot), (obj2_tr, obj2_rot), (obj1_name, obj2_name))]], batch2[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], ..., obj_rand[(camera_pos, camera_rot), ((obj1_tr, obj1_rot), (obj2_tr, obj2_rot), (obj1_name, obj2_name))], ...]
    """
    delta_tr_permute = permute_list([[i / 100 for i in range(-12, 16)],
                                     [i / 100 for i in range(-12, 16)],
                                     [i / 100 for i in range(-12, 16)]])
    delta_rot_permute = permute_list([[i for i in range(-45, 45)],
                                      [i for i in range(-45, 45)],
                                      [i for i in range(-45, 45)]])

    for b_index, batch in enumerate(tr_rot_list):
        for e_index, elm in enumerate(batch[-1]):
            delta_tr = rnd_sample(delta_tr_permute, 1)[0]
            delta_rot = rnd_sample(delta_rot_permute, 1)[0]
            objs = tuple(rnd_sample(obj_names, 2))
            second_obj_tr = (round(elm[1][0][0] + delta_tr[0], 1), round(elm[1][0][1] + delta_tr[1], 1), round(elm[1][0][2] + delta_tr[2], 1))
            second_obj_rot = (round(elm[1][1][0] + delta_rot[0], 0), round(elm[1][1][1] + delta_rot[1], 0), round(elm[1][1][2] + delta_rot[2], 0))
            second_obj_tr_rot = (second_obj_tr, second_obj_rot)
            tr_rot_list[b_index][-1][e_index] = (elm[0], (elm[1], second_obj_tr_rot, objs))

    return tr_rot_list


def save_dts_xml(template_path: Union[str, Path], file_path: Union[str, Path],
                 cam_pos: list, cam_rot: list, roughness: str,
                 obj_file_name: str = None, obj_file_name_1: str = None, obj_file_name_2: str = None,
                 col: int = None, row: int = None) -> None:
    """
    Function that save the dts xml file
    :param template_path: path to the template file
    :param file_path: path to the file to save
    :param obj_file_name: name of the single object
    :param obj_file_name_1: name of the first object in the random object
    :param obj_file_name_2: name of the second object in the random object
    :param cam_pos: list of the camera position
    :param cam_rot: list of the camera rotation
    :param roughness: roughness value of the wall
    :param col: column index
    :param row: row index
    """

    with open(str(template_path), encoding="utf8") as f:  # Open the template file
        tree = et.parse(f)
        root = tree.getroot()

        for elem in root.getiterator():  # For each element in the template file
            try:  # Try to change the value of the element if present
                if elem.attrib["value"] == "obj_name":
                    # noinspection PyUnboundLocalVariable
                    elem.attrib["value"] = str(obj_file_name)
                elif elem.attrib["value"] == "obj_name_1":
                    # noinspection PyUnboundLocalVariable
                    elem.attrib["value"] = str(obj_file_name_1)
                elif elem.attrib["value"] == "obj_name_2":
                    # noinspection PyUnboundLocalVariable
                    elem.attrib["value"] = str(obj_file_name_2)
                elif elem.attrib["value"] == "t_cam_x":
                    elem.attrib["value"] = str(cam_pos[0])
                elif elem.attrib["value"] == "t_cam_y":
                    elem.attrib["value"] = str(cam_pos[1])
                elif elem.attrib["value"] == "t_cam_z":
                    elem.attrib["value"] = str(cam_pos[2])
                elif elem.attrib["value"] == "r_cam_x":
                    elem.attrib["value"] = str(cam_rot[0])
                elif elem.attrib["value"] == "r_cam_y":
                    elem.attrib["value"] = str(cam_rot[1])
                elif elem.attrib["value"] == "r_cam_z":
                    elem.attrib["value"] = str(cam_rot[2])
                elif elem.attrib["value"] == "a_value":
                    elem.attrib["value"] = str(roughness)
                elif elem.attrib["value"] == "o_x":
                    elem.attrib["value"] = str(col)
                elif elem.attrib["value"] == "o_y":
                    elem.attrib["value"] = str(row)
                elif elem.attrib["value"] == "t_name":
                    elem.attrib["value"] = f"bitmap_r{row}_c{col}"
            except KeyError:
                pass
    tree.write(str(file_path), method="xml", encoding="utf8")


def generate_dataset_xml(tr_rot_list: list, n_classes: int, templates_folder: Path, folder_path: Path, obj_names: tuple,
                         obj_base_pos: tuple, roughness: list, img_shape: tuple = None, pattern: tuple = None) -> None:
    """
    Function that given the template.xml file and all the chosen position?translation?rotation combinations generate the correspondent .xml files
    :param tr_rot_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param n_classes: number of classes
    :param templates_folder: template folder
    :param folder_path: path of the output folder
    :param obj_names: tuple that contains the name of every object that will be considered
    :param obj_base_pos: tuple that contains the base position of every object that will be considered
    :param roughness: tuple that contains the value of alpha (roughness) for the front wall material
    :param img_shape: tuple that contains the shape of the image
    :param pattern: tuple that contains the pattern of the emitter grid
    """

    if not folder_path.exists():  # Create the output folder if not already present
        folder_path.mkdir(parents=True)

    for b_index, batch in tqdm(enumerate(tr_rot_list), desc="Generate all the xml files batch by batch", leave=True):  # Cycle through each batch
        batch_path = Path(folder_path / f"batch0{b_index + 1}")
        if not batch_path.exists():  # Create the output folder if not already present
            batch_path.mkdir(parents=True)

        if b_index >= n_classes:  # If the current batch is in the last two classes (the ones with different walls)
            rough_wall = True
        else:
            rough_wall = False

        for o_index, obj in tqdm(enumerate(batch), desc="Objects", leave=False):  # Cycle through each object
            name = obj_names[o_index]  # Extract the object name
            for e_index, elm in tqdm(enumerate(obj), desc="File", leave=False):  # Cycle through each position/translation/rotation combination
                cam_pos = [elm[0][0][0], elm[0][0][1], elm[0][0][2]]  # Extract the camera position
                cam_rot = [elm[0][1][0], elm[0][1][1], elm[0][1][2]]  # Extract the camera rotation
                if name != "Random":
                    obj_tr = [elm[1][0][0], elm[1][0][1], elm[1][0][2]]  # Extract the object translation
                    obj_rot = [elm[1][1][0], elm[1][1][1], elm[1][1][2]]  # Extract the object rotation
                else:
                    obj_tr_1 = [elm[1][0][0][0], elm[1][0][0][1], elm[1][0][0][2]]  # Extract the object translation
                    obj_tr_2 = [elm[1][1][0][0], elm[1][1][0][1], elm[1][1][0][2]]  # Extract the object translation
                    obj_rot_1 = [elm[1][0][1][0], elm[1][0][1][1], elm[1][0][1][2]]  # Extract the object rotation
                    obj_rot_2 = [elm[1][1][1][0], elm[1][1][1][1], elm[1][1][1][2]]  # Extract the object rotation

                if len(roughness[b_index][0]) == 0:  # If the current batch doesn't require roughness
                    roughness_str = "diffuse"  # Set the roughness to diffuse
                else:  # If the current batch requires roughness
                    roughness_str = str(roughness[b_index][o_index][e_index])  # Set the roughness to the one inside the roughness list

                if name != "Random":
                    # noinspection PyUnboundLocalVariable
                    obj_file_name = f"{name}_batch0{b_index + 1}_tr({obj_tr[0]}_{obj_tr[1]}_{obj_tr[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = f"transient_nlos_{name.lower()}_" \
                                f"cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_" \
                                f"cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_" \
                                f"obj_pos_({round(obj_base_pos[0] + obj_tr[0], 2)}_{round(obj_base_pos[1] + obj_tr[1], 2)}_{round(obj_base_pos[2] + obj_tr[2], 2)})_" \
                                f"obj_rot_({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})_" \
                                f"wall_{roughness_str}.xml"  # Set the output file name in a way that contains all the relevant info
                else:
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_1 = f"{elm[1][2][0]}_batch0{b_index + 1}_tr({obj_tr_1[0]}_{obj_tr_1[1]}_{obj_tr_1[2]})_rot({obj_rot_1[0]}_{obj_rot_1[1]}_{obj_rot_1[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_2 = f"{elm[1][2][1]}_batch0{b_index + 1}_tr({obj_tr_2[0]}_{obj_tr_2[1]}_{obj_tr_2[2]})_rot({obj_rot_2[0]}_{obj_rot_2[1]}_{obj_rot_2[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = f"transient_nlos_{elm[1][2][0].lower()}+{elm[1][2][1].lower()}_" \
                                f"cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_" \
                                f"obj_pos_({round(obj_base_pos[0] + obj_tr_1[0], 2)}_{round(obj_base_pos[1] + obj_tr_1[1], 2)}_{round(obj_base_pos[2] + obj_tr_1[2], 2)})_({round(obj_base_pos[0] + obj_tr_2[0], 2)}_{round(obj_base_pos[1] + obj_tr_2[1], 2)}_{round(obj_base_pos[2] + obj_tr_2[2], 2)})_" \
                                f"obj_rot_({obj_rot_1[0]}_{obj_rot_1[1]}_{obj_rot_1[2]})_({obj_rot_2[0]}_{obj_rot_2[1]}_{obj_rot_2[2]})_" \
                                f"wall_{roughness_str}.xml"  # Set the output file name in a way that contains all the relevant info

                # Convert camera position and rotation from blender to mitsuba coordinates system
                cam_pos, cam_rot = blender2mitsuba_coord_mapping(cam_pos[0], cam_pos[1], cam_pos[2], cam_rot[0], cam_rot[1], cam_rot[2])

                # Modify the template inserting the desired data
                # (code from: https://stackoverflow.com/questions/37868881/how-to-search-and-replace-text-in-an-xml-file-using-python)
                if name != "Random" and not rough_wall:
                    template_path = templates_folder / "xml_template_std_obj_std_wall.xml"
                elif name != "Random" and rough_wall:
                    template_path = templates_folder / "xml_template_std_obj_rough_wall.xml"
                elif name == "Random" and not rough_wall:
                    template_path = templates_folder / "xml_template_rnd_obj_std_wall.xml"
                else:
                    template_path = templates_folder / "xml_template_rnd_obj_rough_wall.xml"

                if img_shape is None and pattern is None:
                    if name != "Random":
                        # noinspection PyUnboundLocalVariable
                        save_dts_xml(template_path=template_path,
                                     file_path=batch_path / file_name.replace(" ", "_"),
                                     obj_file_name=obj_file_name,
                                     cam_pos=cam_pos,
                                     cam_rot=cam_rot,
                                     roughness=roughness_str)
                    else:
                        # noinspection PyUnboundLocalVariable
                        save_dts_xml(template_path=template_path,
                                     file_path=batch_path / file_name.replace(" ", "_"),
                                     obj_file_name_1=obj_file_name_1,
                                     obj_file_name_2=obj_file_name_2,
                                     cam_pos=cam_pos,
                                     cam_rot=cam_rot,
                                     roughness=roughness_str)
                else:
                    mask = spot_bitmap_gen(img_size=list(img_shape), pattern=pattern)
                    non_zero_pos = nonzero(mask)

                    elm_path = batch_path / file_name.replace(" ", "_")[:-4]
                    if not elm_path.exists():  # Create the output folder if not already present
                        elm_path.mkdir(parents=True)
                    for row in unique(non_zero_pos[0]):
                        for col in unique(non_zero_pos[1]):
                            if name != "Random":
                                save_dts_xml(template_path=template_path,
                                             file_path=elm_path / (file_name[:-4] + f"_r{row}_c{col}.xml").replace(" ", "_"),
                                             # noinspection PyUnboundLocalVariable
                                             obj_file_name=obj_file_name,
                                             cam_pos=cam_pos,
                                             cam_rot=cam_rot,
                                             roughness=roughness_str,
                                             row=row,
                                             col=col)
                            else:
                                save_dts_xml(template_path=template_path,
                                             file_path=elm_path / (file_name[:-4] + f"_r{row}_c{col}.xml").replace(" ", "_"),
                                             # noinspection PyUnboundLocalVariable
                                             obj_file_name_1=obj_file_name_1,
                                             # noinspection PyUnboundLocalVariable
                                             obj_file_name_2=obj_file_name_2,
                                             cam_pos=cam_pos,
                                             cam_rot=cam_rot,
                                             roughness=roughness_str,
                                             row=row,
                                             col=col)


def build_mirror_gt(gt_path: Path, out_path: Path, fov: int, exp_time: float) -> None:
    """
    Build the mirror ground truth.
    Load all the output from mitsuba, put them in standard form. From them extracts:
    - depth_map (radial)
    - alpha_map
    Finally save the obtained data in the out_path folder as a h5 file
    :param gt_path: Path to the folder containing the output from mitsuba
    :param out_path: Path to the folder where the output will be saved
    :param fov: Field of view of the camera
    :param exp_time: Exposure time of the camera
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    batches_folder = read_folders(gt_path)  # Get the list of batches
    data_path = []
    for batch_folder in batches_folder:
        data_folder = read_folders(batch_folder)  # Get the list of data in each batch
        data_path = data_path + data_folder  # Put together (in the same list) all the file present in all the batches

    for file_path in tqdm(data_path, desc="Generating ground truth data"):  # For each file
        file_name = str(Path(file_path).name) + "_GT"  # Get the file name and add the suffix
        tr = transient_loader(file_path)  # Load the data and put them in standard form
        d_map = compute_distance_map(tr, fov, exp_time)  # Compute the distance map (depth radial map)
        a_map = np_zeros(d_map.shape)  # Create the alpha map
        a_map[where(np_sum(tr[:, :, :, -1], axis=0) != 0)] = 1  # Set the alpha map to 1 where there is data in d_map
        save_h5(file_path=out_path / file_name, data={"depth_map": d_map, "alpha_map": a_map}, fermat=False)  # Save the data in the out_path folder as a h5 file


def gen_filter(exp_coeff: float, sigma: float) -> ndarray:
    """
    This function generates a Difference of Gaussian filter with exponential falloff
    :param exp_coeff: exponential falloff
    :param sigma: standard deviation of the gaussian kernel
    :return: DoG filter with an exponential falloff
    """

    t = array([i/10 for i in range(-50, 51, 1)], dtype=float32)
    t = t.reshape(t.size, 1)
    ind = where(t == 0)[0][0]
    delta = zeros(t.shape)
    delta[ind] = 1
    deltas = zeros(t.shape)
    deltas[ind + 1] = 1

    pd = delta + deltas * (-np_exp(-exp_coeff * 0.001))
    g = np_exp(-np_square(t) / sigma**2)
    return convolve(pd[:, 0], g[:, 0], "same")


def compute_discont(tr: ndarray, exp_time: float) -> ndarray:
    """
    This function detects path length discontinuities in each transient
    This function computes the discontinuity map of the given transient
    :param tr: n*m matrix, n: #transient, m: #temporal bins
    :param exp_time: exposure time
    :return: n*k matrix, storing discontinuities in transients, n: #transient, k: #discontinuities per transient
    """

    # PARAMETERS
    num_of_discont = 1  # Number of discontinuity to search for
    exp_coeff = [0.3]  # Model the exponential falloff of the SPAD signal
    sigma_blur = [1]  # Difference of Gaussian, standard deviation
    whether_sort_disconts = True  # Sort the discontinuity by their distance from the center of the image

    # COMPUTE THE DISCONTINUITY
    n_samples = tr.shape[0]
    temp_bin_center = compute_bin_center(exp_time, tr.shape[1])
    num_of_bin_center = 1
    if num_of_bin_center == 1:
        x = array(temp_bin_center, dtype=float32)
    else:
        assert num_of_bin_center == n_samples

    all_disconts = empty((n_samples, num_of_discont))

    tr_vec = tr.reshape(tr.shape[0], tr.shape[1])

    for i in range(n_samples):
        if num_of_bin_center > 1:
            x = temp_bin_center[i, :]
        y = tr_vec[i]
        if np_nanmax(y) != 0:
            y = y / np_nanmax(y)

        # Convolve the transient with the DoG f, and keep the maximum f response
        dgy = full(y.shape, -Inf)
        for exp_val in exp_coeff:
            for sigma in sigma_blur:
                f = gen_filter(exp_val, sigma)
                dgy_one = maximum(convolve(y, f, 'same'), convolve(y, np_flip(f), 'same'))
                dgy = maximum(dgy, dgy_one)
        if np_nanmax(dgy) != 0:
            dgy = dgy / np_nanmax(dgy)

        # Discontinuities correspond to larger f responses
        # noinspection PyUnboundLocalVariable
        locs_peak, peak_info = find_peaks(dgy, prominence=0)  # FIX THE LOCS INDEX SHOULD BE A VALUE OF X
        if locs_peak.size != 0:
            # noinspection PyUnboundLocalVariable
            locs_peak = x[locs_peak]
        p = peak_info["prominences"]
        if p.size != 0 and np_all(p == p[0]):
            ind_p = array([i for i in range(0, p.size)], dtype=int)
        else:
            ind_p = np_argsort(p)[::-1]  # Sort the prominence in descending order

        if ind_p.size == 0:
            ind_p = 0
        if locs_peak.size == 0:
            locs_peak = [nan]
        else:
            locs_peak = locs_peak[ind_p]

        disconts = full([1, num_of_discont], nan)
        if np_size(locs_peak) >= num_of_discont:
            disconts = locs_peak[:num_of_discont]
        else:
            disconts[:np_size(locs_peak)] = locs_peak

        if whether_sort_disconts:
            disconts = np_sort(disconts)

        # store discont
        all_disconts[i, :] = disconts

    return all_disconts


def build_fermat_gt(gt_path: Path, out_path: Path, exp_time: float, img_size: list, grid_size: list, fov: float) -> None:
    """
    Build the fermat ground truth.
    Load all the output from mitsuba, put them in standard form. From them extracts:
    - first discontinuity location (index)
    Finally save the obtained data in the out_path folder as a h5 file
    :param gt_path: Path to the folder containing the output from mitsuba
    :param out_path: Path to the folder where the output will be saved
    :param exp_time: Exposure time of the camera
    :param img_size: Image size [column, row]
    :param grid_size: Pattern shape of the emitter [column, row]
    :param fov: Field of view of the camera (horizontal)
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    batches_folder = read_folders(gt_path)  # Get the list of batches
    data_path = []
    for batch_folder in batches_folder:
        data_folder = read_folders(batch_folder)  # Get the list of data in each batch
        data_path = data_path + data_folder  # Put together (in the same list) all the file present in all the batches

    for file_path in tqdm(data_path, desc="Generating ground truth data"):  # For each file
        file_name = str(Path(file_path).name) + "_GT"  # Get the file name and add the suffix
        tr = grid_transient_loader(transient_path=file_path)  # Load the data and put them in standard form
        tr, _ = prepare_fermat_data(data=tr,
                                    grid_size=grid_size,
                                    img_size=img_size,
                                    fov=fov,
                                    data_clean=True,
                                    exp_time=exp_time)   # Clean the data and put them in standard form for Fermat

        all_disconts = compute_discont(tr, exp_time)  # Compute the discontinuity

        save_h5(file_path=out_path / file_name, data={"tr": tr, "discont_loc": all_disconts}, fermat=True)  # Save the data


def load_dataset(d_path: Path, out_path: Path, freqs: ndarray = None) -> None:
    """
    Load the dataset and save it in the out_path folder
    :param d_path: folder containing the dataset (raw output of mitsuba)
    :param out_path: folder where the dataset will be saved after processing
    :param freqs: frequencies used by the iToF sensor
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    batches_folder = read_folders(d_path)  # Get the list of batches
    data_path = []
    for batch_folder in batches_folder:
        data_folder = read_folders(batch_folder)  # Get the list of data in each batch
        data_path = data_path + data_folder  # Put together (in the same list) all the file present in all the batches

    for file_path in tqdm(data_path, desc="Loading dataset"):  # For each file
        file_name = str(Path(file_path).name)  # Get the file name
        tr = transient_loader(file_path)[:, :, :, 1]  # Load the data and put them in standard form (only green channel is considered)

        if freqs is not None:  # If the frequencies are provided, compute the iToF amplitude and phase
            phi_data = phi(freqs)  # Compute the phi function, required to compute the iToF style output
            tr = swapaxes(moveaxis(tr, 0, -1), 0, 1)  # Reshape the tr data to match the layout that will be used in the following
            tr_phi = matmul(tr, phi_data.T)  # Compute the iToF transient data
            amp, phs = amp_phi_compute(tr_phi)  # Compute the amplitude and phase of the iToF transient data

            save_h5(file_path=out_path / file_name, data={"data": tr, "tr_itof": tr_phi, "amp_itof": amp, "phase_itof": phs}, fermat=False)  # Save the data in the out_path folder as a h5 file
        else:
            save_h5(file_path=out_path / file_name, data={"data": tr}, fermat=True)  # Save the data in the out_path folder as a h5 file


def fuse_dt_gt_mirror(d_path: Path, gt_path: Path, out_path: Path, def_obj_pos: list) -> None:
    """
    Fuse the dataset and the ground truth together in the same h5 file
    :param d_path: folder containing the dataset (already processed and in h5 form)
    :param gt_path: folder containing the ground truth (already processed and in h5 form)
    :param out_path: folder where the fused dataset will be saved
    :param def_obj_pos: default object position in the scene
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    d_files_name = [Path(i).name for i in read_files(d_path, "h5")]  # Get the list of files in d_path (only name)
    gt_files_name = [Path(i).name for i in read_files(gt_path, "h5")]  # Get the list of files in gt_path (only name)

    if len(d_files_name) != len(gt_files_name):
        raise ValueError("The number of files in the dataset and the ground truth folder are different")  # Raise an error if the number of files is different

    for d_name in tqdm(d_files_name, desc="Fusing dataset and ground truth"):  # For each file in d_path
        d_name_shortened = d_name[:d_name.find("_wall_")]  # Remove the wall information from the name and also the xml extension

        cam_pos = d_name_shortened[(d_name_shortened.find("cam_pos") + 9):(d_name_shortened.find("cam_rot") - 2)]  # Get the camera position
        cam_rot = d_name_shortened[(d_name_shortened.find("cam_rot") + 9):(d_name_shortened.find("obj_pos") - 2)]  # Get the camera rotation
        cam_pos = [float(i) for i in cam_pos.split("_")]  # Convert the camera position to float
        cam_rot = [int(float(i)) for i in cam_rot.split("_")]  # Convert the camera rotation to int

        obj_name = d_name_shortened[(d_name_shortened.find("nlos") + 5):(d_name_shortened.find("cam") - 1)]  # Get the object name
        obj_pos = d_name_shortened[(d_name_shortened.find("obj_pos") + 9):(d_name_shortened.find("obj_rot") - 2)]  # Get the object position (as string)
        obj_rot = d_name_shortened[(d_name_shortened.find("obj_rot") + 9):(-1)]  # Get the object rotation (as string)
        if d_name_shortened.find("+") == -1:  # If the considered file is characterized by only one object
            obj_pos = [float(i) for i in obj_pos.split("_")]  # Convert the object position to float
            obj_pos = [round(obj_pos[i] - elm, 1) for i, elm in enumerate(def_obj_pos)]  # Remove the offset from the object position (get the object translation)
            obj_rot = [int(float(i)) for i in obj_rot.split("_")]  # Convert the object rotation to int
            gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name}_tr({obj_pos[0]}_{obj_pos[1]}_{obj_pos[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})_GT.h5"  # Compose the name of the gt file starting from the one of the dataset
        else:  # If the considered file is characterized by two objects
            obj_name1 = obj_name.split("+")[0]  # Get the first object name
            obj_name2 = obj_name.split("+")[1]  # Get the second object name
            obj_pos1 = obj_pos.split(")_(")[0]  # Get the first object position
            obj_pos2 = obj_pos.split(")_(")[1]  # Get the second object position
            obj_pos1_raw = [float(i) for i in obj_pos1.split("_")]  # Convert the first object position to float
            obj_pos2_raw = [float(i) for i in obj_pos2.split("_")]  # Convert the second object position to float
            obj_pos1 = [round(obj_pos1_raw[i] - elm, 2) for i, elm in enumerate(def_obj_pos)]  # Remove the offset from the first object position (get the object translation)
            obj_pos2 = [round(obj_pos2_raw[i] - elm, 2) for i, elm in enumerate(def_obj_pos)]  # Remove the offset from the second object position (get the object translation)
            obj_rot1 = obj_rot.split(")_(")[0]  # Get the first object rotation
            obj_rot2 = obj_rot.split(")_(")[1]  # Get the second object rotation
            obj_rot1 = [int(float(i)) for i in obj_rot1.split("_")]  # Convert the first object rotation to int
            obj_rot2 = [int(float(i)) for i in obj_rot2.split("_")]  # Convert the second object rotation to float
            gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"  # Compose the name of the gt file starting from the one of the dataset

        if gt_name not in gt_files_name:  # If the gt file doesn't exist
            if 0.0 in obj_pos2:
                indexes = [index for index, elm in enumerate(obj_pos2) if elm == 0.0]
                for index in indexes:
                    if index != 0:
                        obj_pos2[index] = -0.0
                # noinspection PyUnboundLocalVariable
                gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
            if gt_name not in gt_files_name:
                if obj_pos2[0] == -0.0:
                    obj_pos2[0] = 0.0
                    gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
                if gt_name not in gt_files_name:
                    raise ValueError(
                        f"The ground truth file is missing ({gt_name})")  # If the gt file doesn't exist, raise an error

        d_file = d_path / d_name  # Compose the path of the dataset file
        gt_file = gt_path / gt_name  # Compose the path of the gt file
        d = load_h5(d_file)  # Load the dataset file
        gt = load_h5(gt_file)  # Load the gt file

        file_name = d_name[:d_name.find(".h5")].replace("-", "n").replace("(", "").replace(")", "").replace(".", "dot")  # Compose the name of the file
        try:
            save_h5(file_path=out_path / file_name,
                    data={"data": d["data"], "tr_itof": d["tr_itof"], "amp_itof": d["amp_itof"],
                          "phase_itof": d["phase_itof"], "depth_map": swapaxes(gt["depth_map"], 0, 1),
                          "alpha_map": swapaxes(gt["alpha_map"], 0, 1)}, fermat=False)  # Save the file
        except KeyError:
            save_h5(file_path=out_path / file_name,
                    data={"data": d["data"], "depth_map": swapaxes(gt["depth_map"], 0, 1),
                          "alpha_map": swapaxes(gt["alpha_map"], 0, 1)}, fermat=False)  # Save the file


def fuse_dt_gt_fermat(d_path: Path, gt_path: Path, out_path: Path, img_size: list, grid_size: list) -> None:
    """
    Fuse the dataset and the ground truth together in the same h5 file
    :param d_path: folder containing the dataset (already processed and in h5 form)
    :param gt_path: folder containing the ground truth (already processed and in h5 form)
    :param out_path: folder where the fused dataset will be saved
    :param img_size: Image size [column, row]
    :param grid_size: Pattern shape of the emitter [column, row]
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    d_files_name = [Path(i).name for i in read_files(d_path, "h5")]  # Get the list of files in d_path (only name)
    gt_files_name = [Path(i).name for i in read_files(gt_path, "h5")]  # Get the list of files in gt_path (only name)

    if len(d_files_name) != len(gt_files_name):
        raise ValueError(
            "The number of files in the dataset and the ground truth folder are different")  # Raise an error if the number of files is different

    for d_name in tqdm(d_files_name, desc="Fusing dataset and ground truth"):  # For each file in d_path
        d_name_shortened = d_name[:d_name.find(".h5")]  # Remove the wall information from the name and also the xml extension

        if d_name not in gt_files_name:  # If the gt file doesn't exist
            raise ValueError("The ground truth file is missing")  # If the gt file doesn't exist, raise an error

        d_file = d_path / d_name  # Compose the path of the dataset file
        gt_file = gt_path / d_name  # Compose the path of the gt file
        d = load_h5(d_file)  # Load the dataset file
        gt = load_h5(gt_file)  # Load the gt file

        tr = d["data"]  # Get the measured data
        mask = spot_bitmap_gen(img_size=img_size,
                               pattern=tuple(grid_size))  # Define the mask that identify the location of the illuminated spots
        mskd_tr = tr[mask]  # Get the measured data corresponding to the illuminated spots
        reshaped_tr = mskd_tr.flatten(order="F")  # Flatten the array column by column

        file_name = d_name_shortened.replace("-", "n").replace("(", "").replace(")", "").replace(".", "dot")  # Compose the name of the file
        save_h5(file_path=out_path / file_name,
                data={"tr": reshaped_tr, "discont_gt": gt["discont_loc"]}, fermat=False)  # Save the file


def build_point_cloud(data: ndarray, out_path: Path, fov: int, img_size: tuple[int, int], alpha: ndarray = None,
                      f_mesh: bool = True, visualize: bool = False) -> (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh):
    """
    Build a point cloud (and mesh) from a depth map
    :param data: depth map
    :param out_path: folder where to save the point cloud and the mesh
    :param fov: field of view of the camera
    :param img_size: size of the image in pixel (width, height)
    :param alpha: alpha map
    :param f_mesh: if True, build a mesh
    :param visualize: flag to visualize the point cloud and the mesh
    :return: point cloud and mesh
    """

    if alpha is not None:  # If the alpha map is provided
        data = data * alpha  # Apply the alpha map

    k_matrix = k_matrix_calculator(fov, list(img_size)[::-1])  # Calculate the K matrix

    pc = undistort_depthmap(dph=np_copy(data),
                            dm="RADIAL",
                            k_ideal=k_matrix,
                            k_real=k_matrix,
                            d_real=array([[0, 0, 0, 0, 0]], dtype=float32))[0]  # Find the x, y, z coordinates of the points in the camera coordinates system

    n_points = count_nonzero(pc[:, :, 0])  # Count the number of points that actually corresponds to an object
    t = zeros([n_points, 3])  # Create a matrix to store the coordinates of the points
    t[:, 0] = pc[:, :, 0][where(pc[:, :, 0] != 0)]  # Store the x coordinates of the points
    t[:, 1] = pc[:, :, 1][where(pc[:, :, 1] != 0)]  # Store the y coordinates of the points
    t[:, 2] = pc[:, :, 2][where(pc[:, :, 2] != 0)]  # Store the z coordinates of the points

    pcd = o3d.geometry.PointCloud()  # Create a new point cloud
    pcd.points = o3d.utility.Vector3dVector(t)  # Set the points
    pcd.estimate_normals(fast_normal_computation=True)  # Estimate the normals

    objs = [pcd]

    if f_mesh:  # If the mesh is requested
        radii = [0.005, 0.01, 0.02, 0.04]  # Create a list of radii
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pcd, radii=o3d.utility.DoubleVector(radii))  # Create the mesh
        objs.append(rec_mesh)

    if visualize:  # If the visualization is requested
        o3d.visualization.draw_geometries(geometry_list=objs,
                                          window_name="Point cloud and mesh visualization",
                                          point_show_normal=True,
                                          mesh_show_back_face=True,
                                          mesh_show_wireframe=True)  # Visualize the point cloud and the mesh

    o3d.io.write_point_cloud(str(out_path / "point_cloud.ply"), pcd)  # Save the point cloud
    if f_mesh:
        o3d.io.write_triangle_mesh(str(out_path / "mesh.ply"), rec_mesh)  # Save the mesh

    return objs


def ampl_ratio_hists(dts_path: Path, out_path: Path) -> None:
    """
    Build the histogram of the amplification ratio
    :param dts_path: path of the dataset
    :param out_path: path of the output folder
    :return: None
    """

    if not (out_path.parent / "ampl_ratios").exists():  # If the output list doesn't exist
        dts_files = read_files(dts_path, "h5")  # Get the list of files in dts_path

        if len(dts_files) == 0:  # If there are no files in dts_path
            raise ValueError("The dataset folder is empty")  # Raise an error

        ampl_ratios = []  # Create a list to store the amplification ratio

        for dts_file in tqdm(dts_files, desc="Computing amplification ratio"):  # For each file in dts_path
            dts = load_h5(dts_file)  # Load the file
            amp = dts["amp_itof"]    # Get the amplitude data
            amp_ratio = swapaxes(amp[..., 0] / amp[..., 1], 0, 1)
            ampl_ratios.append(np_nanmax(amp_ratio) - np_nanmin(amp_ratio))  # Compute the amplification ratio and store it

        save_list(ampl_ratios, out_path.parent / "ampl_ratios")  # Save the amplification ratio
    else:
        ampl_ratios = load_list(out_path.parent / "ampl_ratios")

    plt.hist(ampl_ratios)
    plt.title("Amplitude ratio hists")
    plt.savefig(out_path)
