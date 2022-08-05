from pathlib import Path
from random import seed as rnd_seed, sample as rnd_sample
from lxml import etree as et
from tqdm import trange, tqdm
from numpy import nonzero, unique, zeros as np_zeros, where, sum as np_sum, swapaxes, ndarray, copy as np_copy, array, float32, zeros, count_nonzero, empty, nanmax as np_nanmax, Inf, full
import open3d as o3d

from modules.utilities import create_folder, permute_list, load_list, save_list, blender2mitsuba_coord_mapping, spot_bitmap_gen, read_folders, save_h5, load_h5, read_files, k_matrix_calculator
from modules.transient_handler import transient_loader, compute_distance_map
from modules.fermat_tools import undistort_depthmap, compute_bin_center


def generate_dataset_file(tx_rt_list: list, folder_path: Path, objs: dict) -> None:
    """
    Function that build a .txt file containing all the information about how the dataset has been created
    :param tx_rt_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param folder_path: path of the folder where to save the dataset file
    :param objs: dict containing all the name of the used objects (keys) with the relatives default positions (values)
    """

    if folder_path is not None:
        create_folder(folder_path, ignore="all")  # Create the output folder if not already present

    with open(str(folder_path / "dataset.txt"), "w") as f:  # Open the target .txt file (in ase create it)
        for b_index, batch in enumerate(tx_rt_list):  # Cycle through each batches
            f.write(f"BATCH 0{b_index + 1}:\n")  # Write which batch it is under consideration
            for obj_index, obj in enumerate(batch):  # Cycle through each object
                f.write(f"{list(objs.keys())[obj_index]}:\n")  # Write which object is under consideration
                for i, data in enumerate(obj):  # Cycle through each cam_pos/cam_rot/obj_tr_obj_rot combination for the given object
                    f.write(f"\t- Object {i + 1}:\n")
                    f.write(f"\t\t- camera position -> (x: {data[0][0][0]}, y: {data[0][0][1]}, z: {data[0][0][2]})\n")  # Write the position of the camera
                    f.write(f"\t\t- camera rotation -> (x: {data[0][1][0]}, y: {data[0][1][1]}, z: {data[0][1][2]})\n")  # Write the rotation of the camera
                    if list(objs.keys())[obj_index] != "Random":
                        f.write(f"\t\t- object position -> (x: {round(list(objs.values())[obj_index][0] + data[1][0][0], 2)}, y: {round(list(objs.values())[obj_index][1] + data[1][0][1], 2)}, z: {round(list(objs.values())[obj_index][2] + data[1][0][2], 2)})\n")  # Write the position of the object summing its default position with the applied translation
                        f.write(f"\t\t- object rotation -> (x: {data[1][1][0]}, y: {data[1][1][1]}, z: {data[1][1][2]})\n")  # Write the rotation of the object
                    else:
                        f.write(f"\t\t- considered objects -> {data[1][2][0]} + {data[1][2][0]}\n")
                        f.write(f"\t\t- {data[1][2][0]} position -> (x: {round(objs[data[1][2][0]][0] + data[1][0][0][0], 2)}, y: {round(objs[data[1][2][0]][1] + data[1][0][0][1], 2)}, z: {round(objs[data[1][2][0]][2] + data[1][0][0][2], 2)})\n")  # Write the position of the object summing its default position with the applied translation
                        f.write(f"\t\t- {data[1][2][1]} position -> (x: {round(objs[data[1][2][1]][0] + data[1][1][0][0], 2)}, y: {round(objs[data[1][2][1]][1] + data[1][1][0][1], 2)}, z: {round(objs[data[1][2][1]][2] + data[1][1][0][2], 2)})\n")  # Write the position of the object summing its default position with the applied translation
                        f.write(f"\t\t- {data[1][2][0]} rotation -> (x: {data[1][0][1][0]}, y: {data[1][0][1][1]}, z: {data[1][0][1][2]})\n")  # Write the rotation of the object
                        f.write(f"\t\t- {data[1][2][1]} rotation -> (x: {data[1][1][1][0]}, y: {data[1][1][1][1]}, z: {data[1][1][1][2]})\n")  # Write the rotation of the object
            f.write("\n")


def generate_dataset_list(obj_tr_list: list, obj_full_rot_list: list, obj_partial_rot_list: list, cam_rot_list: list,
                          cam_pos_list: list, n_batches: int, obj_names: list, def_cam_pos: tuple, def_cam_rot: tuple,
                          n_tr_rot_cam: int, n_tr_obj: list, n_rot_obj: list, n_tr_sphere: list,
                          n_tr_obj_rnd: list = None, n_rot_obj_rnd: list = None, folder_path: Path = None, seed: int = None, rnd: bool = False) -> list[list[list]]:
    """
    Function that generate the list of list containing all the final combinations of camera and object location/translation/rotations
    :param obj_tr_list: List that contains all the possible translations that is granted to an object [[possible x translations], [possible y translations], [possible z translations]]
    :param obj_full_rot_list: List that contains all the possible rotations that is granted to an object [[possible x rotations], [possible y rotations], [possible z rotations]]
    :param obj_partial_rot_list: List that contains all the possible rotations that is granted to an object with no rotations over the z axis [[possible x rotations], [possible y rotations], 0]
    :param cam_rot_list: List that contains all the possible rotations that is granted to the camera [[possible x rotations], [possible y rotations], [possible z rotations]]
    :param cam_pos_list: List that contains all the possible positions that is granted to the camera [[possible x positions], [possible y positions], [possible z positions]]
    :param n_batches: Number of different batches that will be generated
    :param obj_names: List that contains the name of every object that will be considered
    :param def_cam_pos: Default camera position (x, y, z)
    :param def_cam_rot: Default camera orientation (x, y, z)
    :param n_tr_rot_cam: Number of different position and/or rotations that I require for the camera
    :param n_tr_obj: List that contains the number of different translations that I want for each object (sphere excluded) for each batch (len(list) = n_batches)
    :param n_rot_obj: List that contains the number of different rotations that I want for each object (sphere excluded) for each batch (len(list) = n_batches)
    :param n_tr_sphere: List that contains the number of different translations that I want for sphere for each batch (len(list) = n_batches)
    :param n_rot_obj_rnd: List that contains the number of different translations that I want for each random object (sphere excluded) for each batch (len(list) = n_batches)
    :param n_tr_obj_rnd: List that contains the number of different rotations that I want for each random object (sphere excluded) for each batch (len(list) = n_batches)
    :param folder_path: path of the folder where to store the object translation and rotations permutation needed by blender
    :param seed: random seed
    :param rnd: use random obj
    :return: [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    """

    if folder_path is not None:
        create_folder(folder_path, ignore="all")  # Create the output folder if not already present

    # Compute all the permutations of the parameter lists
    obj_tr_list = permute_list(obj_tr_list, seed)
    obj_full_rot_list = permute_list(obj_full_rot_list, seed)
    obj_partial_rot_list = permute_list(obj_partial_rot_list, seed)
    cam_rot_list = permute_list(cam_rot_list, seed)
    cam_pos_list = permute_list(cam_pos_list, seed)

    if seed is not None:
        rnd_seed(seed)  # Set the random seed

    tr_rot_list = []  # define the list that will contain all the data

    for b_index in trange(n_batches, desc="Batches", leave=True, position=0):  # Cycle through each batch
        tr_rot_batch = []  # Define the list that will contain the data of the current batch

        # Avoid that two consecutive batches have the same data
        if b_index % 2 == 0:
            obj_tr_tmp = obj_tr_list.copy()
            obj_full_rot_tmp = obj_full_rot_list.copy()
            obj_partial_rot_tmp = obj_partial_rot_list.copy()
            cam_rot_tmp = cam_rot_list.copy()
            cam_pos_tmp = cam_pos_list.copy()

        # Set the cam translation and rotation parameters considering that in some batches the camera position and/or rotation is fixed
        # Translation
        if b_index <= 3:
            cam_tr = [(def_cam_pos[0], def_cam_pos[1], def_cam_pos[2])]
        else:
            # noinspection PyUnboundLocalVariable
            cam_tr = rnd_sample(cam_pos_tmp, n_tr_rot_cam)
            cam_pos_tmp = [x for x in cam_pos_tmp if x not in cam_tr]  # Avoid that two consecutive batches have the same data
        # Rotations
        if b_index <= 1 or 4 <= b_index <= 5:
            cam_rot = [(def_cam_rot[0], def_cam_rot[1], def_cam_rot[2])]
        else:
            # noinspection PyUnboundLocalVariable
            cam_rot = rnd_sample(cam_rot_tmp, n_tr_rot_cam)
            cam_rot_tmp = [x for x in cam_rot_tmp if x not in cam_rot]  # Avoid that two consecutive batches have the same data

        # Sample the correct number of rotation and translation couple (at random) of the camera considering that in some batches the camera position and/or rotation is fixed
        if b_index <= 1:
            cam_tr_rot = [(cam_tr[0], cam_rot[0])]
        elif 2 <= b_index <= 3:
            cam_tr_rot = rnd_sample(permute_list([cam_tr, cam_rot]), len(cam_rot))
        elif 4 <= b_index <= 5:
            cam_tr_rot = rnd_sample(permute_list([cam_tr, cam_rot]), len(cam_tr))
        else:
            cam_tr_rot = rnd_sample(permute_list([cam_tr, cam_rot]), len(cam_tr) + len(cam_rot))

        for name in tqdm(obj_names, desc="Objects", leave=False, position=1):  # Cycle through all the objects
            # Set the object translation and rotations parameter
            if folder_path is not None and (folder_path / f"obj_tr_rot_batch_0{b_index + 1}_{name.lower()}").is_file():
                obj_tr_rot = load_list(folder_path / f"obj_tr_rot_batch_0{b_index + 1}_{name.lower()}")
            else:
                # Translations
                if name != "Sphere" and name != "Random":
                    # noinspection PyUnboundLocalVariable
                    obj_tr = rnd_sample(obj_tr_tmp, n_tr_obj[b_index])
                    if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                        obj_tr_tmp = [x for x in obj_tr_tmp if x not in obj_tr]
                elif name == "Random":
                    obj_tr = rnd_sample(obj_tr_tmp, n_tr_obj_rnd[b_index])
                    if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                        obj_tr_tmp = [x for x in obj_tr_tmp if x not in obj_tr]
                else:
                    obj_tr = rnd_sample(obj_tr_tmp, n_tr_sphere[b_index])
                    if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                        obj_tr_tmp = [x for x in obj_tr_tmp if x not in obj_tr]
                # Rotations
                if name == "Cube" or name == "Parallelepiped" or name == "Concave plane" or name == "Cube + sphere":
                    # noinspection PyUnboundLocalVariable
                    obj_rot = rnd_sample(obj_full_rot_tmp, n_rot_obj[b_index])
                    if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                        obj_full_rot_tmp = [x for x in obj_full_rot_tmp if x not in obj_rot]
                elif name == "Cone" or name == "Cylinder" or name == "Cylinder + cone" or name == "Sphere + cone":
                    # noinspection PyUnboundLocalVariable
                    obj_rot = rnd_sample(obj_partial_rot_tmp, n_rot_obj[b_index])
                    if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                        obj_partial_rot_tmp = [x for x in obj_partial_rot_tmp if x not in obj_rot]
                elif name == "Random":
                    obj_rot = rnd_sample(obj_partial_rot_tmp, n_rot_obj_rnd[b_index])
                    if name == obj_names[-1]:  # Avoid that two consecutive batches have the same data
                        obj_partial_rot_tmp = [x for x in obj_partial_rot_tmp if x not in obj_rot]
                else:
                    obj_rot = [(0, 0, 0)]

                # Sample the correct number of rotation and translation couple (at random) of the object
                if name != "Sphere":
                    obj_tr_rot = rnd_sample(permute_list([obj_tr, obj_rot]), len(obj_tr) + len(obj_rot))
                else:
                    obj_tr_rot = rnd_sample(permute_list([obj_tr, obj_rot]), len(obj_tr))

                if folder_path is not None:
                    save_list(obj_tr_rot, folder_path / f"obj_tr_rot_batch0{b_index + 1}_{name.lower()}")  # Save the list of the object translation and rotations

            # Sample the correct number of rotation and translation couple (at random) for both the object and the camera
            if b_index == 0:
                tr_rot_batch.append(rnd_sample(permute_list([cam_tr_rot, obj_tr_rot]), len(obj_tr_rot)))
            else:
                tr_rot_batch.append(rnd_sample(permute_list([cam_tr_rot, obj_tr_rot]), int(len(cam_tr_rot) / len(obj_names)) + len(obj_tr_rot)))

        tr_rot_list.append(tr_rot_batch)

    if rnd:
        tr_rot_list = rnd_obj_data_generator(tr_rot_list, obj_names[:-1])

    if folder_path is not None:
        save_list(tr_rot_list, folder_path / "tr_rot_list")  # Save the list of the object translation and rotations

    return tr_rot_list


def rnd_obj_data_generator(tr_rot_list: list[list[list]], obj_names: list) -> list[list[list]]:
    """
    Function that define the rotation, translation and parent objects of each randomly generated one
    :param tr_rot_list: list of all the rotations and translation
    :param obj_names: list of the names of the possible object to combine
    :return: Updated input list [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], ..., obj_rand[(camera_pos, camera_rot), ((obj1_tr, obj1_rot), (obj2_tr, obj2_rot), (obj1_name, obj2_name))]], batch2[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], ..., obj_rand[(camera_pos, camera_rot), ((obj1_tr, obj1_rot), (obj2_tr, obj2_rot), (obj1_name, obj2_name))], ...]
    """
    delta_tr_permute = permute_list([[i / 100 for i in range(-15, 30)],
                                     [i / 100 for i in range(-15, 30)],
                                     [i / 100 for i in range(-15, 30)]])
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


def generate_dataset_xml(tr_rot_list: list, template: Path, folder_path: Path, objs: dict) -> None:
    """
    Function that given the template.xml file and all the chosen position?translation?rotation combinations generate the correspondent .xml files
    :param tr_rot_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param template: template file (.xml)
    :param folder_path: path of the output folder
    :param objs: Dict that contains the name and default position of every object that will be considered
    """

    create_folder(folder_path)  # Create the output folder if not already present

    for b_index, batch in tqdm(enumerate(tr_rot_list), desc="Batches", leave=True, position=0):  # Cycle through each batch
        create_folder((folder_path / f"batch0{b_index + 1}"))  # Create the batch folder if not already present
        for o_index, obj in tqdm(enumerate(batch), desc="Objects", leave=False, position=1):  # Cycle through each object
            name = list(objs.keys())[o_index]  # Extract the object name
            for elm in tqdm(obj, desc="File", leave=False, position=2):  # Cycle through each position/translation/rotation combination
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

                if name != "Random":
                    # noinspection PyUnboundLocalVariable
                    obj_file_name = f"{name}_batch0{b_index + 1}_tr({obj_tr[0]}_{obj_tr[1]}_{obj_tr[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = f"transient_nlos_{name.lower()}_cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_obj_pos_({round(objs[name][0] + obj_tr[0], 2)}_{round(objs[name][1] + obj_tr[1], 2)}_{round(objs[name][2] + obj_tr[2], 2)})_obj_rot_({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]}).xml"  # Set the output file name in a way that contains all the relevant info
                else:
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_1 = f"{elm[1][2][0]}_batch0{b_index + 1}_tr({obj_tr_1[0]}_{obj_tr_1[1]}_{obj_tr_1[2]})_rot({obj_rot_1[0]}_{obj_rot_1[1]}_{obj_rot_1[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_2 = f"{elm[1][2][1]}_batch0{b_index + 1}_tr({obj_tr_2[0]}_{obj_tr_2[1]}_{obj_tr_2[2]})_rot({obj_rot_2[0]}_{obj_rot_2[1]}_{obj_rot_2[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = f"transient_nlos_{elm[1][2][0].lower()}+{elm[1][2][1].lower()}_" \
                                f"cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_" \
                                f"obj_pos_({round(objs[elm[1][2][0]][0] + obj_tr_1[0], 2)}_{round(objs[elm[1][2][0]][1] + obj_tr_1[1], 2)}_{round(objs[elm[1][2][0]][2] + obj_tr_1[2], 2)})_({round(objs[elm[1][2][1]][0] + obj_tr_2[0], 2)}_{round(objs[elm[1][2][1]][1] + obj_tr_2[1], 2)}_{round(objs[elm[1][2][1]][2] + obj_tr_2[2], 2)})_" \
                                f"obj_rot_({obj_rot_1[0]}_{obj_rot_1[1]}_{obj_rot_1[2]})_({obj_rot_2[0]}_{obj_rot_2[1]}_{obj_rot_2[2]}).xml"  # Set the output file name in a way that contains all the relevant info

                # Convert camera position and rotation from blender to mitsuba coordinates system
                cam_pos, cam_rot = blender2mitsuba_coord_mapping(cam_pos[0], cam_pos[1], cam_pos[2], cam_rot[0], cam_rot[1], cam_rot[2])

                # Modify the template inserting the desired data
                # (code from: https://stackoverflow.com/questions/37868881/how-to-search-and-replace-text-in-an-xml-file-using-python)
                if name == "Random":
                    template_path = template.parent.absolute() / "template_rnd.xml"
                else:
                    template_path = template
                with open(str(template_path), encoding="utf8") as f:
                    tree = et.parse(f)
                    root = tree.getroot()

                    for elem in root.getiterator():
                        try:
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
                        except KeyError:
                            pass
                tree.write(str(folder_path / f"batch0{b_index + 1}" / file_name.replace(" ", "_")), method="xml", encoding="utf8")


def generate_dataset_xml_splitted(tr_rot_list: list, template: Path, folder_path: Path, objs: dict, img_shape: list, pattern: list) -> None:
    """
    Function that given the template.xml file and all the chosen position?translation?rotation combinations generate the correspondent .xml files
    :param tr_rot_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param template: template file (.xml)
    :param folder_path: path of the output folder
    :param img_shape: shape of the image, col, row
    :param pattern: grid pattern
    :param objs: Dict that contains the name and default position of every object that will be considered
    """

    create_folder(folder_path)  # Create the output folder if not already present

    for b_index, batch in tqdm(enumerate(tr_rot_list), desc="Batches", leave=True, position=0):  # Cycle through each batch
        batch_folder = folder_path / f"batch0{b_index + 1}"
        create_folder(batch_folder)  # Create the batch folder if not already present
        for o_index, obj in tqdm(enumerate(batch), desc="Objects", leave=False, position=1):  # Cycle through each object
            name = list(objs.keys())[o_index]  # Extract the object name
            for e_index, elm in tqdm(enumerate(obj), desc="File", leave=False, position=2):  # Cycle through each position/translation/rotation combination
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

                if name != "Random":
                    # noinspection PyUnboundLocalVariable
                    obj_file_name = f"{name}_batch0{b_index + 1}_tr({obj_tr[0]}_{obj_tr[1]}_{obj_tr[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = f"transient_nlos_{name.lower()}_cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_obj_pos_({round(objs[name][0] + obj_tr[0], 2)}_{round(objs[name][1] + obj_tr[1], 2)}_{round(objs[name][2] + obj_tr[2], 2)})_obj_rot_({round(obj_rot[0], 2)}_{round(obj_rot[1], 2)}_{round(obj_rot[2], 2)})"  # Set the output file name in a way that contains all the relevant info
                else:
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_1 = f"{elm[1][2][0]}_batch0{b_index + 1}_tr({obj_tr_1[0]}_{obj_tr_1[1]}_{obj_tr_1[2]})_rot({obj_rot_1[0]}_{obj_rot_1[1]}_{obj_rot_1[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_2 = f"{elm[1][2][1]}_batch0{b_index + 1}_tr({obj_tr_2[0]}_{obj_tr_2[1]}_{obj_tr_2[2]})_rot({obj_rot_2[0]}_{obj_rot_2[1]}_{obj_rot_2[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = f"transient_nlos_{elm[1][2][0].lower()}+{elm[1][2][1].lower()}_" \
                                f"cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_" \
                                f"obj_pos_({round(objs[elm[1][2][0]][0] + obj_tr_1[0], 2)}_{round(objs[elm[1][2][0]][1] + obj_tr_1[1], 2)}_{round(objs[elm[1][2][0]][2] + obj_tr_1[2], 2)})_({round(objs[elm[1][2][1]][0] + obj_tr_2[0], 2)}_{round(objs[elm[1][2][1]][1] + obj_tr_2[1], 2)}_{round(objs[elm[1][2][1]][2] + obj_tr_2[2], 2)})_" \
                                f"obj_rot_({round(obj_rot_1[0], 2)}_{round(obj_rot_1[1], 2)}_{round(obj_rot_1[2], 2)})_({round(obj_rot_2[0], 2)}_{round(obj_rot_2[1], 2)}_{round(obj_rot_2[2], 2)})"  # Set the output file name in a way that contains all the relevant info

                # Convert camera position and rotation from blender to mitsuba coordinates system
                cam_pos, cam_rot = blender2mitsuba_coord_mapping(cam_pos[0], cam_pos[1], cam_pos[2], cam_rot[0], cam_rot[1], cam_rot[2])

                # Modify the template inserting the desired data
                # (code from: https://stackoverflow.com/questions/37868881/how-to-search-and-replace-text-in-an-xml-file-using-python)
                if name == "Random":
                    template_path = template.parent.absolute() / "template_rnd.xml"
                else:
                    template_path = template

                mask = spot_bitmap_gen(img_size=img_shape,
                                       pattern=tuple(pattern))
                non_zero_pos = nonzero(mask)

                elm_path = batch_folder / file_name
                create_folder(elm_path)
                for row in unique(non_zero_pos[0]):
                    for col in unique(non_zero_pos[1]):
                        with open(str(template_path), encoding="utf8") as f:
                            tree = et.parse(f)
                            root = tree.getroot()

                            for elem in root.getiterator():
                                try:
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
                                    elif elem.attrib["value"] == "o_x":
                                        elem.attrib["value"] = str(col)
                                    elif elem.attrib["value"] == "o_y":
                                        elem.attrib["value"] = str(row)
                                    elif elem.attrib["value"] == "t_name":
                                        elem.attrib["value"] = f"bitmap_r{row}_c{col}"
                                except KeyError:
                                    pass
                        tree.write(str(elm_path / (file_name.replace(" ", "_") + f"_r{row}_c{col}.xml")), method="xml", encoding="utf8")


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


def build_fermat_gt(gt_path: Path, out_path: Path, fov: int, exp_time: float) -> None:
    """
    Build the fermat ground truth.
    Load all the output from mitsuba, put them in standard form. From them extracts:
    - first discontinuity location (index)
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
        num_of_discont = [1]  # Number of discontinuity to search for
        exp_coeff = [0.3]  # Model the exponential falloff of the SPAD signal
        sigma_blur = [1]  # Difference of Gaussian, standard deviation

        file_name = str(Path(file_path).name) + "_GT"  # Get the file name and add the suffix
        tr = transient_loader(file_path)  # Load the data and put them in standard form
        tr_samples = tr.size[0] * tr.size[1]
        temp_bin_center = compute_bin_center(exp_time, tr.shape[2])
        num_of_bin_center = len(temp_bin_center)
        if num_of_bin_center == 1:
            x = num_of_bin_center
        else:
            assert num_of_bin_center == tr_samples

        all_disconts = empty((tr_samples, num_of_discont))

        for i in range(tr_samples):
            if num_of_bin_center > 1:
                x = temp_bin_center[i, :]
            y = tr[i, :]
            y = y / np_nanmax(y)
        '''
            # Convolve the transient with the DoG filter, and keep the maximum filter response
            dgy = full((y.size, y.size), -Inf)
            for exp_val in exp_coeff:
                for sigma in sigma_blur:
                    filter = generateFilter(exp_val, sigma)
                    dgy_one = np_nanmax(conv(y, filter, 'same'), conv(y, filter(end:-1: 1), 'same'))
                    dgy = np_nanmax(dgy, dgy_one)

            dgy = dgy / np_nanmax(dgy)

            # Discontinuities correspond to larger filter responses
            [~, locsPeak, ~, p] = findpeaks(dgy, x, 'MinPeakProminence', 0)
            [~, indsP] = sort(p, 'descend');
            locsPeak = locsPeak(indsP);

            disconts = nan(1, num_of_discont)
            if numel(locsPeak) >= num_of_discont:
                disconts = locsPeak(1: num_of_discont);
            else:
                disconts(1: numel(locsPeak)) = locsPeak

            if whetherSortDisconts:
                disconts = sort(disconts, 'ascend')

            # store discont
            all_disconts[i, :] = disconts
            save_h5(file_path=out_path / file_name, data={"depth_map": d_map, "alpha_map": a_map}, fermat=True)  # Save the data
        '''


def load_dataset(d_path: Path, out_path: Path) -> None:
    """
    Load the dataset and save it in the out_path folder
    :param d_path: folder containing the dataset (raw output of mitsuba)
    :param out_path: folder where the dataset will be saved after processing
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
        save_h5(file_path=out_path / file_name, data={"data": tr}, fermat=True)  # Save the data in the out_path folder as a h5 file


def fuse_dt_gt(d_path: Path, gt_path: Path, out_path: Path) -> None:
    """
    Fuse the dataset and the ground truth together in the same h5 file
    :param d_path: folder containing the dataset (already processed and in h5 form)
    :param gt_path: folder containing the ground truth (already processed and in h5 form)
    :param out_path: folder where the fused dataset will be saved
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    d_files_name = [Path(i).name for i in read_files(d_path, "h5")]  # Get the list of files in d_path (only name)
    gt_files_name = [Path(i).name for i in read_files(gt_path, "h5")]  # Get the list of files in gt_path (only name)

    if len(d_files_name) != len(gt_files_name):
        raise ValueError("The number of files in the dataset and the ground truth folder are different")  # Raise an error if the number of files is different

    for d_name in tqdm(d_files_name, desc="Fusing dataset and ground truth"):  # For each file in d_path
        d_name_shortened = d_name.replace("cam_pos_(1.5_-1_1.65)_cam_rot_(90_0_50)_", "")[:-3]  # Remove the cam_pos and cam_rot from the file name since I'm considering just the fixed camera position
        obj_name = d_name_shortened[(d_name_shortened.find("nlos") + 5):(d_name_shortened.find("obj") - 1)]  # Get the object name
        obj_pos = d_name_shortened[(d_name_shortened.find("obj_pos") + 9):(d_name_shortened.find("obj_rot") - 2)]  # Get the object position (as string)
        obj_rot = d_name_shortened[(d_name_shortened.find("obj_rot") + 9):(-1)]  # Get the object rotation (as string)
        if d_name_shortened.find("+") == -1:  # If the considered file is characterized by only one object
            obj_pos = [float(i) for i in obj_pos.split("_")]  # Convert the object position to float
            obj_pos = [round(obj_pos[i] - elm, 1) for i, elm in enumerate([1.5, 1.0, 1.65])]  # Remove the offset from the object position (get the object translation)
            obj_rot = [int(float(i)) for i in obj_rot.split("_")]  # Convert the object rotation to int
            gt_name = f"transient_nlos_{obj_name}_tr({obj_pos[0]}_{obj_pos[1]}_{obj_pos[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})_GT.h5"  # Compose the name of the gt file starting from the one of the dataset
        else:  # If the considered file is characterized by two objects
            obj_name1 = obj_name.split("+")[0]  # Get the first object name
            obj_name2 = obj_name.split("+")[1]  # Get the second object name
            obj_pos1 = obj_pos.split(")_(")[0]  # Get the first object position
            obj_pos2 = obj_pos.split(")_(")[1]  # Get the second object position
            obj_pos1_raw = [float(i) for i in obj_pos1.split("_")]  # Convert the first object position to float
            obj_pos2_raw = [float(i) for i in obj_pos2.split("_")]  # Convert the second object position to float
            obj_pos1 = [round(obj_pos1_raw[i] - elm, 1) for i, elm in enumerate([1.5, 1.0, 1.65])]  # Remove the offset from the first object position (get the object translation)
            obj_pos2 = [round(obj_pos2_raw[i] - elm, 1) for i, elm in enumerate([1.5, 1.0, 1.65])]  # Remove the offset from the second object position (get the object translation)
            obj_rot1 = obj_rot.split(")_(")[0]  # Get the first object rotation
            obj_rot2 = obj_rot.split(")_(")[1]  # Get the second object rotation
            obj_rot1 = [int(float(i)) for i in obj_rot1.split("_")]  # Convert the first object rotation to int
            obj_rot2 = [float(i) for i in obj_rot2.split("_")]  # Convert the second object rotation to float
            gt_name = f"transient_nlos_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"  # Compose the name of the gt file starting from the one of the dataset

        if gt_name not in gt_files_name:  # If the gt file doesn't exist
            # noinspection PyUnboundLocalVariable
            obj_pos1 = [round(obj_pos1_raw[i] - elm, 1) for i, elm in enumerate([1.5, 1.0, 1.55])]  # Change offset to remove and check again if the gt_file exists (this is due ti an error in the labeling of the dataset files)
            # noinspection PyUnboundLocalVariable
            gt_name = f"transient_nlos_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
            if gt_name not in gt_files_name:
                # noinspection PyUnboundLocalVariable
                obj_pos1 = [round(obj_pos1_raw[i] - elm, 1) for i, elm in enumerate([1.5, 1.0, 1.65])]
                # noinspection PyUnboundLocalVariable
                obj_pos2 = [round(obj_pos2_raw[i] - elm, 1) for i, elm in enumerate([1.5, 1.0, 1.55])]
                # noinspection PyUnboundLocalVariable
                gt_name = f"transient_nlos_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
                if gt_name not in gt_files_name:
                    # noinspection PyUnboundLocalVariable
                    obj_pos1 = [round(obj_pos1_raw[i] - elm, 1) for i, elm in enumerate([1.5, 1.0, 1.55])]
                    # noinspection PyUnboundLocalVariable
                    gt_name = f"transient_nlos_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
                    if gt_name not in gt_files_name:
                        raise ValueError("The ground truth file is missing")  # If the gt file doesn't exist, raise an error

        d_file = d_path / d_name  # Compose the path of the dataset file
        gt_file = gt_path / gt_name  # Compose the path of the gt file
        d = load_h5(d_file)  # Load the dataset file
        gt = load_h5(gt_file)  # Load the gt file

        file_name = d_name_shortened.replace("-", "n").replace("(", "").replace(")", "").replace(".", "dot")  # Compose the name of the file

        save_h5(file_path=out_path / file_name, data={"data": d["data"], "depth_map": swapaxes(gt["depth_map"], 0, 1), "alpha_map": swapaxes(gt["alpha_map"], 0, 1)}, fermat=False)  # Save the file


def build_point_cloud(data: ndarray, out_path: Path, fov: int, img_size: tuple[int, int], alpha: ndarray = None, f_mesh: bool = True, visualize: bool = False) -> (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh):
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