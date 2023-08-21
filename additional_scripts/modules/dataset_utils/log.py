from pathlib import Path
from random import seed as rnd_seed, sample as rnd_sample, shuffle as rnd_shuffle
from tqdm import trange, tqdm
from ..utilities import permute_list, load_list, save_list
from .build import rnd_obj_data_generator


def generate_dataset_file(
    tx_rt_list: list,
    folder_path: Path,
    obj_names: tuple,
    obj_base_pos: tuple,
    wall_roughness: list,
) -> None:
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

    with open(
        str(folder_path / "dataset.txt"), "w"
    ) as f:  # Open the target .txt file (in ase create it)
        for b_index, batch in enumerate(tx_rt_list):  # Cycle through each batches
            f.write(
                f"BATCH 0{b_index + 1}:\n"
            )  # Write which batch it is under consideration
            for obj_index, obj in enumerate(batch):  # Cycle through each object
                f.write(
                    f"{obj_names[obj_index]}:\n"
                )  # Write which object is under consideration
                for i, data in enumerate(
                    obj
                ):  # Cycle through each cam_pos/cam_rot/obj_tr_obj_rot combination for the given object
                    f.write(f"\t- Object {i + 1}:\n")
                    f.write(
                        f"\t\t- camera position -> (x: {data[0][0][0]}, y: {data[0][0][1]}, z: {data[0][0][2]})\n"
                    )  # Write the position of the camera
                    f.write(
                        f"\t\t- camera rotation -> (x: {data[0][1][0]}, y: {data[0][1][1]}, z: {data[0][1][2]})\n"
                    )  # Write the rotation of the camera
                    if obj_names[obj_index] != "Random":
                        f.write(
                            f"\t\t- object position -> (x: {round(obj_base_pos[0] + data[1][0][0], 2)}, y: {round(obj_base_pos[1] + data[1][0][1], 2)}, z: {round(obj_base_pos[2] + data[1][0][2], 2)})\n"
                        )  # Write the position of the object summing its default position with the applied translation
                        f.write(
                            f"\t\t- object rotation -> (x: {data[1][1][0]}, y: {data[1][1][1]}, z: {data[1][1][2]})\n"
                        )  # Write the rotation of the object
                        if len(wall_roughness[b_index][0]) == 0:
                            f.write(
                                f"\t\t- wall roughness -> diffuse\n"
                            )  # Write the wall roughness of the object
                        else:  # If the current batch requires roughness
                            f.write(
                                f"\t\t- wall roughness -> {wall_roughness[b_index][obj_index][i]}\n"
                            )  # Write the wall roughness of the object
                    else:
                        f.write(
                            f"\t\t- considered objects -> {data[1][2][0]} + {data[1][2][0]}\n"
                        )
                        f.write(
                            f"\t\t- {data[1][2][0]} position -> (x: {round(obj_base_pos[0] + data[1][0][0][0], 2)}, y: {round(obj_base_pos[1] + data[1][0][0][1], 2)}, z: {round(obj_base_pos[2] + data[1][0][0][2], 2)})\n"
                        )  # Write the position of the object summing its default position with the applied translation
                        f.write(
                            f"\t\t- {data[1][2][1]} position -> (x: {round(obj_base_pos[0] + data[1][1][0][0], 2)}, y: {round(obj_base_pos[1] + data[1][1][0][1], 2)}, z: {round(obj_base_pos[2] + data[1][1][0][2], 2)})\n"
                        )  # Write the position of the object summing its default position with the applied translation
                        f.write(
                            f"\t\t- {data[1][2][0]} rotation -> (x: {data[1][0][1][0]}, y: {data[1][0][1][1]}, z: {data[1][0][1][2]})\n"
                        )  # Write the rotation of the object
                        f.write(
                            f"\t\t- {data[1][2][1]} rotation -> (x: {data[1][1][1][0]}, y: {data[1][1][1][1]}, z: {data[1][1][1][2]})\n"
                        )  # Write the rotation of the object
                        if (
                            len(wall_roughness[b_index][0]) == 0
                        ):  # If the current batch doesn't require roughness
                            f.write(
                                f"\t\t- wall roughness -> diffuse\n"
                            )  # Write the wall roughness of the object
                        else:  # If the current batch requires roughness
                            f.write(
                                f"\t\t- wall roughness -> {wall_roughness[b_index][obj_index][i]}\n"
                            )  # Write the wall roughness of the object
            f.write(
                "\n-------------------------------------------------------------------\n\n"
            )


def generate_dataset_list(
    obj_tr_list: list,
    obj_full_rot_list: list,
    obj_partial_rot_list: list,
    cam_rot_list: list,
    cam_pos_list: list,
    n_classes: int,
    n_batches: int,
    n_elm_per_class: int,
    obj_names: tuple,
    def_cam_pos: tuple,
    def_cam_rot: tuple,
    roughness_values: tuple,
    folder_path: Path = None,
    seed: int = None,
) -> tuple[list[list[list]], list]:
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

    if (
        folder_path is not None
        and (folder_path / "tr_rot_list").is_file()
        and (folder_path / "roughness_list").is_file()
    ):  # If the file already exists, load it
        tr_rot_list = load_list(
            folder_path / "tr_rot_list"
        )  # Load the list of all the possible translations and rotations
        roughness_list = load_list(
            folder_path / "roughness_list"
        )  # Load the list of all the possible roughness values
        return tr_rot_list, roughness_list

        # Compute all the permutations of the parameter lists
    obj_tr_list = permute_list(obj_tr_list, seed)
    obj_full_rot_list = permute_list(obj_full_rot_list, seed)
    obj_partial_rot_list = permute_list(obj_partial_rot_list, seed)
    cam_rot_list = permute_list(cam_rot_list, seed)
    cam_pos_list = permute_list(cam_pos_list, seed)

    # Compute how many elements is required for each class
    n_objs_cl_1_3 = int(n_elm_per_class / len(obj_names))  # fixed camera location
    n_objs_cl_2_4 = int(n_elm_per_class / (len(obj_names) + 1))  # no fixed position

    tr_rot_list = []  # define the list that will contain all the data
    roughness_list = []  # define the list that will contain all the roughness data

    for c_index in trange(
        n_classes,
        desc="Generating all the camera/object location, class by class",
        leave=True,
    ):  # Cycle through each batch
        obj_data = []  # Define the list that will contain the data of the current class
        rough_data = (
            []
        )  # Define the list that will contain the roughness data of the current class

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
            cam_pos_tmp = [
                x for x in cam_pos_tmp if x not in cam_tr
            ]  # Avoid that two consecutive batches have the same data
            # noinspection PyUnboundLocalVariable
            cam_rot = rnd_sample(cam_rot_tmp, int(n_objs_cl_2_4 / 2))
            cam_rot_tmp = [
                x for x in cam_rot_tmp if x not in cam_rot
            ]  # Avoid that two consecutive batches have the same data
            cam_tr_rot = rnd_sample(
                permute_list([cam_tr, cam_rot]), n_objs_cl_2_4
            )  # Randomly select the camera translation and rotation (the correct number of combinations)
            rnd_shuffle(
                cam_tr_rot
            )  # Randomly shuffle the camera translation and rotation

        for name in tqdm(
            obj_names, desc="Defining all the objects locations", leave=False
        ):  # Cycle through all the objects
            # Set the object translation and rotations parameter
            if c_index == 0 or c_index == 2:
                n_obj_to_sample = n_objs_cl_1_3
            else:
                n_obj_to_sample = n_objs_cl_2_4

            # Translations
            if name != "Sphere":
                obj_tr = rnd_sample(obj_tr_tmp, int(n_obj_to_sample / 2))
                if (
                    name == obj_names[-1]
                ):  # Avoid that two consecutive batches have the same data
                    obj_tr_tmp = [x for x in obj_tr_tmp if x not in obj_tr]
            else:
                obj_tr = rnd_sample(obj_tr_tmp, n_obj_to_sample)
                if (
                    name == obj_names[-1]
                ):  # Avoid that two consecutive batches have the same data
                    obj_tr_tmp = [x for x in obj_tr_tmp if x not in obj_tr]
            # Rotations
            if (
                name == "Cube"
                or name == "Parallelepiped"
                or name == "Concave plane"
                or name == "Random"
            ):
                # noinspection PyUnboundLocalVariable
                obj_rot = rnd_sample(obj_full_rot_tmp, int(n_obj_to_sample / 2))
                if (
                    name == obj_names[-1]
                ):  # Avoid that two consecutive batches have the same data
                    obj_full_rot_tmp = [x for x in obj_full_rot_tmp if x not in obj_rot]
            elif name == "Cone" or name == "Cylinder":
                # noinspection PyUnboundLocalVariable
                obj_rot = rnd_sample(obj_partial_rot_tmp, int(n_obj_to_sample / 2))
                if (
                    name == obj_names[-1]
                ):  # Avoid that two consecutive batches have the same data
                    obj_partial_rot_tmp = [
                        x for x in obj_partial_rot_tmp if x not in obj_rot
                    ]
            else:
                obj_rot = [(0, 0, 0)]

            # Sample the correct number of rotation and translation couple (at random) of the object
            obj_tr_rot = rnd_sample(permute_list([obj_tr, obj_rot]), n_obj_to_sample)
            rnd_shuffle(
                obj_tr_rot
            )  # Randomly shuffle the list of translation and rotation

            # Sample the correct number of rotation and translation couple (at random) for both the object and the camera
            obj_data.append(
                rnd_sample(permute_list([cam_tr_rot, obj_tr_rot]), n_objs_cl_1_3)
            )

            # Sample the roughness of the wall
            if c_index > 1:
                elm_roughness = []
                for i in range(n_objs_cl_1_3):
                    elm_roughness.append(rnd_sample(roughness_values, 1)[0])
                rough_data.append(elm_roughness)

        for b_index in range(n_batches):
            batch_data = (
                []
            )  # Define the list that will contain the data of the current batch
            batch_roughness = (
                []
            )  # Define the list that will contain the roughness data of the current batch
            for obj_data_i in obj_data:
                batch_data.append(
                    obj_data_i[
                        (b_index * int(len(obj_data_i) / n_batches)) : (
                            (b_index + 1) * int(len(obj_data_i) / n_batches)
                        )
                    ]
                )
            if c_index <= 1:
                batch_roughness.append([])
            else:
                for rough_data_i in rough_data:
                    batch_roughness.append(
                        rough_data_i[
                            (b_index * int(len(rough_data_i) / n_batches)) : (
                                (b_index + 1) * int(len(rough_data_i) / n_batches)
                            )
                        ]
                    )

            tr_rot_list.append(
                batch_data
            )  # Add the current batch to the list of all the batches
            roughness_list.append(
                batch_roughness
            )  # Add the current batch to the list of all the batches

    tr_rot_list = rnd_obj_data_generator(
        tr_rot_list, obj_names[:-1]
    )  # Add the information relative to the random object

    if folder_path is not None:
        save_list(
            tr_rot_list, folder_path / "tr_rot_list"
        )  # Save the list of the object translation and rotations
        save_list(
            roughness_list, folder_path / "roughness_list"
        )  # Save the list of the wall roughness

    return tr_rot_list, roughness_list
