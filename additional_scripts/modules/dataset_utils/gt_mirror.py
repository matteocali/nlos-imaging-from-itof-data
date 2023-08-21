import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..utilities import read_folders, save_h5, load_h5, read_files
from ..transient_utils.loader import transient_loader
from ..transient_utils.tools import compute_distance_map


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
        data_path = (
            data_path + data_folder
        )  # Put together (in the same list) all the file present in all the batches

    for file_path in tqdm(
        data_path, desc="Generating ground truth data"
    ):  # For each file
        file_name = (
            str(Path(file_path).name) + "_GT"
        )  # Get the file name and add the suffix
        tr = transient_loader(file_path)  # Load the data and put them in standard form
        d_map = compute_distance_map(
            tr, fov, exp_time
        )  # Compute the distance map (depth radial map)
        a_map = np.zeros(d_map.shape)  # Create the alpha map
        a_map[
            np.where(np.sum(tr[:, :, :, -1], axis=0) != 0)
        ] = 1  # Set the alpha map to 1 where there is data in d_map
        save_h5(
            file_path=out_path / file_name,
            data={"gt_tr": tr, "depth_map": d_map, "alpha_map": a_map},
            fermat=False,
        )  # Save the data in the out_path folder as a h5 file


def fuse_dt_gt_mirror(
    d_path: Path, gt_path: Path, out_path: Path, def_obj_pos: list
) -> None:
    """
    Fuse the dataset and the ground truth together in the same h5 file
    :param d_path: folder containing the dataset (already processed and in h5 form)
    :param gt_path: folder containing the ground truth (already processed and in h5 form)
    :param out_path: folder where the fused dataset will be saved
    :param def_obj_pos: default object position in the scene
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    d_files_name = [
        Path(i).name for i in read_files(d_path, "h5")
    ]  # Get the list of files in d_path (only name)
    gt_files_name = [
        Path(i).name for i in read_files(gt_path, "h5")
    ]  # Get the list of files in gt_path (only name)

    if len(d_files_name) != len(gt_files_name):
        raise ValueError(
            "The number of files in the dataset and the ground truth folder are different"
        )  # Raise an error if the number of files is different

    for d_name in tqdm(
        d_files_name, desc="Fusing dataset and ground truth"
    ):  # For each file in d_path
        if len(d_files_name) != 1:
            d_name_shortened = d_name[
                : d_name.find("_wall_")
            ]  # Remove the wall information from the name and also the xml extension

            cam_pos = d_name_shortened[
                (d_name_shortened.find("cam_pos") + 9) : (
                    d_name_shortened.find("cam_rot") - 2
                )
            ]  # Get the camera position
            cam_rot = d_name_shortened[
                (d_name_shortened.find("cam_rot") + 9) : (
                    d_name_shortened.find("obj_pos") - 2
                )
            ]  # Get the camera rotation
            cam_pos = [
                float(i) for i in cam_pos.split("_")
            ]  # Convert the camera position to float
            cam_rot = [
                int(float(i)) for i in cam_rot.split("_")
            ]  # Convert the camera rotation to int

            obj_name = d_name_shortened[
                (d_name_shortened.find("nlos") + 5) : (d_name_shortened.find("cam") - 1)
            ]  # Get the object name
            obj_pos = d_name_shortened[
                (d_name_shortened.find("obj_pos") + 9) : (
                    d_name_shortened.find("obj_rot") - 2
                )
            ]  # Get the object position (as string)
            obj_rot = d_name_shortened[
                (d_name_shortened.find("obj_rot") + 9) : (-1)
            ]  # Get the object rotation (as string)
            if (
                d_name_shortened.find("+") == -1
            ):  # If the considered file is characterized by only one object
                obj_pos = [
                    float(i) for i in obj_pos.split("_")
                ]  # Convert the object position to float
                obj_pos = [
                    round(obj_pos[i] - elm, 1) for i, elm in enumerate(def_obj_pos)
                ]  # Remove the offset from the object position (get the object translation)
                obj_rot = [
                    int(float(i)) for i in obj_rot.split("_")
                ]  # Convert the object rotation to int
                gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name}_tr({obj_pos[0]}_{obj_pos[1]}_{obj_pos[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})_GT.h5"  # Compose the name of the gt file starting from the one of the dataset
            else:  # If the considered file is characterized by two objects
                obj_name1 = obj_name.split("+")[0]  # Get the first object name
                obj_name2 = obj_name.split("+")[1]  # Get the second object name
                obj_pos1 = obj_pos.split(")_(")[0]  # Get the first object position
                obj_pos2 = obj_pos.split(")_(")[1]  # Get the second object position
                obj_pos1_raw = [
                    float(i) for i in obj_pos1.split("_")
                ]  # Convert the first object position to float
                obj_pos2_raw = [
                    float(i) for i in obj_pos2.split("_")
                ]  # Convert the second object position to float
                obj_pos1 = [
                    round(obj_pos1_raw[i] - elm, 2) for i, elm in enumerate(def_obj_pos)
                ]  # Remove the offset from the first object position (get the object translation)
                obj_pos2 = [
                    round(obj_pos2_raw[i] - elm, 2) for i, elm in enumerate(def_obj_pos)
                ]  # Remove the offset from the second object position (get the object translation)
                obj_rot1 = obj_rot.split(")_(")[0]  # Get the first object rotation
                obj_rot2 = obj_rot.split(")_(")[1]  # Get the second object rotation
                obj_rot1 = [
                    int(float(i)) for i in obj_rot1.split("_")
                ]  # Convert the first object rotation to int
                obj_rot2 = [
                    int(float(i)) for i in obj_rot2.split("_")
                ]  # Convert the second object rotation to float
                gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"  # Compose the name of the gt file starting from the one of the dataset

            if gt_name not in gt_files_name:  # If the gt file doesn't exist
                indexes = []
                if 0.0 in obj_pos2:
                    indexes = [
                        index for index, elm in enumerate(obj_pos2) if elm == 0.0
                    ]
                    for index in indexes:
                        if index != 0:
                            obj_pos2[index] = -0.0
                    # noinspection PyUnboundLocalVariable
                    gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
                if gt_name not in gt_files_name:
                    if obj_pos2[0] == 0.0:
                        obj_pos2[0] = -0.0
                        gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
                if gt_name not in gt_files_name:
                    if len(indexes) > 1:
                        obj_pos2[indexes[1]] = 0.0
                        gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
                if gt_name not in gt_files_name:
                    if len(indexes) > 1:
                        obj_pos2[indexes[1]] = -0.0
                        obj_pos2[indexes[2]] = 0.0
                        gt_name = f"transient_nlos_cam_pos({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_{obj_name1}_tr({obj_pos1[0]}_{obj_pos1[1]}_{obj_pos1[2]})_rot({obj_rot1[0]}_{obj_rot1[1]}_{obj_rot1[2]})_{obj_name2}_tr({obj_pos2[0]}_{obj_pos2[1]}_{obj_pos2[2]})_rot({obj_rot2[0]}_{obj_rot2[1]}_{obj_rot2[2]})_GT.h5"
                if gt_name not in gt_files_name:
                    raise ValueError(
                        f"The ground truth file is missing ({gt_name})"
                    )  # If the gt file doesn't exist, raise an error
        else:
            gt_name = gt_files_name[0]

        d_file = d_path / d_name  # Compose the path of the dataset file
        gt_file = gt_path / gt_name  # Compose the path of the gt file
        d = load_h5(d_file)  # Load the dataset file
        gt = load_h5(gt_file)  # Load the gt file

        file_name = (
            d_name[: d_name.find(".h5")]
            .replace("-", "n")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "dot")
        )  # Compose the name of the file
        try:
            save_h5(
                file_path=out_path / file_name,
                data={
                    "data": d["data"],
                    "gt_tr": gt["gt_tr"],
                    "tr_itof": d["tr_itof"],
                    "amp_itof": d["amp_itof"],
                    "phase_itof": d["phase_itof"],
                    "depth_map": np.swapaxes(gt["depth_map"], 0, 1),
                    "alpha_map": np.swapaxes(gt["alpha_map"], 0, 1),
                },
                fermat=False,
            )  # Save the file
        except KeyError:
            save_h5(
                file_path=out_path / file_name,
                data={
                    "data": d["data"],
                    "gt_tr": gt["gt_tr"],
                    "depth_map": np.swapaxes(gt["depth_map"], 0, 1),
                    "alpha_map": np.swapaxes(gt["alpha_map"], 0, 1),
                },
                fermat=False,
            )  # Save the file
