from pathlib import Path
from tqdm import tqdm
from ..utilities import read_folders, save_h5, load_h5, read_files, spot_bitmap_gen
from ..transient_utils.loader import grid_transient_loader
from ..fermat_utils.tools import prepare_fermat_data
from .utils import compute_discont


def build_fermat_gt(
    gt_path: Path,
    out_path: Path,
    exp_time: float,
    img_size: list,
    grid_size: list,
    fov: float,
) -> None:
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
        data_path = (
            data_path + data_folder
        )  # Put together (in the same list) all the file present in all the batches

    for file_path in tqdm(
        data_path, desc="Generating ground truth data"
    ):  # For each file
        file_name = (
            str(Path(file_path).name) + "_GT"
        )  # Get the file name and add the suffix
        tr = grid_transient_loader(
            transient_path=file_path
        )  # Load the data and put them in standard form
        tr, _ = prepare_fermat_data(
            data=tr,
            grid_size=grid_size,
            img_size=img_size,
            fov=fov,
            data_clean=True,
            exp_time=exp_time,
        )  # Clean the data and put them in standard form for Fermat

        all_disconts = compute_discont(tr, exp_time)  # Compute the discontinuity

        save_h5(
            file_path=out_path / file_name,
            data={"tr": tr, "discont_loc": all_disconts},
            fermat=True,
        )  # Save the data


def fuse_dt_gt_fermat(
    d_path: Path, gt_path: Path, out_path: Path, img_size: list, grid_size: list
) -> None:
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
        d_name_shortened = d_name[
            : d_name.find(".h5")
        ]  # Remove the wall information from the name and also the xml extension

        if d_name not in gt_files_name:  # If the gt file doesn't exist
            raise ValueError(
                "The ground truth file is missing"
            )  # If the gt file doesn't exist, raise an error

        d_file = d_path / d_name  # Compose the path of the dataset file
        gt_file = gt_path / d_name  # Compose the path of the gt file
        d = load_h5(d_file)  # Load the dataset file
        gt = load_h5(gt_file)  # Load the gt file

        tr = d["data"]  # Get the measured data
        mask = spot_bitmap_gen(
            img_size=img_size, pattern=tuple(grid_size)
        )  # Define the mask that identify the location of the illuminated spots
        mskd_tr = tr[
            mask
        ]  # Get the measured data corresponding to the illuminated spots
        reshaped_tr = mskd_tr.flatten(order="F")  # Flatten the array column by column

        file_name = (
            d_name_shortened.replace("-", "n")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "dot")
        )  # Compose the name of the file
        save_h5(
            file_path=out_path / file_name,
            data={"tr": reshaped_tr, "discont_gt": gt["discont_loc"]},
            fermat=False,
        )  # Save the file
