import numpy as np
from tqdm import tqdm
from .. import utilities as ut
from OpenEXR import InputFile
from pathlib import Path
from .tools import reshape_frame, img_matrix


def transient_loader(img_path, np_path=None, store=False):
    """
    Function that starting from the raw mitsuba transient output load the transient and reshape it
    :param img_path: path of the transient images
    :param np_path: path of the np dataset
    :param store: boolean value that determines if we want to store the loaded transient in np format
    :return: a np array containing all the transient
    """

    if (
        np_path and not store
    ):  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return np.load(str(np_path))
    else:
        files = ut.read_files(
            str(img_path), "exr"
        )  # Load the path of all the files in the input folder with extension .exr
        channels = reshape_frame(files)  # Reshape the frame in a standard layout
        images = img_matrix(channels)  # Create the image files
        if store:
            ut.create_folder(
                np_path.parent.absolute(), "all"
            )  # Create the output folder if not already present
            np.save(str(np_path), images)  # Save the loaded images as a numpy array
        return images


def grid_transient_loader(
    transient_path: Path, np_path: Path = None, store: bool = False
) -> np.ndarray:
    """
    Function that starting from the raw mitsuba transient output load the transient and reshape it
    :param transient_path: path of the transient images
    :param np_path: path of the np dataset
    :param store: boolean value that determines if we want to store the loaded transient in np format
    :return: a np array containing all the transient
    """

    if (
        np_path and not store
    ):  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return np.load(str(np_path))
    else:
        folder_path = ut.read_folders(folder_path=transient_path, reorder=True)
        dw = InputFile(ut.read_files(str(folder_path[0]), "exr")[0]).header()[
            "dataWindow"
        ]  # Extract the data window dimension from the header of the exr file
        size = (
            dw.max.x - dw.min.x + 1,
            dw.max.y - dw.min.y + 1,
        )  # Define the actual size of the image
        transient = np.empty([len(folder_path), size[0], 3])
        print("Loading all the transient data:\n")
        for index, img_path in enumerate(tqdm(folder_path, desc="loading files")):
            files = ut.read_files(
                str(img_path), "exr"
            )  # Load the path of all the files in the input folder with extension .exr
            channels = reshape_frame(
                files, verbose=False
            )  # Reshape the frame in a standard layout
            images = img_matrix(channels, verbose=False)  # Create the image files
            transient[index, :] = images[:, 0, 0, :-1]
        transient[
            np.where(transient < 0)
        ] = 0  # Remove from the transient all the negative data
        if store:
            np.save(str(np_path), transient)  # Save the loaded images as a numpy array
        print("Loading completed")
        return transient
