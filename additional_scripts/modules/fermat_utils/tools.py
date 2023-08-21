import numpy as np
from os.path import dirname, exists
from pathlib import Path
from time import time, sleep
from tqdm import tqdm
from .. import transient_utils as tr
from ..utilities import spot_bitmap_gen, k_matrix_calculator, plt_3d_surfaces
from .utils import (
    roto_transl,
    coordinates_matrix_reshape,
    reshape_fermat_transient,
    undistort_depthmap,
)


def prepare_fermat_data(
    data: np.ndarray,
    grid_size: list,
    img_size: list,
    fov: float,
    data_clean: bool,
    cl_method: str = None,
    cl_threshold: int = None,
    exp_time: float = 0.01,
    show_plt: bool = False,
    file_path: Path = None,
) -> (np.ndarray, np.ndarray):
    """
    Function to prepare the transient data for the Fermat Flow algorithm
    :param data: np.ndarray containing the transient measurements (n*m matrix, n transient measurements with m temporal bins)
    :param grid_size: [n1, n2], with n1 * n2 = n, grid size of sensing points on the LOS wall (n = number of the transient measurements)
    :param img_size: size of the image [<n_col>, <n_row>]
    :param fov: horizontal field of view of the camera
    :param data_clean: boolean that indicate if the data should be cleaned or not
    :param cl_method: method used to clean the data (None = no cleaning, "balanced" = balanced histogram thresholding, "otsu" = otsu histogram thresholding)
    :param cl_threshold: threshold used to clean the data (in the energy domain)
    :param exp_time: exposure time used, required to compute "temporalBinCenters"
    :param show_plt: boolean that indicate if the data should be plotted or not
    :param file_path: file path and name where to save the .mat file
    :return: data = processed transient data, det_locs = locations of the sensing points
    """

    mask = spot_bitmap_gen(
        img_size=img_size, pattern=tuple(grid_size)
    )  # Define the mask that identify the location of the illuminated spots

    k = k_matrix_calculator(
        h_fov=fov, img_shape=img_size
    )  # Define the intrinsic camera matrix needed to map the dots on the LOS wall
    transient_image = build_matrix_from_dot_projection_data(
        transient=data, mask=mask
    )  # Put all the transient data in the right spot following the mask
    depthmap = compute_los_points_coordinates(
        images=transient_image, mask=mask, k_matrix=k, channel=1, exp_time=exp_time
    )  # Compute the mapping of the coordinate of the illuminated spots to the LOS wall (ignor the z coordinate)
    rt_depthmap = roto_transl(
        coordinates_matrix=np.copy(depthmap)
    )  # Roto-translate the coordinates point in order to move from the camera coordinates system to the world one and also move the plane to be on the plane z = 0

    flip_x_rt_depthmap = np.copy(rt_depthmap)
    for i in range(flip_x_rt_depthmap.shape[2]):
        flip_x_rt_depthmap[:, :, i] = np.flip(flip_x_rt_depthmap[:, :, i], axis=1)
        flip_x_rt_depthmap[:, :, i] = np.roll(flip_x_rt_depthmap[:, :, i], -1, axis=1)

    det_locs = coordinates_matrix_reshape(
        data=flip_x_rt_depthmap[:, :, :-1], mask=mask
    )  # Reshape the coordinate value, putting the coordinate of each row one on the bottom of the previous one

    if show_plt:
        plt_3d_surfaces(
            surfaces=[depthmap, rt_depthmap, flip_x_rt_depthmap],
            mask=mask,
            legends=[
                "Original plane",
                "Roto-translated plane",
                "Flipped roto-translated plane",
            ],
        )

    # Data cleaning
    if data_clean:
        for i in range(data.shape[0]):
            data[i, :, :] = tr.tools.clean_transient_tail(
                transient=data[i, :, :], n_samples=20
            )  # Remove the transient tail
        if cl_method is not None:
            data = tr.utils.clear_tr_ratio(
                data, method="otsu"
            )  # Remove the transient that are too dark
        if cl_threshold is not None:
            data = tr.utils.clear_tr_energy(
                data, threshold=cl_threshold
            )  # Remove noise based on the global energy

    if file_path is not None:
        np_file_path = dirname(file_path) + "\\glb_np_transient.npy"
        data = rmv_first_reflection_fermat_transient(
            transient=data, file_path=np_file_path, store=(not exists(np_file_path))
        )  # Remove the direct component from all the transient data
    else:
        data = rmv_first_reflection_fermat_transient(
            transient=data
        )  # Remove the direct component from all the transient data

    data = reshape_fermat_transient(
        transient=data[:, :, 1],
        grid_shape=(int(grid_size[1]), int(grid_size[0])),
        flip_x=False,
        flip_y=False,
    )  # Reshape the transient data in order to be in the same format used in the Fermat Flow algorithm

    return data, det_locs


def compute_los_points_coordinates(
    images: np.ndarray,
    mask: np.ndarray,
    k_matrix: np.ndarray,
    channel: int,
    exp_time: float,
) -> np.ndarray:
    """
    Function that compute the coordinates of the projected points on the LOS wall
    :param images: transient information
    :param mask: position of the point on the bitmap
    :param k_matrix: intrinsic camera matrix
    :param channel: integer that identify the channel of interest (0 = red, 1 = green, 2 = blue)
    :param exp_time: used exposure time
    :return: matrix where the first dimension contains the x coordinates, the second one the y coordinates and the third one the depth information
    """

    # Compute the depth information only on the point that the mask consider
    dph = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                dph[i, j] = tr.tools.compute_radial_distance(
                    peak_pos=tr.tools.extract_peak(images[:, i, j, :])[0][channel],
                    exposure_time=exp_time,
                )

    # Compute the depthmap and coordinates
    depthmap, _, _ = undistort_depthmap(
        dph=dph,
        dm="RADIAL",
        k_ideal=k_matrix,
        k_real=k_matrix,
        d_real=np.array([[0, 0, 0, 0, 0]], dtype=np.float32),
    )

    return depthmap.astype(np.float32)


def build_matrix_from_dot_projection_data(
    transient: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Function that build the full image from the measurements of the transient of only one dot of a time
    :param transient: measured transient information
    :param mask: dot pattern used
    :return: transient image
    """

    matrix = np.zeros([transient.shape[1], mask.shape[0], mask.shape[1], 3])
    t_index = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                matrix[:, i, j, :] = transient[t_index]
                t_index += 1

    return matrix


def rmv_first_reflection_fermat_transient(
    transient: np.ndarray, file_path: Path = None, store: bool = False
) -> np.ndarray:
    """
    Function that given the transient images in the fermat flow shape remove the first reflection leaving only the global component
    :param transient: input transient images
    :param file_path: path of the np dataset
    :param store: if you want to save the np dataset (if already saved set it to false in order to np.load it)
    :return: Transient images of only the global component as a np np.array
    """

    if (
        file_path and not store
    ):  # If already exists a npy file containing all the transient images np.load it instead of processing everything again
        return np.load(str(file_path))

    print("Extracting the first peak (channel by channel):")
    start = time()

    mono = len(transient.shape) == 2  # Check id=f the images are Mono or RGBA

    if not mono:
        peaks = [
            np.nanargmax(transient[:, :, channel_i], axis=1)
            for channel_i in tqdm(range(transient.shape[2]))
        ]  # Find the index of the maximum value in the third dimension
    else:
        peaks = np.nanargmax(
            transient, axis=1
        )  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    print("Remove the first peak (channel by channel):")
    sleep(0.1)

    glb_images = np.copy(transient)
    first_direct = transient.shape[1]
    if not mono:
        for channel_i in tqdm(range(transient.shape[2])):
            for pixel in range(transient.shape[0]):
                zeros_pos = np.where(transient[pixel, :, channel_i] == 0)[0]
                valid_zero_indexes = zeros_pos[
                    np.where(zeros_pos > peaks[channel_i][pixel])
                ]
                if valid_zero_indexes.size == 0:
                    glb_images[pixel, :, channel_i] = -2
                else:
                    glb_images[pixel, : int(valid_zero_indexes[0]), channel_i] = -1
                    if valid_zero_indexes[0] < first_direct:
                        first_direct = valid_zero_indexes[0]
        print("Shift the transient vector so the global always start at t=0:")
        sleep(0.1)
        glb_images_shifted = np.zeros(
            [transient.shape[0], transient.shape[1] - first_direct, transient.shape[2]]
        )
        for channel_i in tqdm(range(transient.shape[2])):
            for pixel in range(transient.shape[0]):
                if glb_images[pixel, 0, channel_i] != -2:
                    shifted_transient = np.delete(
                        glb_images[pixel, :, channel_i],
                        [np.where(glb_images[pixel, :, channel_i] == -1)],
                    )
                    glb_images_shifted[
                        pixel, : shifted_transient.shape[0], channel_i
                    ] = shifted_transient
    else:
        for pixel in tqdm(range(transient.shape[0])):
            zeros_pos = np.where(transient[pixel, :] == 0)[0]
            valid_zero_indexes = zeros_pos[np.where(zeros_pos > peaks[pixel])]
            if valid_zero_indexes.size == 0:
                glb_images[pixel, :] = -2
            else:
                glb_images[pixel, : int(valid_zero_indexes[0])] = -1
        print("Shift the transient vector so the global always start at t=0:")
        sleep(0.1)
        glb_images_shifted = np.zeros(
            [transient.shape[0], transient.shape[1] - first_direct]
        )
        for pixel in range(transient.shape[0]):
            if glb_images[pixel, 0] != -2:
                shifted_transient = np.delete(
                    glb_images[pixel, :], [np.where(glb_images[pixel, :] == -1)]
                )
                glb_images_shifted[
                    pixel, : shifted_transient.shape[0]
                ] = shifted_transient

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    if file_path and store:
        np.save(
            str(file_path), glb_images_shifted
        )  # Save the loaded images as a numpy np.array
    return glb_images_shifted


def rmv_sparse_fermat_transient(
    transients: np.ndarray, channel: int, threshold: int, remove_data: bool
) -> np.ndarray:
    """
    Function that np.delete or set to zero all the transient that have an active bins percentage lower than <threshold>
    :param transients: transient vectors (set in the fermat flow setup)
    :param channel: chose which channel to check (0 = red, 1 = green, 2 = blu)
    :param threshold: percentage value below which the transient will be discarded
    :param remove_data: flag that decide if the data will be removed or simply set to zero
    :return: cleaned transient
    """

    indexes = (
        []
    )  # List that will contain the indexes of all the transient row that will be discarded
    for i in range(transients.shape[0]):  # Analyze each transient one by one
        peaks, _ = tr.tools.extract_peak(
            transients[i, :, :]
        )  # Extract the position of the direct component in all the three channel
        peak = peaks[channel]  # Keep only the information about the channel of interest
        zeros_pos = np.where(transients[i, :, channel] == 0)[
            0
        ]  # Find where all the np.zeros are located
        first_zero_after_peak = zeros_pos[np.where(zeros_pos > peak)][
            0
        ]  # Keep only the first zero after the direct component
        if (
            tr.utils.active_beans_percentage(transients[i, first_zero_after_peak:, channel])
            < threshold
        ):  # If the percentage of active bins in the global component is below th threshold:
            if remove_data:  # If remove_data is set to True:
                indexes.append(i)  # Add the considered row index to the indexes list
            else:  # If remove_data is set to False:
                transients[i, :, :] = 0  # Set to zero all the row
    if remove_data:  # If remove_data is set to True:
        return np.delete(
            transients, indexes, axis=0
        )  # Remove all the row which index is inside the indexes list
    else:  # If remove_data is set to False:
        return transients  # Return the transient with the modification applied
