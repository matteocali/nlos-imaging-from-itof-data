import numpy as np
from pathlib import Path
from scipy import io
from ..utilities import add_extension
from .tools import prepare_fermat_data
from cv2 import (
    fisheye,
    CV_16SC2,
    remap,
    INTER_LINEAR,
    undistort,
    initUndistortRectifyMap,
)


def np2mat(
    data: np.ndarray,
    file_path: Path,
    data_grid_size: list,
    img_shape: list,
    fov: float,
    store_glb: bool = False,
    data_clean: bool = False,
    cl_method: str = None,
    cl_threshold: int = None,
    show_plt: bool = False,
    exp_time: float = 0.01,
    laser_pos: list = None,
) -> None:
    """
    Function to save a .mat file in the format required from the Fermat flow Matlab script
    :param data: np.ndarray containing the transient measurements (n*m matrix, n transient measurements with m temporal bins)
    :param file_path: file path and name where to save the .mat file
    :param data_grid_size: [n1, n2], with n1 * n2 = n, grid size of sensing points on the LOS wall (n = number of the transient measurements)
    :param img_shape: size of the image [<n_col>, <n_row>]
    :param fov: horizontal field of view of the camera
    :param store_glb: boolean to indicate if the global transient should be stored as a .npy or not
    :param data_clean: boolean that indicate if the data should be cleaned or not
    :param cl_method: method used to clean the data (None = no cleaning, "balanced" = balanced histogram thresholding, "otsu" = otsu histogram thresholding)
    :param cl_threshold: threshold used to clean the data (in the energy domain)
    :param show_plt: boolean that indicate if the data should be plotted or not
    :param exp_time: exposure time used, required to compute "temporalBinCenters"
    :param laser_pos: position of the laser, if none it is confocal with the camera (1*3 vector)
    """

    # Define all the required vectors for the .mat file
    if store_glb:
        data, det_locs = prepare_fermat_data(
            data=data,
            grid_size=data_grid_size,
            img_size=img_shape,
            fov=fov,
            data_clean=data_clean,
            cl_method=cl_method,
            cl_threshold=cl_threshold,
            exp_time=exp_time,
            show_plt=show_plt,
            file_path=file_path,
        )  # Prepare the data for the Fermat Flow algorithm
    else:
        data, det_locs = prepare_fermat_data(
            data=data,
            grid_size=data_grid_size,
            img_size=img_shape,
            fov=fov,
            data_clean=data_clean,
            cl_method=cl_method,
            cl_threshold=cl_threshold,
            exp_time=exp_time,
            show_plt=show_plt,
        )  # Prepare the data for the Fermat Flow algorithm

    file_path = str(
        add_extension(str(file_path), ".mat")
    )  # If not already present add the .h5 extension to the file path

    data_grid_size = np.array(
        data_grid_size, dtype=np.float64
    )  # Convert the data_grid_size from a list to a np.ndarray

    if (
        laser_pos is None
    ):  # If laser_pos is not provided it means that the laser is confocal with the camera
        src_loc = np.array((), dtype=np.float32)
    else:
        src_loc = np.array(laser_pos, dtype=np.float32)

    temp_bin_centers = compute_bin_center(
        exp_time, data.shape[1]
    )  # To build the temp_bin_centers it is required to build a vector where each cell contains the center of the correspondent temporal bin, so the first cell contains half the exposure time

    io.savemat(
        str(file_path),
        mdict={
            "detGridSize": np.flip(data_grid_size),
            "detLocs": det_locs,
            "srcLoc": src_loc,
            "temporalBinCenters": temp_bin_centers,
            "transients": data,
        },
    )  # Save the actual .mat file


def compute_bin_center(exp_time: float, n_bins: int) -> list:
    """
    Compute the center of the temporal bins
    :param exp_time: exposure time
    :param n_bins: number of temporal bins
    :return: list of the center of the temporal bins
    """

    temp_bin_centers = [
        exp_time / 2
    ]  # To build the temp_bin_centers it is required to build a vector where each cell contains the center of the correspondent temporal bin, so the first cell contains half the exposure time
    for i in range(1, n_bins):
        temp_bin_centers.append(
            temp_bin_centers[i - 1] + exp_time
        )  # For all the following cell simply add the exposure time to the value stored in the previous cell
    return temp_bin_centers


def undistort_depthmap(dph, dm, k_ideal, k_real, d_real):
    """
    Undistort depth map using calibration output parameters
    :param dph: depthmap (1 channel image)
    :param dm: string with the name of the camera model (FISHEYE, RADIAL, RATIONAL)
    :param k_ideal: Camera matrix
    :param k_real: Camera matrix
    :param d_real: Distortion coefficient
    :return depthmap: undistorted depthmap with 3 dimension (x-axis coordinates, y-axis coordinates, z-coordinates)
    :return mask_valid_positive: validity mask (1=valid points, 0= oor or invalid dots)
    :return radial_dir: cosine(angle between optical axis and the pixel direction)
    """

    depth = dph.copy()
    mask_valid = 1.0 * (depth < 30000)
    depth[depth > 30000] = 0
    shape_depth = (depth.shape[1], depth.shape[0])

    if dm == "FISHEYE":
        [map1, map2] = fisheye.initUndistortRectifyMap(
            k_real, d_real, np.eye(3), k_ideal, shape_depth, CV_16SC2
        )
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    elif dm == "STANDARD":
        depth = undistort(depth, k_real, d_real, None, k_ideal)
        mask_valid = undistort(mask_valid, k_real, d_real, None, k_ideal)

    else:
        [map1, map2] = initUndistortRectifyMap(
            k_real, d_real, np.eye(3), k_ideal, shape_depth, CV_16SC2
        )
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    mask_valid_positive = mask_valid > 0
    depth[mask_valid_positive] = np.divide(
        depth[mask_valid_positive], mask_valid[mask_valid_positive]
    )

    z_matrix = depth
    x_matrix = (np.tile(np.arange(z_matrix.shape[1]), (z_matrix.shape[0], 1))).astype(
        dtype=float
    )  # [[0,1,2,3,...],[0,1,2,3,..],...]
    y_matrix = np.tile(np.arange(z_matrix.shape[0]), (z_matrix.shape[1], 1)).T.astype(
        dtype=float
    )  # [....,[1,1,1,1, ...][0,0,0,0,...]]

    x_undist_matrix = np.zeros_like(x_matrix, dtype=float)
    y_undist_matrix = np.zeros_like(y_matrix, dtype=float)
    z_undist_matrix = np.zeros_like(z_matrix, dtype=float)

    k_1 = np.linalg.inv(k_ideal)

    radial_dir = np.zeros([x_matrix.shape[0], x_matrix.shape[1], 3])
    for x in range(x_matrix.shape[0]):
        for y in range(x_matrix.shape[1]):
            prod = np.dot(k_1, np.asarray([x_matrix[x, y], y_matrix[x, y], 1]))
            prod = prod / np.linalg.norm(prod)

            x_undist_matrix[x, y] = z_matrix[x, y] * prod[0]
            y_undist_matrix[x, y] = z_matrix[x, y] * prod[1]
            z_undist_matrix[x, y] = z_matrix[x, y] * prod[2]
            radial_dir[x, y, :] = prod

    depthmap = np.stack([x_undist_matrix, y_undist_matrix, z_undist_matrix], axis=2)

    return depthmap, mask_valid_positive, radial_dir


def coordinates_matrix_reshape(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Function that remove all the np.zeros in the coordinates and np.reshape it in a way that it contains the coordinates of each row one on the bottom of the other
    :param data: coordinates matrix
    :param mask: matrix that identify the used grid
    :return: reshaped coordinates matrix (z=0)
    """

    shape = [len(np.unique(np.where(mask != 0)[i])) for i in range(2)]
    data = np.reshape(
        data[mask != 0], [shape[0], shape[1], 2]
    )  # Remove all the zero values

    matrix = np.zeros(
        [data.shape[0] * data.shape[1], 3]
    )  # Create the final matrix (full of np.zeros)
    m_index = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            matrix[m_index, 0] = data[j, i, 0]  # Load each cell with the right value
            matrix[m_index, 1] = data[j, i, 1]
            m_index += 1
    return matrix


def nearest_nonzero_idx(matrix: np.ndarray, x: int, y: int) -> tuple:
    """
    Function to find the closest non-zero element to the point (x, y)
    (code from: https://stackoverflow.com/questions/43306291/find-the-nearest-np.nonzero-element-and-corresponding-index-in-a-2d-numpy-np.array)
    :param matrix: 2D data matrix where to search the location
    :param x: x coordinate of the point of interest
    :param y: y coordinate of the point of interest
    :return: the (x, y) coordinate of the nearest non-zero point
    """

    tmp = matrix[x, y]
    matrix[x, y] = 0
    r, c = np.nonzero(matrix)
    matrix[x, y] = tmp
    min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
    return r[min_idx], c[min_idx]


def roto_transl(coordinates_matrix: np.ndarray) -> np.ndarray:
    """
    Function that move the given coordinates from the camera coordinates system to the one of the world making sure that the points stays on the plane z=0
    :param coordinates_matrix: coordinates matrix in the camera system
    :return: coordinates matrix in te word system
    """

    coordinates_matrix = np.round(
        coordinates_matrix, decimals=2
    )  # exp_time = 0.01 max sensibility is 1cm
    center_pos = (
        int(coordinates_matrix.shape[0] / 2),
        int(coordinates_matrix.shape[1] / 2),
    )  # Find the center position on the coordinates matrix
    nearest_center_pos = nearest_nonzero_idx(
        coordinates_matrix[:, :, 0], center_pos[0], center_pos[1]
    )  # Find the location of the active poit closest to the real center
    nearest_center_coord = [
        coordinates_matrix[nearest_center_pos[0], nearest_center_pos[1], i]
        for i in range(coordinates_matrix.shape[2])
    ]  # Extract from the coordinates' matrix the coordinates of the point closest to the center

    next_x_pos = np.nonzero(coordinates_matrix[nearest_center_pos[0], :, 0])[
        0
    ]  # Find the location of the point on the coordinates' matrix located on the same y of the nearest_center but on the extreme right (max x)
    next_x_pos = (
        nearest_center_pos[0],
        next_x_pos[-1],
    )  # Define the indexes where that point is located in the coordinates' matrix (add the row index)
    next_y_pos = np.nonzero(coordinates_matrix[:, nearest_center_pos[1], 0])[
        0
    ]  # Find the location of the point on the coordinates' matrix located on the same x of the nearest_center but on the extreme top (max y)
    next_y_pos = (
        next_y_pos[0],
        nearest_center_pos[1],
    )  # Define the indexes where that point is located in the coordinates' matrix (add the column index)
    next_x_coord = coordinates_matrix[
        next_x_pos[0], next_x_pos[1], :
    ]  # Extract from the coordinates' matrix the coordinates of next_x
    next_y_coord = coordinates_matrix[
        next_y_pos[0], next_y_pos[1], :
    ]  # Extract from the coordinates' matrix the coordinates of next_y

    normal = np.cross(
        np.subtract(next_x_coord, nearest_center_coord),
        np.subtract(next_y_coord, nearest_center_coord),
    )  # Compute the normal of the plane making the cross product of the vector between the center and the far right point and of the vector between the center and the far top point
    normalized_normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    n_x = normalized_normal[0]  # x component of the normal vector
    n_y = normalized_normal[1]  # y component of the normal vector
    n_z = normalized_normal[2]  # z component of the normal vector

    if n_x != 0 or n_y != 0:
        sqrt_squared_sum_nx_ny = np.sqrt((n_x**2) + (n_y**2))
        rot_matrix = np.array(
            [
                [n_y / sqrt_squared_sum_nx_ny, -n_x / sqrt_squared_sum_nx_ny, 0],
                [
                    (n_x * n_z) / sqrt_squared_sum_nx_ny,
                    (n_y * n_z) / sqrt_squared_sum_nx_ny,
                    -sqrt_squared_sum_nx_ny,
                ],
                [n_x, n_y, n_z],
            ],
            dtype=np.float32,
        )  # Build the rotation matrix (code from: https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector)

        o_rot_matrix = np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32
        )  # Define a second rotation matrix to compensate for the rotation 90° over the y-axis
        rot_matrix = np.matmul(
            o_rot_matrix, rot_matrix
        )  # Define the complete rotation matrix multiplying together the two previously defined ones
    else:
        rot_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    tr_vector = np.matmul(
        rot_matrix, np.reshape(nearest_center_coord, [3, 1]), dtype=np.float32
    )  # Define a translation vector that aims to center the plane on the nearest_center point

    # Build the final roto-transl matrix
    rototransl_matrix = np.concatenate(
        (rot_matrix, np.negative(tr_vector)), axis=1
    )  # Add to the left of the rotation matrix the translation one
    rototransl_matrix = np.concatenate(
        (rototransl_matrix, np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0
    )  # Add to the bottom the vector [0 0 0 1]

    non_zero_pos = np.nonzero(
        coordinates_matrix[:, :, 0]
    )  # Find the location where the coordinates' matrix is not zero
    for r in np.unique(non_zero_pos[0]):
        for c in np.unique(non_zero_pos[1]):
            coord_vector = np.concatenate(
                (
                    np.reshape(coordinates_matrix[r, c, :], (3, 1)),
                    np.array([[1]], dtype=np.float32),
                ),
                axis=0,
            )  # For every non-zero point of the coordinates' matrix np.concatenate a 1 at the end and np.reshape it from [3], to [4, 1]
            coordinates_matrix[r, c, :] = np.matmul(
                rototransl_matrix, coord_vector, dtype=np.float32
            )[
                :-1, 0
            ]  # Apply the roto-translation to each non-zero point and remove the final 1
    coordinates_matrix[
        np.where(coordinates_matrix == -0.0)
    ] = 0  # Change all the -0.0 to 0

    return np.round(coordinates_matrix, decimals=2)


def reshape_fermat_transient(
    transient: np.ndarray, grid_shape: tuple, flip_x: bool = False, flip_y: bool = False
) -> np.ndarray:
    """
    Function that take the transient data where the grid is read row by row and rearrange the data to follow the reading column by column
    :param transient: transient data in the shape [<number of measurements>, <number of beans>]
    :param grid_shape: dimension of the grid [column, row]
    :param flip_x: flag that define if it is required to np.flip the order of the x coordinates
    :param flip_y: flag that define if it is required to np.flip the order of the x coordinates
    :return: the transient values rearranged to follow the Fermat Flow requirements
    """

    if flip_x:
        for i in range(0, transient.shape[0], grid_shape[1]):
            transient[i : (i + grid_shape[1]), :] = np.flip(
                transient[i : (i + grid_shape[1]), :], axis=0
            )  # If flip_x is True revers the order of the data in the x coordinate

    reshaped_transient = np.zeros(
        transient.shape
    )  # Initialize a np.ndarray full of np.zeros of the same size of the input transient one
    index = 0  # Initialize to zero the index that will cycle through the original transient vector
    for column_index in range(grid_shape[1]):  # Cycle column by column
        for row_index in range(grid_shape[0]):
            reshaped_transient[index] = transient[
                row_index * grid_shape[1] + column_index
            ]  # Put the right column value in the new transient vector
            index += 1  # Update the index

    if flip_y:
        for i in range(0, reshaped_transient.shape[0], grid_shape[0]):
            reshaped_transient[i : (i + grid_shape[0]), :] = np.flip(
                reshaped_transient[i : (i + grid_shape[0]), :], axis=0
            )  # If flip_y is True revers the order of the data in the x coordinate

    return reshaped_transient
