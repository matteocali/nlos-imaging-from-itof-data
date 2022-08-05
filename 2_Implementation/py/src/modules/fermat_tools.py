from os.path import dirname, exists
from pathlib import Path
from time import time, sleep
from cv2 import fisheye, CV_16SC2, remap, INTER_LINEAR, undistort, initUndistortRectifyMap
from numpy import ndarray, array, float64, float32, eye, divide, tile, arange, zeros_like, linalg, zeros, dot, asarray, \
    stack, reshape, flip, load, nanargmax, copy, where, delete, save, nonzero, cross, sqrt, concatenate, matmul, unique, \
    subtract, negative, roll, round as np_round, flip as np_flip
from scipy import io
from tqdm import tqdm
from modules import transient_handler as tr
from modules.transient_handler import extract_peak, active_beans_percentage, clear_tr_ratio, clear_tr_energy
from modules.utilities import add_extension, spot_bitmap_gen, k_matrix_calculator, plt_3d_surfaces


def np2mat(data: ndarray, file_path: Path, data_grid_size: list, img_shape: list, fov: float, store_glb: bool = False, data_clean: bool = False, cl_method: str = None, cl_threshold: int = None, show_plt: bool = False, exp_time: float = 0.01, laser_pos: list = None) -> None:
    """
    Function to save a .mat file in the format required from the Fermat flow Matlab script
    :param data: ndarray containing the transient measurements (n*m matrix, n transient measurements with m temporal bins)
    :param file_path: file path and name where to save the .mat file
    :param data_grid_size: [n1, n2], with n1 * n2 = n, grid size of sensing points on the LOS wall (n = number of the transient measurements)
    :param img_shape: size of the image [<n_row>, <n_col>]
    :param fov: horizontal field of view of the camera
    :param store_glb: boolean to indicate if the global transient should be stored as a .npy or not
    :param data_clean: boolean that indicate if the data should be cleaned or not
    :param cl_method: method used to clean the data (None = no cleaning, "balanced" = balanced histogram thresholding, "otsu" = otsu histogram thresholding)
    :param cl_threshold: threshold used to clean the data (in the energy domain)
    :param show_plt: boolean that indicate if the data should be plotted or not
    :param exp_time: exposure time used, required to compute "temporalBinCenters"
    :param laser_pos: position of the laser, if none it is confocal with the camera (1*3 vector)
    """

    np_file_path = dirname(file_path) + "\\glb_np_transient.npy"
    file_path = str(add_extension(str(file_path), ".mat"))  # If not already present add the .h5 extension to the file path

    # Define all the required vectors for the .mat file
    data_grid_size = array(data_grid_size, dtype=float64)  # Convert the data_grid_size from a list to a ndarray

    mask = spot_bitmap_gen(img_size=img_shape,
                           pattern=tuple(data_grid_size))  # Define the mask that identify the location of the illuminated spots

    k = k_matrix_calculator(h_fov=fov, img_shape=img_shape)  # Define the intrinsic camera matrix needed to map the dots on the LOS wall
    transient_image = build_matrix_from_dot_projection_data(transient=data, mask=mask)  # Put all the transient data in the right spot following the mask
    depthmap = compute_los_points_coordinates(images=transient_image,
                                              mask=mask,
                                              k_matrix=k,
                                              channel=1,
                                              exp_time=0.01)  # Compute the mapping of the coordinate of the illuminated spots to the LOS wall (ignor the z coordinate)
    rt_depthmap = roto_transl(coordinates_matrix=copy(depthmap))  # Roto-translate the coordinates point in order to move from the camera coordinates system to the world one and also move the plane to be on the plane z = 0

    flip_x_rt_depthmap = copy(rt_depthmap)
    for i in range(flip_x_rt_depthmap.shape[2]):
        flip_x_rt_depthmap[:, :, i] = flip(flip_x_rt_depthmap[:, :, i], axis=1)
        flip_x_rt_depthmap[:, :, i] = roll(flip_x_rt_depthmap[:, :, i], -1, axis=1)

    det_locs = coordinates_matrix_reshape(data=flip_x_rt_depthmap[:, :, :-1],
                                          mask=mask)  # Reshape the coordinate value, putting the coordinate of each row one on the bottom of the previous one

    if show_plt:
        plt_3d_surfaces(surfaces=[depthmap, rt_depthmap, flip_x_rt_depthmap],
                        mask=mask,
                        legends=["Original plane", "Roto-translated plane", "Flipped roto-translated plane"])

    if laser_pos is None:   # If laser_pos is not provided it means that the laser is confocal with the camera
        src_loc = array((), dtype=float32)
    else:
        src_loc = array(laser_pos, dtype=float32)

    # Data cleaning
    if data_clean:
        for i in range(data.shape[0]):
            data[i, :, :] = tr.clean_transient_tail(transient=data[i, :, :], n_samples=20)  # Remove the transient tail
        if cl_method is not None:
            data = clear_tr_ratio(data, method="otsu")  # Remove the transient that are too dark
        if cl_threshold is not None:
            data = clear_tr_energy(data, threshold=cl_threshold)  # Remove noise based on the global energy

    if store_glb:
        data = rmv_first_reflection_fermat_transient(transient=data,
                                                     file_path=np_file_path,
                                                     store=(not exists(np_file_path)))  # Remove the direct component from all the transient data
    else:
        data = rmv_first_reflection_fermat_transient(transient=data)  # Remove the direct component from all the transient data

    data = reshape_fermat_transient(transient=data[:, :, 1],
                                    grid_shape=(int(data_grid_size[1]), int(data_grid_size[0])),
                                    flip_x=False,
                                    flip_y=False)  # Reshape the transient data in order to be in the same format used in the Fermat Flow algorithm

    temp_bin_centers = compute_bin_center(exp_time, data.shape[1])  # To build the temp_bin_centers it is required to build a vector where each cell contains the center of the correspondent temporal bin, so the first cell contains half the exposure time

    io.savemat(str(file_path), mdict={"detGridSize": np_flip(data_grid_size), "detLocs": det_locs, "srcLoc": src_loc, "temporalBinCenters": temp_bin_centers, "transients": data})  # Save the actual .mat file


def compute_bin_center(exp_time: float, n_bins: int) -> list:
    """
    Compute the center of the temporal bins
    :param exp_time: exposure time
    :param n_bins: number of temporal bins
    :return: list of the center of the temporal bins
    """

    temp_bin_centers = [exp_time / 2]  # To build the temp_bin_centers it is required to build a vector where each cell contains the center of the correspondent temporal bin, so the first cell contains half the exposure time
    for i in range(1, n_bins):
        temp_bin_centers.append(temp_bin_centers[i - 1] + exp_time)  # For all the following cell simply add the exposure time to the value stored in the previous cell
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
    mask_valid = 1.0*(depth < 30000)
    depth[depth > 30000] = 0
    shape_depth = (depth.shape[1], depth.shape[0])

    if dm == 'FISHEYE':
        [map1, map2] = fisheye.initUndistortRectifyMap(k_real, d_real, eye(3), k_ideal, shape_depth, CV_16SC2)
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    elif dm == 'STANDARD':
        depth = undistort(depth, k_real, d_real, None, k_ideal)
        mask_valid = undistort(mask_valid, k_real, d_real, None, k_ideal)

    else:
        [map1, map2] = initUndistortRectifyMap(k_real, d_real, eye(3), k_ideal, shape_depth, CV_16SC2)
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    mask_valid_positive = mask_valid > 0
    depth[mask_valid_positive] = divide(depth[mask_valid_positive], mask_valid[mask_valid_positive])

    z_matrix = depth
    x_matrix = (tile(arange(z_matrix.shape[1]), (z_matrix.shape[0], 1))).astype(dtype=float)    # [[0,1,2,3,...],[0,1,2,3,..],...]
    y_matrix = tile(arange(z_matrix.shape[0]), (z_matrix.shape[1], 1)).T.astype(dtype=float)  # [....,[1,1,1,1, ...][0,0,0,0,...]]

    x_undist_matrix = zeros_like(x_matrix, dtype=float)
    y_undist_matrix = zeros_like(y_matrix, dtype=float)
    z_undist_matrix = zeros_like(z_matrix, dtype=float)

    k_1 = linalg.inv(k_ideal)

    radial_dir = zeros([x_matrix.shape[0], x_matrix.shape[1], 3])
    for x in range(x_matrix.shape[0]):
        for y in range(x_matrix.shape[1]):
            prod = dot(k_1, asarray([x_matrix[x, y], y_matrix[x, y], 1]))
            prod = prod/linalg.norm(prod)

            x_undist_matrix[x, y] = z_matrix[x, y]*prod[0]
            y_undist_matrix[x, y] = z_matrix[x, y]*prod[1]
            z_undist_matrix[x, y] = z_matrix[x, y]*prod[2]
            radial_dir[x, y, :] = prod

    depthmap = stack([x_undist_matrix, y_undist_matrix, z_undist_matrix], axis=2)

    return depthmap, mask_valid_positive, radial_dir


def compute_los_points_coordinates(images: ndarray, mask: ndarray, k_matrix: ndarray, channel: int, exp_time: float) -> ndarray:
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
    dph = zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                dph[i, j] = tr.compute_radial_distance(peak_pos=tr.extract_peak(images[:, i, j, :])[0][channel], exposure_time=exp_time)

    # Compute the depthmap and coordinates
    depthmap, _, _ = undistort_depthmap(dph=dph,
                                        dm="RADIAL",
                                        k_ideal=k_matrix,
                                        k_real=k_matrix,
                                        d_real=array([[0, 0, 0, 0, 0]], dtype=float32))

    return depthmap.astype(float32)


def build_matrix_from_dot_projection_data(transient: ndarray, mask: ndarray) -> ndarray:
    """
    Function that build the full image from the measurements of the transient of only one dot of a time
    :param transient: measured transient information
    :param mask: dot pattern used
    :return: transient image
    """

    matrix = zeros([transient.shape[1], mask.shape[0], mask.shape[1], 3])
    t_index = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                matrix[:, i, j, :] = transient[t_index]
                t_index += 1

    return matrix


def coordinates_matrix_reshape(data: ndarray, mask: ndarray) -> ndarray:
    """
    Function that remove all the zeros in the coordinates and reshape it in a way that it contains the coordinates of each row one on the bottom of the other
    :param data: coordinates matrix
    :param mask: matrix that identify the used grid
    :return: reshaped coordinates matrix (z=0)
    """

    shape = [len(unique(where(mask != 0)[i])) for i in range(2)]
    data = reshape(data[mask != 0], [shape[0], shape[1], 2])  # Remove all the zero values

    matrix = zeros([data.shape[0]*data.shape[1], 3])  # Create the final matrix (full of zeros)
    m_index = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            matrix[m_index, 0] = data[j, i, 0]  # Load each cell with the right value
            matrix[m_index, 1] = data[j, i, 1]
            m_index += 1
    return matrix


def reshape_fermat_transient(transient: ndarray, grid_shape: tuple, flip_x: bool = False, flip_y: bool = False) -> ndarray:
    """
    Function that take the transient data where the grid is read row by row and rearrange the data to follow the reading column by column
    :param transient: transient data in the shape [<number of measurements>, <number of beans>]
    :param grid_shape: dimension of the grid [column, row]
    :param flip_x: flag that define if it is required to flip the order of the x coordinates
    :param flip_y: flag that define if it is required to flip the order of the x coordinates
    :return: the transient values rearranged to follow the Fermat Flow requirements
    """

    if flip_x:
        for i in range(0, transient.shape[0], grid_shape[1]):
            transient[i:(i + grid_shape[1]), :] = flip(transient[i:(i + grid_shape[1]), :], axis=0)  # If flip_x is True revers the order of the data in the x coordinate

    reshaped_transient = zeros(transient.shape)  # Initialize a ndarray full of zeros of the same size of the input transient one
    index = 0  # Initialize to zero the index that will cycle through the original transient vector
    for column_index in range(grid_shape[1]):  # Cycle column by column
        for row_index in range(grid_shape[0]):
            reshaped_transient[index] = transient[row_index*grid_shape[1] + column_index]  # Put the right column value in the new transient vector
            index += 1  # Update the index

    if flip_y:
        for i in range(0, reshaped_transient.shape[0], grid_shape[0]):
            reshaped_transient[i:(i + grid_shape[0]), :] = flip(reshaped_transient[i:(i + grid_shape[0]), :], axis=0)  # If flip_y is True revers the order of the data in the x coordinate

    return reshaped_transient


def rmv_first_reflection_fermat_transient(transient: ndarray, file_path: Path = None, store: bool = False) -> ndarray:
    """
    Function that given the transient images in the fermat flow shape remove the first reflection leaving only the global component
    :param transient: input transient images
    :param file_path: path of the np dataset
    :param store: if you want to save the np dataset (if already saved set it to false in order to load it)
    :return: Transient images of only the global component as a np array
    """

    if file_path and not store:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return load(str(file_path))

    print("Extracting the first peak (channel by channel):")
    start = time()

    mono = len(transient.shape) == 2  # Check id=f the images are Mono or RGBA

    if not mono:
        peaks = [nanargmax(transient[:, :, channel_i], axis=1) for channel_i in tqdm(range(transient.shape[2]))]  # Find the index of the maximum value in the third dimension
    else:
        peaks = nanargmax(transient, axis=1)  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    print("Remove the first peak (channel by channel):")
    sleep(0.1)

    glb_images = copy(transient)
    first_direct = transient.shape[1]
    if not mono:
        for channel_i in tqdm(range(transient.shape[2])):
            for pixel in range(transient.shape[0]):
                zeros_pos = where(transient[pixel, :, channel_i] == 0)[0]
                valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[channel_i][pixel])]
                if valid_zero_indexes.size == 0:
                    glb_images[pixel, :, channel_i] = -2
                else:
                    glb_images[pixel, :int(valid_zero_indexes[0]), channel_i] = -1
                    if valid_zero_indexes[0] < first_direct:
                        first_direct = valid_zero_indexes[0]
        print("Shift the transient vector so the global always start at t=0:")
        sleep(0.1)
        glb_images_shifted = zeros([transient.shape[0], transient.shape[1] - first_direct, transient.shape[2]])
        for channel_i in tqdm(range(transient.shape[2])):
            for pixel in range(transient.shape[0]):
                if glb_images[pixel, 0, channel_i] != -2:
                    shifted_transient = delete(glb_images[pixel, :, channel_i], [where(glb_images[pixel, :, channel_i] == -1)])
                    glb_images_shifted[pixel, :shifted_transient.shape[0], channel_i] = shifted_transient
    else:
        for pixel in tqdm(range(transient.shape[1])):
            zeros_pos = where(transient[pixel, :] == 0)[0]
            valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[pixel])]
            if valid_zero_indexes.size == 0:
                glb_images[pixel, :] = -2
            else:
                glb_images[pixel, :int(valid_zero_indexes[0])] = -1
        print("Shift the transient vector so the global always start at t=0:")
        sleep(0.1)
        glb_images_shifted = zeros([transient.shape[0], transient.shape[1] - first_direct])
        for pixel in range(transient.shape[0]):
            if glb_images[pixel, 0] != -2:
                shifted_transient = delete(glb_images[pixel, :], [where(glb_images[pixel, :] == -1)])
                glb_images_shifted[pixel, :shifted_transient.shape[0]] = shifted_transient

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    if file_path and store:
        save(str(file_path), glb_images_shifted)  # Save the loaded images as a numpy array
    return glb_images_shifted


def rmv_sparse_fermat_transient(transients: ndarray, channel: int, threshold: int, remove_data: bool) -> ndarray:
    """
    Function that delete or set to zero all the transient that have an active bins percentage lower than <threshold>
    :param transients: transient vectors (set in the fermat flow setup)
    :param channel: chose which channel to check (0 = red, 1 = green, 2 = blu)
    :param threshold: percentage value below which the transient will be discarded
    :param remove_data: flag that decide if the data will be removed or simply set to zero
    :return: cleaned transient
    """

    indexes = []  # List that will contain the indexes of all the transient row that will be discarded
    for i in range(transients.shape[0]):  # Analyze each transient one by one
        peaks, _ = extract_peak(transients[i, :, :])  # Extract the position of the direct component in all the three channel
        peak = peaks[channel]  # Keep only the information about the channel of interest
        zeros_pos = where(transients[i, :, channel] == 0)[0]  # Find where all the zeros are located
        first_zero_after_peak = zeros_pos[where(zeros_pos > peak)][0]  # Keep only the first zero after the direct component
        if active_beans_percentage(transients[i, first_zero_after_peak:, channel]) < threshold:  # If the percentage of active bins in the global component is below th threshold:
            if remove_data:  # If remove_data is set to True:
                indexes.append(i)  # Add the considered row index to the indexes list
            else:  # If remove_data is set to False:
                transients[i, :, :] = 0  # Set to zero all the row
    if remove_data:  # If remove_data is set to True:
        return delete(transients, indexes, axis=0)  # Remove all the row which index is inside the indexes list
    else:  # If remove_data is set to False:
        return transients  # Return the transient with the modification applied


def nearest_nonzero_idx(matrix: ndarray, x: int, y: int) -> tuple:
    """
    Function to find the closest non-zero element to the point (x, y)
    (code from: https://stackoverflow.com/questions/43306291/find-the-nearest-nonzero-element-and-corresponding-index-in-a-2d-numpy-array)
    :param matrix: 2D data matrix where to search the location
    :param x: x coordinate of the point of interest
    :param y: y coordinate of the point of interest
    :return: the (x, y) coordinate of the nearest non-zero point
    """

    tmp = matrix[x, y]
    matrix[x, y] = 0
    r, c = nonzero(matrix)
    matrix[x, y] = tmp
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]


def roto_transl(coordinates_matrix: ndarray) -> ndarray:
    """
    Function that move the given coordinates from the camera coordinates system to the one of the world making sure that the points stays on the plane z=0
    :param coordinates_matrix: coordinates matrix in the camera system
    :return: coordinates matrix in te word system
    """

    coordinates_matrix = np_round(coordinates_matrix, decimals=2)  # exp_time = 0.01 max sensibility is 1cm
    center_pos = (int(coordinates_matrix.shape[0] / 2), int(coordinates_matrix.shape[1] / 2))  # Find the center position on the coordinates matrix
    nearest_center_pos = nearest_nonzero_idx(coordinates_matrix[:, :, 0], center_pos[0], center_pos[1])  # Find the location of the active poit closest to the real center
    nearest_center_coord = [coordinates_matrix[nearest_center_pos[0], nearest_center_pos[1], i] for i in range(coordinates_matrix.shape[2])]  # Extract from the coordinates' matrix the coordinates of the point closest to the center

    next_x_pos = nonzero(coordinates_matrix[nearest_center_pos[0], :, 0])[0]  # Find the location of the point on the coordinates' matrix located on the same y of the nearest_center but on the extreme right (max x)
    next_x_pos = (nearest_center_pos[0], next_x_pos[-1])  # Define the indexes where that point is located in the coordinates' matrix (add the row index)
    next_y_pos = nonzero(coordinates_matrix[:, nearest_center_pos[1], 0])[0]  # Find the location of the point on the coordinates' matrix located on the same x of the nearest_center but on the extreme top (max y)
    next_y_pos = (next_y_pos[0], nearest_center_pos[1])  # Define the indexes where that point is located in the coordinates' matrix (add the column index)
    next_x_coord = coordinates_matrix[next_x_pos[0], next_x_pos[1], :]  # Extract from the coordinates' matrix the coordinates of next_x
    next_y_coord = coordinates_matrix[next_y_pos[0], next_y_pos[1], :]  # Extract from the coordinates' matrix the coordinates of next_y

    normal = cross(subtract(next_x_coord, nearest_center_coord), subtract(next_y_coord, nearest_center_coord))  # Compute the normal of the plane making the cross product of the vector between the center and the far right point and of the vector between the center and the far top point
    normalized_normal = normal / linalg.norm(normal)  # Normalize the normal vector

    n_x = normalized_normal[0]  # x component of the normal vector
    n_y = normalized_normal[1]  # y component of the normal vector
    n_z = normalized_normal[2]  # z component of the normal vector

    if n_x != 0 or n_y != 0:
        sqrt_squared_sum_nx_ny = sqrt((n_x ** 2) + (n_y ** 2))
        rot_matrix = array([[n_y / sqrt_squared_sum_nx_ny, -n_x / sqrt_squared_sum_nx_ny, 0],
                            [(n_x * n_z) / sqrt_squared_sum_nx_ny, (n_y * n_z) / sqrt_squared_sum_nx_ny, -sqrt_squared_sum_nx_ny],
                            [n_x, n_y, n_z]], dtype=float32)  # Build the rotation matrix (code from: https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector)

        o_rot_matrix = array([[0, 1, 0],
                              [-1, 0, 0],
                              [0, 0, 1]], dtype=float32)  # Define a second rotation matrix to compensate for the rotation 90° over the y-axis
        rot_matrix = matmul(o_rot_matrix, rot_matrix)  # Define the complete rotation matrix multiplying together the two previously defined ones
    else:
        rot_matrix = array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]], dtype=float32)

    tr_vector = matmul(rot_matrix, reshape(nearest_center_coord, [3, 1]), dtype=float32)  # Define a translation vector that aims to center the plane on the nearest_center point

    # Build the final roto-transl matrix
    rototransl_matrix = concatenate((rot_matrix, negative(tr_vector)), axis=1)  # Add to the left of the rotation matrix the translation one
    rototransl_matrix = concatenate((rototransl_matrix, array([[0, 0, 0, 1]], dtype=float32)), axis=0)  # Add to the bottom the vector [0 0 0 1]

    non_zero_pos = nonzero(coordinates_matrix[:, :, 0])  # Find the location where the coordinates' matrix is not zero
    for r in unique(non_zero_pos[0]):
        for c in unique(non_zero_pos[1]):
            coord_vector = concatenate((reshape(coordinates_matrix[r, c, :], (3, 1)), array([[1]], dtype=float32)), axis=0)  # For every non-zero point of the coordinates' matrix concatenate a 1 at the end and reshape it from [3], to [4, 1]
            coordinates_matrix[r, c, :] = matmul(rototransl_matrix, coord_vector, dtype=float32)[:-1, 0]  # Apply the roto-translation to each non-zero point and remove the final 1
    coordinates_matrix[where(coordinates_matrix == -0.0)] = 0  # Change all the -0.0 to 0

    return np_round(coordinates_matrix, decimals=2)
