from os.path import dirname, exists
from pathlib import Path
from time import time, sleep

from cv2 import fisheye, CV_16SC2, remap, INTER_LINEAR, undistort, initUndistortRectifyMap
from numpy import ndarray, array, float64, float32, eye, divide, tile, arange, zeros_like, linalg, zeros, dot, asarray, \
    stack, reshape, flip, load, nanargmax, copy, where, delete, save
from scipy import io
from tqdm import tqdm

from modules import transient_handler as tr
from modules.transient_handler import extract_peak, active_beans_percentage
from modules.utilities import add_extension, spot_bitmap_gen


def np2mat(data: ndarray, file_path: Path, data_grid_size: list, img_shape: list, exp_time: float = 0.01, laser_pos: list = None) -> None:
    """
    Function to save a .mat file in the format required from the Fermat flow Matlab script
    :param data: ndarray containing the transient measurements (n*m matrix, n transient measurements with m temporal bins)
    :param file_path: file path and name where to save the .mat file
    :param data_grid_size: [n1, n2], with n1 * n2 = n, grid size of sensing points on the LOS wall (n = number of the transient measurements)
    :param img_shape: size of the image [<n_row>, <n_col>]
    :param exp_time: exposure time used, required to compute "temporalBinCenters"
    :param laser_pos: position of the laser, if none it is confocal with the camera (1*3 vector)
    """

    np_file_path = dirname(file_path) + "\\glb_np_transient.npy"
    file_path = add_extension(str(file_path), ".mat")  # If not already present add the .h5 extension to the file path
    pattern_interval = (int(img_shape[0] / data_grid_size[0]), int(img_shape[1] / data_grid_size[1]))

    # Define all the required vectors for the .mat file
    data_grid_size = array(data_grid_size, dtype=float64)  # Convert the data_grid_size from a list to an ndarray

    mask = spot_bitmap_gen(img_size=img_shape,
                           pattern=pattern_interval)  # Define the mask that identify the location of the illuminated spots
    k = array([[276.2621, 0, 159.5],
               [0, 276.2621, 119.5],
               [0, 0, 1]], dtype=float32)  # Define the intrinsic camera matrix needed to map the dots on the LOS wall
    transient_image = build_matrix_from_dot_projection_data(transient=data, mask=mask)  # Put all the transient data in the right spot following the mask
    depthmap = compute_los_points_coordinates(images=transient_image,
                                              mask=mask,
                                              k_matrix=k)[:, :, :-1]  # Compute the mapping of the coordinate of the illuminated spots to the LOS wall (ignor the z coordinate)
    det_locs = coordinates_matrix_reshape(data=depthmap, shape=(int(data_grid_size[0]), int(data_grid_size[1])))  # Reshape the coordinate value, putting the coordinate of each row one on the bottom of the previous one

    if laser_pos is None:   # If laser_pos is not provided it means that the laser is confocal with the camera
        src_loc = array((), dtype=float32)
    else:
        src_loc = array(laser_pos, dtype=float32)

    data = rmv_first_reflection_fermat_transient(transient=data, file_path=np_file_path, store=(not exists(np_file_path)))  # Remove the direct component from all the transient data

    data = reshape_fermat_transient(transient=data[:, :, 1],
                                    grid_shape=(int(data_grid_size[1]), int(data_grid_size[0])),
                                    flip_x=True,
                                    flip_y=True)  # Reshape the transient data in order to be in the same format used in the Fermat Flow algorithm
    data = data[2*16:20*16, :]
    det_locs = det_locs[2*16:20*16, :]

    temp_bin_centers = [exp_time / 2]  # To build the temp_bin_centers it is required to build a vector where each cell contains the center of the correspondent temporal bin, so the first cell contains half the exposure time
    for i in range(1, data.shape[1]):
        temp_bin_centers.append(temp_bin_centers[i - 1] + exp_time)  # For all the following cell simply add the exposure time to the value stored in the previous cell

    io.savemat(str(file_path), mdict={"detGridSize": data_grid_size, "detLocs": det_locs, "srcLoc": src_loc, "temporalBinCenters": temp_bin_centers, "transients": data[:, :]})  # Save the actual .mat file


def undistort_depthmap(dph, dm, k_ideal, k_real, d_real):
    """
    Undistort depth map using calibration output parameters
    :param dph: depthmap (1 channel image)
    :param dm: string with the name of the camera model (FISHEYE, RADIAL, RATIONAL)
    :param k_ideal: Camera matrix
    :param k_real: Camera matrix
    :param d_real: Distortion coefficient
    :return depthmap: undistorted depthmap with 3 dimension (x-axis coordinates, y-axis coordinates, z-coordinates)
    :return mask_valid_positive: validity mask (1=valid points, 0= oor or invalid doths)
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


def compute_los_points_coordinates(images: ndarray, mask: ndarray, k_matrix: ndarray) -> ndarray:
    """
    Function that compute the coordinates of the projected points on the LOS wall
    :param images: transient information
    :param mask: position of the point on the bitmap
    :param k_matrix: intrinsic camera matrix
    :return: matrix where the first dimension contains the x coordinates, the second one the y coordinates and the third one the depth information
    """

    # Compute the depth information only on the point that the mask consider
    dph = zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                dph[i, j] = tr.compute_radial_distance(peak_pos=tr.extract_peak(images[:, i, j, :])[0][1], exposure_time=0.01)

    # Compute the depthmap and coordinates
    depthmap, _, _ = undistort_depthmap(dph=dph,
                                        dm="RADIAL",
                                        k_ideal=k_matrix,
                                        k_real=k_matrix,
                                        d_real=array([[0, 0, 0, 0, 0]], dtype=float32))

    return depthmap


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


def coordinates_matrix_reshape(data: ndarray, shape: tuple) -> ndarray:
    """
    Function that remove all the zeros in the coordinates and reshape it in a way that it contains the coordinates of each row one on the bottom of the other
    :param data: coordinates matrix
    :param shape: shape of the matrix without zeros
    :return: reshaped coordinates matrix (z=0)
    """

    data = reshape(data[data != 0], [shape[1], shape[0], 2])  # Remove all the zero values
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

    reshaped_transient = zeros(transient.shape)  # Initialize a ndarray full of zeros of the same size of the input transient one
    index = 0  # Initialize to zero the index that will cycle through the original transient vector
    for column_index in range(grid_shape[1]):  # Cycle column by column
        for row_index in range(grid_shape[0]):
            reshaped_transient[index] = transient[row_index*grid_shape[1] + column_index]  # Put the right column value in the new transient vector
            index += 1  # Update the index

    if flip_x:
        reshaped_transient = flip(reshaped_transient, axis=0)  # If flip_x is True revers the order of the data in the x coordinate
    if flip_y:
        for i in range(0, reshaped_transient.shape[0], grid_shape[0]):
            reshaped_transient[i:(i + grid_shape[0]), :] = flip(reshaped_transient[i:(i + grid_shape[0]), :], axis=0)  # If flip_x is True revers the order of the data in the x coordinate

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
        peaks = nanargmax(transient, axis=0)  # Find the index of the maximum value in the third dimension

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
