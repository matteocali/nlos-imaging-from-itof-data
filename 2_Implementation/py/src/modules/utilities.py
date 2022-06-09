from modules import transient_handler as tr
from numpy import sum, linspace, zeros, where, nanmin, nanmax, array, ndarray, copy, eye, divide, tile, arange, zeros_like, linalg, dot, asarray, stack, reshape
from numpy import uint8, float32
from os import path, listdir, remove, makedirs
from pathlib import Path
from glob import glob
from natsort import natsorted
from cv2 import imwrite, cvtColor, fisheye, remap, undistort, initUndistortRectifyMap
from cv2 import COLOR_RGBA2BGRA, COLOR_RGB2BGR, CV_16SC2, INTER_LINEAR
from matplotlib import pyplot as plt
from h5py import File
from scipy import io
from math import floor


def add_extension(name: str, ext: str) -> str:
    """
    Function that checks the name of a file and if not already present adds the given extension
    :param name: name of the file
    :param ext: desired extension
    :return: name of the file with the correct extension attached
    """
    if name[-len(ext):] == ext:
        return name
    else:
        return name + ext


def reed_files(file_path, extension, reorder=True):
    """
    Function to load all the files in a folder and if needed reorder them using the numbers in the final part of the name
    :param reorder: flag to toggle the reorder process (default = true)
    :param file_path: source folder path
    :param extension: extension of the files to load
    :return: list of file paths
    """

    files = [file_name for file_name in glob(str(file_path) + "\\*." + extension)]  # Load the path of all the files in the input folder with the target extension
                                                                                    # (code from: https://www.delftstack.com/howto/python/python-open-all-files-in-directory/)
    if reorder:
        files = natsorted(files, key=lambda y: y.lower())  # Sort alphanumeric in ascending order
                                                           # (code from: https://studysection.com/blog/how-to-sort-a-list-in-alphanumeric-order-python/)
    return files


def read_folders(folder_path, reorder=True):
    """
    Function that return the path of all the subdirectories in a given directory
    :param folder_path: path of the main directory
    :param reorder: flag to toggle the reorder process (default = true)
    :return: list of all the subdirectory folder
    """
    folders = []
    # Extract from the main directory only the subdirectories' path avoiding the file ones
    # (code from: https://www.techiedelight.com/list-all-subdirectories-in-directory-python/)
    for file in listdir(folder_path):
        d = path.join(folder_path, file)
        if path.isdir(d):
            folders.append(d)

    if reorder:
        folders = natsorted(folders, key=lambda y: y.lower())  # Sort alphanumeric in ascending order
                                                               # (code from: https://studysection.com/blog/how-to-sort-a-list-in-alphanumeric-order-python/)

    return folders


def create_folder(file_path, ignore: str = "") -> None:
    """
    Function to create a new folder if not already present.
    If it already exists, empty it
    :param file_path: path of the folder to create
    :param ignore: file name to not delete
    """

    if path.exists(file_path):  # If the folder is already present remove all its child files
                                # (code from: https://pynative.com/python-delete-files-and-directories/#h-delete-all-files-from-a-directory)
        for file_name in listdir(file_path):
            file = file_path / file_name  # Construct full file path
            if path.isfile(file) and file_name != ignore and ignore != "all":  # If the file is a file remove it
                remove(file)
    else:  # Create the required folder if not already present
        makedirs(file_path)


def generate_quadratic_model(min_x: float, max_x: float, max_y: float, precision: int) -> list:
    """
    Function that define the ground truth quadratic model
    :param min_x: minimum measured value of distances
    :param max_x: maximum measured value of distances
    :param max_y: maximum measured value of radiance
    :param precision: number of samples inside the linear space
    :return: x and y value of the quadratic model
    """

    x = linspace(min_x, max_x, precision)  # Define the x vector as a linear space
    scaling = max_y * pow(min_x, 2)  # Define the scale factor to center the quadratic model on the measured data
    y = scaling / pow(x, 2)  # Build the quadratic model
    return [x, y]


def compute_mse(x: ndarray, y: ndarray) -> float:
    """
    Compute the MSE error between two images
    The 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    The lower the error, the more "similar" the two images are
    (code from: https://pyimagesearch.com/2014/09/15/python-compare-two-images/)
    :param x: image 1
    :param y: image 2 (must have same dimensions of image 1)
    :return The MSE value rounded at the fourth value after comma
    """

    err = sum((x.astype("float") - y.astype("float")) ** 2)  # Convert the images to floating point
                                                                # Take the difference between the images by subtracting the pixel intensities
                                                                # Square these difference and sum them up
    err /= float(x.shape[0] * x.shape[1])  # Handles the mean of the MSE

    return round(float(err), 4)


def normalize_img(img: ndarray) -> ndarray:
    """
    Normalize the image value in the range [0, 1]
    :param img: np aray corresponding to an image
    :return np containing the normalized img
    """

    img[where(img < 0)] = 0
    min_val = nanmin(img)
    max_val = nanmax(img)

    if max_val - min_val != 0:
        for i in range(img.shape[2]):
            img[:, :, i] = (img[:, :, i] - min_val) / (max_val - min_val)  # Normalize each image in [0, 1] ignoring the alpha channel
    return img


def save_png(img: ndarray, file_path: Path) -> None:
    """
    Function to save an image as a png
    :param img: image to save
    :param file_path: path and name
    """
    img = (255 * img).astype(uint8)  # Rescale the input value from [0, 1] to [0, 255] and convert them to unit8
    if img.shape[2] == 4:
        imwrite(str(file_path), cvtColor(img, COLOR_RGBA2BGRA))  # Save the image
    else:
        imwrite(str(file_path), cvtColor(img, COLOR_RGB2BGR))  # Save the image


def save_plt(img: ndarray, file_path: Path, alpha: bool) -> None:
    """
    Function to save an image as a matplotlib png
    :param alpha: define if the output will use or not the alpha channel (True/False)
    :param img: image to save
    :param file_path: path and name
    """

    if not alpha:
        plt.imsave(file_path, img[:, :, :-1])
    else:
        plt.imsave(file_path, img)


def spot_bitmap_gen(img_size: list, file_path: Path = None, spot_size: list = None, exact: bool = False, pattern: tuple = None, split: bool = False) -> ndarray:
    """
    Function that generate a black bitmap image of size img_path with a white square in the middle of size (spot_size * spot_size)
    :param file_path: path where to save the generated image
    :param img_size: size of the desired image [columns * rows]
    :param spot_size: size of the desired white spot [columns * rows]
    :param exact: flag to set white a specific pixel
    :param pattern: list made as follows [x, y] where x represent the interval between white pixel on each row and y on each column
    :return generated image
    """

    img = zeros([img_size[1], img_size[0]], dtype=uint8)  # Generate the base black image

    # Change ve value to white of only the desired center pixels
    if exact:
        img[spot_size[1], spot_size[0]] = 255
    elif not exact and spot_size is not None:
        spot_size = [int(spot_size[0] / 2), int(spot_size[1] / 2)]
        if img_size[0] % 2 == 0 and img_size[1] % 2 == 0:
            img[(int(img_size[1] / 2) - spot_size[1]):(int(img_size[1] / 2) + spot_size[1]), (int(img_size[0] / 2) - spot_size[0]):(int(img_size[0] / 2) + spot_size[0])] = 255
        elif img_size[0] % 2 == 0 and img_size[1] % 2 != 0:
            img[int(img_size[1] / 2), (int(img_size[0] / 2) - spot_size[0]):(int(img_size[0] / 2) + spot_size[0])] = 255
        if img_size[0] % 2 != 0 and img_size[1] % 2 == 0:
            img[int((img_size[1] / 2) - spot_size[1]):(int(img_size[1] / 2) + spot_size[1]), int(img_size[0] / 2)] = 255
        if img_size[0] % 2 != 0 and img_size[1] % 2 != 0:
            img[int(img_size[1] / 2), int(img_size[0] / 2)] = 255
    elif not exact and pattern is not None:
        offset_x = floor(((img_size[0] - 1) - ((int(img_size[0] / pattern[0]) - 1) * pattern[0])) / 2)
        offset_y = floor(((img_size[1] - 1) - ((int(img_size[1] / pattern[1]) - 1) * pattern[1])) / 2)
        for i in range(offset_y, img.shape[0], pattern[1]):
            for j in range(offset_x, img.shape[1], pattern[0]):
                if not split:
                    img[i, j] = 255
                else:
                    create_folder(file_path, "all")
                    tmp = copy(img)
                    tmp[i, j] = 255
                    imwrite(str(file_path / f"bitmap_r{i}_c{j}.png"), tmp)  # Save the image

    if file_path is not None and not split:
        file_path = add_extension(str(file_path), ".png")
        imwrite(file_path, img)  # Save the image


    return img


def load_h5(file_path: Path):
    """
    Function that load a .h5 file and return its content as a np array. If the .h5 file contains more than one keys it returns a list of np arrays one for each key
    :param file_path: path of the .h5 file
    :return: a np array containing the content of the .h5 file or a list of np array each one containing the content of a key of the .h5 file
    """

    h5_file = File(str(file_path), 'r')  # Open the .h5 file
    keys = list(h5_file.keys())  # Obtain the list of keys contained in the .h5 file

    # Check if the .h5 file has only one key or more than one
    if len(keys) == 1:
        return array(h5_file[keys[0]])  # Load the .h5 content and put it inside a np array
    else:
        return [array(h5_file[key]) for key in keys]  # Load the .h5 content (key by key) and put it inside a np array


def save_h5(data: ndarray, file_path: Path, name: str = None) -> None:
    """
    Function to save a transient image into an .h5 file (also perform reshaping [<n_beans>, <n_row>, <col>] -> [<n_row>, <col>, <n_beans>])
    :param data: ndarray containing the transient image (only one channel)
    :param file_path: path (with name) where to save the file
    :param name: name of the key of the data inside the .h5 file
    """

    file_path = add_extension(str(file_path), ".h5")  # If not already present add the .h5 extension to the file path
    data = copy(data)  # Copy the ndarray in order to avoid overriding
    data = data.reshape([data.shape[1], data.shape[2], data.shape[0]])  # Reshape the array in order to match the required layout
    h5f = File(file_path, "w")  # Create the .h5 file and open it
    # Save the ndarray in the just created .h5 file
    if name:
        h5f.create_dataset(name=name,
                           data=data,
                           shape=data.shape,
                           dtype=float32)
    else:
        h5f.create_dataset(name=file_path.split("\\")[-1][-3:],  # If a key name is not provided use the name of the name of the file as key name
                           data=data,
                           shape=data.shape,
                           dtype=float32)


def np2mat(data: ndarray, file_path: Path, data_grid_size: list, img_shape: list, pattern_interval: tuple, exp_time: float = 0.01, laser_pos: list = None) -> None:
    """
    Function to save a .mat file in the format required from the Fermat flow Matlab script
    :param data: ndarray containing the transient measurements (n*m matrix, n transient measurements with m temporal bins)
    :param file_path: file path and name where to save the .mat file
    :param data_grid_size: [n1, n2], with n1 * n2 = n, grid size of sensing points on the LOS wall (n = number of the transient measurements)
    :param img_shape: size of the image [<n_row>, <n_col>]
    :param pattern_interval: tuple made as follows [x, y] where x represent the interval between white pixel on each row and y on each column
    :param exp_time: exposure time used, required to compute "temporalBinCenters"
    :param laser_pos: position of the laser, if none it is confocal with the camera (1*3 vector)
    """

    file_path = add_extension(str(file_path), ".mat")  # If not already present add the .h5 extension to the file path

    # Define all the required vectors for the .mat file
    data_grid_size = array(data_grid_size, dtype=ndarray)  # Convert the data_grid_size from a list to an ndarray

    mask = spot_bitmap_gen(img_size=img_shape,
                           pattern=pattern_interval)  # Define the mask that identify the location of the illuminated spots
    k = array([[276.2621, 0, 159.5],
               [0, 276.2621, 119.5],
               [0, 0, 1]], dtype=float32)  # Define the intrinsic camera matrix needed to map the dots on the LOS wall
    transient_image = build_matrix_from_dot_projection_data(transient=data, mask=mask)  # Put all the transient data in the right spot following the mask
    depthmap = compute_los_points_coordinates(images=transient_image,
                                              mask=mask,
                                              k_matrix=k)[:, :, :-1]  # Compute the mapping of the coordinate of the illuminated spots to the LOS wall (ignor the z coordinate)
    det_locs = coordinates_matrix_reshape(data=depthmap, shape=(int(img_shape[0] / pattern_interval[0]), int(img_shape[1] / pattern_interval[1])))  # Reshape the coordinate value, putting the coordinate of each row one on the bottom of the previous one

    if laser_pos is None:   # If laser_pos is not provided it means that the laser is confocal with the camera
        src_loc = array((), dtype=float32)
    else:
        src_loc = array(laser_pos, dtype=float32)

    temp_bin_centers = [exp_time/2]  # To build the temp_bin_centers it is required to build a vector where each cell contains the center of the correspondent temporal bin, so the first cell contains half the exposure time
    for i in range(1, data.shape[0]):
        temp_bin_centers.append(temp_bin_centers[i - 1] + exp_time)  # For all the following cell simply add the exposure time to the value stored in the previous cell

    io.savemat(str(file_path), mdict={"detGridSize": data_grid_size, "detLocs": det_locs, "srcLoc": src_loc, "temporalBinCenters": temp_bin_centers, "transients": data})  # Save the actual .mat file


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

            x_undist_matrix[x, y] = z_matrix[x, y]*prod[0]
            y_undist_matrix[x, y] = z_matrix[x, y]*prod[1]
            z_undist_matrix[x, y] = z_matrix[x, y]*prod[2]
            radial_dir[x, y, :] = prod/linalg.norm(prod)

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
                dph[i, j] = tr.compute_radial_distance(peak_pos=tr.extract_peak(images[:, i, j, :])[0][1],
                                                       exposure_time=0.01)

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

    matrix = zeros(mask.shape)
    t_index = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                matrix[i, j] = transient[t_index]
                t_index += 1

    return matrix


def coordinates_matrix_reshape(data: ndarray, shape: tuple) -> ndarray:
    """
    Function that remove all the zeros in the coordinates and reshape it in a way that it contains the coordinates of each row one on the bottom of the other
    :param data: coordinates matrix
    :param shape: shape of the matrix without zeros
    :return: reshaped coordinates matrix (z=0)
    """

    data = reshape(data[data != 0], [shape[0], shape[1], 2])  # Remove all the zero values
    matrix = zeros([data.shape[0]*data.shape[1], 3])  # Create the final matrix (full of zeros)
    m_index = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            matrix[m_index, 0] = data[0, i, 0]  # Load each cell with the right value
            matrix[m_index, 1] = data[j, 0, 1]
            m_index += 1
    return matrix
