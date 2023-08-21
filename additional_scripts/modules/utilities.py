import numpy as np
from os import path, listdir, remove, makedirs
from pathlib import Path
from glob import glob
from natsort import natsorted
from cv2 import imwrite, cvtColor, COLOR_RGBA2BGRA, COLOR_RGB2BGR
from matplotlib import pyplot as plt
from h5py import File
from math import floor
from pickle import dump, load
from itertools import product
from random import seed as rnd_seed, shuffle
from tqdm import trange
from typing import Union


def add_extension(name: Union[str, Path], ext: str) -> Path:
    """
    Function that checks the name of a file and if not already present adds the given extension
    :param name: name of the file
    :param ext: desired extension
    :return: name of the file with the correct extension attached
    """
    if type(name) is not str:  # If the name is not a string convert it to a string
        name = str(name)
    if name[-len(ext) :] == ext:  # If the extension is already present return the name
        return Path(name)
    else:  # If the extension is not present add it
        return Path(name + ext)


def read_files(file_path: Union[str, Path], extension: str, reorder: bool = True):
    """
    Function to load all the files in a folder and if needed reorder them using the numbers in the final part of the name
    :param reorder: flag to toggle the reorder process (default = true)
    :param file_path: source folder path
    :param extension: extension of the files to load
    :return: list of file paths
    """

    if type(file_path) is not Path:
        file_path = Path(file_path)
    files = [
        file_name for file_name in glob(str(file_path / f"*.{extension}"))
    ]  # Load the path of all the files in the input folder with the target extension
    # (code from: https://www.delftstack.com/howto/python/python-open-all-files-in-directory/)
    if reorder:
        files = natsorted(
            files, key=lambda y: y.lower()
        )  # Sort alphanumeric in ascending order
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
        folders = natsorted(
            folders, key=lambda y: y.lower()
        )  # Sort alphanumeric in ascending order
        # (code from: https://studysection.com/blog/how-to-sort-a-list-in-alphanumeric-order-python/)

    return folders


def create_folder(file_path, ignore: str = "") -> None:
    """
    Function to create a new folder if not already present.
    If it already exists, empty it
    :param file_path: path of the folder to create
    :param ignore: file name to not delete
    """

    if path.exists(
        file_path
    ):  # If the folder is already present remove all its child files
        # (code from: https://pynative.com/python-delete-files-and-directories/#h-delete-all-files-from-a-directory)
        for file_name in listdir(file_path):
            file = file_path / file_name  # Construct full file path
            if (
                path.isfile(file) and file_name != ignore and ignore != "all"
            ):  # If the file is a file remove it
                remove(file)
    else:  # Create the required folder if not already present
        makedirs(file_path)


def generate_quadratic_model(
    min_x: float, max_x: float, max_y: float, precision: int
) -> list:
    """
    Function that define the ground truth quadratic model
    :param min_x: minimum measured value of distances
    :param max_x: maximum measured value of distances
    :param max_y: maximum measured value of radiance
    :param precision: number of samples inside the linear space
    :return: x and y value of the quadratic model
    """

    x = np.linspace(min_x, max_x, precision)  # Define the x vector as a linear space
    scaling = max_y * pow(
        min_x, 2
    )  # Define the scale factor to center the quadratic model on the measured data
    y = scaling / pow(x, 2)  # Build the quadratic model
    return [x, y]


def compute_mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the MSE error between two images
    The 'Mean Squared Error' between the two images is the np.sum of the squared difference between the two images
    The lower the error, the more "similar" the two images are
    (code from: https://pyimagesearch.com/2014/09/15/python-compare-two-images/)
    :param x: image 1
    :param y: image 2 (must have same dimensions of image 1)
    :return The MSE value rounded at the fourth value after comma
    """

    err = np.sum(
        (x.astype("float") - y.astype("float")) ** 2
    )  # Convert the images to floating point
    # Take the difference between the images by subtracting the pixel intensities
    # Square these difference and np.sum them up
    err /= float(x.shape[0] * x.shape[1])  # Handles the mean of the MSE

    return round(float(err), 4)


def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize the image value in the range [0, 1]
    :param img: np aray corresponding to an image
    :return np containing the normalized img
    """

    img[np.where(img < 0)] = 0
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)

    if max_val - min_val != 0:
        for i in range(img.shape[2]):
            img[:, :, i] = (img[:, :, i] - min_val) / (
                max_val - min_val
            )  # Normalize each image in [0, 1] ignoring the alpha channel
    return img


def exr2rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert an EXR image to RGB
    :param img: np np.array corresponding to an EXR image
    :return: np np.array corresponding to an RGB image
    """

    # linear to standard RGB
    img[..., :3] = np.where(
        img[..., :3] <= 0.0031308,
        12.92 * img[..., :3],
        1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055,
    )

    # sanitize image to be in range [0, 1]
    img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))
    return img


def save_png(img: np.ndarray, file_path: Path) -> None:
    """
    Function to save an image as a png
    :param img: image to save
    :param file_path: path and name
    """
    img = (255 * img).astype(
        np.uint8
    )  # Rescale the input value from [0, 1] to [0, 255] and convert them to unit8
    if img.shape[2] == 4:
        imwrite(str(file_path), cvtColor(img, COLOR_RGBA2BGRA))  # Save the image
    else:
        imwrite(str(file_path), cvtColor(img, COLOR_RGB2BGR))  # Save the image


def save_plt(img: np.ndarray, file_path: Path, alpha: bool) -> None:
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


def spot_bitmap_gen(
    img_size: list,
    file_path: Path = None,
    spot_size: list = None,
    exact: bool = False,
    pattern: tuple = None,
    split: bool = False,
) -> np.ndarray:
    """
    Function that generate a black bitmap image of size img_path with a white square in the middle of size (spot_size * spot_size)
    :param file_path: path where to save the generated image
    :param img_size: size of the desired image [columns * rows]
    :param spot_size: size of the desired white spot [columns * rows]
    :param exact: flag to set white a specific pixel
    :param pattern: list made as follows [x, y] where x represent the number of white pixel in each row and y the number of white pixels in each column
    :param split: if true the grid is split dot by dot
    :return generated image
    """

    img = np.zeros(
        [img_size[1], img_size[0]], dtype=np.uint8
    )  # Generate the base black image

    if exact:  # Change the value to white of only the desired pixel
        img[spot_size[1], spot_size[0]] = 255
    elif (
        not exact and spot_size is not None
    ):  # Change the value to white of only the desired central pixels
        spot_size = [int(spot_size[0] / 2), int(spot_size[1] / 2)]
        if img_size[0] % 2 == 0 and img_size[1] % 2 == 0:
            img[
                (int(img_size[1] / 2) - spot_size[1]) : (
                    int(img_size[1] / 2) + spot_size[1]
                ),
                (int(img_size[0] / 2) - spot_size[0]) : (
                    int(img_size[0] / 2) + spot_size[0]
                ),
            ] = 255
        elif img_size[0] % 2 == 0 and img_size[1] % 2 != 0:
            img[
                int(img_size[1] / 2),
                (int(img_size[0] / 2) - spot_size[0]) : (
                    int(img_size[0] / 2) + spot_size[0]
                ),
            ] = 255
        if img_size[0] % 2 != 0 and img_size[1] % 2 == 0:
            img[
                int((img_size[1] / 2) - spot_size[1]) : (
                    int(img_size[1] / 2) + spot_size[1]
                ),
                int(img_size[0] / 2),
            ] = 255
        if img_size[0] % 2 != 0 and img_size[1] % 2 != 0:
            img[int(img_size[1] / 2), int(img_size[0] / 2)] = 255
    elif (
        not exact and pattern is not None
    ):  # Generate a grid bitmap and if required save each np.dot as a single image
        increase_x = floor(
            (img_size[0] - pattern[0]) / (pattern[0] - 1)
        )  # Define the number of black pixels between two white one on each row
        offset_x = (
            floor((img_size[0] - ((increase_x * (pattern[0] + 1)) + pattern[0])) / 2)
            + increase_x
        )  # Define the number of black pixel on the left before the first white np.dot
        increase_y = floor(
            (img_size[1] - pattern[1]) / (pattern[1] - 1)
        )  # Define the number of black pixels between two white one on each column
        offset_y = (
            floor((img_size[1] - ((increase_y * (pattern[1] + 1)) + pattern[1])) / 2)
            + increase_y
        )  # Define the number of black pixel on the top before the first white np.dot
        for i in trange(
            offset_y,
            img.shape[0] - offset_y,
            increase_y + 1,
            desc="generating the images row by row",
            leave=False,
        ):
            for j in range(offset_x, img.shape[1] - offset_x, increase_x + 1):
                if not split:
                    img[i, j] = 255
                else:
                    create_folder(file_path, "all")
                    tmp = np.copy(img)
                    tmp[i, j] = 255
                    imwrite(
                        str(file_path / f"bitmap_r{i}_c{j}.png"), tmp
                    )  # Save the image

    if file_path is not None and not split:
        file_path = str(add_extension(str(file_path), ".png"))
        imwrite(file_path, img)  # Save the image

    return img


def load_h5(file_path: Path) -> dict:
    """
    Function that load a .h5 file and return its content as a np np.array. If the .h5 file contains more than one keys it returns a list of np arrays one for each key
    :param file_path: path of the .h5 file
    :return: a data containing as data a np np.array containing the content of the .h5 file
    """

    h5_file = File(str(file_path), "r")  # Open the .h5 file
    keys = list(h5_file.keys())  # Obtain the list of keys contained in the .h5 file

    # Check if the .h5 file has only one key or more than one
    data = {}
    for key in keys:
        data[key] = np.array(
            h5_file[key]
        )  # Load the .h5 content and put it inside a np np.array
    return data


def save_h5(
    data: Union[np.ndarray, dict],
    file_path: Path,
    name: str = None,
    fermat: bool = False,
    compression: bool = True,
) -> None:
    """
    Function to save a transient image into an .h5 file (also perform reshaping [<n_beans>, <n_row>, <col>] -> [<n_row>, <col>, <n_beans>])
    :param data: np.ndarray containing the transient image (only one channel) or dict containing multiple transient images (one for each key)
    :param file_path: path (with name) where to save the file
    :param name: name of the key of the data inside the .h5 file
    :param fermat: if true the data is reshaped to [<n_beans>, <n_row>, <col>]
    :param compression: if true the data is compressed
    """

    file_path = add_extension(
        file_path, ".h5"
    )  # If not already present add the .h5 extension to the file path

    with File(str(file_path), "w") as h5f:  # Create the .h5 file and open it
        # Save the data in the just created .h5 file
        if isinstance(data, dict):
            for key, value in data.items():
                if fermat:
                    value = np.copy(
                        value
                    )  # Copy the np.ndarray in order to avoid overriding
                    value = np.moveaxis(
                        value, 0, -1
                    )  # Move the transient length from index 0 to the last one in the np.ndarray
                    value = np.swapaxes(
                        value, 0, 1
                    )  # Reshape the np.array in order to match the required layout [col, row, beans]
                if compression:
                    h5f.create_dataset(
                        name=key,
                        data=value,
                        compression="gzip",
                        compression_opts=9,
                        shape=value.shape,
                        dtype=np.float32,
                    )
                else:
                    h5f.create_dataset(
                        name=key, data=value, shape=value.shape, dtype=np.float32
                    )
        else:
            if fermat:
                data = np.copy(data)  # Copy the np.ndarray in order to avoid overriding
                data = np.moveaxis(
                    data, 0, -1
                )  # Move the transient length from index 0 to the last one in the np.ndarray
                data = data.np.reshape(
                    [data.shape[1], data.shape[0], data.shape[2]]
                )  # Reshape the np.array in order to match the required layout
            if not name:
                name = (
                    file_path.stem
                )  # If a key name is not provided use the name of the file as key name
            if compression:
                h5f.create_dataset(
                    name=name,
                    data=data,
                    compression="gzip",
                    compression_opts=9,
                    shape=data.shape,
                    dtype=np.float32,
                )
            else:
                h5f.create_dataset(
                    name=name, data=data, shape=data.shape, dtype=np.float32
                )


def plt_3d_surfaces(
    surfaces: list,
    mask: np.ndarray = None,
    x_ticks: tuple = None,
    y_ticks: tuple = None,
    z_ticks: tuple = None,
    legends: list = None,
) -> None:
    """
    Function to plot one or more 3d surfaces given a set of 3D points
    :param surfaces: list of np.ndarray each one containing the (x, y, z) coordinates of each point of a surface -> [np.array(surface1), np.array(surface2), ...]
    :param mask: (if necessary) represents the grid shape that the data in surfaces follows
    :param x_ticks: where to put the ticks on the x-axis
    :param y_ticks: where to put the ticks on the y-axis
    :param z_ticks: where to put the ticks on the z-axis
    :param legends: list containing the label for each surfaces
    """

    fig = plt.figure()  # Create the matplotlib figure
    plt3d = fig.gca(projection="3d")  # Create the 3D plot
    plt3d.set_xlabel("X")  # Ad the label on the x-axis
    plt3d.set_ylabel("Y")  # Ad the label on the y-axis
    plt3d.set_zlabel("Z")  # Ad the label on the z-axis

    for index, graph in enumerate(surfaces):  # For each surface in the surfaces list
        if mask is not None:  # If a mask is provided:
            shape = [
                len(np.unique(np.where(mask != 0)[i])) for i in range(2)
            ]  # Compute the shape of the grid (number of pixel active on the column and row)
            x = np.reshape(
                graph[:, :, 0][mask != 0], [shape[0], shape[1]]
            )  # Remove all the zero values from the 2D x coordinates matrix
            y = np.reshape(
                graph[:, :, 1][mask != 0], [shape[0], shape[1]]
            )  # Remove all the zero values from the 2D y coordinates matrix
            z = np.reshape(
                graph[:, :, 2][mask != 0], [shape[0], shape[1]]
            )  # Remove all the zero values from the 2D z coordinates matrix
        else:
            x = graph[
                :, :, 0
            ]  # Extract from the surfaces' matrix the 2D x coordinates' matrix
            y = graph[
                :, :, 1
            ]  # Extract from the surfaces' matrix the 2D y coordinates' matrix
            z = graph[
                :, :, 2
            ]  # Extract from the surfaces' matrix the 2D z coordinates' matrix
        if legends is not None:  # If a legend is provided
            surf = plt3d.plot_surface(
                x, y, z, label=legends[index]
            )  # Plot the surface with its related label
            # noinspection PyProtectedMember
            surf._facecolors2d = (
                surf._facecolor3d
            )  # Necessary to visualize the legend color
            # noinspection PyProtectedMember
            surf._edgecolors2d = surf._edgecolor3d
        else:
            plt3d.plot_surface(x, y, z)  # Plot the surface

    if legends is not None:  # If a legend is provided
        plt3d.legend()  # Show the legend
    if x_ticks is not None:  # if a list of x-ticks is provided:
        plt3d.set_xticks(x_ticks)  # Set the desired ticks
    if y_ticks is not None:  # if a list of y-ticks is provided:
        plt3d.set_yticks(y_ticks)  # Set the desired ticks
    if z_ticks is not None:  # if a list of y-ticks is provided:
        plt3d.set_zticks(z_ticks)  # Set the desired ticks

    fig.tight_layout()
    plt.show(block=True)


def blender2mitsuba_coord_mapping(
    x_pos: float,
    y_pos: float,
    z_pos: float,
    x_angle: float,
    y_angle: float,
    z_angle: float,
) -> [tuple[float, float, float], tuple[float, float, float]]:
    """
    Function that maps (x, y, z) coordinates and rotations from the blender coordinates system to the mitsuba one
    :param x_pos: x coordinate
    :param y_pos: y coordinate
    :param z_pos: z coordinate
    :param x_angle: rotation over the x-axis
    :param y_angle: rotation over the y-axis
    :param z_angle: rotation over the z-axis
    :return: [(x, y, z) coordinates, (x, y, z) rotations]
    """

    # Compute cosine and sine of each angle (x, y, z)
    cos_x = np.cos(np.radians(x_angle))
    cos_y = np.cos(np.radians(y_angle))
    cos_z = np.cos(np.radians(z_angle))
    sin_x = np.sin(np.radians(x_angle))
    sin_y = np.sin(np.radians(y_angle))
    sin_z = np.sin(np.radians(z_angle))

    # Compute the rotation matrix for each axis
    rot_x = np.array([[1, 0, 0], [0, cos_x, sin_x], [0, -sin_x, cos_x]])
    rot_y = np.array([[cos_y, 0, -sin_y], [0, 1, 0], [sin_y, 0, cos_y]])
    roy_z = np.array([[cos_z, sin_z, 0], [-sin_z, cos_z, 0], [0, 0, 1]])

    # Compute the full rotation matrix multiplying together the "by-axis" rotation matrix
    rot_matrix = np.matmul(rot_x, rot_y)
    rot_matrix = np.matmul(rot_matrix, roy_z)
    rot_matrix = rot_matrix.T

    # Generates the roto-transl matrix combining the rotational matrix with the translation vector and making it a 4x4 matrix
    rototrasl_matrix = np.concatenate(
        (rot_matrix, np.array([[x_pos], [y_pos], [z_pos]])), axis=1
    )
    rototrasl_matrix = np.concatenate(
        (rototrasl_matrix, np.array([[0, 0, 0, 1]])), axis=0
    )

    # Define to additional matrix to compensate for the different axis rotations of mitsuba wrt blender
    init_rot = np.array(
        [
            [-1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, -1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )
    axis_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    # Compute the final transformation matrix combining al the previous ones
    matrix = np.matmul(rototrasl_matrix, init_rot)
    matrix = np.matmul(axis_mat, matrix)
    matrix[np.where(abs(matrix) < 1e-6)] = 0

    # Extract the rotation angles (the mitsuba ones) from the final transformation matrix
    sy = np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    ax = np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2]))
    ay = np.degrees(np.arctan2(-matrix[2, 0], sy))
    az = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))

    # Extract the mitsuba coordinates from the final transformation matrix
    px = matrix[0, 3]
    py = matrix[1, 3]
    pz = matrix[2, 3]

    return (px, py, pz), (ax, ay, az)


def permute_list(data: list, s: int = None) -> list:
    """
    Function that given a list of list compute all the possible permutations of the given data
    :param data: original list
    :param s: random seed (not mandatory)
    :return: list of all the possible permutations of the data list
    """

    if s is not None:
        rnd_seed(s)  # If provided define a random seed
    lst = list(product(*data))  # Permute the list
    shuffle(lst)  # Shuffle it
    return lst


def save_list(data: list, data_path: Path) -> None:
    """
    Function to save a given list to a pickle file
    :param data: list to save
    :param data_path: path where to save it
    """

    with open(str(data_path), "wb") as fp:  # Open the target file
        dump(data, fp)  # Save the data


def load_list(data_path: Path) -> list:
    """
    Function to load a list from a pickle file
    :param data_path: path of the file to load
    :return: the loaded list
    """

    with open(str(data_path), "rb") as fp:  # Open the target file
        return load(fp)  # Load the list from the file and return


def k_matrix_calculator(h_fov: float, img_shape: list) -> np.ndarray:
    """
    Function that compute the k matrix of a camera (matrix of the intrinsic parameters)
    :param h_fov: horizontal FOV (filed of view) of the camera (in np.degrees)
    :param img_shape: image size in pixel [n_pixel_row, n_pixel_col]
    :return: k matrix
    """

    v_fov = 2 * np.degrees(
        np.arctan((img_shape[1] / img_shape[0]) * np.tan(np.radians(h_fov / 2)))
    )
    f_x = (img_shape[0] / 2) / np.tan(np.radians(h_fov / 2))
    f_y = (img_shape[1] / 2) / np.tan(np.radians(v_fov / 2))
    if img_shape[0] % 2 == 0:
        x = (img_shape[0] - 1) / 2
    else:
        x = img_shape[0] / 2
    if img_shape[1] % 2 == 0:
        y = (img_shape[1] - 1) / 2
    else:
        y = img_shape[1] / 2

    return np.array([[f_x, 0, x], [0, f_y, y], [0, 0, 1]], dtype=np.float32)


def balanced_hist_thresholding(b: tuple[np.ndarray, np.ndarray]) -> (float, int):
    """
    Function to compute the balanced threshold of a histogram
    (code from: https://theailearner.com/2019/07/19/balanced-histogram-thresholding/)
    :param b: histogram values [bins, value]
    :return: the value and the bin where the threshold is located
    """

    # Starting point of histogram
    i_s = np.min(np.where(b[0] > 0))
    # End point of histogram
    i_e = np.max(np.where(b[0] > 0))
    # Center of histogram
    i_m = (i_s + i_e) // 2
    # Left side weight
    w_l = np.sum(b[0][0 : i_m + 1])
    # Right side weight
    w_r = np.sum(b[0][i_m + 1 : i_e + 1])
    # Until starting point not equal to endpoint
    while i_s != i_e:
        # If right side is heavier
        if w_r > w_l:
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) >= i_m:
                w_l += b[0][i_m + 1]
                w_r -= b[0][i_m + 1]
                i_m += 1
    return b[1][i_m], i_m


def recursive_otsu(
    hist_data: tuple[np.ndarray, np.ndarray],
    w_0: float,
    w_1: float,
    weighted_sum_0: float,
    weighted_sum_1: object,
    thres: int,
    fn_max: float,
    thresh: int,
    total: object,
) -> (float, int):
    """
    Function to recursively compute the Otsu's threshold
    (code from: https://theailearner.com/2019/07/19/optimum-global-thresholding-using-otsus-method/)
    :param hist_data: histogram data
    :param w_0: left weights
    :param w_1: right weights
    :param weighted_sum_0: weighted np.sum using the left weight
    :param weighted_sum_1: weighted np.sum using the right weight
    :param thres: recursive index
    :param fn_max: current value of the data in the threshold
    :param thresh: current position of the threshold
    :param total: total number of bins
    :return: variance_value, thresh_value
    """

    if thres <= 255:
        # To pass the division by zero warning
        if (
            np.sum(hist_data[0][: thres + 1]) != 0
            and np.sum(hist_data[0][thres + 1 :]) != 0
        ):
            # Update the weights
            w_0 += hist_data[0][thres] / total
            w_1 -= hist_data[0][thres] / total
            # Update the mean
            weighted_sum_0 += hist_data[0][thres] * hist_data[1][thres]
            mean_0 = weighted_sum_0 / np.sum(hist_data[0][: thres + 1])
            weighted_sum_1 -= hist_data[0][thres] * hist_data[1][thres]
            if thres == 255:
                mean_1 = 0.0
            else:
                mean_1 = weighted_sum_1 / np.sum(hist_data[0][thres + 1 :])
            # Calculate the between-class variance
            out = w_0 * w_1 * ((mean_0 - mean_1) ** 2)
            # # if variance maximum, update it
            if out > fn_max:
                fn_max = out
                thresh = thres
        return recursive_otsu(
            hist_data,
            w_0=w_0,
            w_1=w_1,
            weighted_sum_0=weighted_sum_0,
            weighted_sum_1=weighted_sum_1,
            thres=thres + 1,
            fn_max=fn_max,
            thresh=thresh,
            total=total,
        )
    # Stopping condition
    else:
        return fn_max, thresh


def otsu_hist_threshold(hist_data: tuple[np.ndarray, np.ndarray]) -> (float, int):
    """
    Function to compute the otsu threshold of a histogram
    :param hist_data: histogram data
    :return: value and location of the threshold
    """

    # Total pixels in the image
    total = np.sum(hist_data[0])
    # calculate the initial weights and the means
    left, right = np.hsplit(hist_data[0], [0])
    left_bins, right_bins = np.hsplit(hist_data[1], [0])
    # left weights
    w_0 = 0.0
    # Right weights
    w_1 = np.sum(right) / total
    # Left mean
    weighted_sum_0 = 0.0
    # Right mean
    weighted_sum_1 = np.dot(right, right_bins[:-1])

    # Compute the threshold value and position recursively
    _, thresh_value = recursive_otsu(
        hist_data,
        w_0=w_0,
        w_1=w_1,
        weighted_sum_0=weighted_sum_0,
        weighted_sum_1=weighted_sum_1,
        thres=1,
        fn_max=-np.inf,
        thresh=0,
        total=total,
    )
    return hist_data[1][thresh_value], thresh_value
