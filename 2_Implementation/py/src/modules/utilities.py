from numpy import sum, linspace, zeros, where, nanmin, nanmax, array, ndarray, copy, cos, sin, matmul, sqrt, radians, \
    degrees, arctan2, uint8, float32, reshape, unique, concatenate
from os import path, listdir, remove, makedirs
from pathlib import Path
from glob import glob
from natsort import natsorted
from cv2 import imwrite, cvtColor
from cv2 import COLOR_RGBA2BGRA, COLOR_RGB2BGR
from matplotlib import pyplot as plt
from h5py import File
from math import floor
from pickle import dump, load
from itertools import product
from random import seed as rnd_seed, shuffle, sample as rnd_sample
from tqdm import tqdm, trange
import lxml.etree as et


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

    files = [file_name for file_name in glob(
        str(file_path) + "\\*." + extension)]  # Load the path of all the files in the input folder with the target extension
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
            img[:, :, i] = (img[:, :, i] - min_val) / (
                    max_val - min_val)  # Normalize each image in [0, 1] ignoring the alpha channel
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


def spot_bitmap_gen(img_size: list, file_path: Path = None, spot_size: list = None, exact: bool = False,
                    pattern: tuple = None, split: bool = False) -> ndarray:
    """
    Function that generate a black bitmap image of size img_path with a white square in the middle of size (spot_size * spot_size)
    :param file_path: path where to save the generated image
    :param img_size: size of the desired image [columns * rows]
    :param spot_size: size of the desired white spot [columns * rows]
    :param exact: flag to set white a specific pixel
    :param pattern: list made as follows [x, y] where x represent the interval between white pixel on each row and y on each column
    :param split: if true the grid is splitted dot by dot
    :return generated image
    """

    img = zeros([img_size[1], img_size[0]], dtype=uint8)  # Generate the base black image

    if exact:  # Change the value to white of only the desired pixel
        img[spot_size[1], spot_size[0]] = 255
    elif not exact and spot_size is not None:  # Change the value to white of only the desired central pixels
        spot_size = [int(spot_size[0] / 2), int(spot_size[1] / 2)]
        if img_size[0] % 2 == 0 and img_size[1] % 2 == 0:
            img[(int(img_size[1] / 2) - spot_size[1]):(int(img_size[1] / 2) + spot_size[1]), (int(img_size[0] / 2) - spot_size[0]):(int(img_size[0] / 2) + spot_size[0])] = 255
        elif img_size[0] % 2 == 0 and img_size[1] % 2 != 0:
            img[int(img_size[1] / 2), (int(img_size[0] / 2) - spot_size[0]):(int(img_size[0] / 2) + spot_size[0])] = 255
        if img_size[0] % 2 != 0 and img_size[1] % 2 == 0:
            img[int((img_size[1] / 2) - spot_size[1]):(int(img_size[1] / 2) + spot_size[1]), int(img_size[0] / 2)] = 255
        if img_size[0] % 2 != 0 and img_size[1] % 2 != 0:
            img[int(img_size[1] / 2), int(img_size[0] / 2)] = 255
    elif not exact and pattern is not None:  # Generate a grid bitmap and if required save each dot as a single image
        offset_x = floor(((img_size[0] - 1) - ((int(img_size[0] / pattern[0]) - 1) * pattern[
            0])) / 2)  # Define the number of black pixel on the left before the first white dot
        offset_y = floor(((img_size[1] - 1) - ((int(img_size[1] / pattern[1]) - 1) * pattern[
            1])) / 2)  # Define the number of black pixel on the top before the first white dot
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
    data = data.reshape(
        [data.shape[1], data.shape[2], data.shape[0]])  # Reshape the array in order to match the required layout
    h5f = File(file_path, "w")  # Create the .h5 file and open it
    # Save the ndarray in the just created .h5 file
    if name:
        h5f.create_dataset(name=name,
                           data=data,
                           shape=data.shape,
                           dtype=float32)
    else:
        h5f.create_dataset(name=file_path.split("\\")[-1][-3:],
                           # If a key name is not provided use the name of the name of the file as key name
                           data=data,
                           shape=data.shape,
                           dtype=float32)


def plt_3d_surfaces(surfaces: tuple, mask: ndarray = None, x_ticks: tuple = None, y_ticks: tuple = None,
                    z_ticks: tuple = None, legends: tuple = None) -> None:
    """
    Function to plot one or more 3d surfaces given a set of 3D points
    :param surfaces: list of ndarray each one containing the (x, y, z) coordinates of each point of a surface -> [array(surface1), array(surface2), ...]
    :param mask: (if necessary) represents the grid shape that the data in surfaces follows
    :param x_ticks: where to put the ticks on the x-axis
    :param y_ticks: where to put the ticks on the y-axis
    :param z_ticks: where to put the ticks on the z-axis
    :param legends: list containing the label for each surfaces
    """

    fig = plt.figure()  # Create the matplotlib figure
    plt3d = fig.gca(projection='3d')  # Create the 3D plot
    plt3d.set_xlabel("X")  # Ad the label on the x-axis
    plt3d.set_ylabel("Y")  # Ad the label on the y-axis
    plt3d.set_zlabel("Z")  # Ad the label on the z-axis

    for index, graph in enumerate(surfaces):  # For each surface in the surfaces list
        if mask is not None:  # If a mask is provided:
            shape = [len(unique(where(mask != 0)[i])) for i in
                     range(2)]  # Compute the shape of the grd (number of pixel active on the column and row)
            x = reshape(graph[:, :, 0][mask != 0],
                        [shape[0], shape[1]])  # Remove all the zero values from the 2D x coordinates matrix
            y = reshape(graph[:, :, 1][mask != 0],
                        [shape[0], shape[1]])  # Remove all the zero values from the 2D y coordinates matrix
            z = reshape(graph[:, :, 2][mask != 0],
                        [shape[0], shape[1]])  # Remove all the zero values from the 2D z coordinates matrix
        else:
            x = graph[:, :, 0]  # Extract from the surfaces' matrix the 2D x coordinates' matrix
            y = graph[:, :, 1]  # Extract from the surfaces' matrix the 2D y coordinates' matrix
            z = graph[:, :, 2]  # Extract from the surfaces' matrix the 2D z coordinates' matrix
        if legends is not None:  # If a legend is provided
            surf = plt3d.plot_surface(x, y, z, label=legends[index])  # Plot the surface with its related label
            surf._facecolors2d = surf._facecolor3d  # Necessary to visualize the legend color
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


def blender2mitsuba_coord_mapping(x_pos: float, y_pos: float, z_pos: float, x_angle: float, y_angle: float,
                                  z_angle: float) -> [tuple[float, float, float], tuple[float, float, float]]:
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
    cos_x = cos(radians(x_angle))
    cos_y = cos(radians(y_angle))
    cos_z = cos(radians(z_angle))
    sin_x = sin(radians(x_angle))
    sin_y = sin(radians(y_angle))
    sin_z = sin(radians(z_angle))

    # Compute the rotation matrix for each axis
    rot_x = array([[1, 0, 0], [0, cos_x, sin_x], [0, -sin_x, cos_x]])
    rot_y = array([[cos_y, 0, -sin_y], [0, 1, 0], [sin_y, 0, cos_y]])
    roy_z = array([[cos_z, sin_z, 0], [-sin_z, cos_z, 0], [0, 0, 1]])

    # Compute the full rotation matrix multiplying together the "by-axis" rotation matrix
    rot_matrix = matmul(rot_x, rot_y)
    rot_matrix = matmul(rot_matrix, roy_z)
    rot_matrix = rot_matrix.T

    # Generates the roto-transl matrix combining the rotational matrix with the translation vector and making it a 4x4 matrix
    rototrasl_matrix = concatenate((rot_matrix, array([[x_pos], [y_pos], [z_pos]])), axis=1)
    rototrasl_matrix = concatenate((rototrasl_matrix, array([[0, 0, 0, 1]])), axis=0)

    # Define to additional matrix to compensate for the different axis rotations of mitsuba wrt blender
    init_rot = array([[-1.0000, 0.0000, 0.0000, 0.0000],
                      [0.0000, 1.0000, 0.0000, 0.0000],
                      [0.0000, 0.0000, -1.0000, 0.0000],
                      [0.0000, 0.0000, 0.0000, 1.0000]])
    axis_mat = array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]])

    # Compute the final transformation matrix combining al the previous ones
    matrix = matmul(rototrasl_matrix, init_rot)
    matrix = matmul(axis_mat, matrix)
    matrix[where(abs(matrix) < 1e-6)] = 0

    # Extract the rotation angles (the mitsuba ones) from the final transformation matrix
    sy = sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    ax = degrees(arctan2(matrix[2, 1], matrix[2, 2]))
    ay = degrees(arctan2(-matrix[2, 0], sy))
    az = degrees(arctan2(matrix[1, 0], matrix[0, 0]))

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


def generate_dataset_file(tx_rt_list: list, folder_path: Path, objs: dict) -> None:
    """
    Function that build a .txt file containing all the information about how the dataset has been created
    :param tx_rt_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param folder_path: path of the folder where to save the dataset file
    :param objs: dict containing all the name of the used objects (keys) with the relatives default positions (values)
    """

    if folder_path is not None:
        create_folder(folder_path, ignore="all")  # Create the output folder if not already present

    with open(str(folder_path / "dataset.txt"), "w") as f:  # Open the target .txt file (in ase create it)
        for b_index, batch in enumerate(tx_rt_list):  # Cycle through each batches
            f.write(f"BATCH 0{b_index + 1}:\n")  # Write which batch it is under consideration
            for obj_index, obj in enumerate(batch):  # Cycle through each object
                f.write(f"{list(objs.keys())[obj_index]}:\n")  # Write which object is under consideration
                for data in obj:  # Cycle through each cam_pos/cam_rot/obj_tr_obj_rot combination for the given object
                    f.write(f"\t- camera position -> (x: {data[0][0][0]}, y: {data[0][0][1]}, z: {data[0][0][2]})\n")  # Write the position of the camera
                    f.write(f"\t- camera rotation -> (x: {data[0][1][0]}, y: {data[0][1][1]}, z: {data[0][1][2]})\n")  # Write the rotation of the camera
                    f.write(f"\t- object position -> (x: {list(objs.values())[obj_index][0] + data[1][0][0]}, y: {list(objs.values())[obj_index][1] + data[1][0][1]}, z: {list(objs.values())[obj_index][2] + data[1][0][2]})\n")  # Write the position of the object summing its default position with the applied translation
                    f.write(f"\t- object rotation -> (x: {data[1][1][0]}, y: {data[1][1][1]}, z: {data[1][1][2]})\n")  # Write the rotation of the object
            f.write("\n")


def generate_dataset_list(obj_tr_list: list, obj_full_rot_list: list, obj_partial_rot_list: list, cam_rot_list: list,
                          cam_pos_list: list, n_batches: int, obj_names: list, def_cam_pos: tuple, def_cam_rot: tuple,
                          n_tr_rot_cam: int, n_tr_obj: list, n_rot_obj: list, n_tr_sphere: list,
                          folder_path: Path = None, seed: int = None) -> list[list[list]]:
    """
    Function that generate the list of list containing all the final combinations of camera and object location/translation/rotations
    :param obj_tr_list: List that contains all the possible translations that is granted to an object [[possible x translations], [possible y translations], [possible z translations]]
    :param obj_full_rot_list: List that contains all the possible rotations that is granted to an object [[possible x rotations], [possible y rotations], [possible z rotations]]
    :param obj_partial_rot_list: List that contains all the possible rotations that is granted to an object with no rotations over the z axis [[possible x rotations], [possible y rotations], 0]
    :param cam_rot_list: List that contains all the possible rotations that is granted to the camera [[possible x rotations], [possible y rotations], [possible z rotations]]
    :param cam_pos_list: List that contains all the possible positions that is granted to the camera [[possible x positions], [possible y positions], [possible z positions]]
    :param n_batches: Number of different batches that will be generated
    :param obj_names: List that contains the name of every object that will be considered
    :param def_cam_pos: Default camera position (x, y, z)
    :param def_cam_rot: Default camera orientation (x, y, z)
    :param n_tr_rot_cam: Number of different position and/or rotations that I require for the camera
    :param n_tr_obj: List that contains the number of different translations that I want for each object (sphere excluded) for each batch (len(list) = n_batches)
    :param n_rot_obj: List that contains the number of different rotations that I want for each object (sphere excluded) for each batch (len(list) = n_batches)
    :param n_tr_sphere: List that contains the number of different translations that I want for sphere for each batch (len(list) = n_batches)
    :param folder_path: path of the folder where to store the object translation and rotations permutation needed by blender
    :param seed: random seed
    :return: [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    """

    if folder_path is not None:
        create_folder(folder_path, ignore="all")  # Create the output folder if not already present

    # Compute all the permutations of the parameter lists
    obj_tr_list = permute_list(obj_tr_list, seed)
    obj_full_rot_list = permute_list(obj_full_rot_list, seed)
    obj_partial_rot_list = permute_list(obj_partial_rot_list, seed)
    cam_rot_list = permute_list(cam_rot_list, seed)
    cam_pos_list = permute_list(cam_pos_list, seed)

    rnd_seed(seed)  # Set the random seed

    tr_rot_list = []  # define the list that will contain all the data

    for b_index in trange(n_batches, desc="Batches", leave=True, position=0):  # Cycle through each batch
        tr_rot_batch = []  # Define the list that will contain the data of the current batch

        # Set the cam translation and rotation parameters considering that in some batches the camera position and/or rotation is fixed
        # Translation
        if (b_index + 1) == 1 or (b_index + 1) == 2:
            cam_tr = [(def_cam_pos[0], def_cam_pos[1], def_cam_pos[2])]
        else:
            cam_tr = rnd_sample(cam_pos_list, n_tr_rot_cam)
        # Rotations
        if (b_index + 1) == 1 or (b_index + 1) == 3:
            cam_rot = [(def_cam_rot[0], def_cam_rot[1], def_cam_rot[2])]
        else:
            cam_rot = rnd_sample(cam_rot_list, n_tr_rot_cam)

        # Sample the correct number of rotation and translation couple (at random) of the camera considering that in some batches the camera position and/or rotation is fixed
        if b_index == 0:
            cam_tr_rot = [(cam_tr[0], cam_rot[0])]
        elif b_index == 1:
            cam_tr_rot = rnd_sample(permute_list([cam_tr, cam_rot], seed), len(cam_rot))
        elif b_index == 2:
            cam_tr_rot = rnd_sample(permute_list([cam_tr, cam_rot], seed), len(cam_tr))
        else:
            cam_tr_rot = rnd_sample(permute_list([cam_tr, cam_rot], seed), len(cam_tr) + len(cam_rot))

        for name in tqdm(obj_names, desc="Objects", leave=False, position=1):  # Cycle through all the objects
            # Set the object translation and rotations parameter
            if folder_path is not None and (folder_path / f"obj_tr_rot_batch_0{b_index + 1}_{name.lower()}").is_file():
                obj_tr_rot = load_list(folder_path / f"obj_tr_rot_batch_0{b_index + 1}_{name.lower()}")
            else:
                # Translations
                if name != "Sphere":
                    obj_tr = rnd_sample(obj_tr_list, n_tr_obj[b_index])
                else:
                    obj_tr = rnd_sample(obj_tr_list, n_tr_sphere[b_index])
                # Rotations
                if name == "Cube" or name == "Parallelepiped" or name == "Concave plane" or name == "Cube + sphere":
                    obj_rot = rnd_sample(obj_full_rot_list, n_rot_obj[b_index])
                elif name == "Cone" or name == "Cylinder" or name == "Cylinder + cone" or name == "Sphere + cone":
                    obj_rot = rnd_sample(obj_partial_rot_list, n_rot_obj[b_index])
                else:
                    obj_rot = [(0, 0, 0)]

                # Sample the correct number of rotation and translation couple (at random) of the object
                if name != "Sphere":
                    obj_tr_rot = rnd_sample(permute_list([obj_tr, obj_rot], seed), len(obj_tr) + len(obj_rot))
                else:
                    obj_tr_rot = rnd_sample(permute_list([obj_tr, obj_rot], seed), len(obj_tr))

                if folder_path is not None:
                    save_list(obj_tr_rot, folder_path / f"obj_tr_rot_batch0{b_index + 1}_{name.lower()}")  # Save the list of the object translation and rotations

            # Sample the correct number of rotation and translation couple (at random) for both the object and the camera
            if (b_index + 1) == 1:
                tr_rot_batch.append(rnd_sample(permute_list([cam_tr_rot, obj_tr_rot], seed), len(obj_tr_rot)))
            else:
                tr_rot_batch.append(rnd_sample(permute_list([cam_tr_rot, obj_tr_rot], seed), int(len(cam_tr_rot) / len(obj_names)) + len(obj_tr_rot)))

        tr_rot_list.append(tr_rot_batch)

    return tr_rot_list


def generate_dataset_xml(tr_rot_list: list, template: Path, folder_path: Path, objs: dict) -> None:
    """
    Function that given the template.xml file and all the chosen position?translation?rotation combinations generate the correspondent .xml files
    :param tr_rot_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param template: template file (.xml)
    :param folder_path: path of the output folder
    :param obj_names: List that contains the name of every object that will be considered
    """

    create_folder(folder_path)  # Create the output folder if not already present

    for b_index, batch in tqdm(enumerate(tr_rot_list), desc="Batches", leave=True, position=0):  # Cycle through each batch
        create_folder((folder_path / f"batch0{b_index + 1}"))  # Create the batch folder if not already present

        for o_index, obj in tqdm(enumerate(batch), desc="Objects", leave=False, position=1):  # Cycle through each object
            for elm in tqdm(obj, desc="File", leave=False, position=2):  # Cycle through each position/translation/rotation combination
                name = list(objs.keys())[o_index]  # Extract the object name
                cam_pos = [elm[0][0][0], elm[0][0][1], elm[0][0][2]]  # Extract the camera position
                cam_rot = [elm[0][1][0], elm[0][1][1], elm[0][1][2]]  # Extract the camera rotation
                obj_tr = [elm[1][0][0], elm[1][0][1], elm[1][0][2]]  # Extract the object translation
                obj_rot = [elm[1][1][0], elm[1][1][1], elm[1][1][2]]  # Extract the object rotation

                obj_file_name = f"{name}_batch0{b_index + 1}_tr({obj_tr[0]}_{obj_tr[1]}_{obj_tr[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                file_name = f"transient_nlos_{name.lower()}_[cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_obj_pos_({objs[name][0] + obj_tr[0]}_{objs[name][1] + obj_tr[1]}_{objs[name][2] + obj_tr[2]})_obj_rot_({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})].xml"  # Set the output file name in a way that contains all the relevant info

                # Convert camera position and rotation from blender to mitsuba coordinates system
                cam_pos, cam_rot = blender2mitsuba_coord_mapping(cam_pos[0], cam_pos[1], cam_pos[2], cam_rot[0], cam_rot[1], cam_rot[2])

                # Modify the template inserting the desired data
                # (code from: https://stackoverflow.com/questions/37868881/how-to-search-and-replace-text-in-an-xml-file-using-python)
                with open(str(template), encoding="utf8") as f:
                    tree = et.parse(f)
                    root = tree.getroot()

                    for elem in root.getiterator():
                        try:
                            if elem.attrib["value"] == "obj_name":
                                elem.attrib["value"] = str(obj_file_name)
                            elif elem.attrib["value"] == "t_cam_x":
                                elem.attrib["value"] = str(cam_pos[0])
                            elif elem.attrib["value"] == "t_cam_y":
                                elem.attrib["value"] = str(cam_pos[1])
                            elif elem.attrib["value"] == "t_cam_z":
                                elem.attrib["value"] = str(cam_pos[2])
                            elif elem.attrib["value"] == "r_cam_x":
                                elem.attrib["value"] = str(cam_rot[0])
                            elif elem.attrib["value"] == "r_cam_y":
                                elem.attrib["value"] = str(cam_rot[1])
                            elif elem.attrib["value"] == "r_cam_z":
                                elem.attrib["value"] = str(cam_rot[2])
                        except KeyError:
                            pass
                tree.write(str(folder_path / f"batch0{b_index + 1}" / file_name.replace(" ", "_")), xml_declaration=True, method="xml", encoding="utf8")
