from os.path import dirname

from numpy import sum, linspace, zeros, where, nanmin, nanmax, array, ndarray, copy
from numpy import uint8, float32, reshape, unique
from os import path, listdir, remove, makedirs
from pathlib import Path
from glob import glob
from natsort import natsorted
from cv2 import imwrite, cvtColor
from cv2 import COLOR_RGBA2BGRA, COLOR_RGB2BGR
from matplotlib import pyplot as plt
from h5py import File
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
        offset_x = floor(((img_size[0] - 1) - ((int(img_size[0] / pattern[0]) - 1) * pattern[0])) / 2)  # Define the number of black pixel on the left before the first white dot
        offset_y = floor(((img_size[1] - 1) - ((int(img_size[1] / pattern[1]) - 1) * pattern[1])) / 2)  # Define the number of black pixel on the top before the first white dot
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


def plt_3d_surfaces(surfaces: tuple, mask: ndarray = None, x_ticks: tuple = None, y_ticks: tuple = None, z_ticks: tuple = None, legends: tuple = None) -> None:
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
            shape = [len(unique(where(mask != 0)[i])) for i in range(2)]  # Compute the shape of the grd (number of pixel active on the column and row)
            x = reshape(graph[:, :, 0][mask != 0], [shape[0], shape[1]])  # Remove all the zero values from the 2D x coordinates matrix
            y = reshape(graph[:, :, 1][mask != 0], [shape[0], shape[1]])  # Remove all the zero values from the 2D y coordinates matrix
            z = reshape(graph[:, :, 2][mask != 0], [shape[0], shape[1]])  # Remove all the zero values from the 2D z coordinates matrix
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
