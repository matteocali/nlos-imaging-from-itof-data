from numpy import sum, linspace
from numpy import uint8
from os import path, listdir, remove, makedirs
from glob import glob
from natsort import natsorted
from cv2 import imwrite, cvtColor, COLOR_RGBA2BGRA, COLOR_RGB2BGR
from matplotlib import pyplot as plt


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


def create_folder(file_path):
    """
    Function to create a new folder if not already present.
    If it already exists, empty it
    :param file_path: path of the folder to create
    """

    if path.exists(file_path):  # If the folder is already present remove all its child files
                              # (code from: https://pynative.com/python-delete-files-and-directories/#h-delete-all-files-from-a-directory)
        for file_name in listdir(file_path):
            file = file_path / file_name  # Construct full file path
            if path.isfile(file):  # If the file is a file remove it
                remove(file)
    else:  # Create the required folder if not already present
        makedirs(file_path)


def generate_quadratic_model(min_x, max_x, max_y, precision):
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


def compute_mse(x, y):
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


def save_png(img, file_path, alpha):
    """
    Function to save an image as a png
    :param alpha: define if the output will use or not the alpha channel (True/False)
    :param img: image to save
    :param file_path: path and name
    """
    img = (255 * img).astype(uint8)  # Rescale the input value from [0, 1] to [0, 255] and convert them to unit8
    if alpha:
        imwrite(str(file_path), cvtColor(img, COLOR_RGBA2BGRA))  # Save the image
    else:
        imwrite(str(file_path), cvtColor(img[:, :, :-1], COLOR_RGB2BGR))  # Save the image


def save_plt(img, file_path, alpha):
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
