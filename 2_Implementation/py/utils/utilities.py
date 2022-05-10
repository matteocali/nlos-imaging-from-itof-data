# To import the function inside this file add the following line in the beginning of the desired script
# import sys
# sys.path.append("C:\Users\DECaligM\Documents\thesis-nlos-for-itof\2_Implementation\py\utils")
#
# from utilities import *


import os
import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import cv2


def create_folder(path):
    """
    Function to create a new folder if not already present.
    If it already exists, empty it
    :param path: path of the folder to create
    """
    if os.path.exists(path):  # If the folder is already present remove all its child files
                              # (code from: https://pynative.com/python-delete-files-and-directories/#h-delete-all-files-from-a-directory)
        for file_name in os.listdir(path):
            file = path / file_name  # Construct full file path
            if os.path.isfile(file):  # If the file is a file remove it
                os.remove(file)
    else:  # Create the required folder if not already present
        os.makedirs(path)


def reed_files(path, extension, reorder=True):
    """
    Function to load all the files in a folder and if needed reorder them using the numbers in the final part of the name
    :param reorder: flag to toggle the reorder process (default = true)
    :param path: source folder path
    :param extension: extension of the files to load
    :return: list of file paths
    """

    files = [file_name for file_name in glob.glob(str(path) + "\\*." + extension)]  # Load the path of all the files in the input folder with the target extension
                                                                                   # (code from: https://www.delftstack.com/howto/python/python-open-all-files-in-directory/)
    if reorder:
        files = natsorted(files, key=lambda y: y.lower())  # Sort alphanumeric in ascending order
                                                           # (code from: https://studysection.com/blog/how-to-sort-a-list-in-alphanumeric-order-python/)
    return files


def save_png(img, path, alpha):
    """
    Function to save an image as a png
    :param alpha: define if the output will use or not the alpha channel (True/False)
    :param img: image to save
    :param path: path and name
    """
    img = (255 * img).astype(np.uint8)  # Rescale the input value from [0, 1] to [0, 255] and convert them to unit8
    if alpha:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))  # Save the image
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img[:, :, :-1], cv2.COLOR_RGB2BGR))  # Save the image


def save_plt(img, path, alpha):
    """
    Function to save an image as a matplotlib png
    :param alpha: define if the output will use or not the alpha channel (True/False)
    :param img: image to save
    :param path: path and name
    """
    if not alpha:
        plt.imsave(path, img[:, :, :-1])
    else:
        plt.imsave(path, img)


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

    err = np.sum((x.astype("float") - y.astype("float")) ** 2)  # Convert the images to floating point
                                                                # Take the difference between the images by subtracting the pixel intensities
                                                                # Square these difference and sum them up
    err /= float(x.shape[0] * x.shape[1])  # Handles the mean of the MSE

    return round(float(err), 4)


def generate_quadratic_model(min_x, max_x, max_y, precision):
    """
    Function that define the ground truth quadratic model
    :param min_x: minimum measured value of distances
    :param max_x: maximum measured value of distances
    :param max_y: maximum measured value of radiance
    :param precision: number of samples inside the linear space
    :return: x and y value of the quadratic model
    """
    x = np.linspace(min_x, max_x, precision)  # Define the x vector as a linear space
    scaling = max_y * pow(min_x, 2)  # Define the scale factor to center the quadratic model on the measured data
    y = scaling / pow(x, 2)  # Build the quadratic model
    return [x, y]
