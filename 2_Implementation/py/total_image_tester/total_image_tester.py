import sys
import getopt
import os
from pathlib import Path
import glob
import time
import warnings
from natsort import natsorted

import numpy
from tqdm import tqdm

import OpenEXR
import imageio
import Imath
import numpy as np


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_input = os.getcwd()  # Argument containing the input directory
    arg_output = "total_image"  # Argument containing the output directory
    arg_help = "{0} -i <input> -o <output>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "input=", "output="])  # Recover the passed options and arguments from the command line (if any)
    except:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = Path(arg)  # Set the input directory
        elif opt in ("-o", "--output"):
            arg_output = arg  # Set the output directory

    print('Input folder:', arg_input)
    print('Output file name:', arg_output)
    print()

    return [arg_input, arg_output]


def create_folder(path):
    """
    Function to create a new folder if not already present or empty it
    :param path: path of the folder to create
    """
    if os.path.exists(path):  # If the folder is already present remove all its child files (code from: https://pynative.com/python-delete-files-and-directories/#h-delete-all-files-from-a-directory)
        for file_name in os.listdir(path):
            file = path / file_name  # Construct full file path
            if os.path.isfile(file):  # If the file is a file remove it
                os.remove(file)
    else:  # Create the required folder if not already present
        os.makedirs(path)


def reed_files(path, extension, reorder=True):
    """
    Function to load all the files in a folder and if needed reoder them using the numbers in the name
    :param reorder: flag to toggle the reorder process
    :param path: folder path
    :param extension: extension of the files to load
    :return: list of file paths
    """
    files = [file_name for file_name in glob.glob(str(path) + "\*." + extension)]  # Load the path of all the files in the input folder with extension .exr
                                                                                   # code from: https://www.delftstack.com/howto/python/python-open-all-files-in-directory/
    if reorder:
        files = natsorted(files, key=lambda y: y.lower())  # Sort alphanumeric in ascending order
                                                           # code from: https://studysection.com/blog/how-to-sort-a-list-in-alphanumeric-order-python/
    return files


def reshape_frame(files):
    """
    Function that load al the exr file in the input folder and reshape it in order to have three matrices, one for each channel containing all the temporal value
    :param files: list off all the file path to analyze
    :return: list containing the reshaped frames for each channel
    """

    print(f"Reshaping {len(files)} frames:")
    start = time.time()  # Compute the execution time

    dw = OpenEXR.InputFile(files[0]).header()['dataWindow']    # Extract the data window dimension from the header of the exr file
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image
    pt = OpenEXR.InputFile(files[0]).header()['channels']['R'].type  # Recover the pixel type from the header

    # Check if the pt is HALF (pt.v == 1) or FLOAT (pt.v == 2)
    if pt.v == Imath.PixelType.HALF:
        np_pt = np.float16  # Define the correspondent value in numpy (float16 or float32)
    elif pt.v == Imath.PixelType.FLOAT:
        np_pt = np.float32

    # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
    frame_A = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_R = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_G = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_B = np.empty([len(files), size[1], size[0]], dtype=np.float32)

    for index, file in enumerate(tqdm(files)):  # For each provided file in the input folder
        img = OpenEXR.InputFile(file)  # Open each file

        (A, R, G, B) = [np.frombuffer(img.channel(Chan, pt), dtype=np_pt) for Chan in ("A", "R", "G", "B")]  # Extract the four channel from each image
        if np_pt == np.float16:
            (A, R, G, B) = [data.reshape(size[1], -1).astype(np.float32) for data in [A, R, G, B]]  # Reshape each vector to match the image size
        else:
            (A, R, G, B) = [data.reshape(size[1], -1) for data in [A, R, G, B]]  # Reshape each vector to match the image size

        # Perform the reshaping saving the results in frame_i for i in A, R,G , B
        for i in range(size[0]):
            frame_A[index, :, i] = A[:, i]
            frame_R[index, :, i] = R[:, i]
            frame_G[index, :, i] = G[:, i]
            frame_B[index, :, i] = B[:, i]

        warnings.filterwarnings('ignore')  # Remove warning about the presence of matrix completely empty (full of nan)

    time.sleep(0.05)  # Wait a bit to allow a proper visualization in the console
    end = time.time()
    print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    return [frame_A, frame_R, frame_G, frame_B]


def img_matrix(channels):
    """
    Function that from the single channel matrices generate a proper image matrix fusing them
    :param channels: list of the 4 channels
    :return: list of image matrix [R, G, B, A]
    """

    print("Generating the image files:")
    start = time.time()  # Compute the execution time

    print(f"Build the {np.shape(frame_A)[2]} image matrices:")
    time.sleep(0.02)
    images = []  # Empty list that will contain all the images
    # Fuse the channels together to obtain a proper [A, R, G, B] image
    for i in tqdm(range(np.shape(frame_A)[2])):
        img = numpy.empty([np.shape(frame_A)[0], np.shape(frame_A)[1], len(channels)], dtype=np.float32)  # Create an empty numpy array of the correct shape

        img[:, :, 0] = frame_R[:, :, i]
        img[:, :, 1] = frame_G[:, :, i]
        img[:, :, 2] = frame_B[:, :, i]
        img[:, :, 3] = frame_A[:, :, i]

        img[np.isnan(frame_A[:, :, i])] = 0  # Remove all the nan value following the Alpha matrix

        images.append(img)

    end = time.time()
    print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def total_img(images, output):
    """
    Function to build the image obtained by sum all the temporal instant of the transient
    :param images: list of all the images
    :param output: output path
    """
    print("Generate the total image = sum over all the time instants")
    start = time.time()  # Compute the execution time

    summed_images = np.nansum(np.asarray(images)[:, :, :, :-1], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel

    # Generate a mask matrix that will contain the number of active beans in each pixel (needed to normalize the image)
    mask = np.zeros([images[0].shape[0], images[0].shape[1]])
    for img in images:
        tmp = np.nansum(img, axis=2)
        mask[tmp.nonzero()] += 1
    mask = np.stack((mask, mask, mask), axis=2)  # make the mask a three layer matrix

    total_image = np.divide(summed_images, mask).astype(np.float32)

    imageio.plugins.freeimage.download()  # Download (if needed the required plugin in order to export .exr file)
    imageio.imwrite(output + ".exr", total_image)  # Save yhe image

    end = time.time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


if __name__ == '__main__':
    arg_input, arg_output = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    files = reed_files(str(arg_input), "exr")  # Load the path of all the files in the input folder with extension .exr

    (frame_A, frame_R, frame_G, frame_B) = reshape_frame(files)  # Reshape the frame in a standard layout

    images = img_matrix([frame_A, frame_R, frame_G, frame_B])  # Create the image files

    total_img(images, arg_output)  # Create and save the total image
