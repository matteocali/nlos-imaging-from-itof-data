import sys
import getopt
import os
from pathlib import Path
import glob
import time
import warnings
from natsort import natsorted
from tqdm import tqdm

import OpenEXR
import Imath
import numpy as np


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_input = os.getcwd()  # Argument containing the input directory
    arg_output = ""  # Argument containing the output directory
    arg_exposure = ""  # Argument containing the used exposure time
    arg_help = "{0} -i <input> -o <output> -p <ptype> -v <video> (default = True) -t <type> (default = plt) -a <alpha> (default = False)".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:e:", ["help", "input=", "output=", "exposure="])  # Recover the passed options and arguments from the command line (if any)
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
            arg_output = arg + ".txt"  # Set the output directory
        elif opt in ("-e", "--exposure"):
            arg_exposure = float(arg)  # Set the exposure time

    if arg_output == "":  # if no output folder is provided define the default one
        arg_output = Path("output")
        if not os.path.exists(arg_output):
            os.makedirs(arg_output)

    print('Input folder:', arg_input)
    print('Output file:', arg_output)
    print()

    return [arg_input, arg_output, arg_exposure]


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


def extract_first_peak(channels):
    """
    Function that extract the position and the value of the first peak in the middle pixel in each channel alpha excluded
    :param channels: list of all the channels
    :return: return a list of two list the firs containing the peak position for each channel the second one the value of each peak
    """

    try:
        max_index_R, max_index_G, max_index_B = [np.nanargmax(channel, axis=2) for channel in channels]  # Find the index of the maximum value in the third dimension
    except ValueError:  # Manage the all NaN situation
        channels_no_nan = []
        for channel in channels:
            temp = channel
            temp[np.isnan(channel)] = 0
            channels_no_nan.append(temp)
        max_index_R, max_index_G, max_index_B = [np.nanargmax(channel, axis=2) for channel in channels_no_nan]
    peak_pos = [data[int(data.shape[0]/2), int(data.shape[1]/2)] for data in [max_index_R, max_index_G, max_index_B]]  # Extract the position of the maximum value in the middle pixel
    peak_values = [channel[int(channel.shape[0]/2), int(channel.shape[1]/2), peak_pos[index]] for index, channel in enumerate(channels)]  # Extract the radiance value in the peak position of the middle pixel
    return [peak_pos, peak_values]


def compute_distance(peak_pos, exposure_time):
    """
    Function that take the position of the peak and the exposure time and compute the measured distance
    :param peak_pos: position of the peak in the transient
    :param exposure_time: size of each time bean
    :return:
    """
    return round(((peak_pos * exposure_time) / 2), 4)  # General equation to compute the distance given the peak_position (p) and the exposure_time (e): ((p + 1) * e) / 2
                                                       # We have to add the correction term, - e/2, to compensate for the rounding in the been size: ((p + 1) * e) / 2 - e/2
                                                       # -> ((p + 1) * e) / 2 - e/2 = (pe + e)/2 - e/2 = pe/2 + e/2 - e/2 = pe/2


if __name__ == '__main__':
    arg_input, arg_output, arg_exposure = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    files = reed_files(str(arg_input), "exr")  # Load the path of all the files in the input folder with extension .exr

    frame_A, frame_R, frame_G, frame_B = reshape_frame(files)  # Reshape the frame in a standard layout

    peak_pos, peak_values = extract_first_peak([frame_R, frame_G, frame_B])  # Extract the peak position and value in the middle pixel

    distances = [compute_distance(pos, arg_exposure) for pos in peak_pos]  # Compute the distance value

    # print the results
    print(f"Computed distance on the pixel ({int(frame_A.shape[0]/2)}, {int(frame_A.shape[1]/2)}) are:")
    print(f"\t- On channel Red: {distances[0]}")
    print(f"\t  peak value: {peak_values[0]}")
    print(f"\t- On channel Green: {distances[1]}")
    print(f"\t  peak value: {peak_values[1]}")
    print(f"\t- On channel Blu: {distances[2]}")
    print(f"\t  peak value: {peak_values[2]}")

    # Save the results to a file
    with open(arg_output, "w") as f:
        f.write(f"Computed distance on the pixel ({int(frame_A.shape[0] / 2)}, {int(frame_A.shape[1] / 2)}) are:\n")
        f.write(f"\t- On channel Red: {distances[0]}\n")
        f.write(f"\t  peak value: {peak_values[0]}\n")
        f.write(f"\t- On channel Green: {distances[1]}\n")
        f.write(f"\t  peak value: {peak_values[1]}\n")
        f.write(f"\t- On channel Blu: {distances[2]}\n")
        f.write(f"\t  peak value: {peak_values[2]}")